"""Generador de datos sintéticos calibrados para oficinas IoT.

Modela cuatro oficinas (Madrid, Sevilla, Barcelona, Oviedo) con dos años de
histórico horario, incluyendo:

    * Temperatura y humedad exteriores estacionales por climatología.
    * Ocupación según horario laboral, día de la semana y festivos.
    * Termodinámica simplificada del edificio (T interior).
    * Demanda eléctrica: HVAC, iluminación, equipos y carga base.
    * Sensores IoT internos (T, HR, CO2).
    * Inyección controlada de anomalías para validar detectores.

El modelo es paramétrico y se inspira en patrones reales de datasets públicos
(ASHRAE Great Energy Predictor III, Building Data Genome Project) sin reusar
datos directamente, lo que permite calibrar cada sede independientemente.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import holidays
import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.config import SEDES, SYNTHETIC_DIR, settings

# ---------------------------------------------------------------------------
# Configuración climática por ciudad
# ---------------------------------------------------------------------------
# Parámetros de temperatura exterior (°C):
#   t_media       media anual
#   t_amplitud    amplitud estacional (verano-invierno)/2
#   t_dia_amp     amplitud diaria
#   t_ruido       sigma del ruido gaussiano horario
#   hr_media      humedad relativa media (%)
#   hr_ruido      ruido humedad
CLIMA_PARAMS: dict[str, dict[str, float]] = {
    "continental": {
        "t_media": 15.0,
        "t_amplitud": 12.0,
        "t_dia_amp": 6.0,
        "t_ruido": 1.5,
        "hr_media": 55.0,
        "hr_ruido": 8.0,
    },
    "mediterraneo_calido": {
        "t_media": 19.0,
        "t_amplitud": 11.0,
        "t_dia_amp": 7.0,
        "t_ruido": 1.5,
        "hr_media": 60.0,
        "hr_ruido": 8.0,
    },
    "mediterraneo": {
        "t_media": 17.0,
        "t_amplitud": 9.0,
        "t_dia_amp": 5.0,
        "t_ruido": 1.3,
        "hr_media": 70.0,
        "hr_ruido": 6.0,
    },
    "oceanico": {
        "t_media": 13.5,
        "t_amplitud": 7.0,
        "t_dia_amp": 4.0,
        "t_ruido": 1.2,
        "hr_media": 78.0,
        "hr_ruido": 7.0,
    },
}


# ---------------------------------------------------------------------------
# Modelo de ocupación
# ---------------------------------------------------------------------------
def occupancy_factor(ts: pd.Timestamp, festivos: set[date]) -> float:
    """Devuelve factor de ocupación en [0, 1] para un instante dado.

    Refleja horario laboral 8-19h con pico ~10-13 y 15-18, fuera = 0,
    fines de semana y festivos = 0 con cola residual de mantenimiento.
    """
    if ts.date() in festivos or ts.weekday() >= 5:
        return 0.05  # personal de mantenimiento puntual

    h = ts.hour
    if h < 7 or h >= 20:
        return 0.02
    if 7 <= h < 9:
        return 0.35 + (h - 7) * 0.25            # rampa entrada
    if 9 <= h < 14:
        return 0.85 + np.sin((h - 9) / 5 * np.pi) * 0.1
    if 14 <= h < 16:
        return 0.55                               # comida
    if 16 <= h < 19:
        return 0.80
    if 19 <= h < 20:
        return 0.30                               # rampa salida
    return 0.02


# ---------------------------------------------------------------------------
# Configuración de la simulación
# ---------------------------------------------------------------------------
@dataclass
class SimulationConfig:
    """Parámetros físicos y de negocio de la simulación."""

    # Setpoints de confort (°C)
    t_setpoint_invierno: float = 21.0
    t_setpoint_verano: float = 24.0

    # Termodinámica del edificio
    k_envoltura: float = 0.12          # transferencia con exterior (1/h)
    aporte_ocupacion: float = 4.0      # °C aporte por ocupación plena
    aporte_solar_max: float = 2.5      # °C aporte solar mediodía

    # Control del HVAC
    banda_confort: float = 1.0         # °C de tolerancia antes de actuar
    capacidad_termica_hora: float = 3.0  # °C/h que corrige el equipo a plena potencia
    modulacion_minima: float = 0.15    # potencia mínima del equipo al arrancar
    ocupacion_en_servicio: float = 0.15  # umbral de ocupación programada para operar
    t_antihielo: float = 12.0          # °C: protección nocturna en invierno
    modulacion_antihielo: float = 0.30

    # Cargas eléctricas (kW)
    carga_base: float = 5.0            # servidores, neveras, standby
    hvac_pot_nominal_per_m2: float = 0.06   # kW/m² nominal del clima
    hvac_delta_t_nominal: float = 15.0      # ΔT (°C) al que se alcanza la potencia nominal
    iluminacion_max_per_m2: float = 0.012   # kW/m²
    equipos_max_per_persona: float = 0.25   # kW/persona

    # CO2 (ppm)
    co2_base: float = 420.0
    co2_aporte_persona: float = 8.0    # ppm por % ocupación máx

    # Anomalías
    prob_anomalia_dia: float = 0.04    # ~4% de los días con alguna anomalía
    modulacion_atascado: float = 0.80  # potencia a la que se queda un HVAC atascado
    seed: int = 42


# ---------------------------------------------------------------------------
# Simulador de una sede
# ---------------------------------------------------------------------------
class OfficeSimulator:
    """Genera dos años de datos horarios para una sede."""

    def __init__(
        self,
        sede_id: str,
        sede_meta: dict,
        start: date,
        end: date,
        cfg: SimulationConfig,
        meteo_real: pd.DataFrame | None = None,
    ) -> None:
        """
        Args:
            meteo_real: serie horaria con `temperatura_exterior_c` y
                `humedad_exterior_pct` observadas. Si se aporta, **gobierna la
                física del edificio**: la temperatura interior y el consumo de
                climatización se derivan de ella. La meteorología sintética se
                conserva en las columnas `*_sintetica` para poder compararlas.

                Es imprescindible pasarla aquí y no sustituir la columna después
                de simular: hacerlo a posteriori dejaba cada fila declarando una
                temperatura exterior que no era la que había generado su propio
                consumo, con un desajuste medio de 3,8 °C.
        """
        self.meteo_real = meteo_real
        self.sede_id = sede_id
        self.sede = sede_meta
        self.start = start
        self.end = end
        self.cfg = cfg
        # `hash()` sobre str está aleatorizado por proceso (PYTHONHASHSEED), así que
        # usar hash() aquí rompería la reproducibilidad: el mismo seed daría datos
        # distintos en cada ejecución. crc32 es estable entre procesos y máquinas.
        desplazamiento = zlib.crc32(sede_id.encode("utf-8")) % 1000
        self.rng = np.random.default_rng(cfg.seed + desplazamiento)

        self.clima = CLIMA_PARAMS[sede_meta["clima"]]
        self.festivos = self._build_holidays()

    def _build_holidays(self) -> set[date]:
        prov_map = {
            "madrid": "MD",
            "sevilla": "AN",
            "barcelona": "CT",
            "oviedo": "AS",
        }
        prov = prov_map.get(self.sede_id, "MD")
        years = list(range(self.start.year, self.end.year + 1))
        es = holidays.country_holidays("ES", subdiv=prov, years=years)
        return set(es.keys())

    # ---------------- Bloques físicos -----------------
    def _temperatura_exterior(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Genera serie horaria de T exterior con estacionalidad anual y diaria."""
        # Día del año normalizado a [0, 2π], pico verano ~día 200
        doy = idx.dayofyear.values + idx.hour.values / 24.0
        anual = -np.cos(2 * np.pi * (doy - 20) / 365.25)
        # Ciclo diario: mínimo ~6h, máximo ~16h
        hourly = -np.cos(2 * np.pi * (idx.hour.values - 6) / 24.0)

        t = (
            self.clima["t_media"]
            + self.clima["t_amplitud"] * anual
            + self.clima["t_dia_amp"] * hourly * 0.5
            + self.rng.normal(0, self.clima["t_ruido"], len(idx))
        )
        return t

    def _humedad_exterior(self, t_ext: np.ndarray) -> np.ndarray:
        """HR exterior anticorrelada con temperatura."""
        hr = (
            self.clima["hr_media"]
            - 0.6 * (t_ext - self.clima["t_media"])
            + self.rng.normal(0, self.clima["hr_ruido"], len(t_ext))
        )
        return np.clip(hr, 15, 100)

    def _resolver_meteo(
        self,
        idx: pd.DatetimeIndex,
        t_sintetica: np.ndarray,
        hr_sintetica: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Elige la meteorología que gobierna la simulación.

        Si se aportó observación real, esa es la que manda; la sintética queda
        solo como respaldo para las horas sin dato. La radiación solar sigue
        siendo sintética, porque la observación diaria gratuita de AEMET no la
        incluye.
        """
        if self.meteo_real is None or self.meteo_real.empty:
            return t_sintetica, hr_sintetica

        real = self.meteo_real.reindex(idx)
        t_real = real.get("temperatura_exterior_c")
        hr_real = real.get("humedad_exterior_pct")

        t_ext = t_sintetica if t_real is None else t_real.to_numpy(dtype=float)
        hr_ext = hr_sintetica if hr_real is None else hr_real.to_numpy(dtype=float)

        # Respaldo sintético en los huecos de la estación
        t_ext = np.where(np.isnan(t_ext), t_sintetica, t_ext)
        hr_ext = np.clip(np.where(np.isnan(hr_ext), hr_sintetica, hr_ext), 15, 100)
        return t_ext, hr_ext

    def _radiacion_solar(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Radiación solar relativa [0, 1]. Cero por la noche."""
        doy = idx.dayofyear.values
        h = idx.hour.values + idx.minute.values / 60.0

        # Duración del día (estimación simple latitud media España)
        amanecer = 6 + 1.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)
        anochecer = 18 - 1.5 * np.cos(2 * np.pi * (doy - 172) / 365.25)

        rad = np.zeros(len(idx))
        mask = (h >= amanecer) & (h <= anochecer)
        # Curva sinusoidal entre amanecer y anochecer
        rad[mask] = np.sin(
            np.pi * (h[mask] - amanecer[mask]) / (anochecer[mask] - amanecer[mask])
        )
        # Variabilidad por nubes
        rad = rad * (1 - self.rng.beta(2, 8, len(idx)))
        return np.clip(rad, 0, 1)

    def _ocupacion_programada(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Ocupación teórica según horario, sin ruido.

        Es la señal que usa el control del HVAC: un BMS real opera contra el
        calendario de la oficina, no contra la lectura instantánea del sensor.
        """
        return np.array([occupancy_factor(ts, self.festivos) for ts in idx])

    def _ocupacion(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Serie de ocupación observada [0, 1] = programada + ruido residual."""
        noise = self.rng.normal(0, 0.05, len(idx))
        return np.clip(self._ocupacion_programada(idx) + noise, 0, 1)

    def _setpoint_estacional(self, idx: pd.DatetimeIndex) -> np.ndarray:
        """Consigna de confort por hora según el mes real del calendario.

        Invierno (dic-mar) 21 °C, verano (jun-sep) 24 °C y 22,5 °C en las
        estaciones intermedias, siguiendo el criterio del RITE para oficinas.
        """
        meses = idx.month.values
        t_obj = np.full(len(idx), 22.5)
        t_obj[np.isin(meses, (6, 7, 8, 9))] = self.cfg.t_setpoint_verano
        t_obj[np.isin(meses, (12, 1, 2, 3))] = self.cfg.t_setpoint_invierno
        return t_obj

    def _temperatura_interior(
        self,
        idx: pd.DatetimeIndex,
        t_ext: np.ndarray,
        ocup_prog: np.ndarray,
        rad: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Modelo termodinámico con control proporcional del HVAC.

        Devuelve `(t_int, hvac_mod)` donde `hvac_mod ∈ [-1, 1]` es la fracción
        de potencia nominal demandada: negativa en refrigeración, positiva en
        calefacción y 0 con el equipo parado.

        El control replica el funcionamiento de un BMS real:

          * Opera contra la **ocupación programada** (calendario de oficina), no
            contra la lectura ruidosa del sensor: el equipo no arranca de
            madrugada porque el sensor fluctúe.
          * **Modula** la potencia en proporción al error de temperatura, en vez
            de conmutar todo/nada. Es lo que deja margen a la optimización de la
            fase 7 para desplazar carga sin perder confort.
          * Fuera de horario solo actúa la protección antihielo.
        """
        n = len(t_ext)
        t_int = np.zeros(n)
        t_int[0] = self.cfg.t_setpoint_invierno
        hvac = np.zeros(n)

        t_obj_serie = self._setpoint_estacional(idx)
        banda = self.cfg.banda_confort
        capacidad = self.cfg.capacidad_termica_hora
        mod_min = self.cfg.modulacion_minima

        for i in range(1, n):
            t_obj = t_obj_serie[i]

            # Drift natural: tiende al exterior, aporte ocupación y solar
            drift = (
                self.cfg.k_envoltura * (t_ext[i] - t_int[i - 1])
                + self.cfg.aporte_ocupacion * ocup_prog[i] * 0.1
                + self.cfg.aporte_solar_max * rad[i] * 0.08
            )
            t_natural = t_int[i - 1] + drift
            error = t_obj - t_natural

            if ocup_prog[i] > self.cfg.ocupacion_en_servicio:
                if error > banda:      # hace frío: calefacción modulada
                    mod = min((error - banda) / capacidad, 1.0)
                    hvac[i] = max(mod, mod_min)
                elif error < -banda:   # hace calor: refrigeración modulada
                    mod = min((-error - banda) / capacidad, 1.0)
                    hvac[i] = -max(mod, mod_min)
                else:                  # dentro de banda: equipo parado
                    hvac[i] = 0.0
            elif t_natural < self.cfg.t_antihielo:
                # Edificio cerrado: solo protección antihielo
                hvac[i] = self.cfg.modulacion_antihielo
            else:
                hvac[i] = 0.0

            # Respuesta térmica proporcional a la potencia aplicada
            t_int[i] = t_natural + hvac[i] * capacidad + self.rng.normal(0, 0.15)

        return t_int, hvac

    def _consumo_electrico(
        self,
        t_int: np.ndarray,
        t_ext: np.ndarray,
        hvac: np.ndarray,
        ocup: np.ndarray,
        rad: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Calcula desglose de consumo eléctrico (kWh por hora)."""
        sup = self.sede["superficie_m2"]
        ocup_max = self.sede["ocupacion_max"]

        # HVAC: potencia = modulación × potencia nominal × dificultad térmica.
        # La dificultad crece con el salto respecto al exterior y satura en 1
        # al alcanzar el ΔT nominal, de modo que el equipo nunca supera su placa.
        delta_t = np.abs(t_int - t_ext)
        dificultad = np.clip(delta_t / self.cfg.hvac_delta_t_nominal, 0.25, 1.0)
        carga_hvac = (
            np.abs(hvac) * self.cfg.hvac_pot_nominal_per_m2 * sup * dificultad
        )

        # Iluminación: mayor cuando hay ocupación y poca radiación
        carga_ilum = (
            ocup
            * self.cfg.iluminacion_max_per_m2
            * sup
            * np.clip(1 - rad, 0.2, 1.0)
        )

        # Equipos: por persona
        carga_equipos = ocup * ocup_max * self.cfg.equipos_max_per_persona

        # Base
        carga_base = np.full(len(t_int), self.cfg.carga_base)

        total = carga_hvac + carga_ilum + carga_equipos + carga_base
        # Ruido del 2%
        total = total * (1 + self.rng.normal(0, 0.02, len(total)))

        return {
            "consumo_hvac_kwh": carga_hvac,
            "consumo_iluminacion_kwh": carga_ilum,
            "consumo_equipos_kwh": carga_equipos,
            "consumo_base_kwh": carga_base,
            "consumo_total_kwh": np.clip(total, 0, None),
        }

    def _co2(self, ocup: np.ndarray) -> np.ndarray:
        """Modelo simple de CO2 interior basado en ocupación."""
        ocup_max = self.sede["ocupacion_max"]
        co2 = (
            self.cfg.co2_base
            + ocup * ocup_max * self.cfg.co2_aporte_persona
            + self.rng.normal(0, 15, len(ocup))
        )
        return np.clip(co2, 380, 2500)

    def _humedad_interior(self, hr_ext: np.ndarray, ocup: np.ndarray) -> np.ndarray:
        """HR interior amortiguada respecto al exterior + aporte de ocupación."""
        hr = 0.4 * hr_ext + 0.6 * 50.0 + 5.0 * ocup + self.rng.normal(0, 3, len(ocup))
        return np.clip(hr, 20, 95)

    # ---------------- Anomalías -----------------
    def _inyectar_anomalias(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inyecta anomalías controladas en la serie y marca con flag."""
        df = df.copy()
        df["es_anomalia"] = False
        df["tipo_anomalia"] = ""

        n_dias = (self.end - self.start).days
        n_anomalias = int(n_dias * self.cfg.prob_anomalia_dia)
        if n_anomalias == 0:
            return df

        tipos = [
            "HVAC_STUCK_ON",
            "EQUIPMENT_LEAK",
            "SENSOR_FROZEN",
            "CONSUMPTION_SPIKE",
        ]

        for _ in range(n_anomalias):
            i_start = self.rng.integers(0, len(df) - 24)
            dur = int(self.rng.integers(3, 12))
            tipo = self.rng.choice(tipos)
            sl = slice(i_start, i_start + dur)

            if tipo == "HVAC_STUCK_ON":
                # Un equipo atascado se queda funcionando aunque no haga falta.
                # Multiplicar el consumo previo NO reproduce eso: si el HVAC
                # estaba parado, 0 × 3,5 sigue siendo 0 y la avería no dejaba
                # ninguna huella en los datos pese a quedar etiquetada.
                nominal = self.cfg.hvac_pot_nominal_per_m2 * self.sede["superficie_m2"]
                atascado = self.cfg.modulacion_atascado * nominal
                df.loc[df.index[sl], "consumo_hvac_kwh"] = np.maximum(
                    df.loc[df.index[sl], "consumo_hvac_kwh"], atascado
                )
                df.loc[df.index[sl], "hvac_estado"] = self.cfg.modulacion_atascado
            elif tipo == "EQUIPMENT_LEAK":
                df.loc[df.index[sl], "consumo_equipos_kwh"] += 6.0
            elif tipo == "SENSOR_FROZEN":
                valor = df.iloc[i_start]["temperatura_interior_c"]
                df.loc[df.index[sl], "temperatura_interior_c"] = valor
            elif tipo == "CONSUMPTION_SPIKE":
                df.loc[df.index[sl], "consumo_total_kwh"] *= 2.2

            df.loc[df.index[sl], "es_anomalia"] = True
            df.loc[df.index[sl], "tipo_anomalia"] = tipo

        # Recalcular total si HVAC/equipos cambiaron
        cols_parc = [
            "consumo_hvac_kwh",
            "consumo_iluminacion_kwh",
            "consumo_equipos_kwh",
            "consumo_base_kwh",
        ]
        df["consumo_total_kwh"] = np.where(
            df["tipo_anomalia"] == "CONSUMPTION_SPIKE",
            df["consumo_total_kwh"],
            df[cols_parc].sum(axis=1),
        )
        return df

    # ---------------- Orquestación -----------------
    def generate(self) -> pd.DataFrame:
        """Devuelve DataFrame horario con todas las variables de la sede."""
        idx = pd.date_range(
            start=self.start,
            end=pd.Timestamp(self.end) + pd.Timedelta(hours=23),
            freq="h",
            tz="Europe/Madrid",
        )

        t_ext_sintetica = self._temperatura_exterior(idx)
        hr_ext_sintetica = self._humedad_exterior(t_ext_sintetica)
        t_ext, hr_ext = self._resolver_meteo(idx, t_ext_sintetica, hr_ext_sintetica)
        rad = self._radiacion_solar(idx)
        ocup_prog = self._ocupacion_programada(idx)
        ocup = np.clip(ocup_prog + self.rng.normal(0, 0.05, len(idx)), 0, 1)
        t_int, hvac = self._temperatura_interior(idx, t_ext, ocup_prog, rad)
        consumos = self._consumo_electrico(t_int, t_ext, hvac, ocup, rad)
        co2 = self._co2(ocup)
        hr_int = self._humedad_interior(hr_ext, ocup)

        df = pd.DataFrame(
            {
                "timestamp": idx,
                "sede": self.sede_id,
                "sede_nombre": self.sede["nombre"],
                "temperatura_exterior_c": t_ext,
                "humedad_exterior_pct": hr_ext,
                # La meteorología sintética se conserva para poder contrastarla
                # con la observada en la memoria
                "temperatura_exterior_c_sintetica": t_ext_sintetica,
                "humedad_exterior_pct_sintetica": hr_ext_sintetica,
                "radiacion_solar_rel": rad,
                "ocupacion_rel": ocup,
                "temperatura_interior_c": t_int,
                "humedad_interior_pct": hr_int,
                "co2_ppm": co2,
                "hvac_estado": hvac,
                **consumos,
                "es_festivo": [d.date() in self.festivos for d in idx],
                "es_finde": [d.weekday() >= 5 for d in idx],
            }
        )

        df = self._inyectar_anomalias(df)
        return df


# ---------------------------------------------------------------------------
# Orquestador multi-sede
# ---------------------------------------------------------------------------
def generate_all_sedes(
    start: date | None = None,
    end: date | None = None,
    cfg: SimulationConfig | None = None,
    output_dir: Path | None = None,
    usar_meteo_real: bool = True,
) -> dict[str, pd.DataFrame]:
    """Genera los datasets de las cuatro sedes y los guarda en disco.

    Args:
        usar_meteo_real: si hay CSV de AEMET descargados, la temperatura y la
            humedad observadas gobiernan la simulación, de modo que el consumo
            de climatización deriva de la meteorología real. Sin ellos, o con
            este parámetro desactivado, se emplea la meteorología sintética.
    """
    start = start or settings.synthetic_start_date
    end = end or settings.synthetic_end_date
    cfg = cfg or SimulationConfig(seed=settings.synthetic_seed)
    output_dir = output_dir or SYNTHETIC_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    resultados: dict[str, pd.DataFrame] = {}
    frames = []

    idx_completo = pd.date_range(
        start=start,
        end=pd.Timestamp(end) + pd.Timedelta(hours=23),
        freq="h",
        tz="Europe/Madrid",
    )

    for sede_id, sede_meta in SEDES.items():
        logger.info(f"Generando sede: {sede_meta['nombre']} ({sede_id})")

        meteo = None
        if usar_meteo_real:
            from tfm_energia.data.meteo_horaria import meteo_real_horaria

            meteo = meteo_real_horaria(sede_id, idx_completo)
            if meteo.empty:
                meteo = None

        sim = OfficeSimulator(sede_id, sede_meta, start, end, cfg, meteo_real=meteo)
        df = sim.generate()
        resultados[sede_id] = df

        # CSV individual
        csv_path = output_dir / f"sede_{sede_id}.csv"
        df.to_csv(csv_path, index=False)
        # Parquet individual (más eficiente)
        parquet_path = output_dir / f"sede_{sede_id}.parquet"
        df.to_parquet(parquet_path, index=False)
        logger.info(
            f"  → {len(df):,} registros | "
            f"anomalías: {df['es_anomalia'].sum()} | "
            f"{csv_path.name}"
        )
        frames.append(df)

    # Dataset consolidado
    consolidado = pd.concat(frames, ignore_index=True)
    consolidado.to_parquet(output_dir / "consolidado.parquet", index=False)
    logger.info(f"Consolidado: {len(consolidado):,} registros totales")

    return resultados


def to_iot_json_payload(df: pd.DataFrame) -> list[dict]:
    """Convierte un DataFrame a formato JSON tipo evento IoT (uno por sensor).

    Esto simula lo que enviaría un data logger real a un topic Kafka,
    para ser consumido por StreamSets.
    """
    payloads = []
    for row in df.itertuples(index=False):
        ts = row.timestamp.isoformat()
        # Sensor 1: ambiente exterior
        payloads.append(
            {
                "sensor_id": f"{row.sede}-S1",
                "tipo": "ambiente_exterior",
                "timestamp": ts,
                "sede": row.sede,
                "temperatura_c": float(row.temperatura_exterior_c),
                "humedad_pct": float(row.humedad_exterior_pct),
            }
        )
        # Sensor 2: ambiente interior
        payloads.append(
            {
                "sensor_id": f"{row.sede}-S2",
                "tipo": "ambiente_interior",
                "timestamp": ts,
                "sede": row.sede,
                "temperatura_c": float(row.temperatura_interior_c),
                "humedad_pct": float(row.humedad_interior_pct),
                "co2_ppm": float(row.co2_ppm),
            }
        )
        # Sensor 3: consumo eléctrico
        payloads.append(
            {
                "sensor_id": f"{row.sede}-S3",
                "tipo": "consumo_electrico",
                "timestamp": ts,
                "sede": row.sede,
                "consumo_total_kwh": float(row.consumo_total_kwh),
                "consumo_hvac_kwh": float(row.consumo_hvac_kwh),
                "consumo_iluminacion_kwh": float(row.consumo_iluminacion_kwh),
                "consumo_equipos_kwh": float(row.consumo_equipos_kwh),
            }
        )
    return payloads
