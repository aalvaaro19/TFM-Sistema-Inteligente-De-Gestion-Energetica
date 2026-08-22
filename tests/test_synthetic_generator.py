from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.synthetic_generator import (
    OfficeSimulator,
    SimulationConfig,
    occupancy_factor,
)


@pytest.fixture
def small_sim() -> OfficeSimulator:
    cfg = SimulationConfig(seed=42)
    return OfficeSimulator(
        "madrid",
        SEDES["madrid"],
        start=date(2024, 6, 1),
        end=date(2024, 6, 7),
        cfg=cfg,
    )


def test_occupancy_factor_horario_laboral() -> None:
    martes_10am = pd.Timestamp("2024-06-04 10:00:00")
    assert occupancy_factor(martes_10am, set()) > 0.8


def test_occupancy_factor_finde() -> None:
    domingo_10am = pd.Timestamp("2024-06-02 10:00:00")
    assert occupancy_factor(domingo_10am, set()) < 0.1


def test_occupancy_factor_madrugada() -> None:
    martes_3am = pd.Timestamp("2024-06-04 03:00:00")
    assert occupancy_factor(martes_3am, set()) < 0.1


def test_generate_estructura(small_sim: OfficeSimulator) -> None:
    df = small_sim.generate()
    cols_esperadas = {
        "timestamp",
        "sede",
        "temperatura_exterior_c",
        "temperatura_interior_c",
        "humedad_exterior_pct",
        "humedad_interior_pct",
        "co2_ppm",
        "ocupacion_rel",
        "consumo_total_kwh",
        "consumo_hvac_kwh",
        "consumo_iluminacion_kwh",
        "consumo_equipos_kwh",
        "consumo_base_kwh",
        "es_anomalia",
        "es_festivo",
        "es_finde",
    }
    assert cols_esperadas.issubset(df.columns)
    # 7 días × 24 h
    assert len(df) == 7 * 24


def test_generate_rangos_fisicos(small_sim: OfficeSimulator) -> None:
    df = small_sim.generate()
    assert df["temperatura_exterior_c"].between(-10, 50).all()
    assert df["humedad_exterior_pct"].between(15, 100).all()
    assert df["co2_ppm"].between(380, 2500).all()
    assert (df["consumo_total_kwh"] >= 0).all()
    assert df["ocupacion_rel"].between(0, 1).all()


def test_generate_estacionalidad_anual() -> None:
    """Madrid verano > Madrid invierno en T exterior."""
    cfg = SimulationConfig(seed=42)
    sim_verano = OfficeSimulator(
        "madrid", SEDES["madrid"], date(2024, 7, 15), date(2024, 7, 21), cfg
    )
    sim_invierno = OfficeSimulator(
        "madrid", SEDES["madrid"], date(2024, 1, 15), date(2024, 1, 21), cfg
    )
    assert (
        sim_verano.generate()["temperatura_exterior_c"].mean()
        > sim_invierno.generate()["temperatura_exterior_c"].mean()
    )


def test_climas_diferenciados() -> None:
    """Sevilla más cálida que Oviedo en promedio."""
    cfg = SimulationConfig(seed=42)
    sev = OfficeSimulator(
        "sevilla", SEDES["sevilla"], date(2024, 7, 1), date(2024, 7, 31), cfg
    ).generate()
    ovd = OfficeSimulator(
        "oviedo", SEDES["oviedo"], date(2024, 7, 1), date(2024, 7, 31), cfg
    ).generate()
    assert sev["temperatura_exterior_c"].mean() > ovd["temperatura_exterior_c"].mean()


def test_consumo_correlacionado_con_ocupacion(small_sim: OfficeSimulator) -> None:
    df = small_sim.generate()
    df_lab = df[~df["es_finde"] & ~df["es_festivo"]]
    corr = df_lab[["ocupacion_rel", "consumo_total_kwh"]].corr().iloc[0, 1]
    assert corr > 0.3


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------
def test_mismo_seed_mismos_datos() -> None:
    """Dos simulaciones con el mismo seed deben ser idénticas."""
    args = ("madrid", SEDES["madrid"], date(2024, 3, 1), date(2024, 3, 10))
    a = OfficeSimulator(*args, SimulationConfig(seed=42)).generate()
    b = OfficeSimulator(*args, SimulationConfig(seed=42)).generate()
    pd.testing.assert_frame_equal(a, b)


def test_seed_estable_entre_procesos() -> None:
    """El desplazamiento por sede no puede depender de `hash()`.

    Regresión: `hash()` sobre str está aleatorizado por proceso
    (PYTHONHASHSEED), así que la semilla efectiva cambiaba en cada ejecución y
    el dataset del TFM no era reproducible.
    """
    import subprocess
    import sys

    codigo = (
        "from datetime import date;"
        "from tfm_energia.config import SEDES;"
        "from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig;"
        "df = OfficeSimulator('madrid', SEDES['madrid'], date(2024,3,1), date(2024,3,3),"
        " SimulationConfig(seed=42)).generate();"
        "print(round(df['consumo_total_kwh'].sum(), 6))"
    )
    salidas = {
        subprocess.run(
            [sys.executable, "-c", codigo], capture_output=True, text=True, check=True
        ).stdout.strip()
        for _ in range(2)
    }
    assert len(salidas) == 1, f"La generación no es reproducible entre procesos: {salidas}"


def test_seeds_distintos_dan_datos_distintos() -> None:
    args = ("madrid", SEDES["madrid"], date(2024, 3, 1), date(2024, 3, 10))
    a = OfficeSimulator(*args, SimulationConfig(seed=1)).generate()
    b = OfficeSimulator(*args, SimulationConfig(seed=2)).generate()
    assert a["consumo_total_kwh"].sum() != b["consumo_total_kwh"].sum()


# ---------------------------------------------------------------------------
# Control del HVAC
# ---------------------------------------------------------------------------
def _simular(sede_id: str, inicio: date, fin: date) -> pd.DataFrame:
    return OfficeSimulator(
        sede_id, SEDES[sede_id], inicio, fin, SimulationConfig(seed=42)
    ).generate()


def test_setpoint_usa_el_mes_real_del_calendario() -> None:
    """El setpoint debe seguir el mes del timestamp, también en el segundo año.

    Regresión: la versión anterior derivaba el mes de la posición en el array
    (`i // 730`), por lo que a partir de la hora 8760 nunca volvía a invierno
    ni a verano y todo el año 2 quedaba con consigna fija de 22,5 °C.
    """
    sim = OfficeSimulator(
        "madrid", SEDES["madrid"], date(2024, 1, 1), date(2025, 12, 31), SimulationConfig(seed=42)
    )
    idx = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Madrid")
    setpoints = pd.Series(sim._setpoint_estacional(idx), index=idx)

    for anio in (2024, 2025):
        assert setpoints[f"{anio}-01"].eq(21.0).all()   # invierno
        assert setpoints[f"{anio}-07"].eq(24.0).all()   # verano
        assert setpoints[f"{anio}-04"].eq(22.5).all()   # intermedia
        assert setpoints[f"{anio}-12"].eq(21.0).all()


def test_hvac_no_arranca_de_madrugada_con_oficina_vacia() -> None:
    """Regresión: el umbral se comparaba con la ocupación ruidosa del sensor,
    lo que disparaba la calefacción a plena potencia a las 3 de la mañana."""
    df = _simular("madrid", date(2025, 12, 1), date(2025, 12, 31)).set_index("timestamp")
    madrugada = df[df.index.hour.isin([1, 2, 3, 4])]

    # Solo puede haber protección antihielo, nunca potencia de confort
    assert madrugada["hvac_estado"].max() <= SimulationConfig().modulacion_antihielo + 1e-9
    assert (madrugada["hvac_estado"] >= 0).all()  # de madrugada no se refrigera
    # Ninguna hora nocturna normal alcanza el consumo típico de una de oficina
    # (se excluyen las anomalías inyectadas a propósito, p. ej. HVAC_STUCK_ON)
    pico_oficina = df[df.index.hour == 10]["consumo_hvac_kwh"].mean()
    normales = madrugada[~madrugada["es_anomalia"]]
    assert normales["consumo_hvac_kwh"].max() < pico_oficina


def test_hvac_modula_y_no_es_binario() -> None:
    """El equipo debe repartir su potencia, no conmutar todo/nada."""
    df = _simular("madrid", date(2025, 1, 1), date(2025, 1, 31))
    activo = df[df["hvac_estado"].abs() > 0]["consumo_hvac_kwh"]

    assert len(activo.round(1).unique()) > 20      # continuo, no un par de valores
    assert activo.std() > 1.0


def test_hvac_no_supera_la_potencia_nominal() -> None:
    """La carga HVAC está acotada por la potencia de placa del equipo."""
    cfg = SimulationConfig(seed=42)
    df = _simular("madrid", date(2025, 1, 1), date(2025, 1, 31))
    nominal = cfg.hvac_pot_nominal_per_m2 * SEDES["madrid"]["superficie_m2"]

    sin_anomalias = df[~df["es_anomalia"]]
    assert sin_anomalias["consumo_hvac_kwh"].max() <= nominal + 1e-6
    assert df["hvac_estado"].abs().max() <= 1.0 + 1e-9


def test_invierno_calienta_y_verano_enfria() -> None:
    """El signo del HVAC debe seguir la estación en el segundo año de la serie."""
    invierno = _simular("madrid", date(2025, 1, 7), date(2025, 1, 21))
    verano = _simular("madrid", date(2025, 7, 7), date(2025, 7, 21))

    lab_inv = invierno[~invierno["es_finde"] & ~invierno["es_festivo"]]
    lab_ver = verano[~verano["es_finde"] & ~verano["es_festivo"]]

    assert lab_inv["hvac_estado"].mean() > 0   # domina la calefacción
    assert lab_ver["hvac_estado"].mean() < 0   # domina la refrigeración


def test_hvac_atascado_deja_huella_en_el_consumo() -> None:
    """Regresión: la avería debe ser visible en los datos.

    La versión anterior multiplicaba el consumo del HVAC por 3,5. Si el equipo
    estaba parado, 0 × 3,5 = 0 y la anomalía quedaba etiquetada pero sin dejar
    rastro alguno, con lo que era imposible de detectar por construcción.
    """
    df = _simular("madrid", date(2025, 1, 1), date(2025, 12, 31))
    stuck = df[df["tipo_anomalia"] == "HVAC_STUCK_ON"]
    if stuck.empty:
        pytest.skip("No se inyectó ninguna anomalía de este tipo en el periodo")

    cfg = SimulationConfig()
    nominal = cfg.hvac_pot_nominal_per_m2 * SEDES["madrid"]["superficie_m2"]
    esperado = cfg.modulacion_atascado * nominal

    assert (stuck["consumo_hvac_kwh"] >= esperado - 1e-6).all()
    # Y muy por encima del consumo típico
    normal = df[~df["es_anomalia"]]["consumo_hvac_kwh"].mean()
    assert stuck["consumo_hvac_kwh"].mean() > 2 * normal


def test_temperatura_interior_en_rango_de_confort() -> None:
    """Una vez recuperado el edificio, la T interior se mantiene en la banda.

    El control es proporcional con banda muerta, así que estabiliza en el borde
    de la banda (setpoint − banda_confort), no exactamente en el setpoint.
    """
    df = _simular("madrid", date(2025, 1, 7), date(2025, 1, 21)).set_index("timestamp")
    oficina = df[(df.index.hour.isin(range(11, 20))) & (df.index.dayofweek < 5)]
    assert oficina["temperatura_interior_c"].between(18, 26).mean() > 0.95


def test_rampa_matinal_tras_paro_nocturno() -> None:
    """Al abrir, el equipo arranca a plena potencia hasta recuperar consigna.

    Este pico matinal es el margen que explota la optimización de la fase 7:
    parte de esa carga puede adelantarse a horas valle.
    """
    df = _simular("madrid", date(2025, 1, 7), date(2025, 1, 21)).set_index("timestamp")
    lab = df[df.index.dayofweek < 5]
    perfil = lab.groupby(lab.index.hour)["consumo_hvac_kwh"].mean()

    assert perfil[8] > perfil[6]          # arranque al abrir la oficina
    assert perfil[9] > perfil[16]         # la punta está por la mañana
    # Y la temperatura sube de forma sostenida durante la rampa
    t_media = lab.groupby(lab.index.hour)["temperatura_interior_c"].mean()
    assert t_media[11] > t_media[7] + 4
