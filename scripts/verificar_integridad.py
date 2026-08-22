from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tfm_energia.config import PROCESSED_DIR, RAW_DIR, SEDES, SYNTHETIC_DIR


@dataclass
class Comprobacion:
    nombre: str
    ok: bool
    detalle: str
    vigila: str = ""


@dataclass
class Informe:
    resultados: list[Comprobacion] = field(default_factory=list)

    def añadir(self, *args, **kwargs) -> None:
        self.resultados.append(Comprobacion(*args, **kwargs))

    @property
    def todo_correcto(self) -> bool:
        return all(c.ok for c in self.resultados)

    def imprimir(self) -> None:
        print("\n" + "=" * 78)
        print("  VERIFICACIÓN DE INTEGRIDAD")
        print("=" * 78)
        for c in self.resultados:
            print(f"  [{'OK ' if c.ok else 'MAL'}] {c.nombre}")
            print(f"        {c.detalle}")
            if not c.ok and c.vigila:
                print(f"        Vigila: {c.vigila}")
        fallos = [c for c in self.resultados if not c.ok]
        print("-" * 78)
        print(f"  {len(self.resultados) - len(fallos)} de {len(self.resultados)} correctas")
        print("=" * 78)


# ---------------------------------------------------------------------------
def comprobar_reproducibilidad(inf: Informe) -> None:
    """La misma semilla debe dar el mismo dataset en procesos distintos."""
    codigo = (
        "from datetime import date;"
        "from tfm_energia.config import SEDES;"
        "from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig;"
        "df = OfficeSimulator('madrid', SEDES['madrid'], date(2024,3,1), date(2024,3,3),"
        " SimulationConfig(seed=42)).generate();"
        "print(round(df['consumo_total_kwh'].sum(), 6))"
    )
    salidas = {
        subprocess.run([sys.executable, "-c", codigo], capture_output=True, text=True).stdout.strip()
        for _ in range(2)
    }
    inf.añadir(
        "Reproducibilidad de la semilla",
        len(salidas) == 1,
        f"Dos procesos independientes producen {'el mismo' if len(salidas)==1 else 'distinto'} "
        f"dataset: {salidas}",
        "hash() sobre texto está aleatorizado por proceso (PYTHONHASHSEED)",
    )


def comprobar_meteo_coherente(inf: Informe) -> None:
    """La temperatura de cada fila debe ser la que generó su consumo."""
    from tfm_energia.optimization.thermal_model import (
        aporte_termico_observado,
        deriva_natural,
        parametros_de_sede,
    )

    errores = []
    for sede, meta in SEDES.items():
        d = pd.read_parquet(PROCESSED_DIR / f"enriquecido_{sede}.parquet")
        d = d.set_index("timestamp").sort_index()
        p = parametros_de_sede(meta)

        t_real = d["temperatura_interior_c"].to_numpy()
        t_ext = d["temperatura_exterior_c"].to_numpy()
        aporte = aporte_termico_observado(
            d["consumo_hvac_kwh"].to_numpy(), d["hvac_estado"].to_numpy(), t_real, t_ext, p
        )
        pred = (
            t_real[:-1]
            + deriva_natural(
                t_real[:-1], t_ext[1:], d["ocupacion_rel"].to_numpy()[1:],
                d["radiacion_solar_rel"].to_numpy()[1:], p,
            )
            + aporte[1:]
        )
        anom = d["es_anomalia"].to_numpy()
        err = (pred - t_real[1:])[~anom[1:] & ~anom[:-1]]
        errores.append((sede, float(np.mean(np.abs(err))), float(np.mean(err))))

    peor = max(e[1] for e in errores)
    sesgo = max(abs(e[2]) for e in errores)
    inf.añadir(
        "Coherencia meteorológica y fidelidad del modelo térmico",
        peor < 0.20 and sesgo < 0.05,
        f"Error del modelo lineal: MAE máximo {peor:.4f} °C, sesgo máximo {sesgo:+.4f} °C "
        f"(el ruido del simulador impone un suelo de ~0,12 °C)",
        "sustituir la temperatura exterior después de simular la física",
    )


def comprobar_interpolacion_aemet(inf: Informe) -> None:
    """La serie horaria debe reproducir las estadísticas diarias de AEMET."""
    from tfm_energia.data.meteo_horaria import cargar_diario, perfil_horario

    idx = pd.date_range("2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Madrid")
    sesgos = []
    for sede in SEDES:
        diario = cargar_diario(sede)
        if diario.empty:
            continue
        horaria = perfil_horario(idx, diario)["temperatura_exterior_c"]
        sesgos.append((sede, float(horaria.mean() - diario["tmed"].mean())))

    peor = max(abs(s) for _, s in sesgos) if sesgos else 0.0
    inf.añadir(
        "Interpolación horaria de AEMET sin sesgo",
        peor < 0.05,
        f"Desviación máxima entre la media horaria reconstruida y la media diaria "
        f"publicada: {peor:+.4f} °C",
        "una curva diaria cuyo factor de forma no tiene media nula",
    )


def comprobar_anomalias_visibles(inf: Informe) -> None:
    """Toda avería etiquetada debe dejar rastro medible en los datos."""
    problemas = []
    for sede, meta in SEDES.items():
        d = pd.read_parquet(SYNTHETIC_DIR / f"sede_{sede}.parquet")
        normal = d[~d["es_anomalia"]]
        for tipo, columna in (
            ("HVAC_STUCK_ON", "consumo_hvac_kwh"),
            ("EQUIPMENT_LEAK", "consumo_equipos_kwh"),
            ("CONSUMPTION_SPIKE", "consumo_total_kwh"),
        ):
            g = d[d["tipo_anomalia"] == tipo]
            if g.empty:
                continue
            if g[columna].mean() <= normal[columna].mean():
                problemas.append(f"{sede}/{tipo}")

        congelado = d[d["tipo_anomalia"] == "SENSOR_FROZEN"]
        if not congelado.empty:
            std = d["temperatura_interior_c"].rolling(3, min_periods=2).std()
            if std[d["tipo_anomalia"] == "SENSOR_FROZEN"].median() >= 0.01:
                problemas.append(f"{sede}/SENSOR_FROZEN")

    inf.añadir(
        "Las anomalías inyectadas dejan rastro",
        not problemas,
        "Todas las averías alteran la señal que las delata"
        if not problemas else f"Sin rastro medible en: {problemas}",
        "multiplicar un consumo que estaba a cero (0 × 3,5 = 0)",
    )


def comprobar_precios_completos(inf: Informe) -> None:
    """Ninguna hora del histórico de precios debe perderse."""
    path = RAW_DIR / "esios" / "pvpc_horario.csv"
    if not path.exists():
        inf.añadir("Serie de precios completa", False, f"No existe {path.name}")
        return

    from tfm_energia.data.api_loader import preparar_precios

    docs = preparar_precios(pd.read_csv(path))
    from datetime import timezone

    instantes = {d["fecha_local"].astimezone(timezone.utc) for d in docs}
    inf.añadir(
        "Serie de precios sin horas solapadas",
        len(instantes) == len(docs),
        f"{len(docs):,} filas → {len(instantes):,} instantes únicos"
        + ("" if len(instantes) == len(docs) else f" ({len(docs)-len(instantes)} colapsan)"),
        "pymongo codifica mal un pandas.Timestamp en la hora ambigua del cambio horario",
    )


def comprobar_capas_sincronizadas(inf: Informe, con_mongo: bool) -> None:
    """Los parquet, el flujo de eventos y MongoDB deben describir lo mismo."""
    from tfm_energia.config import PROJECT_ROOT

    base = PROJECT_ROOT / "data" / "stream"
    if not base.exists():
        inf.añadir("Flujo de eventos generado", False, "No existe data/stream")
        return

    from tfm_energia.data.sensor_stream import leer_jsonl

    eventos = sum(1 for f in base.rglob("*.jsonl") for _ in leer_jsonl(f))
    filas = sum(
        len(pd.read_parquet(SYNTHETIC_DIR / f"sede_{s}.parquet")) for s in SEDES
    )
    inf.añadir(
        "Volumen del flujo de eventos",
        eventos == filas * 3,
        f"{filas:,} registros × 3 sensores = {filas*3:,} esperados · {eventos:,} en el flujo",
        "regenerar el dataset sin regenerar el flujo",
    )

    # Contar eventos NO basta: el flujo puede tener el número correcto y valores
    # obsoletos si se regeneró el dataset y no el flujo. Hay que comparar valores.
    coinciden = comprobados = 0
    for sede in SEDES:
        d = pd.read_parquet(SYNTHETIC_DIR / f"sede_{sede}.parquet").set_index("timestamp")
        ficheros = sorted((base / f"sede={sede}").glob("*.jsonl"))
        if not ficheros:
            continue
        for evento in leer_jsonl(ficheros[0]):
            if evento.get("tipo") != "consumo_electrico" or "timestamp" not in evento:
                continue
            ts = pd.Timestamp(evento["timestamp"])
            if ts not in d.index:
                continue
            comprobados += 1
            coinciden += abs(
                float(evento["consumo_total_kwh"]) - float(d.loc[ts, "consumo_total_kwh"])
            ) < 1e-6
            if comprobados >= 100:
                break

    inf.añadir(
        "Contenido del flujo coincide con el dataset",
        comprobados > 0 and coinciden == comprobados,
        f"{coinciden}/{comprobados} valores contrastados coinciden",
        "el recuento de eventos cuadra aunque los valores sean de una versión anterior",
    )

    if not con_mongo:
        return

    try:
        from tfm_energia.data.mongo_repository import COL_SENSORES, MongoRepository

        repo = MongoRepository()
        col = repo.db[COL_SENSORES]
        total = col.count_documents({})
        como_fecha = col.count_documents({"timestamp": {"$type": "date"}})

        # Contraste puntual con el parquet
        rng = np.random.default_rng(7)
        coinciden = comprobados = 0
        for sede in SEDES:
            d = pd.read_parquet(SYNTHETIC_DIR / f"sede_{sede}.parquet").set_index("timestamp")
            for pos in rng.choice(len(d), size=25, replace=False):
                ts = d.index[pos]
                doc = col.find_one(
                    {"sede": sede, "tipo": "consumo_electrico", "timestamp": ts.to_pydatetime()}
                )
                if doc is None:
                    continue
                comprobados += 1
                coinciden += abs(
                    doc["consumo_total_kwh"] - float(d.iloc[pos]["consumo_total_kwh"])
                ) < 1e-6
        repo.close()

        inf.añadir(
            "Fechas tipadas en MongoDB",
            total > 0 and como_fecha == total,
            f"{como_fecha:,} de {total:,} documentos con timestamp como fecha BSON",
            "guardar las fechas como texto: las consultas por rango fallan sin avisar",
        )
        inf.añadir(
            "MongoDB coincide con el dataset",
            comprobados > 0 and coinciden == comprobados,
            f"{coinciden}/{comprobados} timestamps contrastados coinciden al valor",
            "recargar solo una de las capas",
        )
    except Exception as exc:  # noqa: BLE001
        inf.añadir("Conexión con MongoDB", False, f"{type(exc).__name__}: {exc}")


def comprobar_resultados_al_dia(inf: Informe) -> None:
    """Los resultados deben ser posteriores a los datos con los que se calculan."""
    datos = max(
        (PROCESSED_DIR / f"enriquecido_{s}.parquet").stat().st_mtime
        for s in SEDES
        if (PROCESSED_DIR / f"enriquecido_{s}.parquet").exists()
    )
    obsoletos = []
    for nombre in (
        "metricas_modelos_todas.csv", "anomalias_metricas.csv",
        "optimizacion_resumen.csv",
    ):
        path = PROCESSED_DIR / nombre
        if path.exists() and path.stat().st_mtime < datos:
            obsoletos.append(nombre)

    inf.añadir(
        "Resultados calculados sobre los datos actuales",
        not obsoletos,
        "Todos los resultados son posteriores al dataset"
        if not obsoletos else f"Anteriores al dataset: {obsoletos}",
        "recalcular los datos y no relanzar las fases que dependen de ellos",
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Verificación de integridad")
    ap.add_argument("--sin-mongo", action="store_true")
    args = ap.parse_args()

    inf = Informe()
    comprobar_reproducibilidad(inf)
    comprobar_interpolacion_aemet(inf)
    comprobar_meteo_coherente(inf)
    comprobar_anomalias_visibles(inf)
    comprobar_precios_completos(inf)
    comprobar_capas_sincronizadas(inf, con_mongo=not args.sin_mongo)
    comprobar_resultados_al_dia(inf)
    inf.imprimir()
    return 0 if inf.todo_correcto else 1


if __name__ == "__main__":
    sys.exit(main())
