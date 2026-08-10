"""Acceso a los resultados de las fases previas para servirlos por la API.

La API no recalcula nada: lee los artefactos que dejan los scripts de cada fase
—métricas de modelos, anomalías detectadas, comparativa de optimización— y los
sirve. Separar el cálculo de la consulta mantiene la API rápida y hace que un
fallo en el servicio no arrastre a los resultados ya obtenidos.

Los artefactos se cargan **una sola vez y se memorizan**, porque son ficheros
estáticos que solo cambian al reejecutar una fase.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd
from loguru import logger

from tfm_energia.config import PROCESSED_DIR, SEDES


# Artefactos que producen las distintas fases
ARTEFACTOS = {
    "enriquecido": "enriquecido_{sede}.parquet",
    "metricas_modelos": "metricas_modelos_todas.csv",
    "metricas_horizonte": "metricas_horizonte_{sede}.csv",
    "backtest": "backtest_{sede}.parquet",
    "anomalias_metricas": "anomalias_metricas.csv",
    "anomalias_detectadas": "anomalias_detectadas.csv",
    "optimizacion_resumen": "optimizacion_resumen.csv",
    "optimizacion_horaria": "optimizacion_horaria.parquet",
}


def ruta(clave: str, sede: str | None = None) -> Path:
    plantilla = ARTEFACTOS[clave]
    return PROCESSED_DIR / (plantilla.format(sede=sede) if sede else plantilla)


def disponibles() -> dict[str, bool]:
    """Qué artefactos existen. Permite que la API informe de su estado real."""
    estado = {}
    for clave, plantilla in ARTEFACTOS.items():
        if "{sede}" in plantilla:
            estado[clave] = all(ruta(clave, s).exists() for s in SEDES)
        else:
            estado[clave] = ruta(clave).exists()
    return estado


@lru_cache(maxsize=8)
def datos_sede(sede: str) -> pd.DataFrame:
    """Dataset enriquecido de una sede, indexado por tiempo."""
    if sede not in SEDES:
        raise KeyError(sede)
    path = ruta("enriquecido", sede)
    if not path.exists():
        raise FileNotFoundError(f"No existe {path.name}: ejecuta la fase de datos")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


@lru_cache(maxsize=1)
def metricas_modelos() -> pd.DataFrame:
    path = ruta("metricas_modelos")
    if not path.exists():
        raise FileNotFoundError("Faltan las métricas: ejecuta scripts/train_predictivo.py")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def anomalias() -> pd.DataFrame:
    path = ruta("anomalias_detectadas")
    if not path.exists():
        raise FileNotFoundError("Faltan las anomalías: ejecuta scripts/detectar_anomalias.py")
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df.index.name = "timestamp"
    return df


@lru_cache(maxsize=1)
def optimizacion() -> pd.DataFrame:
    path = ruta("optimizacion_resumen")
    if not path.exists():
        raise FileNotFoundError("Falta la optimización: ejecuta scripts/optimizar_costes.py")
    return pd.read_csv(path)


@lru_cache(maxsize=1)
def optimizacion_horaria() -> pd.DataFrame:
    path = ruta("optimizacion_horaria")
    if not path.exists():
        raise FileNotFoundError("Falta la optimización horaria")
    return pd.read_parquet(path)


def mongo_conectado() -> bool:
    """Comprueba la conexión con MongoDB sin propagar el fallo.

    La API debe seguir sirviendo los resultados aunque la base de datos no esté
    disponible: los artefactos están en disco y no dependen de ella.
    """
    try:
        from tfm_energia.data.mongo_repository import MongoRepository

        repo = MongoRepository()
        repo.db.command("ping")
        repo.close()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"MongoDB no disponible: {type(exc).__name__}")
        return False


def limpiar_cache() -> None:
    """Fuerza la relectura de los artefactos, tras reejecutar alguna fase."""
    for f in (datos_sede, metricas_modelos, anomalias, optimizacion, optimizacion_horaria):
        f.cache_clear()
