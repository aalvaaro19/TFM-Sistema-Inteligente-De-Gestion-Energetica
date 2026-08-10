"""Preparación de los datos de las APIs externas para su carga en MongoDB.

Los clientes de AEMET y e·sios dejan su resultado en CSV. Este módulo los
convierte en documentos listos para persistir, resolviendo las dos rarezas del
formato de AEMET:

  * **Decimales con coma.** Las columnas de presión llegan como `"945,1"`
    mientras que las de temperatura usan punto (`6.6`). Conviven ambos estilos
    en el mismo fichero.
  * **Marcadores de texto en columnas de hora.** `horaHrMax` puede valer
    `"Varias"` cuando el máximo se alcanzó en más de un momento del día.

Las funciones son puras y no tocan la base de datos, de modo que pueden
probarse sin MongoDB.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from loguru import logger


# Columnas numéricas de AEMET. Algunas usan coma decimal según la estación.
COLUMNAS_NUMERICAS_AEMET = (
    "altitud",
    "tmed",
    "prec",
    "tmin",
    "tmax",
    "dir",
    "velmedia",
    "racha",
    "presMax",
    "presMin",
    "hrMedia",
    "hrMax",
    "hrMin",
)

# Columnas que no aportan al proyecto y solo engordan los documentos
COLUMNAS_DESCARTABLES_AEMET = ("horatmin", "horatmax", "horaracha", "horaPresMax", "horaPresMin")


def _a_numero(valor: Any) -> float | None:
    """Convierte a float admitiendo coma decimal. Devuelve None si no se puede."""
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    if isinstance(valor, (int, float)):
        return float(valor)
    texto = str(valor).strip()
    if not texto:
        return None
    try:
        return float(texto.replace(",", "."))
    except ValueError:
        return None


def preparar_meteo(df: pd.DataFrame, sede_id: str) -> list[dict[str, Any]]:
    """Convierte el CSV diario de AEMET en documentos por sede y fecha."""
    df = df.copy()
    df = df.drop(columns=[c for c in COLUMNAS_DESCARTABLES_AEMET if c in df.columns])

    for col in COLUMNAS_NUMERICAS_AEMET:
        if col in df.columns:
            df[col] = df[col].map(_a_numero)

    df["fecha"] = pd.to_datetime(df["fecha"]).dt.tz_localize("Europe/Madrid")
    df["sede"] = sede_id
    df["fuente"] = "AEMET OpenData"

    documentos = []
    for registro in df.to_dict(orient="records"):
        # Los NaN de pandas no son JSON válidos ni tipos BSON útiles
        documentos.append(
            {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in registro.items()}
        )
    return documentos


def preparar_precios(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convierte el CSV horario de e·sios en documentos de precio."""
    df = df.copy()
    df["fecha_local"] = pd.to_datetime(df["fecha_local"], utc=True).dt.tz_convert(
        "Europe/Madrid"
    )
    for col in ("precio_eur_mwh", "precio_eur_kwh"):
        df[col] = df[col].map(_a_numero)
    df["fuente"] = "e·sios REE"

    n_nulos = int(df["precio_eur_kwh"].isna().sum())
    if n_nulos:
        logger.warning(f"{n_nulos} precios sin valor numérico")

    return df.to_dict(orient="records")


def resumen_precios(documentos: list[dict[str, Any]]) -> dict[str, Any]:
    """Estadísticas del histórico de precios, útiles para la memoria."""
    precios = [d["precio_eur_kwh"] for d in documentos if d.get("precio_eur_kwh") is not None]
    franjas: dict[str, int] = {}
    for d in documentos:
        franjas[d.get("franja_pvpc", "?")] = franjas.get(d.get("franja_pvpc", "?"), 0) + 1

    return {
        "n": len(documentos),
        "precio_medio": float(np.mean(precios)) if precios else None,
        "precio_min": float(np.min(precios)) if precios else None,
        "precio_max": float(np.max(precios)) if precios else None,
        "por_franja": franjas,
    }
