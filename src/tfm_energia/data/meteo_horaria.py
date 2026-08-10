"""Conversión de la observación diaria de AEMET en una serie horaria.

AEMET OpenData publica gratuitamente **valores diarios** (mínima, máxima, media,
humedad), pero la simulación del edificio y los modelos trabajan a resolución
horaria. Este módulo reconstruye el ciclo diario a partir de la mínima y la
máxima mediante una curva sinusoidal con el mínimo al amanecer y el máximo a
media tarde, que es la aproximación estándar en modelización energética de
edificios cuando solo se dispone de extremos diarios.

Esta serie es la que **gobierna la física del edificio**: se pasa al generador
como temperatura exterior, de modo que la temperatura interior y el consumo de
climatización derivan de la meteorología real. Antes se sustituía la columna
*después* de haber simulado, lo que dejaba cada fila afirmando una temperatura
exterior que no era la que había producido su propio consumo: un desajuste medio
de 3,8 °C que invalidaba cualquier modelo térmico construido sobre el dataset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.config import RAW_DIR


# Columnas diarias de AEMET que se utilizan
COLUMNAS_AEMET = ("tmed", "tmin", "tmax", "hrMedia")

# Perfil diario: mínimo al amanecer y máximo doce horas después.
#
# Se usa un coseno porque su media a lo largo de las 24 horas es CERO, de modo
# que el promedio del día reconstruido coincide con la temperatura media que
# publica AEMET. Una curva sin esa propiedad introduce un sesgo sistemático: la
# versión anterior combinaba un seno diurno con un valor fijo nocturno cuya
# media era +0,43, lo que elevaba la temperatura reconstruida unos 2 °C y hacía
# que el edificio necesitara mucha menos calefacción de la real.
HORA_MINIMO = 6


def _a_numero(serie: pd.Series) -> pd.Series:
    """Convierte a numérico admitiendo la coma decimal de algunas estaciones."""
    if serie.dtype.kind in "if":
        return serie
    return pd.to_numeric(
        serie.astype(str).str.strip().str.replace(",", ".", regex=False), errors="coerce"
    )


def cargar_diario(sede_id: str) -> pd.DataFrame:
    """Lee el CSV diario de AEMET de una sede y limpia sus columnas."""
    path = RAW_DIR / "aemet" / f"meteo_{sede_id}.csv"
    if not path.exists():
        logger.warning(f"No existe {path.name}: la sede {sede_id} usará meteorología sintética.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    for col in COLUMNAS_AEMET:
        df[col] = _a_numero(df[col]) if col in df.columns else np.nan

    # Huecos puntuales de la estación: se interpolan antes de derivar el perfil
    df[list(COLUMNAS_AEMET)] = df[list(COLUMNAS_AEMET)].interpolate(
        method="linear", limit_direction="both"
    )
    df["tmin"] = df["tmin"].fillna(df["tmed"] - 5)
    df["tmax"] = df["tmax"].fillna(df["tmed"] + 5)
    df["hrMedia"] = df["hrMedia"].fillna(60.0)
    return df


def perfil_horario(idx: pd.DatetimeIndex, df_diario: pd.DataFrame) -> pd.DataFrame:
    """Expande los valores diarios al índice horario dado.

    Devuelve un DataFrame indexado por `idx` con las columnas
    `temperatura_exterior_c` y `humedad_exterior_pct`.
    """
    if df_diario.empty:
        return pd.DataFrame(index=idx)

    fechas = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()
    diario = df_diario.set_index("fecha")[["tmed", "tmin", "tmax", "hrMedia"]]
    alineado = diario.reindex(fechas)

    # Coseno de media nula: vale −1 en `HORA_MINIMO` y +1 doce horas después
    horas = idx.hour.values
    factor = -np.cos(2 * np.pi * (horas - HORA_MINIMO) / 24.0)

    # Se ancla en la media diaria observada, no en el punto medio de los
    # extremos: AEMET publica `tmed` y es un dato más fiable que la semisuma
    media = alineado["tmed"].to_numpy()
    amplitud = (alineado["tmax"].to_numpy() - alineado["tmin"].to_numpy()) / 2

    return pd.DataFrame(
        {
            "temperatura_exterior_c": media + amplitud * factor,
            "humedad_exterior_pct": alineado["hrMedia"].to_numpy(),
        },
        index=idx,
    )


def meteo_real_horaria(sede_id: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Serie horaria de temperatura y humedad reales para una sede.

    Devuelve un DataFrame vacío si no hay datos descargados, para que el
    generador pueda recurrir a su meteorología sintética.
    """
    diario = cargar_diario(sede_id)
    if diario.empty:
        return pd.DataFrame(index=idx)

    horaria = perfil_horario(idx, diario)
    cobertura = horaria["temperatura_exterior_c"].notna().mean()
    logger.info(
        f"  AEMET {sede_id}: {len(diario):,} días → {len(horaria):,} horas "
        f"(cobertura {cobertura:.1%})"
    )
    return horaria
