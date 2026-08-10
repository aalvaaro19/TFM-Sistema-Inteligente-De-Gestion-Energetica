"""Construcción de variables (feature engineering) para los modelos predictivos.

Convierte el dataset crudo en un conjunto de features rico que cubre:

  * Variables calendario (hora, día, mes, festivo) con codificación cíclica.
  * Lags del target (1h, 24h, 168h) para capturar memoria temporal.
  * Estadísticas móviles (rolling mean / std) en distintas ventanas.
  * Variables exógenas climáticas y de ocupación.
  * Variables económicas (precio PVPC) cuando estén disponibles.
  * Particionado train / validation / test respetando el orden temporal.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger


# Lags y ventanas por defecto
LAGS_HORARIOS_DEFAULT = (1, 2, 3, 6, 12, 24, 48, 168)
ROLLING_WINDOWS_DEFAULT = (3, 6, 24, 168)


@dataclass
class FeatureConfig:
    """Configuración del feature builder."""

    target: str = "consumo_total_kwh"
    lags: tuple[int, ...] = LAGS_HORARIOS_DEFAULT
    rolling_windows: tuple[int, ...] = ROLLING_WINDOWS_DEFAULT
    incluir_exogenas: tuple[str, ...] = (
        "temperatura_exterior_c",
        "humedad_exterior_pct",
        "radiacion_solar_rel",
        "ocupacion_rel",
    )
    incluir_precio: bool = True
    dropna_final: bool = True


# ---------------------------------------------------------------------------
# Variables calendario
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame, ts_col: str | None = None) -> pd.DataFrame:
    """Añade variables de calendario y codificación cíclica.

    Si `ts_col` es None, asume que el índice es DatetimeIndex.
    """
    df = df.copy()
    if ts_col is not None:
        idx = pd.to_datetime(df[ts_col])
    else:
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            raise ValueError("El índice debe ser DatetimeIndex o pasar ts_col.")

    df["hora"] = idx.hour
    df["dia_semana"] = idx.dayofweek
    df["dia_mes"] = idx.day
    df["mes"] = idx.month
    df["trimestre"] = idx.quarter
    df["semana_anio"] = idx.isocalendar().week.values.astype(int)
    df["es_finde"] = idx.dayofweek >= 5

    # Codificación cíclica para evitar discontinuidades
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["dia_semana_sin"] = np.sin(2 * np.pi * df["dia_semana"] / 7)
    df["dia_semana_cos"] = np.cos(2 * np.pi * df["dia_semana"] / 7)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    return df


# ---------------------------------------------------------------------------
# Lags y rolling windows
# ---------------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame, col: str, lags: Iterable[int]
) -> pd.DataFrame:
    """Añade columnas de lag (shift) de una variable."""
    df = df.copy()
    for lag in lags:
        df[f"{col}_lag_{lag}h"] = df[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, col: str, windows: Iterable[int]
) -> pd.DataFrame:
    """Añade media y desviación rolling de una variable.

    Importante: se usa `.shift(1)` para evitar lookahead bias: el valor
    en t solo usa información de hasta t-1.
    """
    df = df.copy()
    serie_shifted = df[col].shift(1)
    for w in windows:
        df[f"{col}_roll_mean_{w}h"] = serie_shifted.rolling(window=w, min_periods=w).mean()
        df[f"{col}_roll_std_{w}h"] = serie_shifted.rolling(window=w, min_periods=w).std()
    return df


# ---------------------------------------------------------------------------
# Variables económicas (precios)
# ---------------------------------------------------------------------------
def merge_precios_pvpc(
    df_consumo: pd.DataFrame, df_precios: pd.DataFrame, on_col: str = "fecha_local"
) -> pd.DataFrame:
    """Hace merge del DataFrame de consumo con la serie horaria PVPC.

    El DF de consumo debe tener índice DatetimeIndex; el de precios debe tener
    una columna `fecha_local` con timestamps horarios.
    """
    df = df_consumo.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df_consumo debe indexarse por DatetimeIndex.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("Europe/Madrid")
    precios = df_precios.copy()
    precios[on_col] = pd.to_datetime(precios[on_col])
    if precios[on_col].dt.tz is None:
        precios[on_col] = precios[on_col].dt.tz_localize("Europe/Madrid")

    # merge sobre timestamp
    merged = df.merge(
        precios[[on_col, "precio_eur_kwh", "franja_pvpc"]],
        left_index=True,
        right_on=on_col,
        how="left",
    ).set_index(on_col)
    merged.index.name = "timestamp"
    return merged


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------
class FeatureBuilder:
    """Orquesta la construcción de features para una sede."""

    def __init__(self, cfg: FeatureConfig | None = None) -> None:
        self.cfg = cfg or FeatureConfig()

    def build(
        self,
        df: pd.DataFrame,
        df_precios: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Aplica todo el pipeline de feature engineering."""
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            else:
                raise ValueError("Se requiere columna timestamp o índice temporal.")
        df = df.sort_index()

        # 1. Calendario
        df = add_calendar_features(df)

        # 2. Lags del target
        df = add_lag_features(df, self.cfg.target, self.cfg.lags)

        # 3. Rolling del target
        df = add_rolling_features(df, self.cfg.target, self.cfg.rolling_windows)

        # 4. Lags de variables exógenas críticas
        for ex in self.cfg.incluir_exogenas:
            if ex in df.columns:
                df = add_lag_features(df, ex, (1, 24))

        # 5. Variables económicas (si hay precios)
        if self.cfg.incluir_precio and df_precios is not None:
            df = merge_precios_pvpc(df, df_precios)
            # franja como dummies
            df = pd.get_dummies(df, columns=["franja_pvpc"], prefix="franja", dummy_na=False)

        # 6. Drop de filas con NaN derivados de los lags
        n_antes = len(df)
        if self.cfg.dropna_final:
            df = df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c])
        n_despues = len(df)
        logger.info(
            f"FeatureBuilder: {n_antes:,} → {n_despues:,} filas tras dropna "
            f"({n_antes - n_despues} filas iniciales descartadas)"
        )
        return df

    def split_temporal(
        self,
        df: pd.DataFrame,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split temporal estricto: train | val | test sin solape.

        En series temporales NUNCA se debe usar shuffle. Siempre se valida con
        datos posteriores al entrenamiento.
        """
        assert train_frac + val_frac < 1.0, "Suma de fracciones debe ser <1"
        n = len(df)
        i_train = int(n * train_frac)
        i_val = int(n * (train_frac + val_frac))
        train = df.iloc[:i_train]
        val = df.iloc[i_train:i_val]
        test = df.iloc[i_val:]
        logger.info(
            f"Split temporal: train={len(train):,} "
            f"({train.index.min()} → {train.index.max()}) | "
            f"val={len(val):,} | test={len(test):,}"
        )
        return train, val, test

    def get_feature_cols(self, df: pd.DataFrame) -> list[str]:
        """Lista las columnas de features (excluye target y metadatos)."""
        excluir = {
            self.cfg.target,
            "sede",
            "sede_nombre",
            "tipo_anomalia",
            "es_anomalia",
        }
        return [c for c in df.columns if c not in excluir]


# ---------------------------------------------------------------------------
# Función de conveniencia para cargar una sede
# ---------------------------------------------------------------------------
def cargar_features_sede(
    sede_id: str,
    parquet_path: str | None = None,
    df_precios: pd.DataFrame | None = None,
    cfg: FeatureConfig | None = None,
) -> pd.DataFrame:
    """Carga el parquet de una sede y construye sus features de un tirón."""
    from tfm_energia.config import SYNTHETIC_DIR

    path = parquet_path or (SYNTHETIC_DIR / f"sede_{sede_id}.parquet")
    df = pd.read_parquet(path)
    builder = FeatureBuilder(cfg)
    return builder.build(df, df_precios=df_precios)
