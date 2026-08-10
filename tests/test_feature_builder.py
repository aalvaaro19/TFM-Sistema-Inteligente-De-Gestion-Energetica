"""Tests del feature builder."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig
from tfm_energia.features.feature_builder import (
    FeatureBuilder,
    FeatureConfig,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
)


@pytest.fixture
def df_demo() -> pd.DataFrame:
    """DF pequeño de 14 días para no tardar."""
    sim = OfficeSimulator(
        "madrid",
        SEDES["madrid"],
        start=date(2024, 6, 1),
        end=date(2024, 6, 14),
        cfg=SimulationConfig(seed=42),
    )
    df = sim.generate().set_index("timestamp")
    return df


def test_add_calendar_features(df_demo: pd.DataFrame) -> None:
    df = add_calendar_features(df_demo)
    assert "hora" in df.columns
    assert "hora_sin" in df.columns
    assert "hora_cos" in df.columns
    assert "mes" in df.columns
    assert df["hora"].between(0, 23).all()
    assert df["dia_semana"].between(0, 6).all()
    # Codificación cíclica en [-1, 1]
    assert df["hora_sin"].between(-1, 1).all()


def test_codificacion_ciclica_continua(df_demo: pd.DataFrame) -> None:
    """hora_sin/cos en hora 0 y hora 23 deben estar cerca (continuidad cíclica)."""
    df = add_calendar_features(df_demo)
    h0 = df[df["hora"] == 0].iloc[0]
    h23 = df[df["hora"] == 23].iloc[0]
    # En representación cartesiana, distancia entre 0h y 23h pequeña (60°)
    dist = np.sqrt(
        (h0["hora_sin"] - h23["hora_sin"]) ** 2
        + (h0["hora_cos"] - h23["hora_cos"]) ** 2
    )
    assert dist < 0.6  # son adyacentes en el círculo


def test_add_lag_features(df_demo: pd.DataFrame) -> None:
    df = add_lag_features(df_demo, "consumo_total_kwh", [1, 24])
    assert "consumo_total_kwh_lag_1h" in df.columns
    assert "consumo_total_kwh_lag_24h" in df.columns
    # Primer registro debe tener NaN en el lag
    assert pd.isna(df["consumo_total_kwh_lag_1h"].iloc[0])
    # Valor lag_1h en i debe ser valor original en i-1
    assert df["consumo_total_kwh_lag_1h"].iloc[5] == df["consumo_total_kwh"].iloc[4]


def test_rolling_sin_lookahead(df_demo: pd.DataFrame) -> None:
    """El rolling debe usar .shift(1) para evitar lookahead."""
    df = add_rolling_features(df_demo, "consumo_total_kwh", [3])
    # En posición i, el rolling de 3 debe ser media de [i-3, i-2, i-1] (no incluir i)
    serie = df["consumo_total_kwh"]
    expected = serie.iloc[1:4].mean()
    assert abs(df["consumo_total_kwh_roll_mean_3h"].iloc[4] - expected) < 1e-9


def test_feature_builder_pipeline_completo(df_demo: pd.DataFrame) -> None:
    builder = FeatureBuilder()
    out = builder.build(df_demo)
    # Debe contener cosas de calendario, lags y rolling
    assert any("hora_sin" in c for c in out.columns)
    assert any("lag_1h" in c for c in out.columns)
    assert any("roll_mean" in c for c in out.columns)
    # No debe haber NaN si dropna activo
    feature_cols = [c for c in out.columns if "lag" in c or "roll" in c]
    assert out[feature_cols].isna().sum().sum() == 0


def test_split_temporal_sin_solape(df_demo: pd.DataFrame) -> None:
    builder = FeatureBuilder()
    out = builder.build(df_demo)
    train, val, test = builder.split_temporal(out, 0.7, 0.15)
    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()
    assert len(train) + len(val) + len(test) == len(out)


def test_get_feature_cols_excluye_target(df_demo: pd.DataFrame) -> None:
    builder = FeatureBuilder()
    out = builder.build(df_demo)
    cols = builder.get_feature_cols(out)
    assert "consumo_total_kwh" not in cols
    assert "sede" not in cols
    assert any("hora" in c for c in cols)
