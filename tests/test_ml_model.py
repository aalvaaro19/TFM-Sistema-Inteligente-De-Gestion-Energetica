from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_energia.models.metrics import mae
from tfm_energia.models.ml_model import GradientBoostingForecaster


@pytest.fixture
def serie_exog() -> tuple[pd.Series, pd.DataFrame]:
    """120 días horarios con patrón semanal, térmico y ruido."""
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=24 * 120, freq="h", tz="Europe/Madrid")
    temperatura = 15 + 8 * np.sin(2 * np.pi * (idx.dayofyear - 20) / 365) + rng.normal(0, 1, len(idx))
    ocupacion = np.where((idx.hour >= 8) & (idx.hour < 19) & (idx.dayofweek < 5), 0.85, 0.03)
    consumo = (
        5
        + 20 * ocupacion
        + 0.9 * np.abs(temperatura - 21)
        + rng.normal(0, 0.8, len(idx))
    )
    X = pd.DataFrame(
        {
            "temperatura_exterior_c": temperatura,
            "humedad_exterior_pct": np.clip(70 - temperatura, 20, 95),
            "radiacion_solar_rel": np.clip(np.sin(np.pi * (idx.hour - 7) / 12), 0, 1),
            "ocupacion_rel": ocupacion,
        },
        index=idx,
    )
    return pd.Series(consumo, index=idx, name="consumo_total_kwh"), X


def test_rechaza_lags_menores_que_el_horizonte() -> None:
    """Un lag de 24 h no es observable al predecir a 48 h: debe fallar."""
    with pytest.raises(ValueError, match="lags deben ser"):
        GradientBoostingForecaster(horizonte=48, lags=(24, 48))


def test_prediccion_forma_y_dominio(serie_exog) -> None:
    y, X = serie_exog
    modelo = GradientBoostingForecaster(max_iter=60).fit(y, X)
    pred = modelo.predict(48, X.iloc[:48])

    assert len(pred) == 48
    assert pred.notna().all()
    assert (pred >= 0).all()
    assert pred.index[0] == y.index[-1] + pd.Timedelta(hours=1)
    assert pred.name == modelo.nombre


def test_no_predice_mas_alla_del_horizonte(serie_exog) -> None:
    y, X = serie_exog
    modelo = GradientBoostingForecaster(horizonte=48, max_iter=30).fit(y, X)
    with pytest.raises(ValueError, match="horizonte máximo"):
        modelo.predict(72, X.iloc[:72])


def test_exige_exogenas(serie_exog) -> None:
    y, X = serie_exog
    modelo = GradientBoostingForecaster(max_iter=30)
    with pytest.raises(ValueError, match="requiere exógenas"):
        modelo.fit(y)
    with pytest.raises(ValueError, match="Faltan exógenas"):
        modelo.fit(y, X.drop(columns=["ocupacion_rel"]))


def test_bate_al_naive_estacional(serie_exog) -> None:
    from tfm_energia.models.baseline import NaiveEstacionalSemanal

    y, X = serie_exog
    y_train, y_test = y.iloc[:-48], y.iloc[-48:]
    X_train, X_test = X.iloc[:-48], X.iloc[-48:]

    pred_ml = GradientBoostingForecaster(max_iter=200).fit(y_train, X_train).predict(48, X_test)
    pred_naive = NaiveEstacionalSemanal().fit(y_train).predict(48)

    assert mae(y_test, pred_ml) < mae(y_test, pred_naive)


def test_importancia_permutacion_identifica_ocupacion(serie_exog) -> None:
    """La ocupación domina el consumo en la serie sintética del test."""
    y, X = serie_exog
    modelo = GradientBoostingForecaster(max_iter=120).fit(y, X)
    imp = modelo.importancia_permutacion(y.iloc[-24 * 30 :], X.iloc[-24 * 30 :], n_repeats=3)

    assert list(imp.columns) == ["feature", "importancia", "std"]
    assert imp.iloc[0]["importancia"] > 0
    assert "ocupacion_rel" in imp.head(4)["feature"].tolist()
