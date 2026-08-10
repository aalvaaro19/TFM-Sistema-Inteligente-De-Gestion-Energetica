"""Tests de las métricas de evaluación."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_energia.models.metrics import (
    calcular_metricas,
    comparar_modelos,
    mae,
    mape,
    mbe,
    mejora_relativa,
    r2,
    rmse,
    smape,
)


@pytest.fixture
def serie_perfecta() -> tuple[pd.Series, pd.Series]:
    idx = pd.date_range("2024-01-01", periods=48, freq="h")
    y = pd.Series(np.linspace(10, 20, 48), index=idx)
    return y, y.copy()


def test_prediccion_perfecta_error_cero(serie_perfecta) -> None:
    y, yhat = serie_perfecta
    assert mae(y, yhat) == pytest.approx(0.0)
    assert rmse(y, yhat) == pytest.approx(0.0)
    assert mape(y, yhat) == pytest.approx(0.0)
    assert smape(y, yhat) == pytest.approx(0.0)
    assert r2(y, yhat) == pytest.approx(1.0)


def test_valores_conocidos() -> None:
    y = np.array([10.0, 20.0, 30.0, 40.0])
    yhat = np.array([12.0, 18.0, 33.0, 36.0])
    # errores: +2, -2, +3, -4 -> MAE = 11/4
    assert mae(y, yhat) == pytest.approx(2.75)
    assert rmse(y, yhat) == pytest.approx(np.sqrt((4 + 4 + 9 + 16) / 4))
    # MAPE = media(0.2, 0.1, 0.1, 0.1) * 100
    assert mape(y, yhat) == pytest.approx(12.5)


def test_mbe_detecta_sesgo() -> None:
    y = np.array([10.0, 10.0, 10.0])
    assert mbe(y, y + 3) == pytest.approx(3.0)   # sobreestima
    assert mbe(y, y - 3) == pytest.approx(-3.0)  # infraestima


def test_rmse_penaliza_mas_que_mae() -> None:
    """Un error grande concentrado sube el RMSE por encima del MAE."""
    y = np.zeros(10)
    yhat = np.zeros(10)
    yhat[0] = 10.0
    assert rmse(y, yhat) > mae(y, yhat)


def test_smape_acotado_en_200() -> None:
    y = np.array([1.0, 2.0, 3.0])
    yhat = np.array([-1.0, -2.0, -3.0])
    assert 0 <= smape(y, yhat) <= 200


def test_mape_ignora_ceros() -> None:
    y = np.array([0.0, 10.0])
    yhat = np.array([5.0, 11.0])
    # solo se evalúa el segundo par -> 10%
    assert mape(y, yhat) == pytest.approx(10.0)


def test_alineacion_por_indice() -> None:
    """Series con distinto rango se alinean por timestamp, no por posición."""
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    y = pd.Series(np.arange(10, dtype=float), index=idx)
    yhat = pd.Series(np.arange(5, 10, dtype=float), index=idx[5:])
    assert mae(y, yhat) == pytest.approx(0.0)


def test_series_sin_solape_falla() -> None:
    y = pd.Series([1.0], index=pd.date_range("2024-01-01", periods=1, freq="h"))
    yhat = pd.Series([1.0], index=pd.date_range("2025-01-01", periods=1, freq="h"))
    with pytest.raises(ValueError):
        mae(y, yhat)


def test_dimensiones_incompatibles_falla() -> None:
    with pytest.raises(ValueError):
        mae(np.zeros(5), np.zeros(4))


def test_calcular_metricas_devuelve_cuadro_completo(serie_perfecta) -> None:
    y, yhat = serie_perfecta
    res = calcular_metricas(y, yhat, nombre="test")
    assert res["modelo"] == "test"
    assert res["n"] == 48
    for m in ("MAE", "RMSE", "MAPE", "sMAPE", "R2", "MBE"):
        assert m in res


def test_comparar_modelos_ordena_por_mae() -> None:
    idx = pd.date_range("2024-01-01", periods=24, freq="h")
    y = pd.Series(np.full(24, 10.0), index=idx)
    resultados = {
        "malo": (y, y + 5),
        "bueno": (y, y + 1),
    }
    tabla = comparar_modelos(resultados, ordenar_por="MAE")
    assert tabla.iloc[0]["modelo"] == "bueno"
    assert tabla.iloc[-1]["modelo"] == "malo"


def test_mejora_relativa() -> None:
    assert mejora_relativa(8.0, 10.0) == pytest.approx(20.0)
    assert mejora_relativa(12.0, 10.0) == pytest.approx(-20.0)
