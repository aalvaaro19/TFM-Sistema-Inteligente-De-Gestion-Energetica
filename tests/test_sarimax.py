from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_energia.models.sarimax_model import (
    FOURIER_DEFAULT,
    SarimaxForecaster,
    terminos_fourier,
)


@pytest.fixture
def serie_exog() -> tuple[pd.Series, pd.DataFrame]:
    """21 días horarios con ciclo diario, tendencia térmica y ruido."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-06-01", periods=24 * 21, freq="h", tz="Europe/Madrid")
    temperatura = 22 + 6 * np.sin(2 * np.pi * (idx.hour - 6) / 24) + rng.normal(0, 0.5, len(idx))
    ocupacion = np.where((idx.hour >= 8) & (idx.hour < 19) & (idx.dayofweek < 5), 0.8, 0.05)
    consumo = 5 + 0.8 * np.clip(temperatura - 21, 0, None) + 12 * ocupacion + rng.normal(0, 0.3, len(idx))

    X = pd.DataFrame(
        {"temperatura_exterior_c": temperatura, "ocupacion_rel": ocupacion}, index=idx
    )
    return pd.Series(consumo, index=idx, name="consumo_total_kwh"), X


# ---------------------------------------------------------------------------
# Términos de Fourier
# ---------------------------------------------------------------------------
def test_terminos_fourier_forma_y_rango() -> None:
    idx = pd.date_range("2024-01-01", periods=200, freq="h")
    F = terminos_fourier(idx, FOURIER_DEFAULT)
    # 3 armónicos (24h) + 2 armónicos (168h), seno y coseno cada uno
    assert F.shape == (200, (3 + 2) * 2)
    assert F.to_numpy().min() >= -1.0 and F.to_numpy().max() <= 1.0


def test_terminos_fourier_periodicidad_diaria() -> None:
    """El primer armónico de 24 h debe repetirse exactamente cada 24 horas."""
    idx = pd.date_range("2024-01-01", periods=72, freq="h")
    F = terminos_fourier(idx, ((24, 1),))
    np.testing.assert_allclose(
        F["fourier_sin_24_1"].iloc[:24].to_numpy(),
        F["fourier_sin_24_1"].iloc[24:48].to_numpy(),
        atol=1e-9,
    )


def test_terminos_fourier_alineados_con_hora_local() -> None:
    """Con tz-aware se usa la hora de reloj local, no la UTC."""
    idx = pd.date_range("2024-06-01", periods=48, freq="h", tz="Europe/Madrid")
    F = terminos_fourier(idx, ((24, 1),))
    horas = idx.hour
    # Todos los timestamps con la misma hora local comparten valor del armónico
    for h in (0, 9, 15):
        valores = F.loc[horas == h, "fourier_sin_24_1"].round(9).unique()
        assert len(valores) == 1


# ---------------------------------------------------------------------------
# Ajuste y predicción
# ---------------------------------------------------------------------------
def test_sarimax_fourier_con_exogenas(serie_exog) -> None:
    y, X = serie_exog
    modelo = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 14)
    modelo.fit(y, X)
    pred = modelo.predict(48, X.iloc[:48])

    assert len(pred) == 48
    assert pred.notna().all()
    assert (pred >= 0).all()  # el consumo nunca es negativo
    assert pred.index[0] == y.index[-1] + pd.Timedelta(hours=1)
    assert pred.name == modelo.nombre


def test_sarimax_univariante_no_requiere_exogenas(serie_exog) -> None:
    y, _ = serie_exog
    modelo = SarimaxForecaster(
        order=(1, 0, 0), seasonal_order=(0, 0, 0, 0), exogenas=None, max_train_horas=24 * 10
    )
    modelo.fit(y)
    pred = modelo.predict(24)
    assert len(pred) == 24
    assert "univar" in modelo.nombre


def test_sarimax_falla_si_faltan_exogenas(serie_exog) -> None:
    y, X = serie_exog
    modelo = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 7)
    with pytest.raises(ValueError, match="no se pasó X"):
        modelo.fit(y)

    with pytest.raises(ValueError, match="Faltan exógenas"):
        modelo.fit(y, X.drop(columns=["ocupacion_rel"]))


def test_sarimax_ventana_de_entrenamiento_recorta(serie_exog) -> None:
    """Con max_train_horas solo se usa la cola de la serie, pero el índice
    de predicción sigue arrancando tras el último dato disponible."""
    y, X = serie_exog
    modelo = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 5)
    modelo.fit(y, X)
    assert modelo.resultado.nobs == 24 * 5
    pred = modelo.predict(12, X.iloc[:12])
    assert pred.index[0] == y.index[-1] + pd.Timedelta(hours=1)


def test_sarimax_genera_intervalo_de_confianza(serie_exog) -> None:
    y, X = serie_exog
    modelo = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 10)
    modelo.fit(y, X)
    pred = modelo.predict(24, X.iloc[:24])

    ic = modelo.ultimo_intervalo
    assert ic is not None and len(ic) == 24
    assert (ic["inferior"] <= ic["superior"]).all()
    # La incertidumbre no decrece al alejarse del origen
    amplitud = ic["superior"] - ic["inferior"]
    assert amplitud.iloc[-1] >= amplitud.iloc[0] - 1e-6


def test_sarimax_incluye_constante_si_no_diferencia(serie_exog) -> None:
    """Sin diferenciación hace falta constante, o el modelo revierte hacia 0.

    Regresión: statsmodels usa `trend=None` por defecto, lo que provocaba una
    infraestimación sistemática del nivel de consumo (MBE muy negativo).
    """
    y, X = serie_exog
    modelo = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 10)
    assert modelo.trend == "c"
    modelo.fit(y, X)
    pred = modelo.predict(48, X.iloc[:48])
    # El nivel medio predicho debe parecerse al de la serie, no caer hacia 0
    assert abs(pred.mean() - y.mean()) < 0.4 * y.mean()

    # Con diferenciación la constante sobra
    assert SarimaxForecaster(order=(1, 1, 0), usar_fourier=True).trend == "n"


def test_sarimax_bate_a_media_movil(serie_exog) -> None:
    """Control de sanidad: con exógenas informativas debe superar al baseline plano."""
    from tfm_energia.models.baseline import MediaMovil
    from tfm_energia.models.metrics import mae

    y, X = serie_exog
    y_train, y_test = y.iloc[:-48], y.iloc[-48:]
    X_train, X_test = X.iloc[:-48], X.iloc[-48:]

    sarimax = SarimaxForecaster(order=(1, 0, 0), usar_fourier=True, max_train_horas=24 * 14)
    pred_sarimax = sarimax.fit(y_train, X_train).predict(48, X_test)
    pred_baseline = MediaMovil(ventana=24).fit(y_train).predict(48)

    assert mae(y_test, pred_sarimax) < mae(y_test, pred_baseline)
