from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_energia.models.base import backtest_horizonte, metricas_por_horizonte
from tfm_energia.models.baseline import (
    MediaMovil,
    MediaPerfilSemanal,
    NaiveEstacional,
    NaiveEstacionalSemanal,
    NaivePersistente,
    baselines_estandar,
)


@pytest.fixture
def serie_diaria() -> pd.Series:
    """Serie horaria de 30 días con patrón diario perfectamente periódico."""
    idx = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    patron = 10 + 5 * np.sin(2 * np.pi * np.arange(24) / 24)
    return pd.Series(np.tile(patron, 30), index=idx)


@pytest.fixture
def serie_semanal() -> pd.Series:
    """Serie con patrón semanal: laborables altos, fin de semana bajos."""
    idx = pd.date_range("2024-01-01", periods=24 * 70, freq="h")  # 10 semanas
    base = 10 + 5 * np.sin(2 * np.pi * idx.hour / 24)
    factor = np.where(idx.dayofweek >= 5, 0.4, 1.0)
    return pd.Series(base * factor, index=idx)


def test_naive_persistente_repite_ultimo_valor(serie_diaria: pd.Series) -> None:
    modelo = NaivePersistente().fit(serie_diaria)
    pred = modelo.predict(48)
    assert len(pred) == 48
    assert (pred == serie_diaria.iloc[-1]).all()


def test_naive_estacional_reproduce_patron(serie_diaria: pd.Series) -> None:
    """Con un patrón diario exacto, el naïve-24h debe acertar de pleno."""
    modelo = NaiveEstacional(m=24).fit(serie_diaria)
    pred = modelo.predict(48)
    esperado = np.tile(serie_diaria.iloc[-24:].to_numpy(), 2)
    np.testing.assert_allclose(pred.to_numpy(), esperado)


def test_naive_estacional_semanal_capta_finde(serie_semanal: pd.Series) -> None:
    """El naïve-168h debe batir al naïve-24h cuando hay efecto fin de semana."""
    y_train = serie_semanal.iloc[: -24 * 7]
    y_test = serie_semanal.iloc[-24 * 7 :]

    pred_24 = NaiveEstacional(m=24).fit(y_train).predict(len(y_test))
    pred_168 = NaiveEstacionalSemanal().fit(y_train).predict(len(y_test))

    err_24 = np.mean(np.abs(y_test.to_numpy() - pred_24.to_numpy()))
    err_168 = np.mean(np.abs(y_test.to_numpy() - pred_168.to_numpy()))
    assert err_168 < err_24


def test_indice_futuro_continua_la_serie(serie_diaria: pd.Series) -> None:
    modelo = NaiveEstacional(m=24).fit(serie_diaria)
    pred = modelo.predict(48)
    assert pred.index[0] == serie_diaria.index[-1] + pd.Timedelta(hours=1)
    assert pred.index.freqstr.lower() in {"h", "1h"}


def test_media_perfil_semanal_usa_dia_y_hora(serie_semanal: pd.Series) -> None:
    modelo = MediaPerfilSemanal().fit(serie_semanal)
    pred = modelo.predict(168)
    # El perfil medio debe reproducir la caída de fin de semana
    finde = pred[pred.index.dayofweek >= 5].mean()
    laborable = pred[pred.index.dayofweek < 5].mean()
    assert finde < laborable


def test_media_movil_es_constante(serie_diaria: pd.Series) -> None:
    modelo = MediaMovil(ventana=24).fit(serie_diaria)
    pred = modelo.predict(48)
    assert pred.nunique() == 1
    assert pred.iloc[0] == pytest.approx(serie_diaria.iloc[-24:].mean())


def test_predecir_sin_fit_falla() -> None:
    with pytest.raises(RuntimeError):
        NaivePersistente().predict(24)


def test_naive_estacional_exige_historico_suficiente() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="h")
    y = pd.Series(np.arange(10, dtype=float), index=idx)
    with pytest.raises(ValueError):
        NaiveEstacional(m=24).fit(y)


def test_steps_invalido_falla(serie_diaria: pd.Series) -> None:
    modelo = NaivePersistente().fit(serie_diaria)
    with pytest.raises(ValueError):
        modelo.predict(0)


# ---------------------------------------------------------------------------
# Backtesting
# ---------------------------------------------------------------------------
def test_backtest_estructura_y_sin_lookahead(serie_diaria: pd.Series) -> None:
    bt = backtest_horizonte(
        NaiveEstacional(m=24), serie_diaria, horizonte=48, paso=24, min_train=24 * 20
    )
    assert set(bt.columns) == {"origen", "timestamp", "h", "real", "pred"}
    assert bt["h"].between(1, 48).all()
    # Cada predicción es siempre posterior a su origen: no hay fuga de futuro
    assert (bt["timestamp"] > bt["origen"]).all()


def test_backtest_n_origenes(serie_diaria: pd.Series) -> None:
    bt = backtest_horizonte(
        NaiveEstacional(m=24),
        serie_diaria,
        horizonte=24,
        paso=24,
        n_origenes=3,
        min_train=24 * 20,
    )
    assert bt["origen"].nunique() == 3
    assert len(bt) == 3 * 24


def test_backtest_serie_corta_falla(serie_diaria: pd.Series) -> None:
    with pytest.raises(ValueError):
        backtest_horizonte(
            NaivePersistente(), serie_diaria.iloc[:30], horizonte=48, min_train=24 * 20
        )


def test_metricas_por_horizonte(serie_semanal: pd.Series) -> None:
    bt = backtest_horizonte(
        NaivePersistente(), serie_semanal, horizonte=12, paso=24, min_train=24 * 30
    )
    tabla = metricas_por_horizonte(bt)
    assert list(tabla["h"]) == list(range(1, 13))
    # El naïve persistente se degrada al alejarse del origen
    assert tabla.loc[tabla["h"] == 12, "MAE"].iloc[0] > tabla.loc[tabla["h"] == 1, "MAE"].iloc[0]


def test_baselines_estandar_todos_ejecutables(serie_semanal: pd.Series) -> None:
    for modelo in baselines_estandar():
        pred = modelo.fit(serie_semanal).predict(48)
        assert len(pred) == 48
        assert pred.notna().all()
        assert pred.name == modelo.nombre
