from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig
from tfm_energia.models.anomaly_detection import (
    ConfigDeteccion,
    DetectorCompuesto,
    DetectorEstadistico,
    DetectorIsolationForest,
    DetectorSensorCongelado,
    _episodios,
    construir_features_anomalia,
    evaluar_deteccion,
    predecir_con_presupuesto,
    referencia_local,
)


@pytest.fixture
def df_anual() -> pd.DataFrame:
    """Un año de Madrid: suficiente para que la referencia local se estabilice."""
    return (
        OfficeSimulator(
            "madrid",
            SEDES["madrid"],
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            cfg=SimulationConfig(seed=42),
        )
        .generate()
        .set_index("timestamp")
    )


# ---------------------------------------------------------------------------
# Referencia local
# ---------------------------------------------------------------------------
def test_referencia_solo_mira_al_pasado() -> None:
    """Un valor no puede entrar en su propia referencia: sería mirar el futuro."""
    idx = pd.date_range("2025-01-01", periods=24 * 40, freq="h")
    serie = pd.Series(np.arange(len(idx), dtype=float), index=idx)
    ref = referencia_local(serie, dias=5)

    # La referencia de cada punto debe ser menor que el propio valor, porque la
    # serie es creciente y solo puede usar observaciones anteriores
    validos = ref.notna()
    assert (ref[validos] < serie[validos]).all()


def test_referencia_separa_laborables_de_festivos() -> None:
    """Sin separar, la mediana mezcla oficina llena y oficina vacía."""
    idx = pd.date_range("2025-01-06", periods=24 * 28, freq="h")  # empieza en lunes
    laborable = pd.Series(idx.dayofweek < 5, index=idx)
    # Consumo alto en laborables, bajo en fines de semana
    serie = pd.Series(np.where(laborable, 100.0, 10.0), index=idx)

    ref = referencia_local(serie, dias=7, laborable=laborable)
    validos = ref.notna()

    # Cada tipo de día debe compararse consigo mismo, no con la mezcla
    assert ref[validos & laborable].round(1).eq(100.0).all()
    assert ref[validos & ~laborable].round(1).eq(10.0).all()


def test_referencia_local_neutraliza_mejor_que_el_perfil_global() -> None:
    """Es la razón de usar referencia local en vez del perfil de todo el año.

    Con una serie puramente estacional y sin ninguna anomalía, el residuo ideal
    sería cero. El perfil anual deja un residuo enorme —en enero todo parece
    anómalo— mientras que la referencia local lo cancela casi por completo.
    """
    idx = pd.date_range("2025-01-01", periods=24 * 300, freq="h")
    estacional = 50 + 40 * np.cos(2 * np.pi * np.arange(len(idx)) / (24 * 365))
    serie = pd.Series(estacional, index=idx)

    local = (serie - referencia_local(serie, dias=14)).abs()
    # Perfil global: la media de esa hora en TODA la serie
    perfil_global = serie.groupby(serie.index.hour).transform("mean")
    global_ = (serie - perfil_global).abs()

    # La serie es determinista, así que estas cotas no dependen de ninguna
    # semilla. Valores medidos: media 6,65 frente a 22,98 y máximo 18,74 frente
    # a 47,00.
    assert local.mean() < global_.mean() / 3
    assert local.max() < global_.max() / 2


# ---------------------------------------------------------------------------
# Construcción de variables
# ---------------------------------------------------------------------------
def test_features_sin_nulos(df_anual: pd.DataFrame) -> None:
    X = construir_features_anomalia(df_anual)
    assert len(X) == len(df_anual)
    assert not X.isna().any().any(), "Las features no pueden llevar nulos al detector"


def test_features_incluyen_residuo_y_variacion(df_anual: pd.DataFrame) -> None:
    X = construir_features_anomalia(df_anual)
    assert any(c.endswith("_residuo") for c in X.columns)
    assert any("_std_" in c for c in X.columns)


def test_features_exigen_indice_temporal() -> None:
    with pytest.raises(ValueError, match="DatetimeIndex"):
        construir_features_anomalia(pd.DataFrame({"consumo_total_kwh": [1.0, 2.0]}))


def test_la_variacion_delata_al_sensor_congelado(df_anual: pd.DataFrame) -> None:
    """Es la única señal que distingue un sensor bloqueado."""
    X = construir_features_anomalia(df_anual)
    col = [c for c in X.columns if "temperatura_interior_c_std_" in c][0]

    congelado = df_anual["tipo_anomalia"] == "SENSOR_FROZEN"
    if not congelado.any():
        pytest.skip("No se inyectaron sensores congelados en el periodo")

    assert X.loc[congelado, col].median() < X.loc[~df_anual["es_anomalia"], col].median()


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------
def test_regla_detecta_todos_los_episodios_congelados(df_anual: pd.DataFrame) -> None:
    """Cada episodio debe generar al menos un aviso.

    A nivel de hora el recall está acotado por la geometría de la ventana: con
    una ventana de 3 horas, las dos primeras de cada episodio todavía mezclan
    lecturas previas y su desviación no llega a cero, así que solo son
    detectables `duración − 2` horas. Lo que sí está garantizado —y es lo que
    importa en operación— es que ningún episodio pase desapercibido.
    """
    X = construir_features_anomalia(df_anual)
    pred = DetectorSensorCongelado().fit(X).predict(X)

    congelado = df_anual["tipo_anomalia"] == "SENSOR_FROZEN"
    if not congelado.any():
        pytest.skip("Sin sensores congelados en el periodo")

    episodios = _episodios(congelado)
    detectados = [ep for ep in episodios if pred[ep].any()]
    assert len(detectados) == len(episodios), (
        f"Solo se detectaron {len(detectados)} de {len(episodios)} episodios"
    )


def test_regla_apenas_produce_falsos_positivos(df_anual: pd.DataFrame) -> None:
    X = construir_features_anomalia(df_anual)
    pred = DetectorSensorCongelado().fit(X).predict(X)

    normales = ~df_anual["es_anomalia"].to_numpy()
    assert pred[normales].mean() < 0.02


def test_regla_falla_sin_columna_de_variacion() -> None:
    detector = DetectorSensorCongelado()
    with pytest.raises(ValueError, match="desviación típica"):
        detector.predict(pd.DataFrame({"otra": [1.0, 2.0]}))


def test_isolation_forest_excluye_patrones(df_anual: pd.DataFrame) -> None:
    """Las señales con canal propio no deben diluir el bosque."""
    X = construir_features_anomalia(df_anual)
    cfg = ConfigDeteccion(patrones_excluidos=("_residuo_medio_",))
    detector = DetectorIsolationForest(cfg).fit(X)

    assert detector._columnas is not None
    assert not any("_residuo_medio_" in c for c in detector._columnas)
    assert len(detector._columnas) < len(X.columns)


def test_isolation_forest_marca_la_fraccion_pedida(df_anual: pd.DataFrame) -> None:
    X = construir_features_anomalia(df_anual)
    detector = DetectorIsolationForest(ConfigDeteccion(contaminacion=0.05)).fit(X)
    assert detector.predict(X).mean() == pytest.approx(0.05, abs=0.01)


def test_detector_sin_ajustar_falla(df_anual: pd.DataFrame) -> None:
    X = construir_features_anomalia(df_anual)
    with pytest.raises(RuntimeError, match="fit"):
        DetectorIsolationForest().predict(X)


def test_estadistico_exige_su_columna() -> None:
    with pytest.raises(ValueError, match="necesita la columna"):
        DetectorEstadistico(columna="no_existe").fit(pd.DataFrame({"otra": [1.0]}))


def test_compuesto_dispara_si_cualquier_canal_lo_hace() -> None:
    class Fijo(DetectorSensorCongelado):
        def __init__(self, salida: np.ndarray, nombre: str) -> None:
            self.salida = salida
            self.nombre = nombre

        def fit(self, X):  # noqa: ANN001
            return self

        def predict(self, X):  # noqa: ANN001
            return self.salida

        def puntuar(self, X):  # noqa: ANN001
            return self.salida.astype(float)

    a = Fijo(np.array([True, False, False]), "a")
    b = Fijo(np.array([False, True, False]), "b")
    compuesto = DetectorCompuesto([a, b]).fit(pd.DataFrame(index=range(3)))

    np.testing.assert_array_equal(compuesto.predict(None), [True, True, False])
    assert compuesto.nombre == "a+b"


def test_compuesto_exige_algun_detector() -> None:
    with pytest.raises(ValueError, match="al menos un detector"):
        DetectorCompuesto([])


def test_cuotas_reparten_el_presupuesto(df_anual: pd.DataFrame) -> None:
    """Cada canal recibe su parte, para que ninguno silencie a los demás."""
    X = construir_features_anomalia(df_anual)
    isolation = DetectorIsolationForest()
    congelado = DetectorSensorCongelado()
    compuesto = DetectorCompuesto([isolation, congelado]).fit(X)

    pred = compuesto.predecir_con_cuotas(X, presupuesto=0.04)
    # Como mucho el presupuesto total; menos si los canales coinciden
    assert 0 < pred.mean() <= 0.041


# ---------------------------------------------------------------------------
# Presupuesto de avisos
# ---------------------------------------------------------------------------
def test_presupuesto_marca_el_numero_pedido() -> None:
    puntuaciones = np.arange(1000, dtype=float)
    pred = predecir_con_presupuesto(puntuaciones, 0.05)
    assert pred.sum() == 50
    # Y son las de mayor puntuación
    assert pred[-50:].all()
    assert not pred[:-50].any()


def test_presupuesto_marca_al_menos_uno() -> None:
    assert predecir_con_presupuesto(np.arange(10, dtype=float), 0.0001).sum() >= 1


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def test_episodios_agrupa_horas_consecutivas() -> None:
    serie = pd.Series([False, True, True, False, False, True, False])
    eps = _episodios(serie)
    assert len(eps) == 2
    np.testing.assert_array_equal(eps[0], [1, 2])
    np.testing.assert_array_equal(eps[1], [5])


def test_metricas_con_valores_conocidos() -> None:
    idx = pd.date_range("2025-01-01", periods=10, freq="h")
    real = pd.Series([False] * 6 + [True] * 4, index=idx)
    pred = np.array([False] * 5 + [True] * 3 + [False] * 2)

    res = evaluar_deteccion(pred, real, nombre="prueba")
    assert res.verdaderos_positivos == 2   # posiciones 6 y 7
    assert res.falsos_positivos == 1       # posición 5
    assert res.falsos_negativos == 2       # posiciones 8 y 9
    assert res.precision == pytest.approx(2 / 3)
    assert res.recall == pytest.approx(0.5)
    assert res.f1 == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))


def test_recall_por_episodio_es_mayor_que_por_hora() -> None:
    """Detectar una hora de un episodio de seis ya genera el aviso."""
    idx = pd.date_range("2025-01-01", periods=12, freq="h")
    real = pd.Series([False] * 6 + [True] * 6, index=idx)
    pred = np.array([False] * 6 + [True] + [False] * 5)

    res = evaluar_deteccion(pred, real)
    assert res.recall == pytest.approx(1 / 6)
    assert res.episodios_totales == 1
    assert res.recall_episodios == 1.0


def test_recall_por_tipo() -> None:
    idx = pd.date_range("2025-01-01", periods=6, freq="h")
    real = pd.Series([True, True, True, True, False, False], index=idx)
    tipos = pd.Series(["A", "A", "B", "B", "", ""], index=idx)
    pred = np.array([True, True, False, False, False, False])

    res = evaluar_deteccion(pred, real, tipos)
    assert res.recall_por_tipo == {"A": 1.0, "B": 0.0}


def test_metricas_sin_umbral_si_hay_puntuaciones() -> None:
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    real = pd.Series([False] * 90 + [True] * 10, index=idx)
    # Puntuación perfecta: las anomalías son las más altas
    puntuaciones = np.r_[np.zeros(90), np.ones(10)]

    res = evaluar_deteccion(puntuaciones > 0.5, real, puntuaciones=puntuaciones)
    assert res.roc_auc == pytest.approx(1.0)
    assert res.average_precision == pytest.approx(1.0)


def test_dimensiones_incompatibles_fallan() -> None:
    idx = pd.date_range("2025-01-01", periods=5, freq="h")
    with pytest.raises(ValueError, match="Dimensiones"):
        evaluar_deteccion(np.array([True, False]), pd.Series([True] * 5, index=idx))
