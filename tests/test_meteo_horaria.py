"""Tests de la reconstrucción horaria de la observación diaria de AEMET.

Este módulo gobierna la física del edificio: la temperatura que reconstruye es la
que produce la temperatura interior y el consumo de climatización. Un sesgo aquí
se propaga a todo el proyecto sin dar ningún error.

El test central es el de media nula. La versión anterior de la curva combinaba un
seno diurno con un valor fijo nocturno cuya media era +0,43, de modo que la
temperatura reconstruida quedaba unos 2 °C por encima de la que publica AEMET.
Madrid salía a 18,8 °C de media anual cuando la real es 16,7, y el edificio
necesitaba mucha menos calefacción de la que le corresponde.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import RAW_DIR, SEDES
from tfm_energia.data.meteo_horaria import (
    HORA_MINIMO,
    cargar_diario,
    meteo_real_horaria,
    perfil_horario,
)


@pytest.fixture
def diario() -> pd.DataFrame:
    """Diez días con mínima, máxima y media coherentes."""
    fechas = pd.date_range("2025-01-01", periods=10, freq="D")
    return pd.DataFrame({
        "fecha": fechas,
        "tmed": np.linspace(8.0, 14.0, 10),
        "tmin": np.linspace(3.0, 9.0, 10),
        "tmax": np.linspace(13.0, 19.0, 10),
        "hrMedia": np.full(10, 70.0),
    })


@pytest.fixture
def idx(diario: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.date_range(
        diario["fecha"].min(),
        diario["fecha"].max() + pd.Timedelta(hours=23),
        freq="h", tz="Europe/Madrid",
    )


# ---------------------------------------------------------------------------
# La propiedad crítica: media nula
# ---------------------------------------------------------------------------
def test_la_media_horaria_reproduce_la_media_diaria(
    diario: pd.DataFrame, idx: pd.DatetimeIndex
) -> None:
    """Regresión del sesgo de +2 °C: la curva debe tener media nula.

    Si el factor de forma no promedia cero a lo largo de las 24 horas, la serie
    reconstruida se desplaza respecto a la observación y el edificio simulado
    necesita más o menos climatización de la que le corresponde.
    """
    t = perfil_horario(idx, diario)["temperatura_exterior_c"]
    por_dia = t.groupby(t.index.tz_localize(None).date).mean()
    esperada = diario.set_index(diario["fecha"].dt.date)["tmed"]

    for fecha, media in por_dia.items():
        assert media == pytest.approx(esperada[fecha], abs=0.01), (
            f"El día {fecha} se desvía de la media publicada por AEMET"
        )


def test_reproduce_la_minima_y_la_maxima_diarias(
    diario: pd.DataFrame, idx: pd.DatetimeIndex
) -> None:
    t = perfil_horario(idx, diario)["temperatura_exterior_c"]
    agrupado = t.groupby(t.index.tz_localize(None).date)
    d = diario.set_index(diario["fecha"].dt.date)

    for fecha, minimo in agrupado.min().items():
        assert minimo == pytest.approx(d.loc[fecha, "tmin"], abs=0.01)
    for fecha, maximo in agrupado.max().items():
        assert maximo == pytest.approx(d.loc[fecha, "tmax"], abs=0.01)


def test_el_minimo_cae_al_amanecer(diario: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
    t = perfil_horario(idx, diario)["temperatura_exterior_c"]
    primer_dia = t[t.index.tz_localize(None).date == diario["fecha"].dt.date.iloc[0]]
    assert primer_dia.idxmin().hour == HORA_MINIMO


def test_el_maximo_cae_doce_horas_despues(
    diario: pd.DataFrame, idx: pd.DatetimeIndex
) -> None:
    t = perfil_horario(idx, diario)["temperatura_exterior_c"]
    primer_dia = t[t.index.tz_localize(None).date == diario["fecha"].dt.date.iloc[0]]
    assert primer_dia.idxmax().hour == (HORA_MINIMO + 12) % 24


# ---------------------------------------------------------------------------
# Estructura de la salida
# ---------------------------------------------------------------------------
def test_devuelve_una_fila_por_hora(diario: pd.DataFrame, idx: pd.DatetimeIndex) -> None:
    horaria = perfil_horario(idx, diario)
    assert len(horaria) == len(idx)
    assert horaria.index.equals(idx)
    assert {"temperatura_exterior_c", "humedad_exterior_pct"} <= set(horaria.columns)


def test_la_humedad_se_mantiene_constante_en_el_dia(
    diario: pd.DataFrame, idx: pd.DatetimeIndex
) -> None:
    """AEMET solo publica la media diaria: no se inventa un ciclo horario."""
    hr = perfil_horario(idx, diario)["humedad_exterior_pct"]
    assert hr.groupby(hr.index.tz_localize(None).date).nunique().eq(1).all()


def test_sin_datos_diarios_devuelve_vacio(idx: pd.DatetimeIndex) -> None:
    """Permite al generador recurrir a su meteorología sintética."""
    vacio = perfil_horario(idx, pd.DataFrame())
    assert vacio.empty


def test_indice_sin_zona_horaria(diario: pd.DataFrame) -> None:
    idx = pd.date_range("2025-01-01", periods=48, freq="h")
    t = perfil_horario(idx, diario)["temperatura_exterior_c"]
    assert t.notna().all()


# ---------------------------------------------------------------------------
# Limpieza de la fuente
# ---------------------------------------------------------------------------
def test_convierte_la_coma_decimal(tmp_path, monkeypatch) -> None:
    """Algunas estaciones publican la presión y la humedad con coma decimal."""
    from tfm_energia.data import meteo_horaria

    directorio = tmp_path / "aemet"
    directorio.mkdir()
    (directorio / "meteo_prueba.csv").write_text(
        "fecha,tmed,tmin,tmax,hrMedia\n"
        "2025-01-01,\"8,5\",\"3,2\",\"13,8\",\"70,5\"\n"
        "2025-01-02,9.0,4.0,14.0,71.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(meteo_horaria, "RAW_DIR", tmp_path)

    d = meteo_horaria.cargar_diario("prueba")
    assert d["tmed"].iloc[0] == pytest.approx(8.5)
    assert d["hrMedia"].iloc[0] == pytest.approx(70.5)
    assert d["tmed"].iloc[1] == pytest.approx(9.0)


def test_interpola_los_huecos_de_la_estacion(tmp_path, monkeypatch) -> None:
    from tfm_energia.data import meteo_horaria

    directorio = tmp_path / "aemet"
    directorio.mkdir()
    (directorio / "meteo_prueba.csv").write_text(
        "fecha,tmed,tmin,tmax,hrMedia\n"
        "2025-01-01,8.0,3.0,13.0,70.0\n"
        "2025-01-02,,,,\n"
        "2025-01-03,10.0,5.0,15.0,72.0\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(meteo_horaria, "RAW_DIR", tmp_path)

    d = meteo_horaria.cargar_diario("prueba")
    assert d["tmed"].notna().all()
    assert d["tmed"].iloc[1] == pytest.approx(9.0)


def test_sede_sin_fichero_no_rompe(tmp_path, monkeypatch) -> None:
    from tfm_energia.data import meteo_horaria

    monkeypatch.setattr(meteo_horaria, "RAW_DIR", tmp_path)
    assert meteo_horaria.cargar_diario("inexistente").empty

    idx = pd.date_range("2025-01-01", periods=24, freq="h", tz="Europe/Madrid")
    assert meteo_horaria.meteo_real_horaria("inexistente", idx).empty


# ---------------------------------------------------------------------------
# Contra los ficheros reales
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (RAW_DIR / "aemet" / "meteo_madrid.csv").exists(),
    reason="Requiere ejecutar antes scripts/download_aemet.py",
)
def test_las_series_reales_no_tienen_sesgo() -> None:
    """La comprobación que dio con el bug, sobre los datos de verdad."""
    idx = pd.date_range(
        "2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Madrid"
    )
    for sede in SEDES:
        d = cargar_diario(sede)
        if d.empty:
            continue
        t = perfil_horario(idx, d)["temperatura_exterior_c"]
        sesgo = float(t.mean() - d["tmed"].mean())
        assert abs(sesgo) < 0.05, f"{sede} tiene un sesgo de {sesgo:+.3f} °C"


@pytest.mark.skipif(
    not (RAW_DIR / "aemet" / "meteo_madrid.csv").exists(),
    reason="Requiere ejecutar antes scripts/download_aemet.py",
)
def test_cobertura_completa_del_periodo() -> None:
    idx = pd.date_range(
        "2024-01-01", "2025-12-31 23:00", freq="h", tz="Europe/Madrid"
    )
    horaria = meteo_real_horaria("madrid", idx)
    assert len(horaria) == len(idx)
    assert horaria["temperatura_exterior_c"].notna().all()
