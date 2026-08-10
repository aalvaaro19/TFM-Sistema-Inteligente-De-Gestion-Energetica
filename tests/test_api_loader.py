"""Tests de la preparación de datos de las APIs externas (AEMET y e·sios).

El riesgo aquí está en la conversión numérica y en las fechas: AEMET mezcla
decimales con coma y con punto en el mismo fichero, y la serie de precios
contiene las horas ambiguas de los cambios horarios.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import RAW_DIR
from tfm_energia.data.api_loader import (
    COLUMNAS_DESCARTABLES_AEMET,
    _a_numero,
    preparar_meteo,
    preparar_precios,
    resumen_precios,
)


@pytest.fixture
def df_meteo() -> pd.DataFrame:
    """Réplica del formato real de AEMET, con sus rarezas."""
    return pd.DataFrame(
        [
            {
                "fecha": "2024-01-01", "indicativo": "3195", "nombre": "MADRID, RETIRO",
                "provincia": "MADRID", "altitud": 667, "tmed": "6.6", "prec": "0.0",
                "tmin": "3.8", "horatmin": "08:50", "tmax": "9.4", "horatmax": "15:40",
                "dir": "23", "velmedia": "1.4", "racha": "6.4", "horaracha": "15:50",
                "presMax": "945,1", "horaPresMax": "23", "presMin": "940,8",
                "horaPresMin": "03", "hrMedia": "85.0", "hrMax": "99",
                "horaHrMax": "Varias", "hrMin": "62", "horaHrMin": "15:30",
            },
            {
                "fecha": "2024-01-02", "indicativo": "3195", "nombre": "MADRID, RETIRO",
                "provincia": "MADRID", "altitud": 667, "tmed": "5.5", "prec": "2.0",
                "tmin": "2.8", "horatmin": "05:50", "tmax": "8.2", "horatmax": "23:40",
                "dir": "14", "velmedia": "0.3", "racha": "4.4", "horaracha": "02:10",
                "presMax": "945,2", "horaPresMax": "11", "presMin": "941,7",
                "horaPresMin": "24", "hrMedia": np.nan, "hrMax": "99",
                "horaHrMax": "Varias", "hrMin": "83", "horaHrMin": "13:20",
            },
        ]
    )


@pytest.fixture
def df_precios() -> pd.DataFrame:
    """Horas alrededor del cambio a horario de invierno de 2024."""
    return pd.DataFrame(
        [
            {"fecha_local": "2024-10-27 01:00:00+02:00", "precio_eur_mwh": 135.07,
             "precio_eur_kwh": 0.13507, "franja_pvpc": "valle"},
            {"fecha_local": "2024-10-27 02:00:00+02:00", "precio_eur_mwh": 135.26,
             "precio_eur_kwh": 0.13526, "franja_pvpc": "valle"},
            {"fecha_local": "2024-10-27 02:00:00+01:00", "precio_eur_mwh": 134.21,
             "precio_eur_kwh": 0.13421, "franja_pvpc": "valle"},
            {"fecha_local": "2024-10-27 03:00:00+01:00", "precio_eur_mwh": 135.63,
             "precio_eur_kwh": 0.13563, "franja_pvpc": "punta"},
        ]
    )


# ---------------------------------------------------------------------------
# Conversión numérica
# ---------------------------------------------------------------------------
def test_convierte_coma_decimal() -> None:
    """AEMET publica la presión con coma: `"945,1"`."""
    assert _a_numero("945,1") == pytest.approx(945.1)
    assert _a_numero("940,8") == pytest.approx(940.8)


def test_convierte_punto_decimal() -> None:
    """Y la temperatura con punto, en el mismo fichero."""
    assert _a_numero("6.6") == pytest.approx(6.6)


def test_numeros_pasan_tal_cual() -> None:
    assert _a_numero(667) == 667.0
    assert _a_numero(6.6) == pytest.approx(6.6)


def test_valores_no_numericos_dan_none() -> None:
    """`"Varias"` aparece cuando el máximo se alcanzó a varias horas."""
    assert _a_numero("Varias") is None
    assert _a_numero("") is None
    assert _a_numero("   ") is None
    assert _a_numero(None) is None
    assert _a_numero(np.nan) is None


def test_negativos_y_ceros() -> None:
    assert _a_numero("-3,5") == pytest.approx(-3.5)
    assert _a_numero("0.0") == 0.0


# ---------------------------------------------------------------------------
# AEMET
# ---------------------------------------------------------------------------
def test_meteo_un_documento_por_dia(df_meteo: pd.DataFrame) -> None:
    docs = preparar_meteo(df_meteo, "madrid")
    assert len(docs) == 2


def test_meteo_descarta_columnas_de_hora(df_meteo: pd.DataFrame) -> None:
    docs = preparar_meteo(df_meteo, "madrid")
    for columna in COLUMNAS_DESCARTABLES_AEMET:
        assert columna not in docs[0]


def test_meteo_anade_sede_y_fuente(df_meteo: pd.DataFrame) -> None:
    docs = preparar_meteo(df_meteo, "madrid")
    assert all(d["sede"] == "madrid" for d in docs)
    assert all(d["fuente"] == "AEMET OpenData" for d in docs)


def test_meteo_convierte_la_presion(df_meteo: pd.DataFrame) -> None:
    docs = preparar_meteo(df_meteo, "madrid")
    assert docs[0]["presMax"] == pytest.approx(945.1)
    assert docs[0]["presMin"] == pytest.approx(940.8)


def test_meteo_fecha_es_datetime_con_zona(df_meteo: pd.DataFrame) -> None:
    docs = preparar_meteo(df_meteo, "madrid")
    fecha = docs[0]["fecha"]
    assert isinstance(fecha, datetime)
    assert fecha.tzinfo is not None


def test_meteo_sustituye_nan_por_none(df_meteo: pd.DataFrame) -> None:
    """Un NaN de pandas no es un valor útil en MongoDB."""
    docs = preparar_meteo(df_meteo, "madrid")
    assert docs[1]["hrMedia"] is None
    assert not any(
        isinstance(v, float) and np.isnan(v) for d in docs for v in d.values()
    )


def test_meteo_no_muta_el_dataframe_original(df_meteo: pd.DataFrame) -> None:
    columnas_antes = list(df_meteo.columns)
    preparar_meteo(df_meteo, "madrid")
    assert list(df_meteo.columns) == columnas_antes
    assert df_meteo["presMax"].iloc[0] == "945,1"


# ---------------------------------------------------------------------------
# e·sios
# ---------------------------------------------------------------------------
def test_precios_un_documento_por_hora(df_precios: pd.DataFrame) -> None:
    docs = preparar_precios(df_precios)
    assert len(docs) == 4
    assert all(d["fuente"] == "e·sios REE" for d in docs)


def test_precios_fecha_con_zona(df_precios: pd.DataFrame) -> None:
    docs = preparar_precios(df_precios)
    assert all(d["fecha_local"].tzinfo is not None for d in docs)


def test_precios_horas_ambiguas_son_instantes_distintos(df_precios: pd.DataFrame) -> None:
    """Las dos 02:00 del día de 25 horas no pueden colapsar en una.

    Es el caso que provocaba la pérdida de un precio en cada cambio horario.
    """
    docs = preparar_precios(df_precios)
    instantes = [d["fecha_local"].astimezone(timezone.utc) for d in docs]
    assert len(set(instantes)) == 4, "Cada fila debe seguir siendo un instante único"

    # 01:00+02:00 → 23:00 UTC del día anterior; las dos 02:00 → 00:00 y 01:00 UTC
    assert [i.hour for i in instantes] == [23, 0, 1, 2]

    dos_de_la_madrugada = [
        d["fecha_local"].astimezone(timezone.utc)
        for d in docs
        if d["fecha_local"].hour == 2
    ]
    assert len(dos_de_la_madrugada) == 2
    assert dos_de_la_madrugada[1] - dos_de_la_madrugada[0] == timedelta(hours=1)


def test_precios_orden_temporal_correcto(df_precios: pd.DataFrame) -> None:
    """Ordenados por instante, la 02:00 CEST va antes que la 02:00 CET."""
    docs = preparar_precios(df_precios)
    por_instante = sorted(docs, key=lambda d: d["fecha_local"])
    assert [d["precio_eur_kwh"] for d in por_instante] == [
        0.13507, 0.13526, 0.13421, 0.13563
    ]


# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
def test_resumen_precios(df_precios: pd.DataFrame) -> None:
    res = resumen_precios(preparar_precios(df_precios))
    assert res["n"] == 4
    assert res["precio_min"] == pytest.approx(0.13421)
    assert res["precio_max"] == pytest.approx(0.13563)
    assert res["por_franja"] == {"valle": 3, "punta": 1}


def test_resumen_sin_precios_no_falla() -> None:
    res = resumen_precios([])
    assert res["n"] == 0
    assert res["precio_medio"] is None


# ---------------------------------------------------------------------------
# Contra los ficheros reales (se omite si no se han descargado)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not (RAW_DIR / "aemet" / "meteo_madrid.csv").exists(),
    reason="Requiere ejecutar antes scripts/download_aemet.py",
)
def test_formato_real_de_aemet_no_ha_cambiado() -> None:
    """Detecta que AEMET cambie el formato de sus columnas."""
    docs = preparar_meteo(pd.read_csv(RAW_DIR / "aemet" / "meteo_madrid.csv"), "madrid")
    assert len(docs) > 700
    assert all(d["sede"] == "madrid" for d in docs)
    assert any(d.get("tmed") is not None for d in docs)


@pytest.mark.skipif(
    not (RAW_DIR / "esios" / "pvpc_horario.csv").exists(),
    reason="Requiere ejecutar antes scripts/download_esios.py",
)
def test_todas_las_horas_reales_son_unicas() -> None:
    """Ninguna hora del histórico real puede colapsar con otra."""
    docs = preparar_precios(pd.read_csv(RAW_DIR / "esios" / "pvpc_horario.csv"))
    instantes = {d["fecha_local"].astimezone(timezone.utc) for d in docs}
    assert len(instantes) == len(docs), (
        f"{len(docs) - len(instantes)} horas colapsan en el mismo instante"
    )
