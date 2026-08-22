from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tfm_energia.data.mongo_repository import (
    CAMPOS_FECHA,
    normalizar_fechas,
    parsear_fecha_iso,
)


# ---------------------------------------------------------------------------
# parsear_fecha_iso
# ---------------------------------------------------------------------------
def test_convierte_iso_con_offset() -> None:
    dt = parsear_fecha_iso("2024-01-01T00:00:00+01:00")
    assert isinstance(dt, datetime)
    assert dt.tzinfo is not None
    assert dt.utcoffset() == timedelta(hours=1)


def test_convierte_iso_con_sufijo_z() -> None:
    """`fromisoformat` de Python 3.10 no admite 'Z' y hay que traducirla."""
    dt = parsear_fecha_iso("2024-06-01T10:00:00Z")
    assert isinstance(dt, datetime)
    assert dt.utcoffset() == timedelta(0)


def test_es_idempotente() -> None:
    """Aplicarla dos veces no debe romper nada."""
    dt = parsear_fecha_iso("2024-01-01T00:00:00+01:00")
    assert parsear_fecha_iso(dt) == dt


def test_degrada_pandas_timestamp_a_datetime_nativo() -> None:
    """Regresión: pymongo calcula mal el offset de un `pandas.Timestamp`.

    En la hora ambigua del cambio a horario de invierno, las 02:00+01:00 se
    persistían como 02:00 UTC en vez de 01:00 UTC, colisionando con la hora
    siguiente. Se perdía un precio por cada cambio horario.
    """
    import pandas as pd

    ts = pd.Timestamp("2024-10-27 02:00:00+01:00").tz_convert("Europe/Madrid")
    salida = parsear_fecha_iso(ts)

    assert type(salida) is datetime, f"Debe ser datetime nativo, no {type(salida).__name__}"
    assert salida.astimezone(timezone.utc) == datetime(2024, 10, 27, 1, 0, tzinfo=timezone.utc)


def test_las_dos_horas_ambiguas_no_colisionan() -> None:
    """Las dos 02:00 del día de 25 horas son instantes distintos.

    `pandas` les asigna el mismo hash pese a no ser iguales, que es el origen
    del problema anterior.
    """
    import pandas as pd

    cest = parsear_fecha_iso(pd.Timestamp("2024-10-27 02:00:00+02:00"))
    cet = parsear_fecha_iso(pd.Timestamp("2024-10-27 02:00:00+01:00"))

    assert cest != cet
    assert cest.astimezone(timezone.utc).hour == 0
    assert cet.astimezone(timezone.utc).hour == 1


def test_normalizar_fechas_degrada_timestamps() -> None:
    import pandas as pd

    salida = normalizar_fechas({"fecha_local": pd.Timestamp("2025-03-30 04:00:00+02:00")})
    assert type(salida["fecha_local"]) is datetime


def test_deja_intacto_lo_que_no_es_fecha() -> None:
    assert parsear_fecha_iso("madrid") == "madrid"
    assert parsear_fecha_iso(42.5) == 42.5
    assert parsear_fecha_iso(None) is None


# ---------------------------------------------------------------------------
# normalizar_fechas
# ---------------------------------------------------------------------------
def test_normaliza_solo_los_campos_de_fecha() -> None:
    evento = {
        "sensor_id": "madrid-S3",
        "tipo": "consumo_electrico",
        "timestamp": "2024-01-01T00:00:00+01:00",
        "sede": "madrid",
        "consumo_total_kwh": 8.03,
    }
    salida = normalizar_fechas(evento)

    assert isinstance(salida["timestamp"], datetime)
    assert salida["sensor_id"] == "madrid-S3"
    assert salida["consumo_total_kwh"] == 8.03
    assert salida["sede"] == "madrid"


def test_no_muta_el_documento_original() -> None:
    evento = {"timestamp": "2024-01-01T00:00:00+01:00"}
    normalizar_fechas(evento)
    assert isinstance(evento["timestamp"], str)


def test_documento_sin_fechas_pasa_igual() -> None:
    doc = {"sede": "madrid", "valor": 1}
    assert normalizar_fechas(doc) == doc


@pytest.mark.parametrize("campo", CAMPOS_FECHA)
def test_todos_los_campos_declarados_se_convierten(campo: str) -> None:
    salida = normalizar_fechas({campo: "2025-07-15T12:00:00+02:00"})
    assert isinstance(salida[campo], datetime)


# ---------------------------------------------------------------------------
# El motivo de fondo: por qué no vale guardar texto
# ---------------------------------------------------------------------------
def test_las_cadenas_iso_con_distinto_offset_ordenan_mal() -> None:
    """Justifica el cambio: como texto, el orden no respeta el instante real.

    Las dos cadenas representan el mismo momento (una en CET y otra en CEST),
    pero comparadas como texto dan un resultado distinto al comparar fechas.
    """
    invierno = "2024-01-01T12:00:00+01:00"   # 11:00 UTC
    verano = "2024-06-01T12:00:00+02:00"     # 10:00 UTC

    # Como texto, enero parece anterior a junio
    assert invierno < verano
    # Como instante, la hora UTC de junio es MENOR que la de enero
    assert parsear_fecha_iso(verano).timetz() < parsear_fecha_iso(invierno).timetz()


def test_mismo_instante_distinta_representacion() -> None:
    """Dos cadenas diferentes pueden ser el mismo momento; como texto, no."""
    a = "2024-06-01T02:00:00+02:00"
    b = "2024-06-01T00:00:00+00:00"

    assert a != b                                        # distintas como texto
    assert parsear_fecha_iso(a) == parsear_fecha_iso(b)  # idénticas como fecha


def test_conversion_a_utc_conserva_el_instante() -> None:
    """Es lo que hace pymongo al persistir: convierte a UTC sin perder el momento."""
    dt = parsear_fecha_iso("2024-01-01T00:00:00+01:00")
    assert dt.astimezone(timezone.utc) == datetime(2023, 12, 31, 23, 0, tzinfo=timezone.utc)
