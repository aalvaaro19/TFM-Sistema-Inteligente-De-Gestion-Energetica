"""Tests del simulador de gateways IoT.

Cubren tres cosas: que el formato JSON Lines sea válido y legible, que el
particionado sea el esperado, y que la corrupción controlada produzca
exactamente los defectos que la validación debe detectar.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.sensor_stream import (
    CAMPOS_OBLIGATORIOS,
    TIPOS_DEFECTO,
    VALOR_CENTINELA,
    ConfigEmisor,
    EmisorSensores,
    corromper_evento,
    es_valido,
    leer_jsonl,
)
from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig


@pytest.fixture
def df_demo() -> pd.DataFrame:
    """3 días de Madrid: suficiente para probar particionado por día."""
    sim = OfficeSimulator(
        "madrid",
        SEDES["madrid"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 3),
        cfg=SimulationConfig(seed=42),
    )
    return sim.generate()


@pytest.fixture
def emisor_limpio() -> EmisorSensores:
    """Emisor sin defectos, para probar el camino feliz."""
    return EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0, seed=42))


# ---------------------------------------------------------------------------
# Formato JSON Lines
# ---------------------------------------------------------------------------
def test_escribe_jsonl_una_linea_por_evento(
    emisor_limpio: EmisorSensores, df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    eventos = list(emisor_limpio.eventos_desde_df(df_demo.head(10)))
    destino = tmp_path / "lecturas.jsonl"
    n = emisor_limpio.escribir_jsonl(eventos, destino)

    lineas = destino.read_text(encoding="utf-8").strip().split("\n")
    assert n == len(eventos) == len(lineas)
    # Cada línea debe ser un JSON independiente y válido
    for linea in lineas:
        assert isinstance(json.loads(linea), dict)


def test_ida_y_vuelta_por_leer_jsonl(
    emisor_limpio: EmisorSensores, df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    eventos = list(emisor_limpio.eventos_desde_df(df_demo.head(5)))
    destino = tmp_path / "lecturas.jsonl"
    emisor_limpio.escribir_jsonl(eventos, destino)

    recuperados = list(leer_jsonl(destino))
    assert recuperados == eventos


def test_leer_jsonl_no_aborta_ante_una_linea_corrupta(tmp_path: Path) -> None:
    """Una línea ilegible se marca como error, pero la lectura continúa."""
    destino = tmp_path / "roto.jsonl"
    destino.write_text(
        '{"sensor_id": "madrid-S1"}\n'
        "esto no es json\n"
        '{"sensor_id": "madrid-S2"}\n',
        encoding="utf-8",
    )
    leidos = list(leer_jsonl(destino))

    assert len(leidos) == 3
    assert "_error_parseo" in leidos[1]
    assert leidos[1]["_linea"] == 2
    assert leidos[2]["sensor_id"] == "madrid-S2"


# ---------------------------------------------------------------------------
# Metadatos del gateway
# ---------------------------------------------------------------------------
def test_eventos_llevan_metadatos_de_gateway(
    emisor_limpio: EmisorSensores, df_demo: pd.DataFrame
) -> None:
    ev = next(iter(emisor_limpio.eventos_desde_df(df_demo.head(1))))
    assert ev["gateway_id"] == "gw-madrid-01"
    assert "ingest_ts" in ev
    # El instante de emisión es distinto del de la lectura
    assert ev["ingest_ts"] != ev["timestamp"]


def test_tres_eventos_por_hora(emisor_limpio: EmisorSensores, df_demo: pd.DataFrame) -> None:
    """Cada hora produce 3 lecturas: exterior, interior y consumo."""
    eventos = list(emisor_limpio.eventos_desde_df(df_demo.head(4)))
    assert len(eventos) == 12
    tipos = {e["tipo"] for e in eventos}
    assert tipos == {"ambiente_exterior", "ambiente_interior", "consumo_electrico"}


# ---------------------------------------------------------------------------
# Particionado
# ---------------------------------------------------------------------------
def test_particion_por_dia(df_demo: pd.DataFrame, tmp_path: Path) -> None:
    emisor = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0, particion="dia"))
    resumen = emisor.volcar_lote(df_demo, tmp_path)

    assert set(resumen) == {
        "lecturas_20250101.jsonl",
        "lecturas_20250102.jsonl",
        "lecturas_20250103.jsonl",
    }
    # 24 h x 3 sensores por día
    assert all(n == 72 for n in resumen.values())


def test_particion_por_mes(df_demo: pd.DataFrame, tmp_path: Path) -> None:
    emisor = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0, particion="mes"))
    resumen = emisor.volcar_lote(df_demo, tmp_path)

    assert list(resumen) == ["lecturas_202501.jsonl"]
    assert resumen["lecturas_202501.jsonl"] == 3 * 72


def test_ruta_usa_patron_hive(df_demo: pd.DataFrame, tmp_path: Path) -> None:
    """El directorio debe seguir el patrón clave=valor."""
    emisor = EmisorSensores("sevilla", ConfigEmisor(tasa_defectos=0.0, particion="mes"))
    emisor.volcar_lote(df_demo, tmp_path)
    assert (tmp_path / "sede=sevilla").is_dir()


def test_particion_no_soportada_falla() -> None:
    emisor = EmisorSensores("madrid", ConfigEmisor(particion="semana"))
    with pytest.raises(ValueError, match="Partición no soportada"):
        emisor.clave_particion(pd.Timestamp("2025-01-01"))


# ---------------------------------------------------------------------------
# Validación y defectos
# ---------------------------------------------------------------------------
def test_evento_correcto_es_valido(
    emisor_limpio: EmisorSensores, df_demo: pd.DataFrame
) -> None:
    for ev in emisor_limpio.eventos_desde_df(df_demo.head(20)):
        ok, motivo = es_valido(ev)
        assert ok, f"Evento válido rechazado por {motivo}: {ev}"


@pytest.mark.parametrize("defecto", TIPOS_DEFECTO)
def test_cada_defecto_es_detectado(
    defecto: str, emisor_limpio: EmisorSensores, df_demo: pd.DataFrame
) -> None:
    """Todo defecto inyectado debe ser rechazado por la validación.

    Es el contrato entre el simulador y la rama de rechazo del pipeline: si
    algo se cuela, el pipeline lo daría por bueno.
    """
    rng = np.random.default_rng(7)
    # El evento de consumo tiene varios campos numéricos que corromper
    base = [
        e
        for e in emisor_limpio.eventos_desde_df(df_demo.head(3))
        if e["tipo"] == "consumo_electrico"
    ][0]

    for intento in range(20):
        corrupto = corromper_evento(base, defecto, np.random.default_rng(intento))
        ok, motivo = es_valido(corrupto)
        assert not ok, f"El defecto '{defecto}' no fue detectado: {corrupto}"
        assert motivo, "Todo rechazo debe indicar un motivo auditable"


def test_defecto_desconocido_falla() -> None:
    with pytest.raises(ValueError, match="Defecto desconocido"):
        corromper_evento({"sede": "madrid"}, "explosion_cosmica", np.random.default_rng(1))


def test_valor_fuera_de_rango_usa_centinela(
    emisor_limpio: EmisorSensores, df_demo: pd.DataFrame
) -> None:
    """-999 es el valor que devuelven muchos sensores industriales averiados."""
    base = [
        e for e in emisor_limpio.eventos_desde_df(df_demo.head(3))
        if e["tipo"] == "consumo_electrico"
    ][0]
    corrupto = corromper_evento(base, "fuera_de_rango", np.random.default_rng(3))
    assert VALOR_CENTINELA in corrupto.values()


def test_falta_campo_obligatorio_se_detecta() -> None:
    for campo in CAMPOS_OBLIGATORIOS:
        evento = {c: "x" for c in CAMPOS_OBLIGATORIOS}
        evento["timestamp"] = "2025-01-01T00:00:00+01:00"
        del evento[campo]
        ok, motivo = es_valido(evento)
        assert not ok
        assert campo in motivo


def test_tasa_de_defectos_se_respeta(df_demo: pd.DataFrame) -> None:
    """Con tasa 0 no debe haber ningún rechazo; con tasa alta, muchos."""
    limpio = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0, seed=1))
    sucio = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.5, seed=1))

    n_malos_limpio = sum(
        not es_valido(e)[0] for e in limpio.eventos_desde_df(df_demo)
    )
    n_malos_sucio = sum(
        not es_valido(e)[0] for e in sucio.eventos_desde_df(df_demo)
    )
    assert n_malos_limpio == 0
    assert n_malos_sucio > 50


def test_emision_reproducible(df_demo: pd.DataFrame) -> None:
    """Misma semilla, mismos defectos en las mismas posiciones."""
    def firmas() -> list[bool]:
        em = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.2, seed=99))
        return [es_valido(e)[0] for e in em.eventos_desde_df(df_demo)]

    assert firmas() == firmas()


# ---------------------------------------------------------------------------
# Modo streaming
# ---------------------------------------------------------------------------
def test_modo_stream_genera_un_fichero_por_lote(
    df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    emisor = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0))
    total = emisor.emitir_stream(
        df_demo, tmp_path, horas_por_lote=2, intervalo_s=0, max_lotes=3
    )

    ficheros = sorted((tmp_path / "sede=madrid").glob("*.jsonl"))
    assert len(ficheros) == 3
    assert total == 3 * 2 * 3  # 3 lotes x 2 horas x 3 sensores
