from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.ingest_pipeline import (
    ControlOffsets,
    EstadisticasIngesta,
    PipelineIngesta,
)
from tfm_energia.data.sensor_stream import ConfigEmisor, EmisorSensores
from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig


class RepoFalso:
    """Doble del repositorio: acumula en memoria lo que se insertaría."""

    def __init__(self) -> None:
        self.insertados: list[dict] = []

    def insertar_eventos_sensor(self, eventos: list[dict]) -> int:
        self.insertados.extend(eventos)
        return len(eventos)


@pytest.fixture
def df_demo() -> pd.DataFrame:
    sim = OfficeSimulator(
        "madrid",
        SEDES["madrid"],
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        cfg=SimulationConfig(seed=42),
    )
    return sim.generate()


@pytest.fixture
def flujo_limpio(df_demo: pd.DataFrame, tmp_path: Path) -> Path:
    """Directorio de eventos sin ningún defecto."""
    emisor = EmisorSensores("madrid", ConfigEmisor(tasa_defectos=0.0, particion="dia"))
    emisor.volcar_lote(df_demo, tmp_path)
    return tmp_path


@pytest.fixture
def flujo_con_defectos(df_demo: pd.DataFrame, tmp_path: Path) -> Path:
    """Directorio de eventos con una proporción alta de defectos."""
    emisor = EmisorSensores(
        "madrid", ConfigEmisor(tasa_defectos=0.3, particion="dia", seed=7)
    )
    emisor.volcar_lote(df_demo, tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Etapa: Expression Evaluator
# ---------------------------------------------------------------------------
def test_enriquecer_anade_trazabilidad() -> None:
    ev = PipelineIngesta.enriquecer({"sede": "madrid"}, "lecturas_20250101.jsonl")
    assert ev["_origen"] == "lecturas_20250101.jsonl"
    assert "_procesado_ts" in ev


def test_enriquecer_normaliza_la_sede() -> None:
    ev = PipelineIngesta.enriquecer({"sede": "  MADRID "}, "f.jsonl")
    assert ev["sede"] == "madrid"


def test_enriquecer_no_muta_el_original() -> None:
    original = {"sede": "MADRID"}
    PipelineIngesta.enriquecer(original, "f.jsonl")
    assert original == {"sede": "MADRID"}


# ---------------------------------------------------------------------------
# Etapa: Stream Selector
# ---------------------------------------------------------------------------
def test_json_ilegible_tiene_su_propio_motivo() -> None:
    ok, motivo = PipelineIngesta.clasificar({"_error_parseo": "línea corrupta"})
    assert not ok
    assert motivo == "json_ilegible"


# ---------------------------------------------------------------------------
# Recorrido completo
# ---------------------------------------------------------------------------
def test_flujo_limpio_se_inserta_entero(flujo_limpio: Path) -> None:
    repo = RepoFalso()
    stats = PipelineIngesta(repo=repo).ejecutar(flujo_limpio)

    assert stats.ficheros_procesados == 2          # 2 días
    assert stats.eventos_leidos == 2 * 24 * 3      # 2 días x 24 h x 3 sensores
    assert stats.eventos_rechazados == 0
    assert stats.eventos_validos == stats.eventos_leidos
    assert stats.eventos_insertados == stats.eventos_leidos
    assert len(repo.insertados) == stats.eventos_leidos


def test_las_fechas_llegan_tipadas_a_mongo(flujo_limpio: Path) -> None:
    """La etapa Field Type Converter debe aplicarse antes de persistir."""
    repo = RepoFalso()
    PipelineIngesta(repo=repo).ejecutar(flujo_limpio)

    assert all(isinstance(ev["timestamp"], datetime) for ev in repo.insertados)
    assert all(ev["timestamp"].tzinfo is not None for ev in repo.insertados)


def test_los_defectuosos_no_llegan_a_mongo(flujo_con_defectos: Path, tmp_path: Path) -> None:
    repo = RepoFalso()
    rechazados = tmp_path / "rechazados"
    stats = PipelineIngesta(repo=repo, dir_rechazados=rechazados).ejecutar(
        flujo_con_defectos
    )

    assert stats.eventos_rechazados > 0
    assert len(repo.insertados) == stats.eventos_validos
    # Nada de lo insertado debe llevar marca de rechazo
    assert not any("_motivo_rechazo" in ev for ev in repo.insertados)


def test_ningun_evento_se_pierde(flujo_con_defectos: Path, tmp_path: Path) -> None:
    """Válidos + rechazados debe ser exactamente lo leído."""
    stats = PipelineIngesta(
        repo=RepoFalso(), dir_rechazados=tmp_path / "rech"
    ).ejecutar(flujo_con_defectos)
    assert stats.eventos_validos + stats.eventos_rechazados == stats.eventos_leidos


# ---------------------------------------------------------------------------
# Rama de rechazo
# ---------------------------------------------------------------------------
def test_los_rechazados_se_escriben_con_su_motivo(
    flujo_con_defectos: Path, tmp_path: Path
) -> None:
    destino = tmp_path / "rechazados"
    stats = PipelineIngesta(repo=RepoFalso(), dir_rechazados=destino).ejecutar(
        flujo_con_defectos
    )

    ficheros = sorted(destino.rglob("*.jsonl"))
    assert ficheros, "Debe haberse escrito la rama de rechazo"

    total = 0
    for f in ficheros:
        for linea in f.read_text(encoding="utf-8").strip().split("\n"):
            ev = json.loads(linea)
            assert ev["_motivo_rechazo"], "Todo rechazo debe llevar motivo"
            assert ev["_origen"], "Todo rechazo debe indicar de qué fichero vino"
            total += 1
    assert total == stats.eventos_rechazados


def test_el_rechazado_conserva_su_forma_original(
    flujo_con_defectos: Path, tmp_path: Path
) -> None:
    """Se guarda tal como llegó, sin convertir: es lo que hace falta para auditar."""
    destino = tmp_path / "rechazados"
    PipelineIngesta(repo=RepoFalso(), dir_rechazados=destino).ejecutar(flujo_con_defectos)

    con_timestamp = []
    for f in destino.rglob("*.jsonl"):
        for linea in f.read_text(encoding="utf-8").strip().split("\n"):
            ev = json.loads(linea)
            if "timestamp" in ev and ev["timestamp"] is not None:
                con_timestamp.append(ev["timestamp"])

    assert con_timestamp, "Debería haber rechazados que conserven su timestamp"
    assert all(isinstance(t, str) for t in con_timestamp)


def test_no_se_pierden_rechazos_entre_sedes(df_demo: pd.DataFrame, tmp_path: Path) -> None:
    """Regresión: los ficheros de las 4 sedes se llaman igual.

    Si el destino de la rama de error se nombra solo con el nombre del fichero
    de origen, cada sede sobrescribe a la anterior y se pierden los rechazos de
    todas menos la última.
    """
    flujo = tmp_path / "stream"
    sedes = ("madrid", "sevilla", "barcelona", "oviedo")
    for sede in sedes:
        # Cada sede necesita SUS datos: el campo `sede` del evento sale del
        # DataFrame, no del emisor, que solo decide carpeta y gateway_id.
        df_sede = OfficeSimulator(
            sede,
            SEDES[sede],
            start=date(2025, 1, 1),
            end=date(2025, 1, 2),
            cfg=SimulationConfig(seed=42),
        ).generate()
        EmisorSensores(
            sede, ConfigEmisor(tasa_defectos=0.3, particion="mes", seed=11)
        ).volcar_lote(df_sede, flujo)

    destino = tmp_path / "rechazados"
    stats = PipelineIngesta(repo=RepoFalso(), dir_rechazados=destino).ejecutar(flujo)

    escritos = sum(
        1
        for f in destino.rglob("*.jsonl")
        for linea in f.read_text(encoding="utf-8").strip().split("\n")
        if linea
    )
    assert escritos == stats.eventos_rechazados

    # Debe haber rechazos de las cuatro sedes, no solo de la última
    sedes_con_rechazo = {
        json.loads(linea).get("sede")
        for f in destino.rglob("*.jsonl")
        for linea in f.read_text(encoding="utf-8").strip().split("\n")
        if linea
    }
    assert set(sedes) <= sedes_con_rechazo


def test_la_rama_de_error_replica_la_estructura_de_origen(
    df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    flujo = tmp_path / "stream"
    for sede in ("madrid", "oviedo"):
        EmisorSensores(
            sede, ConfigEmisor(tasa_defectos=0.5, particion="mes", seed=5)
        ).volcar_lote(df_demo, flujo)

    destino = tmp_path / "rechazados"
    PipelineIngesta(repo=RepoFalso(), dir_rechazados=destino).ejecutar(flujo)

    assert (destino / "sede=madrid").is_dir()
    assert (destino / "sede=oviedo").is_dir()


def test_se_contabilizan_los_motivos(flujo_con_defectos: Path, tmp_path: Path) -> None:
    stats = PipelineIngesta(
        repo=RepoFalso(), dir_rechazados=tmp_path / "rech"
    ).ejecutar(flujo_con_defectos)

    assert sum(stats.motivos_rechazo.values()) == stats.eventos_rechazados
    # Los motivos se agrupan por familia, sin el detalle del campo concreto
    assert all(":" not in motivo for motivo in stats.motivos_rechazo)


# ---------------------------------------------------------------------------
# Idempotencia
# ---------------------------------------------------------------------------
def test_reejecutar_no_duplica(flujo_limpio: Path, tmp_path: Path) -> None:
    """El control de offsets es lo que hace segura una segunda ejecución."""
    checkpoint = tmp_path / "offsets.json"
    repo = RepoFalso()

    primera = PipelineIngesta(repo=repo, checkpoint=checkpoint).ejecutar(flujo_limpio)
    insertados_tras_primera = len(repo.insertados)

    segunda = PipelineIngesta(repo=repo, checkpoint=checkpoint).ejecutar(flujo_limpio)

    assert primera.ficheros_procesados == 2
    assert segunda.ficheros_procesados == 0
    assert segunda.ficheros_omitidos == 2
    assert segunda.eventos_leidos == 0
    assert len(repo.insertados) == insertados_tras_primera


def test_reejecutar_no_duplica_con_varias_sedes(
    df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    """Regresión: las sedes tienen ficheros con el MISMO nombre.

    `lecturas_20250101.jsonl` existe en las cuatro sedes, así que si el
    checkpoint se indexa por nombre de fichero unas entradas sobrescriben a
    otras y ninguna vuelve a reconocerse como procesada.
    """
    flujo = tmp_path / "stream"
    for sede in ("madrid", "sevilla", "barcelona", "oviedo"):
        EmisorSensores(sede, ConfigEmisor(tasa_defectos=0.0, particion="dia")).volcar_lote(
            df_demo, flujo
        )

    # 4 sedes x 2 días, todos con nombres repetidos entre sedes
    assert len(list(flujo.rglob("*.jsonl"))) == 8
    assert len({f.name for f in flujo.rglob("*.jsonl")}) == 2

    checkpoint = tmp_path / "offsets.json"
    repo = RepoFalso()
    primera = PipelineIngesta(repo=repo, checkpoint=checkpoint).ejecutar(flujo)
    tras_primera = len(repo.insertados)

    segunda = PipelineIngesta(repo=repo, checkpoint=checkpoint).ejecutar(flujo)

    assert primera.ficheros_procesados == 8
    assert segunda.ficheros_procesados == 0
    assert segunda.ficheros_omitidos == 8
    assert len(repo.insertados) == tras_primera


def test_checkpoint_indexa_por_ruta_relativa(
    df_demo: pd.DataFrame, tmp_path: Path
) -> None:
    """La clave del checkpoint debe distinguir la sede."""
    flujo = tmp_path / "stream"
    for sede in ("madrid", "sevilla"):
        EmisorSensores(sede, ConfigEmisor(tasa_defectos=0.0, particion="mes")).volcar_lote(
            df_demo, flujo
        )

    checkpoint = tmp_path / "offsets.json"
    PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo)

    claves = set(json.loads(checkpoint.read_text(encoding="utf-8")))
    assert len(claves) == 2
    assert all("sede=" in c for c in claves), f"Claves sin la sede: {claves}"


def test_sin_checkpoint_si_se_reprocesa(flujo_limpio: Path) -> None:
    """Sin control de offsets, cada ejecución vuelve a leerlo todo."""
    repo = RepoFalso()
    primera = PipelineIngesta(repo=repo).ejecutar(flujo_limpio)
    segunda = PipelineIngesta(repo=repo).ejecutar(flujo_limpio)
    assert segunda.eventos_leidos == primera.eventos_leidos


def test_un_fichero_modificado_se_vuelve_a_procesar(
    flujo_limpio: Path, tmp_path: Path
) -> None:
    """Si el gateway añade eventos a un fichero, hay que releerlo."""
    checkpoint = tmp_path / "offsets.json"
    PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo_limpio)

    fichero = sorted(flujo_limpio.rglob("*.jsonl"))[0]
    with fichero.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"sensor_id": "madrid-S1", "tipo": "ambiente_exterior",
                            "timestamp": "2025-01-01T00:00:00+01:00", "sede": "madrid",
                            "temperatura_c": 5.0, "humedad_pct": 70.0}) + "\n")

    segunda = PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo_limpio)
    assert segunda.ficheros_procesados == 1
    assert segunda.ficheros_omitidos == 1


def test_limpiar_offsets_fuerza_reproceso(flujo_limpio: Path, tmp_path: Path) -> None:
    checkpoint = tmp_path / "offsets.json"
    PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo_limpio)
    ControlOffsets(checkpoint).limpiar()

    segunda = PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo_limpio)
    assert segunda.ficheros_procesados == 2
    assert segunda.ficheros_omitidos == 0


def test_checkpoint_corrupto_no_rompe_la_ingesta(
    flujo_limpio: Path, tmp_path: Path
) -> None:
    checkpoint = tmp_path / "offsets.json"
    checkpoint.write_text("{esto no es json", encoding="utf-8")

    stats = PipelineIngesta(repo=RepoFalso(), checkpoint=checkpoint).ejecutar(flujo_limpio)
    assert stats.ficheros_procesados == 2


# ---------------------------------------------------------------------------
# Modo sin base de datos y estadísticas
# ---------------------------------------------------------------------------
def test_sin_repositorio_valida_pero_no_persiste(flujo_limpio: Path) -> None:
    stats = PipelineIngesta(repo=None).ejecutar(flujo_limpio)
    assert stats.eventos_validos > 0
    assert stats.eventos_insertados == 0


def test_directorio_vacio_no_falla(tmp_path: Path) -> None:
    stats = PipelineIngesta(repo=RepoFalso()).ejecutar(tmp_path)
    assert stats.ficheros_procesados == 0
    assert stats.eventos_leidos == 0


def test_estadisticas_calculadas() -> None:
    stats = EstadisticasIngesta(
        eventos_leidos=1000, eventos_validos=990, eventos_rechazados=10, segundos=2.0
    )
    assert stats.tasa_rechazo == pytest.approx(0.01)
    assert stats.eventos_por_segundo == pytest.approx(500.0)


def test_estadisticas_sin_eventos_no_divide_por_cero() -> None:
    stats = EstadisticasIngesta()
    assert stats.tasa_rechazo == 0.0
    assert stats.eventos_por_segundo == 0.0


def test_resumen_incluye_los_motivos(flujo_con_defectos: Path, tmp_path: Path) -> None:
    stats = PipelineIngesta(
        repo=RepoFalso(), dir_rechazados=tmp_path / "rech"
    ).ejecutar(flujo_con_defectos)
    texto = stats.resumen()

    assert "RESUMEN DE LA INGESTA" in texto
    assert "MOTIVOS DE RECHAZO" in texto
    for motivo in stats.motivos_rechazo:
        assert motivo in texto
