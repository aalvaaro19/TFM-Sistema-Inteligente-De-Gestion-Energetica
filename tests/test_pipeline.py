from __future__ import annotations

from pathlib import Path

import pytest

from tfm_energia.config import SEDES
from tfm_energia.pipeline import (
    ETAPAS,
    POR_NOMBRE,
    Etapa,
    orden_topologico,
    resumen_estado,
    seleccionar,
)


# ---------------------------------------------------------------------------
# El grafo de dependencias
# ---------------------------------------------------------------------------
def test_toda_dependencia_existe() -> None:
    nombres = {e.nombre for e in ETAPAS}
    for etapa in ETAPAS:
        for dep in etapa.depende_de:
            assert dep in nombres, f"{etapa.nombre} depende de '{dep}', que no existe"


def test_el_orden_respeta_las_dependencias() -> None:
    """Ninguna etapa puede aparecer antes que algo de lo que depende."""
    orden = [e.nombre for e in orden_topologico()]
    for etapa in ETAPAS:
        for dep in etapa.depende_de:
            assert orden.index(dep) < orden.index(etapa.nombre), (
                f"'{dep}' debe ir antes de '{etapa.nombre}'"
            )


def test_el_orden_incluye_todas_las_etapas() -> None:
    assert {e.nombre for e in orden_topologico()} == {e.nombre for e in ETAPAS}


def test_los_ciclos_se_detectan() -> None:
    """Un ciclo debe dar error, no un orden arbitrario."""
    a = Etapa("a", "", "a.py", depende_de=("b",))
    b = Etapa("b", "", "b.py", depende_de=("a",))
    with pytest.raises(ValueError, match="circulares"):
        orden_topologico((a, b))


def test_las_etapas_clave_estan_declaradas() -> None:
    """El proceso completo debe cubrir las fases del proyecto."""
    esperadas = {
        "descargar_aemet", "descargar_esios", "generar", "enriquecer",
        "simular_sensores", "ingerir", "cargar_apis", "entrenar",
        "anomalias", "optimizar",
    }
    assert esperadas <= {e.nombre for e in ETAPAS}


def test_los_scripts_declarados_existen() -> None:
    """Una etapa que apunta a un script inexistente fallaría en ejecución."""
    from tfm_energia.config import PROJECT_ROOT

    for etapa in ETAPAS:
        script = PROJECT_ROOT / "scripts" / etapa.script
        assert script.exists(), f"{etapa.nombre} apunta a {etapa.script}, que no existe"


# ---------------------------------------------------------------------------
# Dependencias concretas del proyecto
# ---------------------------------------------------------------------------
def test_la_simulacion_depende_de_la_meteorologia() -> None:
    """La física se genera a partir de la observación real: el orden importa.

    Si se generase antes de descargar AEMET, el dataset saldría con meteorología
    sintética y el consumo no derivaría de datos reales.
    """
    assert "descargar_aemet" in POR_NOMBRE["generar"].depende_de


def test_los_modelos_dependen_del_dataset_enriquecido() -> None:
    for fase in ("entrenar", "anomalias", "optimizar"):
        assert "enriquecer" in POR_NOMBRE[fase].depende_de


def test_la_ingesta_depende_del_flujo_de_eventos() -> None:
    assert "simular_sensores" in POR_NOMBRE["ingerir"].depende_de


def test_las_etapas_de_mongo_estan_marcadas() -> None:
    assert POR_NOMBRE["ingerir"].requiere_mongo
    assert POR_NOMBRE["cargar_apis"].requiere_mongo
    assert not POR_NOMBRE["generar"].requiere_mongo


def test_las_descargas_requieren_token() -> None:
    assert POR_NOMBRE["descargar_aemet"].requiere_token
    assert POR_NOMBRE["descargar_esios"].requiere_token


# ---------------------------------------------------------------------------
# Selección de etapas
# ---------------------------------------------------------------------------
def test_seleccion_completa_por_defecto() -> None:
    assert len(seleccionar()) == len(ETAPAS)


def test_seleccion_desde_una_etapa() -> None:
    elegidas = [e.nombre for e in seleccionar(desde="enriquecer")]
    assert "enriquecer" in elegidas
    assert "generar" not in elegidas
    # Y sigue conservando el orden correcto entre las que quedan
    assert elegidas.index("enriquecer") < elegidas.index("entrenar")


def test_seleccion_de_etapas_concretas() -> None:
    elegidas = [e.nombre for e in seleccionar(solo=["entrenar", "anomalias"])]
    assert set(elegidas) == {"entrenar", "anomalias"}


def test_saltar_etapas() -> None:
    elegidas = [e.nombre for e in seleccionar(saltar=["ingerir", "cargar_apis"])]
    assert "ingerir" not in elegidas
    assert "generar" in elegidas


def test_saltar_las_de_mongo_deja_el_resto() -> None:
    de_mongo = [e.nombre for e in ETAPAS if e.requiere_mongo]
    elegidas = [e.nombre for e in seleccionar(saltar=de_mongo)]
    assert not set(elegidas) & set(de_mongo)
    assert len(elegidas) == len(ETAPAS) - len(de_mongo)


def test_etapa_desconocida_da_error() -> None:
    for kwargs in ({"solo": ["inventada"]}, {"desde": "inventada"}, {"saltar": ["inventada"]}):
        with pytest.raises(ValueError, match="desconocida"):
            seleccionar(**kwargs)


# ---------------------------------------------------------------------------
# Estado de los artefactos
# ---------------------------------------------------------------------------
def test_artefactos_declarados_por_sede() -> None:
    """Las etapas por sede deben declarar una salida por cada una."""
    assert len(POR_NOMBRE["generar"].artefactos) == len(SEDES)
    assert len(POR_NOMBRE["enriquecer"].artefactos) == len(SEDES)


def test_completada_exige_todos_los_artefactos(tmp_path: Path) -> None:
    existente = tmp_path / "hay.txt"
    existente.write_text("x", encoding="utf-8")

    completa = Etapa("a", "", "s.py", artefactos=(existente,))
    incompleta = Etapa("b", "", "s.py", artefactos=(existente, tmp_path / "falta.txt"))
    sin_declarar = Etapa("c", "", "s.py")

    assert completa.completada()
    assert not incompleta.completada()
    assert [p.name for p in incompleta.artefactos_faltantes()] == ["falta.txt"]
    # Sin artefactos declarados no se puede afirmar que esté hecha
    assert not sin_declarar.completada()


def test_resumen_estado_cubre_todas_las_etapas() -> None:
    estado = resumen_estado()
    assert len(estado) == len(ETAPAS)
    assert all(isinstance(hecha, bool) for _, hecha in estado)


def test_estimaciones_de_tiempo_razonables() -> None:
    """Sirven para avisar del coste antes de lanzar: deben estar puestas."""
    assert all(e.minutos_estimados > 0 for e in ETAPAS)
    # El proceso completo es de horas, no de minutos
    assert 30 < sum(e.minutos_estimados for e in ETAPAS) < 300
