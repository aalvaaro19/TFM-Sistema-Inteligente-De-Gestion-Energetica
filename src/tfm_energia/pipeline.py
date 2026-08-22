from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from tfm_energia.config import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, SEDES, SYNTHETIC_DIR


@dataclass(frozen=True)
class Etapa:
    """Una etapa del proceso, con sus dependencias y sus salidas."""

    nombre: str
    descripcion: str
    script: str
    argumentos: tuple[str, ...] = ()
    depende_de: tuple[str, ...] = ()
    artefactos: tuple[Path, ...] = ()
    requiere_mongo: bool = False
    requiere_token: bool = False
    minutos_estimados: float = 1.0

    def completada(self) -> bool:
        """Cierto si todas sus salidas existen."""
        return bool(self.artefactos) and all(a.exists() for a in self.artefactos)

    def artefactos_faltantes(self) -> list[Path]:
        return [a for a in self.artefactos if not a.exists()]


def _por_sede(plantilla: str, base: Path) -> tuple[Path, ...]:
    return tuple(base / plantilla.format(sede=s) for s in SEDES)


ETAPAS: tuple[Etapa, ...] = (
    Etapa(
        nombre="descargar_aemet",
        descripcion="Observación meteorológica diaria de AEMET, por sede",
        script="download_aemet.py",
        artefactos=_por_sede("meteo_{sede}.csv", RAW_DIR / "aemet"),
        requiere_token=True,
        minutos_estimados=3,
    ),
    Etapa(
        nombre="descargar_esios",
        descripcion="Precios PVPC horarios de e·sios (REE)",
        script="download_esios.py",
        artefactos=(RAW_DIR / "esios" / "pvpc_horario.csv",),
        requiere_token=True,
        minutos_estimados=5,
    ),
    Etapa(
        nombre="generar",
        descripcion="Simulación de las cuatro oficinas, gobernada por la meteorología real",
        script="generate_synthetic.py",
        depende_de=("descargar_aemet",),
        artefactos=_por_sede("sede_{sede}.parquet", SYNTHETIC_DIR),
        minutos_estimados=1,
    ),
    Etapa(
        nombre="enriquecer",
        descripcion="Incorporación de precios y cálculo del coste horario",
        script="enrich_with_real_data.py",
        depende_de=("generar", "descargar_esios"),
        artefactos=_por_sede("enriquecido_{sede}.parquet", PROCESSED_DIR),
        minutos_estimados=1,
    ),
    Etapa(
        nombre="simular_sensores",
        descripcion="Emisión del flujo de eventos IoT en JSON Lines",
        script="simulate_sensors.py",
        argumentos=("--modo", "lote", "--particion", "mes", "--limpiar"),
        depende_de=("generar",),
        artefactos=tuple(
            PROJECT_ROOT / "data" / "stream" / f"sede={s}" for s in SEDES
        ),
        minutos_estimados=1,
    ),
    Etapa(
        nombre="ingerir",
        descripcion="Ingesta del flujo a MongoDB, con validación y rama de rechazo",
        script="ingest_stream.py",
        argumentos=("--reiniciar-offsets", "--vaciar-coleccion"),
        depende_de=("simular_sensores",),
        requiere_mongo=True,
        minutos_estimados=6,
    ),
    Etapa(
        nombre="cargar_apis",
        descripcion="Carga de AEMET y precios PVPC en sus colecciones",
        script="load_apis_to_mongo.py",
        depende_de=("descargar_aemet", "descargar_esios"),
        requiere_mongo=True,
        minutos_estimados=1,
    ),
    Etapa(
        nombre="entrenar",
        descripcion="Comparativa de modelos predictivos con validación de origen móvil",
        script="train_predictivo.py",
        argumentos=("--sede", "todas"),
        depende_de=("enriquecer",),
        artefactos=(PROCESSED_DIR / "metricas_modelos_todas.csv",),
        minutos_estimados=20,
    ),
    Etapa(
        nombre="anomalias",
        descripcion="Detección de anomalías y evaluación contra las etiquetas",
        script="detectar_anomalias.py",
        depende_de=("enriquecer",),
        artefactos=(
            PROCESSED_DIR / "anomalias_metricas.csv",
            PROCESSED_DIR / "anomalias_detectadas.csv",
        ),
        minutos_estimados=2,
    ),
    Etapa(
        nombre="optimizar",
        descripcion="Comparativa de estrategias de climatización y sensibilidad",
        script="optimizar_costes.py",
        argumentos=("--sensibilidad",),
        depende_de=("enriquecer",),
        artefactos=(
            PROCESSED_DIR / "optimizacion_resumen.csv",
            PROCESSED_DIR / "optimizacion_horaria.parquet",
        ),
        minutos_estimados=20,
    ),
)

POR_NOMBRE = {e.nombre: e for e in ETAPAS}


def orden_topologico(etapas: tuple[Etapa, ...] = ETAPAS) -> list[Etapa]:
    """Ordena las etapas de modo que toda dependencia vaya antes que su etapa.

    Se resuelve por orden topológico y no por una lista fija para que añadir una
    etapa nueva no obligue a recolocar el resto a mano, y para que un ciclo entre
    dependencias se detecte en vez de producir un orden silenciosamente erróneo.
    """
    pendientes = {e.nombre: set(e.depende_de) for e in etapas}
    disponibles = {e.nombre for e in etapas}
    salida: list[Etapa] = []

    while pendientes:
        libres = sorted(
            n for n, deps in pendientes.items() if not (deps & set(pendientes))
        )
        if not libres:
            raise ValueError(f"Dependencias circulares entre: {sorted(pendientes)}")
        for nombre in libres:
            salida.append(POR_NOMBRE[nombre])
            del pendientes[nombre]

    desconocidas = {
        d for e in etapas for d in e.depende_de if d not in disponibles
    }
    if desconocidas:
        raise ValueError(f"Dependencias inexistentes: {sorted(desconocidas)}")
    return salida


def seleccionar(
    solo: list[str] | None = None,
    desde: str | None = None,
    saltar: list[str] | None = None,
) -> list[Etapa]:
    """Etapas a ejecutar según los filtros pedidos."""
    orden = orden_topologico()
    nombres = [e.nombre for e in orden]

    for valor in filter(None, [*(solo or []), desde, *(saltar or [])]):
        if valor not in POR_NOMBRE:
            raise ValueError(f"Etapa desconocida: {valor}. Opciones: {nombres}")

    if solo:
        elegidas = [e for e in orden if e.nombre in solo]
    elif desde:
        elegidas = orden[nombres.index(desde):]
    else:
        elegidas = list(orden)

    return [e for e in elegidas if e.nombre not in (saltar or [])]


@dataclass
class ResultadoEtapa:
    etapa: str
    estado: str  # ejecutada | omitida | fallida | simulada
    segundos: float = 0.0
    detalle: str = ""


def ejecutar_etapa(etapa: Etapa, python: str | None = None) -> ResultadoEtapa:
    """Lanza el script de una etapa como proceso independiente."""
    python = python or sys.executable
    comando = [python, str(PROJECT_ROOT / "scripts" / etapa.script), *etapa.argumentos]

    logger.info(f"▶ {etapa.nombre}: {etapa.descripcion}")
    t0 = time.perf_counter()
    proceso = subprocess.run(
        comando,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        # Los scripts emiten UTF-8 (símbolos de estado, grados, acentos) y en
        # Windows la codificación por omisión es cp1252, que falla al leerlos.
        # Sin esto, el texto de una etapa fallida —justo cuando hace falta— se
        # pierde con un UnicodeDecodeError.
        encoding="utf-8",
        errors="replace",
    )
    segundos = time.perf_counter() - t0

    if proceso.returncode != 0:
        cola = (proceso.stderr or proceso.stdout or "").strip().splitlines()[-5:]
        logger.error(f"✖ {etapa.nombre} falló ({segundos:.0f}s)")
        for linea in cola:
            logger.error(f"    {linea}")
        return ResultadoEtapa(etapa.nombre, "fallida", segundos, "\n".join(cola))

    faltan = etapa.artefactos_faltantes()
    if faltan:
        detalle = f"terminó sin generar: {[p.name for p in faltan]}"
        logger.error(f"✖ {etapa.nombre}: {detalle}")
        return ResultadoEtapa(etapa.nombre, "fallida", segundos, detalle)

    logger.success(f"✔ {etapa.nombre} ({segundos:.0f}s)")
    return ResultadoEtapa(etapa.nombre, "ejecutada", segundos)


def resumen_estado() -> list[tuple[Etapa, bool]]:
    """Qué etapas tienen ya sus artefactos en disco."""
    return [(e, e.completada()) for e in orden_topologico()]
