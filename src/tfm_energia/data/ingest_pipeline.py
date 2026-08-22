from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from tfm_energia.data.mongo_repository import MongoRepository, normalizar_fechas
from tfm_energia.data.sensor_stream import es_valido, leer_jsonl


NOMBRE_CHECKPOINT = "_ingesta_offsets.json"
LOTE_INSERCION = 5000


# ---------------------------------------------------------------------------
# Estadísticas de ejecución
# ---------------------------------------------------------------------------
@dataclass
class EstadisticasIngesta:
    """Contadores de una ejecución, equivalentes a los de la UI de SDC."""

    ficheros_procesados: int = 0
    ficheros_omitidos: int = 0
    eventos_leidos: int = 0
    eventos_validos: int = 0
    eventos_rechazados: int = 0
    eventos_insertados: int = 0
    motivos_rechazo: Counter = field(default_factory=Counter)
    segundos: float = 0.0

    @property
    def tasa_rechazo(self) -> float:
        """Fracción de eventos que no superaron la validación."""
        return self.eventos_rechazados / self.eventos_leidos if self.eventos_leidos else 0.0

    @property
    def eventos_por_segundo(self) -> float:
        return self.eventos_leidos / self.segundos if self.segundos else 0.0

    def resumen(self) -> str:
        """Informe legible del resultado de la ingesta."""
        lineas = [
            "─" * 58,
            "  RESUMEN DE LA INGESTA",
            "─" * 58,
            f"  Ficheros procesados      {self.ficheros_procesados:>12,}",
            f"  Ficheros omitidos        {self.ficheros_omitidos:>12,}  (sin cambios)",
            f"  Eventos leídos           {self.eventos_leidos:>12,}",
            f"  Eventos válidos          {self.eventos_validos:>12,}",
            f"  Eventos rechazados       {self.eventos_rechazados:>12,}  ({self.tasa_rechazo:.2%})",
            f"  Eventos insertados       {self.eventos_insertados:>12,}",
            f"  Tiempo                   {self.segundos:>12.1f} s  ({self.eventos_por_segundo:,.0f} ev/s)",
        ]
        if self.motivos_rechazo:
            lineas.append("─" * 58)
            lineas.append("  MOTIVOS DE RECHAZO")
            for motivo, n in self.motivos_rechazo.most_common():
                lineas.append(f"    {motivo:<32} {n:>8,}")
        lineas.append("─" * 58)
        return "\n".join(lineas)


# ---------------------------------------------------------------------------
# Control de offsets (idempotencia)
# ---------------------------------------------------------------------------
class ControlOffsets:
    """Registro de qué ficheros se han procesado ya.

    Equivale al control de offsets de StreamSets: permite reanudar sin
    reprocesar. Un fichero se considera ya ingerido si su tamaño y su fecha de
    modificación no han cambiado desde la última vez.
    """

    def __init__(self, path: Path, base: Path | None = None) -> None:
        self.path = path
        self.base = base
        self._estado: dict[str, dict[str, Any]] = {}
        if path.exists():
            try:
                self._estado = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                logger.warning(f"Checkpoint ilegible, se reinicia: {path}")

    def _clave(self, fichero: Path) -> str:
        """Identificador único del fichero dentro del flujo.

        Debe ser la ruta relativa a la raíz, no el nombre: las sedes reparten
        sus eventos en ficheros que se llaman igual (`lecturas_202401.jsonl`
        existe en las cuatro), de modo que usar el nombre haría que unas
        sobrescribieran las entradas de otras y nada se reconocería como ya
        procesado. Se usa relativa —y no absoluta— para que el checkpoint siga
        siendo válido si el repositorio cambia de ubicación.
        """
        if self.base is not None:
            try:
                return fichero.relative_to(self.base).as_posix()
            except ValueError:
                pass
        return fichero.name

    @staticmethod
    def _firma(fichero: Path) -> dict[str, Any]:
        st = fichero.stat()
        return {"tamano": st.st_size, "mtime": int(st.st_mtime)}

    def ya_procesado(self, fichero: Path) -> bool:
        previo = self._estado.get(self._clave(fichero))
        if previo is None:
            return False
        actual = self._firma(fichero)
        return (
            previo.get("tamano") == actual["tamano"]
            and previo.get("mtime") == actual["mtime"]
        )

    def registrar(self, fichero: Path, eventos: int) -> None:
        self._estado[self._clave(fichero)] = {**self._firma(fichero), "eventos": eventos}

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._estado, indent=2), encoding="utf-8")

    def limpiar(self) -> None:
        self._estado = {}
        if self.path.exists():
            self.path.unlink()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
class PipelineIngesta:
    """Orquesta las etapas de ingesta de un directorio de eventos."""

    def __init__(
        self,
        repo: MongoRepository | None = None,
        dir_rechazados: Path | None = None,
        checkpoint: Path | None = None,
        lote: int = LOTE_INSERCION,
    ) -> None:
        self.repo = repo
        self.dir_rechazados = dir_rechazados
        self.checkpoint = checkpoint
        self.lote = lote

    # -- Etapa: Expression Evaluator ---------------------------------------
    @staticmethod
    def enriquecer(evento: dict[str, Any], origen: str) -> dict[str, Any]:
        """Añade metadatos de procesado y normaliza el identificador de sede.

        `_origen` deja trazabilidad de qué fichero trajo cada evento, que es lo
        primero que se necesita cuando hay que investigar un dato raro.
        """
        ev = dict(evento)
        ev["_origen"] = origen
        ev["_procesado_ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if isinstance(ev.get("sede"), str):
            ev["sede"] = ev["sede"].strip().lower()
        return ev

    # -- Etapa: Stream Selector --------------------------------------------
    @staticmethod
    def clasificar(evento: dict[str, Any]) -> tuple[bool, str]:
        """Decide si el evento va a la rama principal o a la de rechazo."""
        if "_error_parseo" in evento:
            return False, "json_ilegible"
        return es_valido(evento)

    # -- Procesado de un fichero -------------------------------------------
    def procesar_fichero(
        self, fichero: Path, stats: EstadisticasIngesta
    ) -> tuple[list[dict], list[dict]]:
        """Aplica las etapas a un fichero. Devuelve (válidos, rechazados)."""
        validos: list[dict] = []
        rechazados: list[dict] = []

        for evento in leer_jsonl(fichero):
            stats.eventos_leidos += 1
            ev = self.enriquecer(evento, fichero.name)

            ok, motivo = self.clasificar(ev)
            if ok:
                # Etapa Field Type Converter: solo sobre lo que ya es válido
                validos.append(normalizar_fechas(ev))
                stats.eventos_validos += 1
            else:
                ev["_motivo_rechazo"] = motivo
                rechazados.append(ev)
                stats.eventos_rechazados += 1
                # Se agrupa por familia de motivo (antes de los dos puntos)
                stats.motivos_rechazo[motivo.split(":")[0]] += 1

        return validos, rechazados

    # -- Destinos ------------------------------------------------------------
    def cargar_en_mongo(self, eventos: list[dict]) -> int:
        """Inserta por lotes. Sin repositorio configurado, simula la carga."""
        if self.repo is None or not eventos:
            return 0
        insertados = 0
        for i in range(0, len(eventos), self.lote):
            insertados += self.repo.insertar_eventos_sensor(eventos[i : i + self.lote])
        return insertados

    def volcar_rechazados(self, eventos: list[dict], relativa: Path) -> None:
        """Escribe la rama de error, conservando el evento tal como llegó.

        Se replica la estructura de directorios del origen (`sede=madrid/...`).
        Nombrar el destino solo con el nombre del fichero haría que las sedes se
        pisaran entre sí, porque todas reparten sus eventos en ficheros
        homónimos, y se perderían los rechazos de todas menos la última.
        """
        if not eventos or self.dir_rechazados is None:
            return
        destino = self.dir_rechazados / relativa.parent / f"rechazados_{relativa.stem}.jsonl"
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8") as f:
            for ev in eventos:
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")

    # -- Ejecución -----------------------------------------------------------
    def ejecutar(self, base: Path, patron: str = "**/*.jsonl") -> EstadisticasIngesta:
        """Procesa todos los ficheros de eventos bajo `base`."""
        stats = EstadisticasIngesta()
        t0 = time.perf_counter()

        offsets = ControlOffsets(self.checkpoint, base) if self.checkpoint else None
        ficheros = sorted(f for f in base.glob(patron) if f.name != NOMBRE_CHECKPOINT)
        if not ficheros:
            logger.warning(f"No se han encontrado ficheros de eventos en {base}")

        for fichero in ficheros:
            if offsets is not None and offsets.ya_procesado(fichero):
                stats.ficheros_omitidos += 1
                logger.debug(f"  omitido (sin cambios): {fichero.name}")
                continue

            validos, rechazados = self.procesar_fichero(fichero, stats)
            stats.eventos_insertados += self.cargar_en_mongo(validos)

            try:
                relativa = fichero.relative_to(base)
            except ValueError:  # pragma: no cover - defensivo
                relativa = Path(fichero.name)
            self.volcar_rechazados(rechazados, relativa)

            stats.ficheros_procesados += 1
            if offsets is not None:
                offsets.registrar(fichero, len(validos) + len(rechazados))
            logger.info(
                f"  {fichero.name}: {len(validos):,} válidos, {len(rechazados):,} rechazados"
            )

        if offsets is not None:
            offsets.guardar()

        stats.segundos = time.perf_counter() - t0
        return stats
