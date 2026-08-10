"""Ingesta los eventos IoT del flujo hacia MongoDB.

Equivalente en Python del pipeline de StreamSets: lee los JSON Lines que dejan
los gateways en `data/stream/`, valida cada evento, tipa las fechas, carga los
correctos en MongoDB y aparta los defectuosos con su motivo.

Uso:
    # Ingesta completa (idempotente: reejecutar no duplica)
    python scripts/ingest_stream.py

    # Prueba en seco, sin escribir en MongoDB
    python scripts/ingest_stream.py --sin-mongo

    # Solo una sede
    python scripts/ingest_stream.py --sede madrid

    # Reprocesar todo desde cero
    python scripts/ingest_stream.py --reiniciar-offsets --vaciar-coleccion
"""
from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from tfm_energia.config import DATA_DIR, SEDES
from tfm_energia.data.ingest_pipeline import (
    NOMBRE_CHECKPOINT,
    ControlOffsets,
    PipelineIngesta,
)
from tfm_energia.data.mongo_repository import COL_SENSORES, MongoRepository


STREAM_DIR = DATA_DIR / "stream"
RECHAZADOS_DIR = DATA_DIR / "stream_rechazados"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ingesta de eventos IoT a MongoDB")
    p.add_argument("--origen", default=str(STREAM_DIR), help="Directorio de eventos")
    p.add_argument("--sede", default=None, help="Procesar solo una sede")
    p.add_argument("--rechazados", default=str(RECHAZADOS_DIR))
    p.add_argument("--sin-mongo", action="store_true", help="No escribe en la base de datos")
    p.add_argument(
        "--reiniciar-offsets",
        action="store_true",
        help="Olvida qué ficheros se procesaron y los vuelve a leer todos",
    )
    p.add_argument(
        "--vaciar-coleccion",
        action="store_true",
        help="Vacía sensores_iot antes de cargar (usar junto a --reiniciar-offsets)",
    )
    p.add_argument("--lote", type=int, default=5000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    origen = Path(args.origen)
    if args.sede:
        if args.sede not in SEDES:
            raise SystemExit(f"Sede desconocida: {args.sede}. Opciones: {list(SEDES)}")
        origen = origen / f"sede={args.sede}"
    if not origen.exists():
        raise SystemExit(
            f"No existe {origen}. Ejecuta antes scripts/simulate_sensors.py"
        )

    checkpoint = Path(args.origen) / NOMBRE_CHECKPOINT
    if args.reiniciar_offsets:
        ControlOffsets(checkpoint).limpiar()
        logger.warning("Offsets reiniciados: se reprocesarán todos los ficheros.")

    repo = None
    if not args.sin_mongo:
        repo = MongoRepository()
        try:
            repo.db.command("ping")
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"No se puede conectar a MongoDB: {type(exc).__name__}") from exc

        if args.vaciar_coleccion:
            n = repo.vaciar_coleccion(COL_SENSORES)
            logger.warning(f"Colección {COL_SENSORES} vaciada: {n:,} documentos eliminados")
        repo.crear_indices()
    else:
        logger.info("Modo --sin-mongo: se valida y transforma, pero no se persiste.")

    pipeline = PipelineIngesta(
        repo=repo,
        dir_rechazados=Path(args.rechazados),
        checkpoint=checkpoint,
        lote=args.lote,
    )

    logger.info(f"Ingiriendo desde {origen}")
    stats = pipeline.ejecutar(origen)
    print("\n" + stats.resumen())

    if repo is not None:
        total = repo.db[COL_SENSORES].count_documents({})
        print(f"\n  Total en {COL_SENSORES}: {total:,} documentos")
        repo.close()


if __name__ == "__main__":
    main()
