from __future__ import annotations

import argparse
import time

import pandas as pd
from loguru import logger
from pymongo.errors import BulkWriteError

from tfm_energia.config import SEDES, SYNTHETIC_DIR
from tfm_energia.data.mongo_repository import COL_SENSORES, MongoRepository
from tfm_energia.data.synthetic_generator import to_iot_json_payload


BATCH_SIZE = 5000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Carga datos sintéticos a MongoDB")
    p.add_argument(
        "--sede",
        type=str,
        default=None,
        choices=list(SEDES.keys()) + [None],
        help="Cargar solo una sede concreta. Por defecto, todas.",
    )
    p.add_argument(
        "--reset",
        action="store_true",
        help="Vaciar la colección sensores_iot antes de cargar.",
    )
    return p.parse_args()


def cargar_sede(repo: MongoRepository, sede_id: str) -> int:
    """Carga el parquet de una sede como eventos IoT."""
    path = SYNTHETIC_DIR / f"sede_{sede_id}.parquet"
    if not path.exists():
        logger.error(f"No encontrado: {path}")
        return 0

    df = pd.read_parquet(path)
    logger.info(f"  Leídos {len(df):,} registros de {path.name}")

    # Conversión a eventos IoT (3 eventos por registro → ~52k por sede)
    eventos = to_iot_json_payload(df)
    logger.info(f"  Generados {len(eventos):,} eventos IoT")

    # Inserción por lotes
    total = 0
    for i in range(0, len(eventos), BATCH_SIZE):
        batch = eventos[i : i + BATCH_SIZE]
        try:
            inserted = repo.insertar_eventos_sensor(batch)
            total += inserted
        except BulkWriteError as e:
            logger.warning(f"  BulkWriteError parcial: {e.details.get('writeErrors', [])[:2]}")
            total += e.details.get("nInserted", 0)
        if (i // BATCH_SIZE) % 5 == 0:
            logger.info(f"  Progreso: {total:,} / {len(eventos):,}")

    return total


def main() -> None:
    args = parse_args()
    repo = MongoRepository()

    try:
        # Sanity check de conexión
        repo.db.command("ping")
        logger.info(f"Conexión a MongoDB OK: {repo.uri.split('@')[-1].split('/')[0]}")
    except Exception as e:
        logger.error(f"No se puede conectar a MongoDB: {e}")
        logger.error("Revisa MONGO_URI en .env y que el cluster esté accesible.")
        return

    # Reset opcional
    if args.reset:
        n = repo.vaciar_coleccion(COL_SENSORES)
        logger.info(f"Reset: eliminados {n:,} documentos de {COL_SENSORES}")

    # Crear índices antes de cargar (más rápido al vacío que después)
    repo.crear_indices()

    sedes_a_cargar = [args.sede] if args.sede else list(SEDES.keys())

    t0 = time.time()
    total_global = 0
    for sede_id in sedes_a_cargar:
        logger.info(f"=== Cargando sede {SEDES[sede_id]['nombre']} ({sede_id}) ===")
        n = cargar_sede(repo, sede_id)
        total_global += n
        logger.info(f"  → {n:,} eventos cargados")

    dt = time.time() - t0
    logger.info(
        f"Carga completa: {total_global:,} eventos en {dt:.1f}s "
        f"({total_global / dt:,.0f} docs/s)"
    )

    # Verificación
    count = repo.db[COL_SENSORES].count_documents({})
    logger.info(f"Total en colección {COL_SENSORES}: {count:,} documentos")

    repo.close()


if __name__ == "__main__":
    main()
