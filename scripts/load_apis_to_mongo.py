from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from tfm_energia.config import RAW_DIR, SEDES
from tfm_energia.data.api_loader import preparar_meteo, preparar_precios, resumen_precios
from tfm_energia.data.mongo_repository import (
    COL_METEO,
    COL_PRECIOS,
    MongoRepository,
)


def cargar_meteo(repo: MongoRepository) -> int:
    """Carga las observaciones diarias de AEMET de las cuatro sedes."""
    total = 0
    for sede_id, meta in SEDES.items():
        path = RAW_DIR / "aemet" / f"meteo_{sede_id}.csv"
        if not path.exists():
            logger.warning(f"No existe {path.name}; se omite {sede_id}")
            continue

        documentos = preparar_meteo(pd.read_csv(path), sede_id)
        n = repo.insertar_meteo(documentos)
        total += n
        logger.info(
            f"  {meta['nombre']:<10} estación {meta['aemet_station']:<6} "
            f"{len(documentos):>4} días → {n:>4} escritos"
        )
    return total


def cargar_precios(repo: MongoRepository) -> int:
    """Carga el histórico horario de precios PVPC."""
    path = RAW_DIR / "esios" / "pvpc_horario.csv"
    if not path.exists():
        logger.warning(f"No existe {path}; se omiten los precios")
        return 0

    documentos = preparar_precios(pd.read_csv(path))
    n = repo.insertar_precios(documentos)

    res = resumen_precios(documentos)
    logger.info(f"  {res['n']:,} horas de precio → {n:,} escritos")
    logger.info(
        f"  Precio medio {res['precio_medio']:.5f} €/kWh "
        f"(min {res['precio_min']:.5f}, max {res['precio_max']:.5f})"
    )
    for franja, cuenta in sorted(res["por_franja"].items()):
        logger.info(f"    franja {franja:<8} {cuenta:>6,} horas")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description="Carga de datos de APIs externas a MongoDB")
    p.add_argument("--solo", choices=["meteo", "precios"], default=None)
    args = p.parse_args()

    repo = MongoRepository()
    try:
        repo.db.command("ping")
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"No se puede conectar a MongoDB: {type(exc).__name__}") from exc

    repo.crear_indices()

    if args.solo in (None, "meteo"):
        logger.info("=== AEMET: observación diaria por sede ===")
        cargar_meteo(repo)

    if args.solo in (None, "precios"):
        logger.info("=== e·sios: precios PVPC horarios ===")
        cargar_precios(repo)

    print("\n=== ESTADO DE LAS COLECCIONES ===")
    for nombre in (COL_METEO, COL_PRECIOS):
        print(f"  {nombre:<16} {repo.db[nombre].count_documents({}):>8,} documentos")
    repo.close()


if __name__ == "__main__":
    main()
