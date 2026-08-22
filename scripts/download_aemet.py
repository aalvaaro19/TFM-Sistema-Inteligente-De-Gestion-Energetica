from __future__ import annotations

import argparse
from datetime import date

from loguru import logger

from tfm_energia.config import RAW_DIR, SEDES
from tfm_energia.data.aemet_client import AemetClient


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sede", type=str, default=None, choices=list(SEDES.keys()) + [None])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = date(2024, 1, 1)
    end = date(2025, 12, 31)
    out_dir = RAW_DIR / "aemet"
    out_dir.mkdir(parents=True, exist_ok=True)

    sedes_a_descargar = [args.sede] if args.sede else list(SEDES.keys())
    resumen = {}

    with AemetClient() as client:
        for sede_id in sedes_a_descargar:
            meta = SEDES[sede_id]
            station = meta["aemet_station"]
            logger.info(f"Descargando AEMET para {meta['nombre']} ({station})")
            try:
                df = client.fetch_historico_por_lotes(station, start, end)
                if df.empty:
                    logger.warning(f"  Sin datos para {sede_id}")
                    resumen[sede_id] = "vacío"
                    continue
                csv_path = out_dir / f"meteo_{sede_id}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"  → {len(df)} registros en {csv_path.name}")
                resumen[sede_id] = f"OK ({len(df)} regs)"
            except Exception as e:
                logger.error(f"  Falló {sede_id} ({station}): {type(e).__name__}: {e}")
                resumen[sede_id] = f"FALLÓ ({type(e).__name__})"

    logger.info("=" * 50)
    logger.info("RESUMEN:")
    for sede_id, estado in resumen.items():
        logger.info(f"  {sede_id:12s} → {estado}")


if __name__ == "__main__":
    main()
