from __future__ import annotations

from datetime import date

from loguru import logger

from tfm_energia.config import RAW_DIR
from tfm_energia.data.esios_client import EsiosClient


def main() -> None:
    start = date(2024, 1, 1)
    end = date(2025, 12, 31)
    out_dir = RAW_DIR / "esios"
    out_dir.mkdir(parents=True, exist_ok=True)

    with EsiosClient() as client:
        logger.info(f"Descargando PVPC entre {start} y {end}")
        df = client.precio_pvpc(start, end)
        if df.empty:
            logger.warning("Sin datos recibidos de e·sios.")
            return
        csv_path = out_dir / "pvpc_horario.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"→ {len(df)} registros en {csv_path.name}")


if __name__ == "__main__":
    main()
