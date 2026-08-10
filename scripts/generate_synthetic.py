"""Genera los datasets sintéticos de las cuatro sedes y los guarda en data/synthetic/.

Uso:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --start 2024-01-01 --end 2025-12-31 --seed 42
"""
from __future__ import annotations

import argparse
from datetime import date

from loguru import logger

from tfm_energia.data.synthetic_generator import (
    SimulationConfig,
    generate_all_sedes,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generador sintético TFM")
    p.add_argument("--start", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    start = date.fromisoformat(args.start) if args.start else None
    end = date.fromisoformat(args.end) if args.end else None

    cfg = SimulationConfig(seed=args.seed)
    logger.info(f"Generando dataset sintético (seed={args.seed})")
    resultados = generate_all_sedes(start=start, end=end, cfg=cfg)

    total_registros = sum(len(df) for df in resultados.values())
    total_anomalias = sum(int(df["es_anomalia"].sum()) for df in resultados.values())
    logger.info(
        f"Generación completa. {total_registros:,} registros. "
        f"{total_anomalias} eventos anómalos."
    )


if __name__ == "__main__":
    main()
