"""Simula los gateways IoT de las sedes emitiendo lecturas en JSON Lines.

Convierte el histórico de `data/synthetic/` en ficheros de eventos que el
pipeline de ingesta (StreamSets o su equivalente Python) puede consumir. Es la
pieza que convierte un dataset estático en un flujo.

Salida: `data/stream/sede={sede}/lecturas_{clave}.jsonl`

Uso:
    # Vuelca todo el histórico particionado por mes (carga inicial)
    python scripts/simulate_sensors.py --modo lote --particion mes

    # Solo Madrid, un rango concreto, particionado por día
    python scripts/simulate_sensors.py --modo lote --sede madrid \
        --desde 2025-12-01 --hasta 2025-12-31 --particion dia

    # Emisión en vivo para la demo: un fichero por hora simulada cada 2 s
    python scripts/simulate_sensors.py --modo stream --sede madrid \
        --intervalo 2 --max-lotes 30
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
from loguru import logger

from tfm_energia.config import DATA_DIR, SEDES, SYNTHETIC_DIR
from tfm_energia.data.sensor_stream import ConfigEmisor, EmisorSensores


STREAM_DIR = DATA_DIR / "stream"


def cargar_historico(sede_id: str, desde: str | None, hasta: str | None) -> pd.DataFrame:
    """Carga el parquet de una sede y lo recorta al rango pedido."""
    path = SYNTHETIC_DIR / f"sede_{sede_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta antes scripts/generate_synthetic.py"
        )
    df = pd.read_parquet(path).sort_values("timestamp").reset_index(drop=True)

    if desde:
        df = df[df["timestamp"] >= pd.Timestamp(desde, tz="Europe/Madrid")]
    if hasta:
        # `hasta` inclusive: se toma hasta el final de ese día
        limite = pd.Timestamp(hasta, tz="Europe/Madrid") + pd.Timedelta(days=1)
        df = df[df["timestamp"] < limite]

    if df.empty:
        raise ValueError(f"El rango solicitado no contiene datos para {sede_id}.")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulador de gateways IoT (JSON Lines)")
    p.add_argument("--modo", choices=["lote", "stream"], default="lote")
    p.add_argument("--sede", default="todas", help="ID de sede o 'todas'")
    p.add_argument("--desde", default=None, help="Fecha inicial YYYY-MM-DD")
    p.add_argument("--hasta", default=None, help="Fecha final YYYY-MM-DD (inclusive)")
    p.add_argument("--particion", choices=["dia", "mes"], default="mes")
    p.add_argument(
        "--tasa-defectos",
        type=float,
        default=0.01,
        help="Fracción de eventos corrompidos, para ejercitar la rama de rechazo",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--salida", default=str(STREAM_DIR))
    p.add_argument("--limpiar", action="store_true", help="Vacía el directorio de salida")
    # Solo modo stream
    p.add_argument("--horas-por-lote", type=int, default=1)
    p.add_argument("--intervalo", type=float, default=1.0, help="Segundos entre lotes")
    p.add_argument("--max-lotes", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.sede != "todas" and args.sede not in SEDES:
        raise SystemExit(f"Sede desconocida: {args.sede}. Opciones: {list(SEDES)} o 'todas'")
    sedes = list(SEDES) if args.sede == "todas" else [args.sede]

    base = Path(args.salida)

    if args.limpiar and base.exists():
        shutil.rmtree(base)
        logger.warning(f"Directorio de salida vaciado: {base}")

    cfg = ConfigEmisor(
        tasa_defectos=args.tasa_defectos,
        seed=args.seed,
        particion=args.particion,
    )

    total = 0
    for sede_id in sedes:
        df = cargar_historico(sede_id, args.desde, args.hasta)
        emisor = EmisorSensores(sede_id, cfg)
        logger.info(
            f"=== {SEDES[sede_id]['nombre']} ({sede_id}): {len(df):,} horas "
            f"[{df['timestamp'].min()} → {df['timestamp'].max()}] ==="
        )

        if args.modo == "lote":
            resumen = emisor.volcar_lote(df, base)
            total += sum(resumen.values())
        else:
            total += emisor.emitir_stream(
                df,
                base,
                horas_por_lote=args.horas_por_lote,
                intervalo_s=args.intervalo,
                max_lotes=args.max_lotes,
            )

    logger.success(
        f"Emisión completa: {total:,} eventos en {base} "
        f"(≈{args.tasa_defectos:.1%} defectuosos por diseño)"
    )


if __name__ == "__main__":
    main()
