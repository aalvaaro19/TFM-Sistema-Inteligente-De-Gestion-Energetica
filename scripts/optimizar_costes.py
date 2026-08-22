from __future__ import annotations

import argparse
import dataclasses

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.config import PROCESSED_DIR, SEDES
from tfm_energia.optimization.optimizer import (
    ResultadoComparativa,
    serie_control_reactivo,
    simular_control,
)
from tfm_energia.optimization.thermal_model import parametros_de_sede


def cargar(sede_id: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"enriquecido_{sede_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta enrich_with_real_data.py")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


def comparar_sede(
    sede_id: str, ventanas: int | None, margen: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ejecuta las tres estrategias en una sede. Devuelve (tabla, series)."""
    df = cargar(sede_id)
    p = parametros_de_sede(SEDES[sede_id])
    precio_plano = np.full(len(df), df["precio_eur_kwh"].mean())

    logger.info(f"=== {SEDES[sede_id]['nombre']} ===")
    logger.info("  control predictivo ciego al precio (tarifa plana)...")
    ciego = simular_control(
        df, p, precios_decision=precio_plano,
        margen_preacondicionamiento=margen, max_ventanas=ventanas,
    )
    logger.info("  control predictivo consciente del precio...")
    consciente = simular_control(
        df, p, margen_preacondicionamiento=margen, max_ventanas=ventanas,
    )
    reactivo = serie_control_reactivo(df, p).loc[ciego.index]

    comp = ResultadoComparativa(
        estrategias={
            "reactivo_actual": reactivo,
            "predictivo_ciego": ciego,
            "predictivo_precio": consciente,
        },
        referencia="predictivo_ciego",
    )
    tabla = comp.tabla().reset_index()
    tabla.insert(0, "sede", sede_id)

    series = pd.concat(
        {k: v for k, v in comp.estrategias.items()}, names=["estrategia", "timestamp"]
    ).reset_index()
    series.insert(0, "sede", sede_id)

    for _, fila in tabla.iterrows():
        logger.info(
            f"  {fila['estrategia']:<20} {fila['energia_kwh']:>9,.0f} kWh  "
            f"{fila['coste_eur']:>8,.0f} €  {fila['precio_medio']:.4f} €/kWh  "
            f"{fila['grados_hora']:>7,.0f} °C·h"
        )
    return tabla, series


def analisis_sensibilidad(sede_id: str, ventanas: int) -> pd.DataFrame:
    """Cómo depende el ahorro de la envolvente y del margen de preacondicionamiento.

    Es el análisis que explica el resultado: el desplazamiento de carga solo
    resulta rentable si el edificio conserva el calor que almacena.
    """
    df = cargar(sede_id)
    base = parametros_de_sede(SEDES[sede_id])
    precio_plano = np.full(len(df), df["precio_eur_kwh"].mean())

    filas = []
    for k in (0.12, 0.08, 0.05, 0.03):
        p = dataclasses.replace(base, k_envoltura=k)
        for margen in (0.0, 3.0):
            ciego = simular_control(
                df, p, precios_decision=precio_plano,
                margen_preacondicionamiento=margen, max_ventanas=ventanas,
            )
            cons = simular_control(
                df, p, margen_preacondicionamiento=margen, max_ventanas=ventanas,
            )
            c0, c1 = ciego["coste_eur"].sum(), cons["coste_eur"].sum()
            filas.append({
                "sede": sede_id,
                "k_envoltura": k,
                "constante_tiempo_h": round(1 / k, 1),
                "margen_preacondicionamiento": margen,
                "coste_ciego": c0,
                "coste_consciente": c1,
                "ahorro_pct": 100 * (c0 - c1) / c0 if c0 else np.nan,
                "energia_extra_pct": 100 * (
                    cons["energia_kwh"].sum() / ciego["energia_kwh"].sum() - 1
                ),
            })
            logger.info(
                f"  k={k:.2f} (tau={1/k:.0f} h) margen={margen:.0f} °C  "
                f"→ ahorro {filas[-1]['ahorro_pct']:+.2f}%"
            )
    return pd.DataFrame(filas)


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimización de costes (fase 7)")
    ap.add_argument("--sede", default="todas")
    ap.add_argument("--ventanas", type=int, default=None, help="Nº de ventanas; None = año completo")
    ap.add_argument("--margen", type=float, default=3.0)
    ap.add_argument("--sensibilidad", action="store_true")
    args = ap.parse_args()

    if args.sede != "todas" and args.sede not in SEDES:
        raise SystemExit(f"Sede desconocida: {args.sede}")
    sedes = list(SEDES) if args.sede == "todas" else [args.sede]

    tablas, series = [], []
    for sede_id in sedes:
        t, s = comparar_sede(sede_id, args.ventanas, args.margen)
        tablas.append(t)
        series.append(s)

    resumen = pd.concat(tablas, ignore_index=True)
    resumen.to_csv(PROCESSED_DIR / "optimizacion_resumen.csv", index=False)
    pd.concat(series, ignore_index=True).to_parquet(
        PROCESSED_DIR / "optimizacion_horaria.parquet", index=False
    )

    print("\n=== RESUMEN POR SEDE ===")
    cols = ["sede", "estrategia", "energia_kwh", "coste_eur", "precio_medio",
            "grados_hora", "ahorro_pct"]
    print(resumen[cols].to_string(index=False, float_format=lambda v: f"{v:,.2f}"))

    print("\n=== AGREGADO DE LAS SEDES ===")
    agg = resumen.groupby("estrategia")[["energia_kwh", "coste_eur", "grados_hora"]].sum()
    base = agg.loc["predictivo_ciego", "coste_eur"]
    agg["ahorro_eur"] = base - agg["coste_eur"]
    agg["ahorro_pct"] = 100 * agg["ahorro_eur"] / base
    print(agg.to_string(float_format=lambda v: f"{v:,.1f}"))

    if args.sensibilidad:
        print("\n=== SENSIBILIDAD (Madrid) ===")
        sens = analisis_sensibilidad("madrid", args.ventanas or 40)
        sens.to_csv(PROCESSED_DIR / "optimizacion_sensibilidad.csv", index=False)
        print(sens.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))


if __name__ == "__main__":
    main()
