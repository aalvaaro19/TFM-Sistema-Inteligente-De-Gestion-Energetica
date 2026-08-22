from __future__ import annotations

import argparse
import time

import pandas as pd
from loguru import logger

from tfm_energia.config import PROCESSED_DIR, SEDES
from tfm_energia.models.base import backtest_horizonte, metricas_por_horizonte
from tfm_energia.models.baseline import baselines_estandar
from tfm_energia.models.metrics import calcular_metricas, comparar_modelos, mejora_relativa
from tfm_energia.models.ml_model import EXOGENAS_ML_DEFAULT, GradientBoostingForecaster
from tfm_energia.models.sarimax_model import EXOGENAS_DEFAULT, SarimaxForecaster


TARGET = "consumo_total_kwh"


def cargar_sede(sede_id: str) -> tuple[pd.Series, pd.DataFrame]:
    """Carga el parquet enriquecido y devuelve (target, exógenas) indexados por tiempo."""
    path = PROCESSED_DIR / f"enriquecido_{sede_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No existe {path}. Ejecuta antes scripts/enrich_with_real_data.py"
        )
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df = df.sort_index()

    # Se carga la unión de exógenas que necesita cualquiera de los modelos;
    # cada uno selecciona después las suyas.
    exogenas = list(dict.fromkeys(EXOGENAS_DEFAULT + EXOGENAS_ML_DEFAULT))
    y = df[TARGET].astype(float)
    X = df[exogenas].astype(float)
    logger.info(f"{sede_id}: {len(y):,} horas ({y.index.min()} → {y.index.max()})")
    return y, X


def modelos_a_evaluar(con_seasonal: bool, max_train_horas: int) -> list:
    """Lista de modelos de la comparativa del día 3."""
    modelos = list(baselines_estandar())

    # SARIMAX univariante: mide cuánto aporta la parte puramente autorregresiva
    modelos.append(
        SarimaxForecaster(
            order=(2, 0, 1),
            usar_fourier=True,
            exogenas=None,
            max_train_horas=max_train_horas,
            nombre="sarimax_fourier_univar",
        )
    )
    # SARIMAX con exógenas reales (temperatura AEMET + ocupación)
    modelos.append(
        SarimaxForecaster(
            order=(2, 0, 1),
            usar_fourier=True,
            exogenas=EXOGENAS_DEFAULT,
            max_train_horas=max_train_horas,
            nombre="sarimax_fourier_exog",
        )
    )
    # Gradient boosting con features conocidas a 48 h vista
    modelos.append(GradientBoostingForecaster(horizonte=48, max_train_horas=24 * 365))

    if con_seasonal:
        # Variante estacional clásica (P,D,Q,24): más lenta, se pide explícitamente
        modelos.append(
            SarimaxForecaster(
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 24),
                exogenas=EXOGENAS_DEFAULT,
                usar_fourier=False,
                max_train_horas=min(max_train_horas, 24 * 30),
                nombre="sarimax_s24_exog",
            )
        )
    return modelos


def evaluar_sede(
    sede_id: str,
    horizonte: int,
    paso: int,
    n_origenes: int,
    con_seasonal: bool,
    max_train_horas: int,
) -> pd.DataFrame:
    """Ejecuta el backtest de todos los modelos sobre una sede y guarda resultados."""
    y, X = cargar_sede(sede_id)

    resultados: list[dict] = []
    backtests: list[pd.DataFrame] = []
    horizontes: list[pd.DataFrame] = []

    for modelo in modelos_a_evaluar(con_seasonal, max_train_horas):
        t0 = time.perf_counter()
        bt = backtest_horizonte(
            modelo,
            y,
            horizonte=horizonte,
            paso=paso,
            n_origenes=n_origenes,
            X=X,
        )
        segundos = time.perf_counter() - t0

        met = calcular_metricas(bt["real"], bt["pred"], nombre=modelo.nombre)
        met["segundos"] = round(segundos, 1)
        met["origenes"] = int(bt["origen"].nunique())
        resultados.append(met)

        bt["modelo"] = modelo.nombre
        backtests.append(bt)

        h_tab = metricas_por_horizonte(bt)
        h_tab["modelo"] = modelo.nombre
        horizontes.append(h_tab)

        logger.success(
            f"{modelo.nombre}: MAE={met['MAE']:.3f} kWh | RMSE={met['RMSE']:.3f} | "
            f"MAPE={met['MAPE']:.2f}% | R2={met['R2']:.3f} ({segundos:.1f}s)"
        )

    tabla = comparar_modelos(resultados, ordenar_por="MAE")

    # Mejora respecto al baseline duro (naïve semanal), que es la referencia a batir
    ref = tabla.loc[tabla["modelo"] == "naive_estacional_168h", "MAE"]
    if not ref.empty:
        tabla["mejora_vs_naive168_%"] = tabla["MAE"].apply(
            lambda m: round(mejora_relativa(m, float(ref.iloc[0])), 2)
        )
    tabla.insert(0, "sede", sede_id)

    tabla.to_csv(PROCESSED_DIR / f"metricas_modelos_{sede_id}.csv", index=False)
    pd.concat(horizontes, ignore_index=True).to_csv(
        PROCESSED_DIR / f"metricas_horizonte_{sede_id}.csv", index=False
    )
    pd.concat(backtests, ignore_index=True).to_parquet(
        PROCESSED_DIR / f"backtest_{sede_id}.parquet", index=False
    )
    logger.info(f"Resultados guardados en {PROCESSED_DIR} para la sede {sede_id}")
    return tabla


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparativa de modelos predictivos (fase 3)")
    parser.add_argument("--sede", default="madrid", help="ID de sede o 'todas'")
    parser.add_argument("--horizonte", type=int, default=48, help="Horas a predecir por origen")
    parser.add_argument("--paso", type=int, default=24, help="Horas entre orígenes")
    parser.add_argument("--n-origenes", type=int, default=30, help="Nº de orígenes del backtest")
    parser.add_argument(
        "--max-train-horas", type=int, default=24 * 60, help="Ventana de entrenamiento SARIMAX"
    )
    parser.add_argument(
        "--con-seasonal", action="store_true", help="Añade SARIMAX estacional (P,D,Q,24), lento"
    )
    args = parser.parse_args()

    sedes = list(SEDES) if args.sede == "todas" else [args.sede]
    if args.sede != "todas" and args.sede not in SEDES:
        raise SystemExit(f"Sede desconocida: {args.sede}. Opciones: {list(SEDES)} o 'todas'")

    tablas = [
        evaluar_sede(
            sede_id,
            horizonte=args.horizonte,
            paso=args.paso,
            n_origenes=args.n_origenes,
            con_seasonal=args.con_seasonal,
            max_train_horas=args.max_train_horas,
        )
        for sede_id in sedes
    ]

    resumen = pd.concat(tablas, ignore_index=True)
    if len(sedes) > 1:
        resumen.to_csv(PROCESSED_DIR / "metricas_modelos_todas.csv", index=False)

    print("\n=== COMPARATIVA DE MODELOS (backtest 48h) ===")
    cols = ["sede", "modelo", "MAE", "RMSE", "MAPE", "sMAPE", "R2", "MBE", "segundos"]
    print(resumen[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
