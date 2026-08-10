"""Detección de anomalías de consumo y evaluación contra las etiquetas reales.

Entrena un Isolation Forest por sede de forma **no supervisada** —sin usar las
etiquetas— y después lo evalúa contra ellas. Se compara con un baseline
estadístico para comprobar que la complejidad añadida se justifica.

Salidas en `data/processed/`:
    * ``anomalias_metricas.csv``    – comparativa de detectores por sede
    * ``anomalias_detectadas.csv``  – detalle de cada detección, para el dashboard

Uso:
    python scripts/detectar_anomalias.py
    python scripts/detectar_anomalias.py --sede madrid --contaminacion 0.03
"""
from __future__ import annotations

import argparse

import pandas as pd
from loguru import logger

from tfm_energia.config import PROCESSED_DIR, SEDES
from tfm_energia.models.anomaly_detection import (
    ConfigDeteccion,
    DetectorCompuesto,
    DetectorEstadistico,
    DetectorIsolationForest,
    DetectorSensorCongelado,
    construir_features_anomalia,
    evaluar_deteccion,
    predecir_con_presupuesto,
)


def cargar(sede_id: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"enriquecido_{sede_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No existe {path}. Ejecuta enrich_with_real_data.py")
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    return df.sort_index()


def construir_detectores(cfg: ConfigDeteccion) -> list:
    """Los detectores de la comparativa, cada uno con su especialidad.

    El compuesto reúne tres canales complementarios: el bosque para consumos
    puntualmente anómalos, la regla de varianza para sensores bloqueados, y el
    residuo acumulado para desviaciones pequeñas pero sostenidas, que hora a
    hora quedan enterradas en el ruido.
    """
    isolation = DetectorIsolationForest(cfg)
    congelado = DetectorSensorCongelado()

    # Cada canal vigila la señal donde su avería deja huella:
    #  - el consumo total delata los picos
    #  - el consumo de equipos delata las fugas: +6 kWh se pierden en un total
    #    con desviación típica de 14, pero duplican una componente que ronda los 5
    #  - la varianza de la temperatura delata los sensores bloqueados
    #  - el bosque recoge lo que no encaja en ningún patrón concreto
    total = DetectorEstadistico(columna="consumo_total_kwh_residuo")
    total.nombre = "residuo_total"
    equipos = DetectorEstadistico(columna="consumo_equipos_kwh_residuo")
    equipos.nombre = "residuo_equipos"

    return [
        total,
        congelado,
        equipos,
        isolation,
        DetectorCompuesto(
            [isolation, congelado, equipos, total], nombre="compuesto_4canales"
        ),
    ]


def procesar_sede(
    sede_id: str, cfg: ConfigDeteccion, presupuesto: float
) -> tuple[list, pd.DataFrame]:
    df = cargar(sede_id)
    X = construir_features_anomalia(
        df, ventana=cfg.ventana_variacion, dias_referencia=cfg.dias_referencia
    )

    logger.info(
        f"=== {SEDES[sede_id]['nombre']}: {len(df):,} horas, "
        f"{int(df['es_anomalia'].sum())} anomalías reales ({df['es_anomalia'].mean():.2%}) ==="
    )

    resultados = []
    detecciones = pd.DataFrame(index=df.index)

    for detector in construir_detectores(cfg):
        # Entrenamiento SIN etiquetas: es lo que ocurriría en producción
        detector.fit(X)
        puntuaciones = detector.puntuar(X)
        # Mismo presupuesto de avisos para todos: comparación en igualdad.
        # El compuesto lo reparte entre sus canales para que ninguno silencie
        # a los demás.
        if isinstance(detector, DetectorCompuesto):
            pred = detector.predecir_con_cuotas(X, presupuesto)
        else:
            pred = predecir_con_presupuesto(puntuaciones, presupuesto)

        res = evaluar_deteccion(
            pred,
            df["es_anomalia"],
            df["tipo_anomalia"],
            nombre=detector.nombre,
            puntuaciones=puntuaciones,
        )
        resultados.append((sede_id, res))
        detecciones[detector.nombre] = pred

        auc = f"{res.roc_auc:.3f}" if res.roc_auc is not None else "  n/a"
        logger.info(
            f"  {detector.nombre:<22} P={res.precision:.3f} R={res.recall:.3f} "
            f"F1={res.f1:.3f} AUC={auc} | episodios "
            f"{res.episodios_detectados}/{res.episodios_totales} ({res.recall_episodios:.1%})"
        )
        for tipo, r in sorted(res.recall_por_tipo.items()):
            logger.info(f"      recall {tipo:<20} {r:.3f}")

    detalle = df.loc[
        detecciones.any(axis=1),
        ["sede", "consumo_total_kwh", "consumo_hvac_kwh", "consumo_equipos_kwh",
         "temperatura_interior_c", "es_anomalia", "tipo_anomalia"],
    ].copy()
    for col in detecciones.columns:
        detalle[f"det_{col}"] = detecciones.loc[detalle.index, col]

    return resultados, detalle


def main() -> None:
    p = argparse.ArgumentParser(description="Detección de anomalías (fase 6)")
    p.add_argument("--sede", default="todas")
    p.add_argument("--contaminacion", type=float, default=0.02)
    p.add_argument("--ventana", type=int, default=3)
    p.add_argument("--dias-referencia", type=int, default=14)
    p.add_argument(
        "--presupuesto",
        type=float,
        default=0.02,
        help="Fracción de horas que se pueden avisar (mismo para todos los detectores)",
    )
    args = p.parse_args()

    if args.sede != "todas" and args.sede not in SEDES:
        raise SystemExit(f"Sede desconocida: {args.sede}")
    sedes = list(SEDES) if args.sede == "todas" else [args.sede]

    cfg = ConfigDeteccion(
        contaminacion=args.contaminacion,
        ventana_variacion=args.ventana,
        dias_referencia=args.dias_referencia,
    )

    filas, detalles = [], []
    for sede_id in sedes:
        resultados, detalle = procesar_sede(sede_id, cfg, args.presupuesto)
        for sede, res in resultados:
            fila = res.a_dict()
            fila["sede"] = sede
            filas.append(fila)
        detalles.append(detalle)

    tabla = pd.DataFrame(filas)
    cols = ["sede", "detector", "precision", "recall", "f1", "recall_episodios",
            "roc_auc", "avg_precision"]
    tipos = [c for c in tabla.columns if c.startswith("recall_") and c != "recall_episodios"]
    tabla = tabla[cols + tipos + ["VP", "FP", "FN"]]
    tabla.to_csv(PROCESSED_DIR / "anomalias_metricas.csv", index=False)
    pd.concat(detalles).to_csv(PROCESSED_DIR / "anomalias_detectadas.csv")

    print(f"\n=== COMPARATIVA (presupuesto de aviso: {args.presupuesto:.1%} de las horas) ===")
    print(tabla[cols].to_string(index=False, float_format=lambda v: f"{v:,.3f}"))

    print("\n=== RECALL POR TIPO DE AVERÍA (media de las sedes) ===")
    print(tabla.groupby("detector")[tipos].mean().to_string(float_format=lambda v: f"{v:,.3f}"))

    print("\n=== MEDIA GLOBAL POR DETECTOR ===")
    resumen = tabla.groupby("detector")[
        ["precision", "recall", "f1", "recall_episodios", "roc_auc", "avg_precision"]
    ].mean().sort_values("f1", ascending=False)
    print(resumen.to_string(float_format=lambda v: f"{v:,.3f}"))


if __name__ == "__main__":
    main()
