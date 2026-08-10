"""Métricas de evaluación para los modelos de predicción de consumo.

Se centraliza aquí el cálculo de errores para que **todos** los modelos del TFM
(baselines, SARIMAX, Prophet, LSTM) se comparen exactamente con el mismo criterio.

Métricas implementadas:

  * **MAE**   – error absoluto medio (kWh). Interpretable en unidades del target.
  * **RMSE**  – penaliza más los errores grandes (picos de consumo).
  * **MAPE**  – error porcentual medio. Inestable si hay valores próximos a 0.
  * **sMAPE** – variante simétrica del MAPE, acotada en [0, 200] %.
  * **R2**    – proporción de varianza explicada.
  * **MBE**   – sesgo medio: >0 el modelo sobreestima, <0 infraestima.

Nota metodológica: en este dataset el consumo nunca baja de la carga base
(~5 kWh), por lo que el MAPE es estable. Aun así se calcula con una máscara de
seguridad para evitar divisiones por cero si se reutiliza en otras series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-9


def _alinear(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Alinea dos series por índice (si lo tienen) y descarta NaN de cualquiera.

    Es la salvaguarda que evita el error clásico de evaluar predicciones
    desplazadas respecto a los valores reales.
    """
    if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
        df = pd.concat([y_true.rename("real"), y_pred.rename("pred")], axis=1, join="inner")
        if df.empty:
            raise ValueError("No hay solape temporal entre y_true e y_pred.")
        df = df.dropna()
        return df["real"].to_numpy(dtype=float), df["pred"].to_numpy(dtype=float)

    yt = np.asarray(y_true, dtype=float).ravel()
    yp = np.asarray(y_pred, dtype=float).ravel()
    if yt.shape != yp.shape:
        raise ValueError(f"Dimensiones incompatibles: {yt.shape} vs {yp.shape}")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]


# ---------------------------------------------------------------------------
# Métricas individuales
# ---------------------------------------------------------------------------
def mae(y_true, y_pred) -> float:
    """Mean Absolute Error (kWh)."""
    yt, yp = _alinear(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error (kWh)."""
    yt, yp = _alinear(y_true, y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (%). Ignora reales ~0."""
    yt, yp = _alinear(y_true, y_pred)
    mask = np.abs(yt) > EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (%), acotado en [0, 200]."""
    yt, yp = _alinear(y_true, y_pred)
    denom = (np.abs(yt) + np.abs(yp)) / 2
    mask = denom > EPSILON
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(yt[mask] - yp[mask]) / denom[mask]) * 100)


def r2(y_true, y_pred) -> float:
    """Coeficiente de determinación."""
    yt, yp = _alinear(y_true, y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot < EPSILON:
        return float("nan")
    return float(1 - ss_res / ss_tot)


def mbe(y_true, y_pred) -> float:
    """Mean Bias Error (kWh). Positivo = el modelo sobreestima."""
    yt, yp = _alinear(y_true, y_pred)
    return float(np.mean(yp - yt))


# ---------------------------------------------------------------------------
# Agregadores
# ---------------------------------------------------------------------------
def calcular_metricas(
    y_true,
    y_pred,
    nombre: str = "modelo",
) -> dict[str, float | str | int]:
    """Devuelve el cuadro completo de métricas de un modelo en un dict."""
    yt, yp = _alinear(y_true, y_pred)
    return {
        "modelo": nombre,
        "n": int(len(yt)),
        "MAE": mae(yt, yp),
        "RMSE": rmse(yt, yp),
        "MAPE": mape(yt, yp),
        "sMAPE": smape(yt, yp),
        "R2": r2(yt, yp),
        "MBE": mbe(yt, yp),
    }


def comparar_modelos(
    resultados: list[dict] | dict[str, tuple],
    ordenar_por: str = "MAE",
) -> pd.DataFrame:
    """Construye la tabla comparativa de modelos ordenada por una métrica.

    Acepta dos formatos:
      * lista de dicts ya calculados con :func:`calcular_metricas`
      * dict ``{nombre: (y_true, y_pred)}`` y calcula las métricas al vuelo
    """
    if isinstance(resultados, dict):
        filas = [calcular_metricas(yt, yp, nombre) for nombre, (yt, yp) in resultados.items()]
    else:
        filas = list(resultados)

    df = pd.DataFrame(filas)
    ascendente = ordenar_por != "R2"  # en R2 mejor es mayor
    return df.sort_values(ordenar_por, ascending=ascendente).reset_index(drop=True)


def mejora_relativa(metrica_modelo: float, metrica_baseline: float) -> float:
    """% de mejora de un modelo respecto al baseline (positivo = mejor).

    Aplicable a métricas donde menos es mejor (MAE, RMSE, MAPE, sMAPE).
    """
    if abs(metrica_baseline) < EPSILON:
        return float("nan")
    return float((metrica_baseline - metrica_modelo) / metrica_baseline * 100)
