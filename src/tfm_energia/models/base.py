"""Interfaz común a todos los modelos de predicción del TFM.

Definir un contrato único (`fit` / `predict`) permite que baselines, SARIMAX,
Prophet y LSTM se evalúen con el mismo código de backtesting y las mismas
métricas, que es justo lo que hace comparable el estudio.

Convenios:

  * La serie objetivo `y` es un ``pd.Series`` con ``DatetimeIndex`` horario.
  * ``predict(steps)`` devuelve una predicción **puramente fuera de muestra**:
    los `steps` timestamps siguientes al final del entrenamiento.
  * Las exógenas `X` son opcionales; los modelos que no las usan las ignoran.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger


FREQ_HORARIA = "h"


class BaseForecaster(ABC):
    """Clase base de todos los predictores de consumo."""

    nombre: str = "base"

    def __init__(self) -> None:
        self._y: pd.Series | None = None
        self._ajustado: bool = False

    # -- API pública --------------------------------------------------------
    def fit(self, y: pd.Series, X: pd.DataFrame | None = None) -> "BaseForecaster":
        """Ajusta el modelo sobre la serie de entrenamiento."""
        if not isinstance(y.index, pd.DatetimeIndex):
            raise ValueError("y debe tener DatetimeIndex.")
        self._y = y.sort_index().astype(float)
        self._fit(self._y, X)
        self._ajustado = True
        return self

    def predict(self, steps: int, X: pd.DataFrame | None = None) -> pd.Series:
        """Predice los `steps` periodos horarios siguientes al entrenamiento."""
        self._check_ajustado()
        if steps <= 0:
            raise ValueError("steps debe ser > 0.")
        pred = self._predict(steps, X)
        pred.index = self.indice_futuro(steps)
        return pred.rename(self.nombre)

    def indice_futuro(self, steps: int) -> pd.DatetimeIndex:
        """Índice horario de los `steps` timestamps posteriores al train."""
        self._check_ajustado()
        inicio = self._y.index[-1] + pd.Timedelta(hours=1)
        return pd.date_range(start=inicio, periods=steps, freq=FREQ_HORARIA)

    # -- A implementar por cada modelo --------------------------------------
    @abstractmethod
    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None: ...

    @abstractmethod
    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series: ...

    # -- Utilidades internas ------------------------------------------------
    def _check_ajustado(self) -> None:
        if not self._ajustado or self._y is None:
            raise RuntimeError(f"{self.nombre}: hay que llamar a fit() antes de predecir.")

    def __repr__(self) -> str:  # pragma: no cover - cosmético
        estado = "ajustado" if self._ajustado else "sin ajustar"
        return f"<{self.__class__.__name__} nombre={self.nombre!r} ({estado})>"


# ---------------------------------------------------------------------------
# Backtesting con origen móvil
# ---------------------------------------------------------------------------
def backtest_horizonte(
    modelo: BaseForecaster,
    y: pd.Series,
    horizonte: int = 48,
    paso: int = 24,
    n_origenes: int | None = None,
    min_train: int | None = None,
    X: pd.DataFrame | None = None,
    reajustar: bool = True,
) -> pd.DataFrame:
    """Backtesting *rolling origin* para un horizonte fijo.

    Reproduce cómo trabajaría el sistema en producción: cada día se reentrena
    con todo lo observado hasta ese momento y se predicen las siguientes
    `horizonte` horas (48 h según el anteproyecto).

    Args:
        modelo: predictor que implementa la interfaz :class:`BaseForecaster`.
        y: serie completa (train + test) sobre la que se hace el backtest.
        horizonte: horas a predecir en cada origen (48 por defecto).
        paso: horas entre orígenes consecutivos (24 = un origen al día).
        n_origenes: nº de orígenes a evaluar. None = todos los que quepan.
        min_train: tamaño mínimo del histórico inicial. Por defecto 1 año.
        X: exógenas alineadas con `y` (se recortan igual que la serie).
        reajustar: si False, ajusta una sola vez con el histórico inicial.

    Returns:
        DataFrame con columnas ``origen``, ``timestamp``, ``h`` (paso del
        horizonte, 1..horizonte), ``real`` y ``pred``.
    """
    y = y.sort_index().astype(float)
    min_train = min_train or min(len(y) // 2, 24 * 365)

    origenes: list[int] = []
    i = min_train
    while i + horizonte <= len(y):
        origenes.append(i)
        i += paso
    if n_origenes is not None:
        origenes = origenes[-n_origenes:]
    if not origenes:
        raise ValueError(
            f"Serie demasiado corta: len={len(y)}, min_train={min_train}, horizonte={horizonte}"
        )

    logger.info(
        f"Backtest {modelo.nombre}: {len(origenes)} orígenes, horizonte={horizonte}h, paso={paso}h"
    )

    filas = []
    for corte in origenes:
        y_train = y.iloc[:corte]
        y_real = y.iloc[corte : corte + horizonte]
        X_train = X.iloc[:corte] if X is not None else None
        X_fut = X.iloc[corte : corte + horizonte] if X is not None else None

        if reajustar or not modelo._ajustado:
            modelo.fit(y_train, X_train)
        pred = modelo.predict(horizonte, X_fut)

        filas.append(
            pd.DataFrame(
                {
                    "origen": y_train.index[-1],
                    "timestamp": y_real.index,
                    "h": range(1, len(y_real) + 1),
                    "real": y_real.to_numpy(),
                    "pred": pred.to_numpy()[: len(y_real)],
                }
            )
        )

    return pd.concat(filas, ignore_index=True)


def metricas_por_horizonte(bt: pd.DataFrame) -> pd.DataFrame:
    """Degradación del error a medida que crece el horizonte de predicción.

    Recibe la salida de :func:`backtest_horizonte` y devuelve MAE/RMSE por
    cada paso `h`. Es la tabla que justifica hasta dónde es fiable predecir.
    """
    from tfm_energia.models.metrics import mae, rmse

    filas = [
        {"h": h, "MAE": mae(g["real"], g["pred"]), "RMSE": rmse(g["real"], g["pred"]), "n": len(g)}
        for h, g in bt.groupby("h")
    ]
    return pd.DataFrame(filas)
