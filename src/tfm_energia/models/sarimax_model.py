from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.models.base import BaseForecaster


# Exógenas por defecto: las que el EDA mostró más correlacionadas con el consumo
EXOGENAS_DEFAULT = ("temperatura_exterior_c", "ocupacion_rel")

# Periodos estacionales y nº de armónicos para la variante Fourier
FOURIER_DEFAULT = ((24, 3), (168, 2))

ORIGEN_FOURIER = pd.Timestamp("2024-01-01")


def terminos_fourier(
    index: pd.DatetimeIndex,
    periodos: tuple[tuple[int, int], ...] = FOURIER_DEFAULT,
) -> pd.DataFrame:
    """Genera pares seno/coseno para cada periodo estacional.

    Se usa la hora de reloj local (se descarta la zona horaria) para que el
    armónico diario siga alineado con el horario de oficina tras los cambios
    de hora.
    """
    idx_naive = index.tz_localize(None) if index.tz is not None else index
    t = (idx_naive - ORIGEN_FOURIER) / pd.Timedelta(hours=1)

    datos: dict[str, np.ndarray] = {}
    for periodo, n_armonicos in periodos:
        for k in range(1, n_armonicos + 1):
            ang = 2 * np.pi * k * t / periodo
            datos[f"fourier_sin_{periodo}_{k}"] = np.sin(ang)
            datos[f"fourier_cos_{periodo}_{k}"] = np.cos(ang)
    return pd.DataFrame(datos, index=index)


class SarimaxForecaster(BaseForecaster):
    """Envoltorio de ``statsmodels.tsa.statespace.SARIMAX`` con la interfaz del TFM."""

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 1),
        seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
        exogenas: tuple[str, ...] | None = EXOGENAS_DEFAULT,
        usar_fourier: bool = False,
        fourier_periodos: tuple[tuple[int, int], ...] = FOURIER_DEFAULT,
        max_train_horas: int | None = 24 * 90,
        trend: str | None = None,
        nombre: str | None = None,
    ) -> None:
        super().__init__()
        self.order = order
        self.seasonal_order = (0, 0, 0, 0) if usar_fourier else seasonal_order

        # statsmodels no incluye constante por defecto (`trend=None`): sin
        # diferenciar, el proceso revertiría hacia 0 en vez de hacia la media de
        # la serie, lo que produce una infraestimación sistemática del consumo.
        # Solo se añade cuando no hay diferenciación que ya elimine el nivel.
        sin_diferenciar = order[1] == 0 and self.seasonal_order[1] == 0
        self.trend = trend if trend is not None else ("c" if sin_diferenciar else "n")
        self.exogenas = tuple(exogenas or ())
        self.usar_fourier = usar_fourier
        self.fourier_periodos = fourier_periodos
        self.max_train_horas = max_train_horas
        self.resultado = None
        self.ultimo_intervalo: pd.DataFrame | None = None

        sufijo = "fourier" if usar_fourier else f"s{self.seasonal_order[3]}"
        con_ex = "exog" if self.exogenas else "univar"
        self.nombre = nombre or f"sarimax_{sufijo}_{con_ex}"

    # -- Construcción de la matriz exógena -----------------------------------
    def _matriz_exog(self, index: pd.DatetimeIndex, X: pd.DataFrame | None) -> pd.DataFrame | None:
        """Combina exógenas reales y armónicos de Fourier para un índice dado."""
        bloques: list[pd.DataFrame] = []

        if self.exogenas:
            if X is None:
                raise ValueError(
                    f"{self.nombre} requiere exógenas {self.exogenas} pero no se pasó X."
                )
            faltan = [c for c in self.exogenas if c not in X.columns]
            if faltan:
                raise ValueError(f"Faltan exógenas en X: {faltan}")
            bloque = X.loc[:, list(self.exogenas)].astype(float)
            # Se rellenan huecos puntuales de las APIs para no romper el ajuste
            bloque = bloque.ffill().bfill()
            bloques.append(bloque.set_axis(index))

        if self.usar_fourier:
            bloques.append(terminos_fourier(index, self.fourier_periodos))

        if not bloques:
            return None
        return pd.concat(bloques, axis=1)

    # -- Interfaz BaseForecaster ---------------------------------------------
    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        from statsmodels.tsa.statespace.sarimax import SARIMAX

        y_fit = y
        X_fit = X
        if self.max_train_horas is not None and len(y) > self.max_train_horas:
            y_fit = y.iloc[-self.max_train_horas :]
            X_fit = X.iloc[-self.max_train_horas :] if X is not None else None
            # `_y` marca el final del train, que define el índice de predicción
            logger.debug(
                f"{self.nombre}: ventana de entrenamiento recortada a {len(y_fit):,} horas"
            )

        exog = self._matriz_exog(y_fit.index, X_fit)
        self._cols_exog = list(exog.columns) if exog is not None else None

        with warnings.catch_warnings():
            # El optimizador avisa de convergencia lenta en series largas; se
            # registra el resultado y se valora con el AIC, no con el warning.
            warnings.simplefilter("ignore")
            modelo = SARIMAX(
                y_fit.to_numpy(dtype=float),
                exog=None if exog is None else exog.to_numpy(dtype=float),
                order=self.order,
                seasonal_order=self.seasonal_order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            self.resultado = modelo.fit(disp=False)

        logger.info(
            f"{self.nombre} ajustado sobre {len(y_fit):,} h | "
            f"AIC={self.resultado.aic:,.1f} BIC={self.resultado.bic:,.1f}"
        )

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        idx_fut = self.indice_futuro(steps)
        exog_fut = self._matriz_exog(idx_fut, X)
        if exog_fut is not None:
            exog_fut = exog_fut.loc[:, self._cols_exog]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self.resultado.get_forecast(
                steps=steps,
                exog=None if exog_fut is None else exog_fut.to_numpy(dtype=float),
            )

        ic = fc.conf_int(alpha=0.05)
        self.ultimo_intervalo = pd.DataFrame(
            {"inferior": np.asarray(ic)[:, 0], "superior": np.asarray(ic)[:, 1]},
            index=idx_fut,
        )
        # El consumo no puede ser negativo: se trunca en 0
        return pd.Series(np.clip(np.asarray(fc.predicted_mean, dtype=float), 0, None))

    # -- Diagnóstico ---------------------------------------------------------
    def resumen(self) -> str:
        """Resumen de statsmodels (coeficientes, significatividad, Ljung-Box)."""
        self._check_ajustado()
        return str(self.resultado.summary())


# ---------------------------------------------------------------------------
# Selección de órdenes por AIC
# ---------------------------------------------------------------------------
def seleccionar_orden(
    y: pd.Series,
    X: pd.DataFrame | None = None,
    ordenes: tuple[tuple[int, int, int], ...] = ((1, 0, 0), (2, 0, 1), (1, 1, 1), (2, 1, 2)),
    usar_fourier: bool = True,
    exogenas: tuple[str, ...] | None = EXOGENAS_DEFAULT,
    max_train_horas: int = 24 * 60,
) -> pd.DataFrame:
    """Búsqueda acotada del orden ARIMA comparando AIC/BIC.

    Alternativa reproducible y barata a un `auto_arima` completo: se prueban
    unos pocos órdenes plausibles sobre una ventana corta y se documenta la
    tabla resultante en la memoria.
    """
    filas = []
    for order in ordenes:
        modelo = SarimaxForecaster(
            order=order,
            usar_fourier=usar_fourier,
            exogenas=exogenas,
            max_train_horas=max_train_horas,
        )
        try:
            modelo.fit(y, X)
            filas.append(
                {
                    "order": str(order),
                    "AIC": float(modelo.resultado.aic),
                    "BIC": float(modelo.resultado.bic),
                    "converge": bool(modelo.resultado.mle_retvals.get("converged", True)),
                }
            )
        except Exception as exc:  # noqa: BLE001 - se documenta el fallo y se sigue
            logger.warning(f"Orden {order} descartado: {exc}")
            filas.append({"order": str(order), "AIC": np.nan, "BIC": np.nan, "converge": False})

    return pd.DataFrame(filas).sort_values("AIC").reset_index(drop=True)
