from __future__ import annotations

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.models.base import BaseForecaster


# Lags seguros para un horizonte de 48 h (todos ≥ 48)
LAGS_SEGUROS_DEFAULT = (48, 72, 96, 168, 336)
ROLLING_SEGUROS_DEFAULT = (24, 168)

EXOGENAS_ML_DEFAULT = (
    "temperatura_exterior_c",
    "humedad_exterior_pct",
    "radiacion_solar_rel",
    "ocupacion_rel",
)


def _features_calendario(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Variables de calendario con codificación cíclica."""
    return pd.DataFrame(
        {
            "hora": idx.hour,
            "dia_semana": idx.dayofweek,
            "mes": idx.month,
            "dia_mes": idx.day,
            "es_finde": (idx.dayofweek >= 5).astype(int),
            "hora_sin": np.sin(2 * np.pi * idx.hour / 24),
            "hora_cos": np.cos(2 * np.pi * idx.hour / 24),
            "dia_semana_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
            "dia_semana_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
            "mes_sin": np.sin(2 * np.pi * idx.month / 12),
            "mes_cos": np.cos(2 * np.pi * idx.month / 12),
        },
        index=idx,
    )


class GradientBoostingForecaster(BaseForecaster):
    """Gradient boosting sobre features conocidas a 48 h vista."""

    def __init__(
        self,
        horizonte: int = 48,
        lags: tuple[int, ...] = LAGS_SEGUROS_DEFAULT,
        rolling: tuple[int, ...] = ROLLING_SEGUROS_DEFAULT,
        exogenas: tuple[str, ...] = EXOGENAS_ML_DEFAULT,
        max_train_horas: int | None = None,
        max_iter: int = 300,
        learning_rate: float = 0.06,
        max_depth: int | None = 8,
        random_state: int = 42,
        nombre: str = "gradient_boosting",
    ) -> None:
        super().__init__()
        if any(lag < horizonte for lag in lags):
            raise ValueError(
                f"Todos los lags deben ser >= horizonte ({horizonte}) para no usar "
                f"información no disponible en el momento de predecir. Recibidos: {lags}"
            )
        self.horizonte = horizonte
        self.lags = lags
        self.rolling = rolling
        self.exogenas = tuple(exogenas)
        self.max_train_horas = max_train_horas
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_state = random_state
        self.nombre = nombre
        self.modelo = None

    # -- Construcción de features -------------------------------------------
    def _construir_features(
        self, idx: pd.DatetimeIndex, historico: pd.Series, X: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Matriz de features para `idx`, usando `historico` solo hasta su final.

        `historico` contiene los valores observados del target. Los lags se
        toman de esa serie: para timestamps futuros el desfase mínimo garantiza
        que el valor exista y sea observable.
        """
        feats = _features_calendario(idx)

        for lag in self.lags:
            objetivo = idx - pd.Timedelta(hours=lag)
            feats[f"lag_{lag}h"] = historico.reindex(objetivo).to_numpy()

        # Medias móviles sobre el histórico desplazado el horizonte completo
        base = historico.shift(self.horizonte)
        fuera_de_muestra = idx[0] > historico.index[-1]
        for w in self.rolling:
            media = base.rolling(window=w, min_periods=max(2, w // 4)).mean()
            desv = base.rolling(window=w, min_periods=max(2, w // 4)).std()
            if fuera_de_muestra:
                # En el futuro no hay ventana que calcular: se congela el último
                # agregado observado, que es la información realmente disponible.
                feats[f"roll_mean_{w}h"] = float(media.dropna().iloc[-1])
                feats[f"roll_std_{w}h"] = float(desv.dropna().iloc[-1])
            else:
                feats[f"roll_mean_{w}h"] = media.reindex(idx).to_numpy()
                feats[f"roll_std_{w}h"] = desv.reindex(idx).to_numpy()

        if self.exogenas:
            if X is None:
                raise ValueError(f"{self.nombre} requiere exógenas {self.exogenas}.")
            faltan = [c for c in self.exogenas if c not in X.columns]
            if faltan:
                raise ValueError(f"Faltan exógenas en X: {faltan}")
            ex = X.loc[:, list(self.exogenas)].astype(float).ffill().bfill()
            ex.index = idx
            feats = pd.concat([feats, ex], axis=1)

        return feats

    # -- Interfaz BaseForecaster ---------------------------------------------
    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        from sklearn.ensemble import HistGradientBoostingRegressor

        y_fit, X_fit = y, X
        if self.max_train_horas is not None and len(y) > self.max_train_horas:
            y_fit = y.iloc[-self.max_train_horas :]
            X_fit = X.iloc[-self.max_train_horas :] if X is not None else None

        feats = self._construir_features(y_fit.index, y, X_fit)
        valido = feats.notna().all(axis=1)
        n_descartadas = int((~valido).sum())

        self._cols = list(feats.columns)
        self.modelo = HistGradientBoostingRegressor(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_state=self.random_state,
            early_stopping=False,
        )
        self.modelo.fit(feats.loc[valido], y_fit.loc[valido])
        logger.info(
            f"{self.nombre} ajustado sobre {int(valido.sum()):,} h "
            f"({n_descartadas} descartadas por lags incompletos), "
            f"{len(self._cols)} features"
        )

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        if steps > self.horizonte:
            raise ValueError(
                f"{self.nombre} está construido para un horizonte máximo de "
                f"{self.horizonte} h; se han pedido {steps}."
            )
        idx_fut = self.indice_futuro(steps)
        feats = self._construir_features(idx_fut, self._y, X).loc[:, self._cols]
        pred = self.modelo.predict(feats.astype(float))
        return pd.Series(np.clip(pred, 0, None))

    # -- Interpretabilidad ---------------------------------------------------
    def importancia_permutacion(
        self, y: pd.Series, X: pd.DataFrame | None, n_repeats: int = 5
    ) -> pd.DataFrame:
        """Importancia por permutación sobre un tramo de validación.

        Es el análisis que justifica en la memoria qué variables aportan de
        verdad (temperatura real de AEMET, ocupación, memoria de la serie…).
        """
        from sklearn.inspection import permutation_importance

        self._check_ajustado()
        feats = self._construir_features(y.index, y, X).loc[:, self._cols]
        valido = feats.notna().all(axis=1)

        res = permutation_importance(
            self.modelo,
            feats.loc[valido],
            y.loc[valido],
            n_repeats=n_repeats,
            random_state=self.random_state,
            scoring="neg_mean_absolute_error",
        )
        return (
            pd.DataFrame(
                {
                    "feature": self._cols,
                    "importancia": res.importances_mean,
                    "std": res.importances_std,
                }
            )
            .sort_values("importancia", ascending=False)
            .reset_index(drop=True)
        )
