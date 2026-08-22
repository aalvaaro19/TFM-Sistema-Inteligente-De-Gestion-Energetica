from __future__ import annotations

import numpy as np
import pandas as pd

from tfm_energia.models.base import BaseForecaster


class NaivePersistente(BaseForecaster):
    """Predicción constante igual al último valor observado."""

    nombre = "naive_persistente"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        self._ultimo = float(y.iloc[-1])

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        return pd.Series(np.full(steps, self._ultimo))


class NaiveEstacional(BaseForecaster):
    """Naïve estacional: ``y_hat(t) = y(t - m)``.

    Con `m=24` reproduce el perfil horario del día anterior. Si el horizonte
    supera el periodo, el patrón se repite cíclicamente (comportamiento
    estándar del *seasonal naive* de Hyndman).
    """

    def __init__(self, m: int = 24) -> None:
        super().__init__()
        self.m = m
        self.nombre = f"naive_estacional_{m}h"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        if len(y) < self.m:
            raise ValueError(f"Se necesitan al menos {self.m} observaciones (hay {len(y)}).")
        self._patron = y.iloc[-self.m :].to_numpy(dtype=float)

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        repeticiones = int(np.ceil(steps / self.m))
        return pd.Series(np.tile(self._patron, repeticiones)[:steps])


class NaiveEstacionalSemanal(NaiveEstacional):
    """Atajo de :class:`NaiveEstacional` con periodo semanal (168 h)."""

    def __init__(self) -> None:
        super().__init__(m=168)
        self.nombre = "naive_estacional_168h"


class MediaPerfilSemanal(BaseForecaster):
    """Media histórica del target por (día de la semana, hora).

    Filtra el ruido del naïve semanal promediando todas las semanas del
    histórico, a costa de perder la tendencia reciente.
    """

    nombre = "media_perfil_semanal"

    def __init__(self, ultimas_semanas: int | None = None) -> None:
        super().__init__()
        self.ultimas_semanas = ultimas_semanas

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        if self.ultimas_semanas is not None:
            y = y.iloc[-self.ultimas_semanas * 168 :]
        perfil = y.groupby([y.index.dayofweek, y.index.hour]).mean()
        perfil.index.names = ["dia_semana", "hora"]
        self._perfil = perfil
        self._media_global = float(y.mean())

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        idx = self.indice_futuro(steps)
        claves = pd.MultiIndex.from_arrays([idx.dayofweek, idx.hour])
        valores = self._perfil.reindex(claves).to_numpy(dtype=float)
        return pd.Series(np.where(np.isnan(valores), self._media_global, valores))


class MediaMovil(BaseForecaster):
    """Media de las últimas `ventana` horas, constante en todo el horizonte."""

    def __init__(self, ventana: int = 24) -> None:
        super().__init__()
        self.ventana = ventana
        self.nombre = f"media_movil_{ventana}h"

    def _fit(self, y: pd.Series, X: pd.DataFrame | None) -> None:
        self._media = float(y.iloc[-self.ventana :].mean())

    def _predict(self, steps: int, X: pd.DataFrame | None) -> pd.Series:
        return pd.Series(np.full(steps, self._media))


# ---------------------------------------------------------------------------
# Conjunto estándar de baselines del TFM
# ---------------------------------------------------------------------------
def baselines_estandar() -> list[BaseForecaster]:
    """Devuelve los baselines que se comparan en la memoria técnica."""
    return [
        NaivePersistente(),
        MediaMovil(ventana=24),
        NaiveEstacional(m=24),
        NaiveEstacionalSemanal(),
        MediaPerfilSemanal(),
    ]
