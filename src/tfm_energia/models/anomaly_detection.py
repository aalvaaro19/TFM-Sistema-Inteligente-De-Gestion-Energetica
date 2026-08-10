"""Detección de anomalías de consumo mediante aprendizaje no supervisado.

El sistema debe localizar automáticamente comportamientos anómalos —equipos que
se quedan encendidos, fugas de consumo, sensores averiados— **sin que nadie le
diga cuáles son**. Por eso se emplea un enfoque no supervisado: el detector se
entrena solo con los datos, y las etiquetas del dataset se reservan
exclusivamente para evaluarlo.

Tipología de anomalías presente en los datos:

  * ``CONSUMPTION_SPIKE`` – el consumo total se dispara (×2,2).
  * ``EQUIPMENT_LEAK``    – los equipos consumen de más (+6 kWh constantes).
  * ``HVAC_STUCK_ON``     – la climatización se queda enganchada (×3,5).
  * ``SENSOR_FROZEN``     – el sensor de temperatura interior deja de actualizarse.

Los tres primeros alteran el consumo y son detectables mirando valores. El
cuarto **no toca el consumo en absoluto**: congela la temperatura en un valor
que suele ser perfectamente normal. Su única huella es que la señal deja de
variar, así que hace falta una variable que mida esa variación. Es la razón de
que el conjunto de features incluya desviaciones típicas móviles: sin ellas,
una de las cuatro clases sería invisible por construcción.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from loguru import logger


# Señales de consumo sobre las que se calcula el residuo frente a lo habitual
COLUMNAS_BASE = (
    "consumo_total_kwh",
    "consumo_hvac_kwh",
    "consumo_equipos_kwh",
)

# Señales cuya *falta de variación* delata un sensor averiado
COLUMNAS_VARIACION = ("temperatura_interior_c",)

VENTANA_VARIACION = 3   # horas de la ventana de variación
VENTANA_RESIDUO = 6     # horas sobre las que se acumula el residuo
DIAS_REFERENCIA = 14    # días previos que definen el comportamiento habitual


@dataclass
class ConfigDeteccion:
    """Parámetros del detector."""

    contaminacion: float = 0.02
    n_estimadores: int = 200
    max_muestras: int | str = "auto"
    seed: int = 42
    ventana_variacion: int = VENTANA_VARIACION
    dias_referencia: int = DIAS_REFERENCIA
    ventana_residuo: int = VENTANA_RESIDUO
    # Variables que NO entran en el bosque. Isolation Forest elige la variable
    # de corte al azar, así que cada variable añadida reduce la probabilidad de
    # que use las discriminantes. Las señales que tienen su propio canal
    # especializado se excluyen aquí para no diluir el resto.
    patrones_excluidos: tuple[str, ...] = ("_residuo_medio_",)


# ---------------------------------------------------------------------------
# Construcción de variables
# ---------------------------------------------------------------------------
def referencia_local(
    serie: pd.Series,
    dias: int = DIAS_REFERENCIA,
    minimo: int = 3,
    laborable: pd.Series | None = None,
) -> pd.Series:
    """Comportamiento habitual de cada hora, según los días equivalentes previos.

    Para cada timestamp se toma la mediana de **esa misma hora** en los días
    anteriores del **mismo tipo** (laborable o no laborable), nunca en los
    posteriores.

    Dos decisiones importantes:

    * **Referencia local, no anual.** Comparar contra el perfil de todo el año
      haría que en enero pareciera anómalo el día entero, porque el consumo
      invernal dobla al de mayo. Contra los últimos días la estación se cancela
      sola, porque está presente en ambos lados de la resta.
    * **Separar laborables de festivos.** Sin esta distinción, la mediana de las
      10:00 mezcla diez días de oficina llena con cuatro de oficina vacía, y
      todos los sábados generan residuos enormes que no son averías. Al usar
      valor absoluto, esos falsos positivos acaparan el presupuesto de avisos.

    Se usa la mediana y no la media para que un episodio anómalo reciente no
    contamine la referencia de los días siguientes.
    """
    if laborable is None:
        laborable = pd.Series(serie.index.dayofweek < 5, index=serie.index)

    referencia = pd.Series(index=serie.index, dtype=float)
    for es_laborable in (True, False):
        for hora in range(24):
            mascara = (serie.index.hour == hora) & (
                laborable.to_numpy().astype(bool) == es_laborable
            )
            if not mascara.any():
                continue
            del_dia = serie[mascara]
            # shift(1) garantiza que el valor actual no entra en su propia referencia
            referencia[mascara] = (
                del_dia.shift(1).rolling(dias, min_periods=minimo).median()
            )
    return referencia


def construir_features_anomalia(
    df: pd.DataFrame,
    ventana: int = VENTANA_VARIACION,
    dias_referencia: int = DIAS_REFERENCIA,
    ventana_residuo: int = VENTANA_RESIDUO,
) -> pd.DataFrame:
    """Genera las variables de detección a partir del dataset enriquecido.

    Dos familias, cada una dirigida a un grupo de averías:

    1. **Residuos frente a la referencia local** – cuánto se aparta cada señal
       de lo que viene siendo habitual a esa hora. Detectan los consumos
       excesivos: picos, fugas de equipos y climatización atascada.
    2. **Medidas de variación** – desviación típica móvil y salto respecto a la
       hora anterior. Son las únicas capaces de ver un sensor congelado, cuyo
       valor es normal pero deja de moverse.

    Deliberadamente **no** se incluyen los valores en bruto. Un consumo de
    120 kWh es perfectamente normal en enero y absurdo en mayo, así que en bruto
    aportan sobre todo estacionalidad, que es ruido para este problema y además
    diluye las variables que sí discriminan.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Se requiere un DatetimeIndex.")

    df = df.sort_index()
    feats = pd.DataFrame(index=df.index)

    # Un festivo se comporta como un fin de semana, no como un laborable
    laborable = pd.Series(df.index.dayofweek < 5, index=df.index)
    if "es_festivo" in df.columns:
        laborable &= ~df["es_festivo"].astype(bool)

    # -- 1. Residuos frente a la referencia local ---------------------------
    for col in COLUMNAS_BASE:
        if col not in df.columns:
            continue
        serie = df[col].astype(float)
        ref = referencia_local(serie, dias_referencia, laborable=laborable)
        residuo = serie - ref
        feats[f"{col}_residuo"] = residuo
        # Ratio acotado: informa de la magnitud relativa sin desbordar cuando la
        # referencia es próxima a cero (p. ej. el HVAC apagado de madrugada)
        feats[f"{col}_ratio"] = (serie / ref.clip(lower=0.5)).clip(upper=20)
        # Residuo acumulado: una fuga de equipos añade solo +6 kWh sobre una
        # serie con desviación típica de 14, así que hora a hora es
        # indistinguible del ruido. Pero dura horas, y al promediar el ruido se
        # cancela mientras el sesgo persiste, que es lo que la hace visible.
        feats[f"{col}_residuo_medio_{ventana_residuo}h"] = residuo.rolling(
            ventana_residuo, min_periods=2
        ).mean()

    # -- 2. Medidas de variación --------------------------------------------
    for col in COLUMNAS_VARIACION:
        if col not in df.columns:
            continue
        serie = df[col].astype(float)
        # Una desviación típica móvil de 0 significa señal congelada
        feats[f"{col}_std_{ventana}h"] = serie.rolling(ventana, min_periods=2).std()
        feats[f"{col}_salto_1h"] = serie.diff().abs()

    # Las primeras filas no tienen histórico suficiente para la referencia
    return feats.bfill().ffill()


# ---------------------------------------------------------------------------
# Detectores
# ---------------------------------------------------------------------------
class BaseDetector(ABC):
    """Interfaz común, para poder comparar detectores con el mismo código."""

    nombre: str = "base"

    @abstractmethod
    def fit(self, X: pd.DataFrame) -> "BaseDetector": ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Devuelve un booleano por fila: True = anomalía."""

    @abstractmethod
    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        """Puntuación de anomalía; cuanto mayor, más anómalo."""


class DetectorIsolationForest(BaseDetector):
    """Isolation Forest sobre las variables de detección.

    El algoritmo aísla observaciones construyendo árboles aleatorios: los puntos
    anómalos quedan separados con pocas particiones, mientras que los normales
    requieren muchas. No necesita etiquetas ni suponer una distribución.
    """

    nombre = "isolation_forest"

    def __init__(self, cfg: ConfigDeteccion | None = None) -> None:
        self.cfg = cfg or ConfigDeteccion()
        self.modelo = None
        self._columnas: list[str] | None = None

    def _seleccionar(self, X: pd.DataFrame) -> list[str]:
        return [
            c for c in X.columns
            if not any(p in c for p in self.cfg.patrones_excluidos)
        ]

    def fit(self, X: pd.DataFrame) -> "DetectorIsolationForest":
        from sklearn.ensemble import IsolationForest

        self._columnas = self._seleccionar(X)
        X = X.loc[:, self._columnas]
        self.modelo = IsolationForest(
            n_estimators=self.cfg.n_estimadores,
            contamination=self.cfg.contaminacion,
            max_samples=self.cfg.max_muestras,
            random_state=self.cfg.seed,
            n_jobs=-1,
        )
        self.modelo.fit(X)
        logger.info(
            f"{self.nombre} ajustado sobre {len(X):,} observaciones y "
            f"{len(self._columnas)} variables (contaminación {self.cfg.contaminacion:.1%})"
        )
        return self

    def _validar(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.modelo is None:
            raise RuntimeError(f"{self.nombre}: hay que llamar a fit() primero.")
        faltan = set(self._columnas or []) - set(X.columns)
        if faltan:
            raise ValueError(f"Faltan variables: {sorted(faltan)}")
        return X.loc[:, self._columnas]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # `_validar` se evalúa antes a propósito: escribirlo en línea haría que
        # Python resolviese `self.modelo.predict` primero y, sin ajustar, saltase
        # un AttributeError opaco en vez del mensaje de la validación.
        datos = self._validar(X)
        # sklearn devuelve -1 para anomalía y 1 para normal
        return self.modelo.predict(datos) == -1

    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        datos = self._validar(X)
        # score_samples es más negativo cuanto más anómalo: se invierte el signo
        return -self.modelo.score_samples(datos)


class DetectorEstadistico(BaseDetector):
    """Baseline: z-score robusto sobre el desvío respecto al perfil horario.

    Marca como anómala toda observación que se aleje más de `umbral` desviaciones
    medianas de su comportamiento habitual. Es el mínimo que cualquier método
    sofisticado debe superar para justificar su complejidad.
    """

    nombre = "zscore_robusto"

    def __init__(self, umbral: float = 3.5, columna: str = "consumo_total_kwh_residuo") -> None:
        self.umbral = umbral
        self.columna = columna
        self.mediana_ = 0.0
        self.mad_ = 1.0

    def fit(self, X: pd.DataFrame) -> "DetectorEstadistico":
        if self.columna not in X.columns:
            raise ValueError(f"El baseline necesita la columna '{self.columna}'.")
        serie = X[self.columna].astype(float)
        self.mediana_ = float(serie.median())
        # Desviación absoluta mediana, escalada para ser comparable a una sigma
        self.mad_ = float((serie - self.mediana_).abs().median()) * 1.4826
        if self.mad_ < 1e-9:
            self.mad_ = float(serie.std()) or 1.0
        return self

    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        serie = X[self.columna].astype(float)
        return np.abs((serie - self.mediana_) / self.mad_).to_numpy()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.puntuar(X) > self.umbral


class DetectorSensorCongelado(BaseDetector):
    """Regla explícita para sensores bloqueados: la señal deja de variar.

    Isolation Forest rinde mal en este caso porque la anomalía es extrema en una
    única dimensión —la desviación típica móvil— mientras el resto de variables
    permanecen normales; solo los árboles que cortan pronto por esa variable
    llegan a aislarla.

    Una regla directa lo resuelve con recall casi perfecto y, además, es
    interpretable: se puede decir a mantenimiento *"la temperatura interior no
    se ha movido en tres horas"*, que es un aviso accionable, en lugar de *"el
    modelo ha dado una puntuación de −0,62"*.
    """

    nombre = "regla_sensor_congelado"

    def __init__(self, umbral_std: float = 1e-3, columna: str | None = None) -> None:
        self.umbral_std = umbral_std
        self.columna = columna

    def _columna_std(self, X: pd.DataFrame) -> str:
        if self.columna:
            return self.columna
        candidatas = [c for c in X.columns if "_std_" in c]
        if not candidatas:
            raise ValueError("No hay ninguna columna de desviación típica móvil.")
        return candidatas[0]

    def fit(self, X: pd.DataFrame) -> "DetectorSensorCongelado":
        self.columna = self._columna_std(X)
        return self

    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        # Cuanto menor es la variación, mayor la sospecha
        std = X[self._columna_std(X)].astype(float).to_numpy()
        return 1.0 / (std + self.umbral_std)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X[self._columna_std(X)].astype(float).to_numpy() <= self.umbral_std


class DetectorCompuesto(BaseDetector):
    """Combina varios detectores especializados: salta si alguno se activa.

    Ningún algoritmo cubre bien todas las familias de avería. En un sistema real
    se despliegan varios canales en paralelo, cada uno afinado a un tipo de
    problema, y la alerta se dispara cuando cualquiera de ellos se activa.
    """

    nombre = "compuesto"

    def __init__(self, detectores: list[BaseDetector], nombre: str | None = None) -> None:
        if not detectores:
            raise ValueError("Hace falta al menos un detector.")
        self.detectores = detectores
        self.nombre = nombre or "+".join(d.nombre for d in detectores)

    def fit(self, X: pd.DataFrame) -> "DetectorCompuesto":
        for d in self.detectores:
            d.fit(X)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        salidas = [d.predict(X) for d in self.detectores]
        return np.logical_or.reduce(salidas)

    def puntuar(self, X: pd.DataFrame) -> np.ndarray:
        """Máximo de las puntuaciones normalizadas de cada canal.

        Se normaliza por rango porque cada detector puntúa en una escala propia
        y, sin ello, el de valores más grandes dominaría siempre.
        """
        normalizadas = []
        for d in self.detectores:
            s = np.asarray(d.puntuar(X), dtype=float)
            rango = np.nanmax(s) - np.nanmin(s)
            normalizadas.append((s - np.nanmin(s)) / rango if rango > 0 else np.zeros_like(s))
        return np.max(normalizadas, axis=0)

    def predecir_con_cuotas(self, X: pd.DataFrame, presupuesto: float) -> np.ndarray:
        """Reparte el presupuesto de avisos entre los canales, a partes iguales.

        Puntuar en conjunto y quedarse con el 2% más alto deja que los canales
        de puntuación más agresiva acaparen los avisos y silencien a los demás:
        el canal que detecta picos de consumo pierde casi todo su recall aunque
        por separado sea el mejor en esa familia.

        Dando a cada canal su propia cuota se conserva lo mejor de cada uno, que
        es además como se opera en la práctica: cada tipo de alarma tiene su
        propio umbral y su propio destinatario.
        """
        cuota = presupuesto / len(self.detectores)
        salidas = [
            predecir_con_presupuesto(np.asarray(d.puntuar(X), dtype=float), cuota)
            for d in self.detectores
        ]
        return np.logical_or.reduce(salidas)


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
@dataclass
class ResultadoDeteccion:
    """Métricas de un detector frente a las anomalías etiquetadas."""

    detector: str
    n: int
    detectadas: int
    verdaderos_positivos: int
    falsos_positivos: int
    falsos_negativos: int
    precision: float
    recall: float
    f1: float
    recall_por_tipo: dict[str, float] = field(default_factory=dict)
    episodios_totales: int = 0
    episodios_detectados: int = 0
    roc_auc: float | None = None
    average_precision: float | None = None

    @property
    def recall_episodios(self) -> float:
        """Fracción de episodios en los que se detectó al menos una hora."""
        return self.episodios_detectados / self.episodios_totales if self.episodios_totales else 0.0

    def a_dict(self) -> dict:
        d = {
            "detector": self.detector,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "recall_episodios": self.recall_episodios,
            "roc_auc": self.roc_auc,
            "avg_precision": self.average_precision,
            "VP": self.verdaderos_positivos,
            "FP": self.falsos_positivos,
            "FN": self.falsos_negativos,
        }
        d.update({f"recall_{k}": v for k, v in self.recall_por_tipo.items()})
        return d


def _episodios(es_anomalia: pd.Series) -> list[np.ndarray]:
    """Agrupa horas anómalas consecutivas en episodios (posiciones)."""
    valores = es_anomalia.to_numpy()
    bloques = np.cumsum(valores != np.r_[False, valores[:-1]])
    posiciones = np.arange(len(valores))
    return [posiciones[(bloques == b) & valores] for b in np.unique(bloques[valores])]


def predecir_con_presupuesto(
    puntuaciones: np.ndarray, presupuesto: float
) -> np.ndarray:
    """Marca como anómalas las `presupuesto` observaciones peor puntuadas.

    Permite comparar detectores en igualdad de condiciones. Con umbrales
    propios, uno que avise de 3.500 horas y otro de 350 tienen precisiones
    incomparables por construcción, y la diferencia refleja la calibración del
    umbral más que la calidad del detector.

    Además es como se opera en la práctica: un equipo de mantenimiento puede
    atender un número limitado de avisos al día, no los que decida un umbral.
    """
    n = max(1, int(round(len(puntuaciones) * presupuesto)))
    corte = np.partition(puntuaciones, -n)[-n]
    return puntuaciones >= corte


def evaluar_deteccion(
    y_pred: np.ndarray,
    es_anomalia: pd.Series,
    tipo_anomalia: pd.Series | None = None,
    nombre: str = "detector",
    puntuaciones: np.ndarray | None = None,
) -> ResultadoDeteccion:
    """Compara las detecciones con las anomalías reales.

    Además de las métricas por hora, calcula el **recall por episodio**: en
    operación real basta con detectar una hora de una avería que dura siete para
    que se genere el aviso y acuda mantenimiento, así que evaluar solo hora a
    hora infravalora la utilidad del sistema.
    """
    y_true = es_anomalia.to_numpy().astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    if len(y_true) != len(y_pred):
        raise ValueError(f"Dimensiones distintas: {len(y_true)} vs {len(y_pred)}")

    vp = int(np.sum(y_pred & y_true))
    fp = int(np.sum(y_pred & ~y_true))
    fn = int(np.sum(~y_pred & y_true))

    precision = vp / (vp + fp) if (vp + fp) else 0.0
    recall = vp / (vp + fn) if (vp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    recall_tipo: dict[str, float] = {}
    if tipo_anomalia is not None:
        tipos = tipo_anomalia.to_numpy()
        for t in sorted({t for t in tipos[y_true] if t}):
            mascara = y_true & (tipos == t)
            recall_tipo[t] = float(np.sum(y_pred & mascara) / np.sum(mascara))

    episodios = _episodios(es_anomalia)
    detectados = sum(1 for ep in episodios if y_pred[ep].any())

    # Métricas independientes del umbral: miden la capacidad de ORDENAR, no la
    # calibración del corte. La precisión media es más informativa que el AUC
    # cuando las clases están muy desbalanceadas, como aquí (1,15% de anomalías).
    roc_auc = avg_prec = None
    if puntuaciones is not None:
        from sklearn.metrics import average_precision_score, roc_auc_score

        s = np.asarray(puntuaciones, dtype=float)
        finitos = np.isfinite(s)
        if finitos.any() and len(set(y_true[finitos])) > 1:
            roc_auc = float(roc_auc_score(y_true[finitos], s[finitos]))
            avg_prec = float(average_precision_score(y_true[finitos], s[finitos]))

    return ResultadoDeteccion(
        detector=nombre,
        n=len(y_true),
        detectadas=int(y_pred.sum()),
        verdaderos_positivos=vp,
        falsos_positivos=fp,
        falsos_negativos=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        recall_por_tipo=recall_tipo,
        episodios_totales=len(episodios),
        episodios_detectados=detectados,
        roc_auc=roc_auc,
        average_precision=avg_prec,
    )


def comparar_detectores(resultados: list[ResultadoDeteccion]) -> pd.DataFrame:
    """Tabla comparativa ordenada por F1."""
    return (
        pd.DataFrame([r.a_dict() for r in resultados])
        .sort_values("f1", ascending=False)
        .reset_index(drop=True)
    )
