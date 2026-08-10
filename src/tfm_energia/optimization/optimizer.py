"""Optimización del coste de climatización mediante programación lineal.

Formulación, para un horizonte de H horas:

    **Variables de decisión** (por hora h)
        cal[h] ≥ 0      energía eléctrica de calefacción (kWh)
        frio[h] ≥ 0     energía eléctrica de refrigeración (kWh)
        T[h]            temperatura interior al final de la hora (°C)
        vinf[h], vsup[h] ≥ 0   holgura de confort (°C), penalizada

    **Balance térmico** (ecuación de estado del edificio)
        T[h] = T[h-1] + k·(Text[h] − T[h-1]) + g·ocup[h] + s·rad[h]
               + η[h]·cal[h] − η[h]·frio[h]

    **Confort**
        T[h] ≥ Tmin[h] − vinf[h]
        T[h] ≤ Tmax[h] + vsup[h]

    **Potencia**
        cal[h] + frio[h] ≤ Emax[h]

    **Objetivo**
        min Σ precio[h]·(cal[h] + frio[h]) + M·Σ (vinf[h] + vsup[h])

Notas sobre el modelo:

* **Por qué es lineal.** El balance es lineal en `T` y en las energías, así que
  el problema es un LP y el óptimo encontrado es global. La única no linealidad
  del edificio real —el rendimiento depende de `|T_int − T_ext|`, y `T_int` es
  variable— se elude calculando `η` con la temperatura exterior prevista y la
  consigna. Es la aproximación habitual en control predictivo y debe declararse.

* **Por qué el confort es blando.** El control reactivo del edificio no siempre
  mantiene la banda: en la rampa matinal el equipo satura y la temperatura va por
  debajo durante horas. Exigir la banda de forma estricta haría el problema
  infactible en esas ventanas. Con holgura penalizada el problema siempre tiene
  solución, y la holgura empleada es **medible**, de modo que puede compararse
  con la que ya incurre el control actual y demostrar que el ahorro no procede de
  pasar frío.

* **Por qué no hacen falta variables binarias.** Calentar y enfriar a la vez
  sería absurdo y costoso, así que el óptimo nunca lo hace: no hace falta
  imponerlo con enteros y el problema se mantiene como LP puro, mucho más rápido.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pulp
from loguru import logger

from tfm_energia.optimization.thermal_model import (
    ParametrosTermicos,
    banda_confort,
    setpoint_estacional,
)


# Penalización por grado y hora de incumplimiento de confort. Debe ser muy
# superior al coste de corregirlo con energía (del orden de 4 €/°C) para que el
# optimizador solo recurra a la holgura cuando no exista alternativa factible.
PENALIZACION_CONFORT = 100.0


@dataclass
class ResultadoVentana:
    """Solución del LP para una ventana de optimización."""

    index: pd.DatetimeIndex
    calefaccion_kwh: np.ndarray
    refrigeracion_kwh: np.ndarray
    temperatura_c: np.ndarray
    violacion_inferior: np.ndarray
    violacion_superior: np.ndarray
    coste_eur: float
    estado: str

    @property
    def energia_kwh(self) -> np.ndarray:
        return self.calefaccion_kwh + self.refrigeracion_kwh

    @property
    def optima(self) -> bool:
        return self.estado == "Optimal"

    @property
    def grados_hora_incumplidos(self) -> float:
        """Suma de °C·h fuera de banda. Es la medida del confort sacrificado."""
        return float(self.violacion_inferior.sum() + self.violacion_superior.sum())


def energia_maxima(
    t_exterior: np.ndarray, objetivo: np.ndarray, p: ParametrosTermicos
) -> np.ndarray:
    """Energía máxima que el equipo puede consumir en cada hora.

    A plena modulación el consumo es `potencia_nominal · dificultad`, y la
    dificultad crece con el salto respecto al exterior. Cuando el edificio está
    cerca de la temperatura exterior el equipo consume menos a plena potencia,
    porque le cuesta menos mantener la consigna.
    """
    return p.potencia_nominal_kw * p.dificultad(objetivo, t_exterior)


def rendimiento_horario(
    t_exterior: np.ndarray, objetivo: np.ndarray, p: ParametrosTermicos
) -> np.ndarray:
    """Grados que aporta cada kWh en cada hora (la linealización declarada)."""
    return p.capacidad_hora / energia_maxima(t_exterior, objetivo, p)


def resolver_ventana(
    index: pd.DatetimeIndex,
    t_exterior: np.ndarray,
    ocupacion: np.ndarray,
    radiacion: np.ndarray,
    precio_eur_kwh: np.ndarray,
    en_servicio: np.ndarray,
    t_inicial: float,
    p: ParametrosTermicos,
    holgura_confort: float = 0.0,
    margen_preacondicionamiento: float = 3.0,
    penalizacion: float = PENALIZACION_CONFORT,
    temperatura_referencia: np.ndarray | None = None,
    iteraciones: int = 3,
    estado_terminal: tuple[int, float] | None = None,
) -> ResultadoVentana:
    """Resuelve el LP para una ventana de optimización.

    Args:
        temperatura_referencia: trayectoria de temperatura que logró el control
            con el que se compara. Si se aporta, los límites de confort se
            relajan hasta ella en las horas en que ese control **no** cumplía la
            banda, de modo que el optimizador queda obligado a ser *al menos tan
            confortable*, pero no más.

            Esto es imprescindible para que la comparación de costes sea legítima.
            El control reactivo del edificio no alcanza la consigna en las
            mañanas de invierno —llega a estar 6 °C por debajo durante horas— y
            resulta barato precisamente por eso. Exigirle al optimizador la banda
            completa mientras se compara con un control que la incumple mide la
            diferencia de servicio, no la de eficiencia.

        iteraciones: pasadas de **linealización sucesiva**. El rendimiento `η`
            depende de `|T_int − T_ext|`, y `T_int` es variable de decisión.
            Estimarlo una sola vez con la consigna introduce un sesgo grande y
            además pesimista: durante la rampa matinal el edificio está muy por
            debajo de la consigna, su dificultad térmica real es menor y el
            equipo rinde bastante más de lo que supondría esa estimación. Tras
            cada solución se recalcula `η` con la trayectoria obtenida y se
            vuelve a resolver, que es el procedimiento habitual para linealizar
            un problema de control con esta estructura.

        estado_terminal: par `(hora, temperatura_máxima)`. Acota la temperatura
            en ese instante del horizonte.

            Sirve para cerrar la miopía del horizonte deslizante. Sin esta cota,
            el optimizador calienta al final de su ventana para aprovechar el
            beneficio que sí ve —las primeras horas de la siguiente jornada—
            mientras las pérdidas de mantener el edificio caliente caen ya fuera
            de la ventana y no las paga. Externalizar ese coste hace que el plan
            parezca mejor de lo que resulta al ejecutarlo día tras día.
    """
    n = len(index)
    objetivo = setpoint_estacional(index, p)
    t_min, t_max = banda_confort(
        index, en_servicio, p, holgura_confort, margen_preacondicionamiento
    )

    if temperatura_referencia is not None:
        # Igualar el servicio: nunca peor que la referencia, tampoco mejor
        t_min = np.minimum(t_min, temperatura_referencia)
        t_max = np.maximum(t_max, temperatura_referencia)

    if estado_terminal is not None:
        hora, tope = estado_terminal
        if 0 <= hora < n:
            t_max = t_max.copy()
            t_max[hora] = min(t_max[hora], tope)
            t_min = np.minimum(t_min, t_max)

    # Primera estimación de la temperatura interior para linealizar
    t_estimada = np.clip(objetivo, t_min, np.where(np.isfinite(t_max), t_max, objetivo))

    resultado: ResultadoVentana | None = None
    for _ in range(max(1, iteraciones)):
        resultado = _resolver_lineal(
            index, t_exterior, ocupacion, radiacion, precio_eur_kwh,
            t_inicial, p, t_min, t_max, t_estimada, penalizacion,
        )
        if not resultado.optima:
            break
        # Se recalcula la linealización con la trayectoria obtenida
        nueva = resultado.temperatura_c
        if np.max(np.abs(nueva - t_estimada)) < 0.05:
            break
        t_estimada = nueva

    assert resultado is not None
    return resultado


def _resolver_lineal(
    index: pd.DatetimeIndex,
    t_exterior: np.ndarray,
    ocupacion: np.ndarray,
    radiacion: np.ndarray,
    precio_eur_kwh: np.ndarray,
    t_inicial: float,
    p: ParametrosTermicos,
    t_min: np.ndarray,
    t_max: np.ndarray,
    t_linealizacion: np.ndarray,
    penalizacion: float,
) -> ResultadoVentana:
    """Una pasada del LP con la linealización fijada."""
    n = len(index)
    e_max = energia_maxima(t_exterior, t_linealizacion, p)
    eta = rendimiento_horario(t_exterior, t_linealizacion, p)

    problema = pulp.LpProblem("coste_climatizacion", pulp.LpMinimize)

    cal = [pulp.LpVariable(f"cal_{h}", lowBound=0, upBound=e_max[h]) for h in range(n)]
    frio = [pulp.LpVariable(f"frio_{h}", lowBound=0, upBound=e_max[h]) for h in range(n)]
    temp = [pulp.LpVariable(f"T_{h}") for h in range(n)]
    vinf = [pulp.LpVariable(f"vinf_{h}", lowBound=0) for h in range(n)]
    vsup = [pulp.LpVariable(f"vsup_{h}", lowBound=0) for h in range(n)]

    # Objetivo: coste energético más la penalización por incumplir confort
    problema += (
        pulp.lpSum(precio_eur_kwh[h] * (cal[h] + frio[h]) for h in range(n))
        + penalizacion * pulp.lpSum(vinf[h] + vsup[h] for h in range(n))
    )

    for h in range(n):
        previa = t_inicial if h == 0 else temp[h - 1]
        deriva = (
            p.k_envoltura * (t_exterior[h] - previa)
            + p.ganancia_ocupacion * ocupacion[h]
            + p.ganancia_solar * radiacion[h]
        )
        problema += temp[h] == previa + deriva + eta[h] * cal[h] - eta[h] * frio[h]

        problema += temp[h] >= t_min[h] - vinf[h]
        problema += temp[h] <= t_max[h] + vsup[h]
        problema += cal[h] + frio[h] <= e_max[h]

    problema.solve(pulp.PULP_CBC_CMD(msg=False))
    estado = pulp.LpStatus[problema.status]

    def valores(variables) -> np.ndarray:
        return np.array([v.value() if v.value() is not None else 0.0 for v in variables])

    calefaccion = valores(cal)
    refrigeracion = valores(frio)

    return ResultadoVentana(
        index=index,
        calefaccion_kwh=calefaccion,
        refrigeracion_kwh=refrigeracion,
        temperatura_c=valores(temp),
        violacion_inferior=valores(vinf),
        violacion_superior=valores(vsup),
        coste_eur=float(np.sum(precio_eur_kwh * (calefaccion + refrigeracion))),
        estado=estado,
    )


# ---------------------------------------------------------------------------
# Control predictivo sobre un periodo largo
# ---------------------------------------------------------------------------
COLUMNAS_REQUERIDAS = frozenset({
    "temperatura_exterior_c", "ocupacion_rel", "radiacion_solar_rel",
    "precio_eur_kwh", "consumo_hvac_kwh", "temperatura_interior_c", "es_festivo",
})


def horario_de_servicio(df: pd.DataFrame) -> np.ndarray:
    """Horas en las que el edificio está ocupado, igual que en el simulador."""
    laborable = (df.index.dayofweek < 5) & ~df["es_festivo"].to_numpy().astype(bool)
    return laborable & df.index.hour.isin(range(7, 20))


def grados_hora_fuera_de_banda(
    temperatura: np.ndarray, t_min: np.ndarray, t_max: np.ndarray
) -> np.ndarray:
    """°C·h por los que una trayectoria se sale de la banda de confort.

    Se aplica **la misma banda estricta a todas las estrategias**. Medir una
    contra una banda relajada y otra contra la estricta produce comparaciones sin
    sentido, como que la estrategia relajada aparezca con cero incumplimientos
    por construcción.
    """
    return np.maximum(t_min - temperatura, 0) + np.maximum(temperatura - t_max, 0)


def simular_control(
    df: pd.DataFrame,
    p: ParametrosTermicos,
    precios_decision: np.ndarray | None = None,
    horizonte: int = 48,
    paso: int = 24,
    holgura_confort: float = 0.0,
    margen_preacondicionamiento: float = 3.0,
    max_ventanas: int | None = None,
) -> pd.DataFrame:
    """Ejecuta control predictivo por horizonte deslizante sobre un periodo.

    Cada `paso` horas se optimizan las siguientes `horizonte`, pero solo se
    ejecutan las primeras `paso`; después se vuelve a optimizar con información
    actualizada. Evita el sesgo optimista de resolver el año entero conociendo
    el futuro completo.

    Args:
        precios_decision: precios con los que el optimizador **toma decisiones**.
            Si es None se usan los reales. Pasar un vector constante produce el
            control ciego al precio, que minimiza energía y sirve de referencia
            limpia: misma física y mismo confort, sin señal económica.

    Returns:
        DataFrame horario con la energía, la temperatura y el coste evaluado
        siempre a **precios reales**, sea cual sea el precio de decisión.
    """
    df = df.sort_index()
    faltan = COLUMNAS_REQUERIDAS - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en el DataFrame: {sorted(faltan)}")

    en_servicio = horario_de_servicio(df)
    t_ext = df["temperatura_exterior_c"].to_numpy(dtype=float)
    ocup = df["ocupacion_rel"].to_numpy(dtype=float)
    rad = df["radiacion_solar_rel"].to_numpy(dtype=float)
    precio_real = df["precio_eur_kwh"].to_numpy(dtype=float)
    precio_dec = precio_real if precios_decision is None else np.asarray(
        precios_decision, dtype=float
    )

    t_min, t_max = banda_confort(
        df.index, en_servicio, p, holgura_confort, margen_preacondicionamiento
    )

    filas = []
    t_actual = float(df["temperatura_interior_c"].iloc[0])
    ventanas = no_optimas = 0

    inicio = 0
    while inicio + paso <= len(df):
        fin = min(inicio + horizonte, len(df))
        sl = slice(inicio, fin)

        res = resolver_ventana(
            df.index[sl], t_ext[sl], ocup[sl], rad[sl], precio_dec[sl],
            en_servicio[sl], t_actual, p, holgura_confort, margen_preacondicionamiento,
        )
        ventanas += 1
        no_optimas += 0 if res.optima else 1

        ejecutadas = min(paso, fin - inicio)
        idx_ej = slice(inicio, inicio + ejecutadas)
        energia = res.energia_kwh[:ejecutadas]
        temperatura = res.temperatura_c[:ejecutadas]

        filas.append(pd.DataFrame({
            "energia_kwh": energia,
            "coste_eur": energia * precio_real[idx_ej],
            "temperatura_c": temperatura,
            "grados_hora": grados_hora_fuera_de_banda(
                temperatura, t_min[idx_ej], t_max[idx_ej]
            ),
            "precio_eur_kwh": precio_real[idx_ej],
            "en_servicio": en_servicio[idx_ej],
        }, index=df.index[idx_ej]))

        t_actual = float(temperatura[-1])
        inicio += paso
        if max_ventanas is not None and ventanas >= max_ventanas:
            break

    if no_optimas:
        logger.warning(f"{no_optimas} de {ventanas} ventanas no alcanzaron el óptimo")
    return pd.concat(filas)


def serie_control_reactivo(df: pd.DataFrame, p: ParametrosTermicos) -> pd.DataFrame:
    """Recoge el comportamiento del control reactivo ya registrado en los datos."""
    en_servicio = horario_de_servicio(df)
    t_min, t_max = banda_confort(df.index, en_servicio, p)
    temperatura = df["temperatura_interior_c"].to_numpy(dtype=float)
    energia = df["consumo_hvac_kwh"].to_numpy(dtype=float)

    return pd.DataFrame({
        "energia_kwh": energia,
        "coste_eur": energia * df["precio_eur_kwh"].to_numpy(dtype=float),
        "temperatura_c": temperatura,
        "grados_hora": grados_hora_fuera_de_banda(temperatura, t_min, t_max),
        "precio_eur_kwh": df["precio_eur_kwh"].to_numpy(dtype=float),
        "en_servicio": en_servicio,
    }, index=df.index)


@dataclass
class ResultadoComparativa:
    """Comparación de las tres estrategias de control sobre el mismo periodo."""

    estrategias: dict[str, pd.DataFrame] = field(default_factory=dict)
    referencia: str = "predictivo_ciego"

    def tabla(self) -> pd.DataFrame:
        filas = []
        for nombre, d in self.estrategias.items():
            filas.append({
                "estrategia": nombre,
                "energia_kwh": d["energia_kwh"].sum(),
                "coste_eur": d["coste_eur"].sum(),
                "precio_medio": (
                    d["coste_eur"].sum() / d["energia_kwh"].sum()
                    if d["energia_kwh"].sum() else np.nan
                ),
                "grados_hora": d["grados_hora"].sum(),
            })
        tabla = pd.DataFrame(filas).set_index("estrategia")

        base = tabla.loc[self.referencia, "coste_eur"]
        tabla["ahorro_eur"] = base - tabla["coste_eur"]
        tabla["ahorro_pct"] = 100 * tabla["ahorro_eur"] / base if base else np.nan
        return tabla
