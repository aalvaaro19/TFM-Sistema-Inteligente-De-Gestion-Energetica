"""Modelo térmico lineal del edificio, base física del optimizador.

El optimizador no puede limitarse a mover kWh de una hora a otra: el consumo de
climatización responde a mantener el edificio en confort, no es un caudal que se
trasvase libremente. Lo que sí permite el desplazamiento de carga es usar la
**masa del edificio como batería térmica**: calentar de más en horas baratas
deja energía almacenada que reduce la necesidad en horas caras.

Este módulo formaliza esa física en una ecuación de estado lineal:

    T[h] = T[h-1] + k·(Text[h] − T[h-1]) + g·ocup[h] + s·rad[h] + η[h]·q[h]

donde `q[h]` es la energía térmica aplicada (positiva calienta, negativa
enfría). Al ser lineal en `T` y en `q`, el problema de optimización resultante es
un programa lineal, con garantía de óptimo global.

**Coherencia con el simulador.** Los parámetros se toman de `SimulationConfig`,
el mismo objeto que gobierna la generación de los datos. Es imprescindible: si el
modelo del optimizador y el del edificio simulado no coincidieran, el ahorro
calculado no sería comparable con el consumo real y la cifra no valdría nada.

**La linealización que hay que declarar.** El rendimiento `η` —grados que sube el
edificio por kWh consumido— depende de la dificultad térmica, que a su vez
depende de `|T_int − T_ext|`. Como `T_int` es variable de decisión, eso haría el
problema no lineal. Se resuelve calculando `η` con la temperatura exterior
prevista y la consigna de confort, no con la variable. Es la aproximación
habitual en control predictivo y hay que documentarla como tal.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tfm_energia.data.synthetic_generator import SimulationConfig


# Factores con los que el simulador escala los aportes internos. Están escritos
# como constantes literales en `_temperatura_interior`, así que se replican aquí
# para que ambos modelos describan el mismo edificio.
ESCALA_APORTE_OCUPACION = 0.1
ESCALA_APORTE_SOLAR = 0.08


@dataclass(frozen=True)
class ParametrosTermicos:
    """Coeficientes del modelo lineal para una sede concreta."""

    k_envoltura: float          # fracción de la diferencia con el exterior por hora
    ganancia_ocupacion: float   # °C/h con ocupación plena
    ganancia_solar: float       # °C/h con radiación máxima
    capacidad_hora: float       # °C/h que corrige el equipo a plena potencia
    potencia_nominal_kw: float  # potencia eléctrica de placa del HVAC
    delta_t_nominal: float      # ΔT al que se alcanza la potencia nominal
    banda_confort: float
    t_antihielo: float
    t_setpoint_invierno: float
    t_setpoint_verano: float

    @property
    def energia_nominal_kwh(self) -> float:
        """Energía consumida en una hora a plena potencia."""
        return self.potencia_nominal_kw

    def rendimiento(self, dificultad: float | np.ndarray) -> float | np.ndarray:
        """Grados que aporta cada kWh consumido, dada la dificultad térmica.

        A plena potencia el equipo consume `potencia_nominal · dificultad` y
        corrige `capacidad_hora` grados, de donde sale la conversión.
        """
        return self.capacidad_hora / (self.potencia_nominal_kw * dificultad)

    def dificultad(self, t_interior, t_exterior):
        """Factor de dificultad térmica, acotado igual que en el simulador.

        Cuanto mayor es el salto con el exterior, más energía cuesta cada grado.
        Satura en 1 al alcanzar el ΔT nominal: el equipo no puede superar su placa.
        """
        salto = np.abs(np.asarray(t_interior, dtype=float) - np.asarray(t_exterior, dtype=float))
        return np.clip(salto / self.delta_t_nominal, 0.25, 1.0)


def parametros_de_sede(sede: dict, cfg: SimulationConfig | None = None) -> ParametrosTermicos:
    """Construye los parámetros térmicos a partir del catálogo de sedes."""
    cfg = cfg or SimulationConfig()
    return ParametrosTermicos(
        k_envoltura=cfg.k_envoltura,
        ganancia_ocupacion=cfg.aporte_ocupacion * ESCALA_APORTE_OCUPACION,
        ganancia_solar=cfg.aporte_solar_max * ESCALA_APORTE_SOLAR,
        capacidad_hora=cfg.capacidad_termica_hora,
        potencia_nominal_kw=cfg.hvac_pot_nominal_per_m2 * sede["superficie_m2"],
        delta_t_nominal=cfg.hvac_delta_t_nominal,
        banda_confort=cfg.banda_confort,
        t_antihielo=cfg.t_antihielo,
        t_setpoint_invierno=cfg.t_setpoint_invierno,
        t_setpoint_verano=cfg.t_setpoint_verano,
    )


# ---------------------------------------------------------------------------
# Consignas de confort
# ---------------------------------------------------------------------------
def setpoint_estacional(index: pd.DatetimeIndex, p: ParametrosTermicos) -> np.ndarray:
    """Consigna de confort por hora, con el mismo criterio que el simulador."""
    meses = index.month.values
    objetivo = np.full(len(index), 22.5)
    objetivo[np.isin(meses, (6, 7, 8, 9))] = p.t_setpoint_verano
    objetivo[np.isin(meses, (12, 1, 2, 3))] = p.t_setpoint_invierno
    return objetivo


def banda_confort(
    index: pd.DatetimeIndex,
    en_servicio: np.ndarray,
    p: ParametrosTermicos,
    holgura: float = 0.0,
    margen_preacondicionamiento: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Límites inferior y superior de temperatura admisibles en cada hora.

    En horario de oficina se exige la banda de confort alrededor de la consigna.
    Fuera de horario el suelo es la protección antihielo, que es lo que permite
    al edificio enfriarse solo de noche sin gastar nada.

    El techo fuera de horario es la consigna más `margen_preacondicionamiento`.
    Ese margen es **el que habilita usar el edificio como batería térmica**:
    calentar por encima de la consigna en horas baratas para necesitar menos
    después. Sin techo, el óptimo matemático llevaría el edificio a temperaturas
    absurdas —30 °C de madrugada para ir enfriándose solo hasta media mañana—
    que ningún BMS real permitiría. Acotarlo mantiene el resultado creíble.

    `holgura` amplía la banda en horario de oficina. **Debe dejarse a cero para
    que la comparación sea honesta**: ensancharla haría que el optimizador
    "ahorre" a base de empeorar el confort, y la cifra dejaría de ser comparable
    con el consumo del control actual. Se expone como parámetro solo para poder
    cuantificar en la memoria cuánto ahorro extra costaría cada grado de confort.
    """
    objetivo = setpoint_estacional(index, p)
    inferior = np.where(en_servicio, objetivo - p.banda_confort - holgura, p.t_antihielo)
    superior = np.where(
        en_servicio,
        objetivo + p.banda_confort + holgura,
        objetivo + p.banda_confort + margen_preacondicionamiento,
    )
    return inferior, superior


# ---------------------------------------------------------------------------
# Simulación del modelo lineal
# ---------------------------------------------------------------------------
def deriva_natural(
    t_previa: float | np.ndarray,
    t_exterior: float | np.ndarray,
    ocupacion: float | np.ndarray,
    radiacion: float | np.ndarray,
    p: ParametrosTermicos,
) -> float | np.ndarray:
    """Cambio de temperatura en una hora **sin** climatización."""
    return (
        p.k_envoltura * (np.asarray(t_exterior, dtype=float) - np.asarray(t_previa, dtype=float))
        + p.ganancia_ocupacion * np.asarray(ocupacion, dtype=float)
        + p.ganancia_solar * np.asarray(radiacion, dtype=float)
    )


def simular_temperatura(
    t_inicial: float,
    t_exterior: np.ndarray,
    ocupacion: np.ndarray,
    radiacion: np.ndarray,
    aporte_termico: np.ndarray,
    p: ParametrosTermicos,
) -> np.ndarray:
    """Evolución de la temperatura interior aplicando el modelo lineal.

    `aporte_termico` son los grados que la climatización añade (positivo) o
    quita (negativo) en cada hora. Sirve tanto para validar el modelo contra los
    datos reales como para reconstruir la temperatura de una solución del
    optimizador.
    """
    n = len(t_exterior)
    temperatura = np.empty(n)
    previa = float(t_inicial)
    for h in range(n):
        previa = (
            previa
            + deriva_natural(previa, t_exterior[h], ocupacion[h], radiacion[h], p)
            + aporte_termico[h]
        )
        temperatura[h] = previa
    return temperatura


def aporte_termico_observado(
    consumo_hvac_kwh: np.ndarray,
    hvac_estado: np.ndarray,
    t_interior: np.ndarray,
    t_exterior: np.ndarray,
    p: ParametrosTermicos,
) -> np.ndarray:
    """Grados aportados por el HVAC según los datos observados.

    Se deduce del consumo y del signo del estado: el consumo da la magnitud y el
    estado indica si se estaba calentando o enfriando, información que el valor
    absoluto del consumo no conserva.
    """
    dificultad = p.dificultad(t_interior, t_exterior)
    modulacion = np.asarray(consumo_hvac_kwh, dtype=float) / (
        p.potencia_nominal_kw * dificultad
    )
    signo = np.sign(np.asarray(hvac_estado, dtype=float))
    return signo * modulacion * p.capacidad_hora
