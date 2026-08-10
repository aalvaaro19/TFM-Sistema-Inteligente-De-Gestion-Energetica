"""Tests del modelo térmico y del optimizador de costes.

Lo crítico aquí es que el modelo térmico sea **fiel al edificio simulado**: si no
lo fuera, el ahorro que calcule el optimizador sería ficción. Por eso el test
central compara la predicción del modelo lineal con la trayectoria registrada.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.synthetic_generator import (
    OfficeSimulator,
    SimulationConfig,
)
from tfm_energia.optimization.optimizer import (
    grados_hora_fuera_de_banda,
    horario_de_servicio,
    resolver_ventana,
    serie_control_reactivo,
    simular_control,
)
from tfm_energia.optimization.thermal_model import (
    aporte_termico_observado,
    banda_confort,
    deriva_natural,
    parametros_de_sede,
    setpoint_estacional,
    simular_temperatura,
)


@pytest.fixture
def df_invierno() -> pd.DataFrame:
    """Tres semanas de enero en Madrid, con precios sintéticos por franjas."""
    df = (
        OfficeSimulator(
            "madrid", SEDES["madrid"],
            start=date(2025, 1, 6), end=date(2025, 1, 26),
            cfg=SimulationConfig(seed=42),
        )
        .generate()
        .set_index("timestamp")
        .sort_index()
    )
    # Precio con la estructura de la tarifa 2.0TD: valle de noche, punta de tarde
    hora = df.index.hour
    precio = np.where(hora < 8, 0.10, np.where((hora >= 18) & (hora < 22), 0.25, 0.15))
    df["precio_eur_kwh"] = precio
    return df


@pytest.fixture
def p():
    return parametros_de_sede(SEDES["madrid"])


# ---------------------------------------------------------------------------
# Modelo térmico
# ---------------------------------------------------------------------------
def test_modelo_termico_reproduce_el_edificio(df_invierno: pd.DataFrame, p) -> None:
    """El test más importante del módulo: la física debe ser exacta.

    Se predice a una hora partiendo de la temperatura real previa y aplicando el
    aporte deducido del consumo observado. El simulador añade ruido N(0, 0,15) a
    la temperatura interior, así que un MAE en torno a 0,12 °C es el suelo
    teórico: no se puede predecir el ruido.
    """
    d = df_invierno
    t_real = d["temperatura_interior_c"].to_numpy()
    t_ext = d["temperatura_exterior_c"].to_numpy()

    aporte = aporte_termico_observado(
        d["consumo_hvac_kwh"].to_numpy(), d["hvac_estado"].to_numpy(), t_real, t_ext, p
    )
    prediccion = (
        t_real[:-1]
        + deriva_natural(
            t_real[:-1], t_ext[1:], d["ocupacion_rel"].to_numpy()[1:],
            d["radiacion_solar_rel"].to_numpy()[1:], p,
        )
        + aporte[1:]
    )
    anomala = d["es_anomalia"].to_numpy()
    error = (prediccion - t_real[1:])[~anomala[1:] & ~anomala[:-1]]

    assert np.mean(np.abs(error)) < 0.20, "El modelo térmico no reproduce el edificio"
    assert abs(np.mean(error)) < 0.05, "El modelo tiene sesgo sistemático"


def test_deriva_tiende_al_exterior(p) -> None:
    """Sin climatización ni aportes, el edificio se acerca al exterior."""
    assert deriva_natural(20.0, 5.0, 0.0, 0.0, p) < 0     # se enfría
    assert deriva_natural(20.0, 30.0, 0.0, 0.0, p) > 0    # se calienta
    assert deriva_natural(20.0, 20.0, 0.0, 0.0, p) == pytest.approx(0.0)


def test_aportes_internos_calientan(p) -> None:
    sin_gente = deriva_natural(20.0, 20.0, 0.0, 0.0, p)
    con_gente = deriva_natural(20.0, 20.0, 1.0, 0.0, p)
    con_sol = deriva_natural(20.0, 20.0, 0.0, 1.0, p)
    assert con_gente > sin_gente
    assert con_sol > sin_gente


def test_setpoint_estacional(p) -> None:
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="MS", tz="Europe/Madrid")
    sp = setpoint_estacional(idx, p)
    assert sp[0] == p.t_setpoint_invierno    # enero
    assert sp[6] == p.t_setpoint_verano      # julio
    assert sp[3] == 22.5                     # abril


def test_banda_mas_estrecha_en_servicio(p) -> None:
    idx = pd.date_range("2025-01-07 00:00", periods=24, freq="h", tz="Europe/Madrid")
    servicio = np.array([8 <= h < 19 for h in idx.hour])
    inf, sup = banda_confort(idx, servicio, p)

    assert (sup[servicio] - inf[servicio]).max() == pytest.approx(2 * p.banda_confort)
    # Fuera de servicio el suelo es el antihielo y hay margen para preacondicionar
    assert inf[~servicio].max() == pytest.approx(p.t_antihielo)
    assert sup[~servicio].min() > sup[servicio].min()


def test_dificultad_acotada(p) -> None:
    """El factor de dificultad satura, igual que en el simulador."""
    assert p.dificultad(20.0, 20.0) == pytest.approx(0.25)   # suelo
    assert p.dificultad(40.0, 0.0) == pytest.approx(1.0)     # techo
    assert 0.25 < p.dificultad(20.0, 12.0) < 1.0


def test_simular_temperatura_es_coherente_con_la_deriva(p) -> None:
    n = 12
    t_ext = np.full(n, 5.0)
    traza = simular_temperatura(20.0, t_ext, np.zeros(n), np.zeros(n), np.zeros(n), p)
    # Sin climatización y con exterior frío, desciende de forma monótona
    assert np.all(np.diff(traza) < 0)
    assert traza[-1] > 5.0  # nunca baja del exterior


# ---------------------------------------------------------------------------
# Medida del confort
# ---------------------------------------------------------------------------
def test_grados_hora_solo_penaliza_fuera_de_banda() -> None:
    t_min = np.array([20.0, 20.0, 20.0])
    t_max = np.array([22.0, 22.0, 22.0])
    gh = grados_hora_fuera_de_banda(np.array([21.0, 18.0, 24.0]), t_min, t_max)
    assert gh[0] == 0.0
    assert gh[1] == pytest.approx(2.0)
    assert gh[2] == pytest.approx(2.0)


def test_horario_de_servicio_excluye_findes_y_festivos() -> None:
    idx = pd.date_range("2025-01-04", periods=48, freq="h", tz="Europe/Madrid")
    df = pd.DataFrame({"es_festivo": False}, index=idx)
    serv = horario_de_servicio(df)
    # 4 y 5 de enero de 2025 son sábado y domingo
    assert not serv.any()

    idx2 = pd.date_range("2025-01-07 10:00", periods=1, freq="h", tz="Europe/Madrid")
    assert horario_de_servicio(pd.DataFrame({"es_festivo": [False]}, index=idx2))[0]
    assert not horario_de_servicio(pd.DataFrame({"es_festivo": [True]}, index=idx2))[0]


# ---------------------------------------------------------------------------
# El LP
# ---------------------------------------------------------------------------
def test_lp_resuelve_al_optimo(df_invierno: pd.DataFrame, p) -> None:
    v = df_invierno.iloc[:48]
    serv = horario_de_servicio(v)
    res = resolver_ventana(
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(), v["precio_eur_kwh"].to_numpy(),
        serv, float(v["temperatura_interior_c"].iloc[0]), p,
    )
    assert res.optima
    assert len(res.energia_kwh) == 48
    assert (res.energia_kwh >= -1e-6).all()


def test_lp_respeta_la_banda_de_confort(df_invierno: pd.DataFrame, p) -> None:
    v = df_invierno.iloc[:48]
    serv = horario_de_servicio(v)
    res = resolver_ventana(
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(), v["precio_eur_kwh"].to_numpy(),
        serv, float(v["temperatura_interior_c"].iloc[0]), p,
    )
    t_min, t_max = banda_confort(v.index, serv, p)
    gh = grados_hora_fuera_de_banda(res.temperatura_c, t_min, t_max)
    # Solo se admite holgura al arrancar frío, y acotada
    assert gh.sum() < 20.0


def test_lp_nunca_calienta_y_enfria_a_la_vez(df_invierno: pd.DataFrame, p) -> None:
    """Sería absurdo y costoso: el óptimo no lo hace, sin necesidad de binarias."""
    v = df_invierno.iloc[:48]
    serv = horario_de_servicio(v)
    res = resolver_ventana(
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(), v["precio_eur_kwh"].to_numpy(),
        serv, float(v["temperatura_interior_c"].iloc[0]), p,
    )
    simultaneo = (res.calefaccion_kwh > 0.1) & (res.refrigeracion_kwh > 0.1)
    assert not simultaneo.any()


def test_lp_desplaza_consumo_a_las_horas_baratas(df_invierno: pd.DataFrame, p) -> None:
    """Con precio barato de noche, debe consumir más de noche que con tarifa plana."""
    v = df_invierno.iloc[:72]
    serv = horario_de_servicio(v)
    args = (
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(),
    )
    t0 = float(v["temperatura_interior_c"].iloc[0])

    con_precio = resolver_ventana(*args, v["precio_eur_kwh"].to_numpy(), serv, t0, p)
    plano = resolver_ventana(
        *args, np.full(len(v), float(v["precio_eur_kwh"].mean())), serv, t0, p
    )

    barata = v.index.hour < 8
    assert con_precio.energia_kwh[barata].sum() > plano.energia_kwh[barata].sum()


def test_lp_paga_menos_por_kwh_al_ver_el_precio(df_invierno: pd.DataFrame, p) -> None:
    """El arbitraje debe reducir el precio medio pagado, aunque suba la energía."""
    v = df_invierno.iloc[:72]
    serv = horario_de_servicio(v)
    precio = v["precio_eur_kwh"].to_numpy()
    args = (
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(),
    )
    t0 = float(v["temperatura_interior_c"].iloc[0])

    con_precio = resolver_ventana(*args, precio, serv, t0, p)
    plano = resolver_ventana(*args, np.full(len(v), float(precio.mean())), serv, t0, p)

    def precio_medio(res) -> float:
        return float((res.energia_kwh * precio).sum() / res.energia_kwh.sum())

    assert precio_medio(con_precio) < precio_medio(plano)


def test_estado_terminal_acota_la_temperatura(df_invierno: pd.DataFrame, p) -> None:
    v = df_invierno.iloc[:48]
    serv = horario_de_servicio(v)
    res = resolver_ventana(
        v.index, v["temperatura_exterior_c"].to_numpy(), v["ocupacion_rel"].to_numpy(),
        v["radiacion_solar_rel"].to_numpy(), v["precio_eur_kwh"].to_numpy(),
        serv, float(v["temperatura_interior_c"].iloc[0]), p,
        estado_terminal=(23, 15.0),
    )
    assert res.temperatura_c[23] <= 15.0 + 1e-3


# ---------------------------------------------------------------------------
# Control sobre un periodo
# ---------------------------------------------------------------------------
def test_simular_control_cubre_el_periodo(df_invierno: pd.DataFrame, p) -> None:
    d = simular_control(df_invierno, p, max_ventanas=5)
    assert len(d) == 5 * 24
    assert {"energia_kwh", "coste_eur", "temperatura_c", "grados_hora"} <= set(d.columns)
    assert (d["energia_kwh"] >= -1e-6).all()


def test_simular_control_exige_las_columnas(p) -> None:
    with pytest.raises(ValueError, match="Faltan columnas"):
        simular_control(pd.DataFrame({"otra": [1.0]}, index=pd.date_range("2025-01-01", periods=1, freq="h", tz="Europe/Madrid")), p)


def test_el_predictivo_mejora_el_confort_del_reactivo(df_invierno: pd.DataFrame, p) -> None:
    """Hallazgo del proyecto: el control reactivo es barato porque incumple confort."""
    opt = simular_control(df_invierno, p, max_ventanas=10)
    reactivo = serie_control_reactivo(df_invierno, p).loc[opt.index]

    assert opt["grados_hora"].sum() < reactivo["grados_hora"].sum()
    assert reactivo["grados_hora"].sum() > 0, "El reactivo debería incumplir la banda"


def test_serie_reactiva_usa_los_datos_registrados(df_invierno: pd.DataFrame, p) -> None:
    d = serie_control_reactivo(df_invierno, p)
    assert d["energia_kwh"].sum() == pytest.approx(df_invierno["consumo_hvac_kwh"].sum())
    assert len(d) == len(df_invierno)
