"""Tests del cuadro de mando.

Se prueban el tema y los gráficos, que son módulos puros y no requieren levantar
Streamlit. Lo que se verifica no es la estética sino las reglas que hacen la
figura legible y honesta: **ningún gráfico con doble eje vertical**, leyenda
cuando hay varias series, colores de estado nunca usados como serie, y rótulos
que no se recortan contra el borde.

La paleta está verificada aparte con el validador de accesibilidad del sistema de
diseño; aquí se comprueba que el código respeta el orden de asignación y no
inventa tonos cuando se agotan las ranuras.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tfm_energia.config import SEDES
from tfm_energia.data.synthetic_generator import OfficeSimulator, SimulationConfig
from tfm_energia.dashboard import graficos as gr
from tfm_energia.dashboard.tema import CLARA, OSCURA, Paleta, layout_base


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    """Dos semanas de Madrid con precios y franjas, como el dataset real."""
    d = (
        OfficeSimulator(
            "madrid", SEDES["madrid"],
            start=date(2025, 1, 6), end=date(2025, 1, 19),
            cfg=SimulationConfig(seed=42),
        )
        .generate()
        .set_index("timestamp")
        .sort_index()
    )
    hora = d.index.hour
    d["precio_eur_kwh"] = np.where(hora < 8, 0.10, np.where(hora >= 18, 0.25, 0.15))
    d["franja_pvpc"] = np.where(hora < 8, "valle", np.where(hora >= 18, "punta", "llano"))
    d["coste_eur"] = d["consumo_total_kwh"] * d["precio_eur_kwh"]
    return d


# ---------------------------------------------------------------------------
# Tema
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("p", [CLARA, OSCURA], ids=["clara", "oscura"])
def test_la_paleta_declara_todas_sus_ranuras(p: Paleta) -> None:
    assert len(p.series) == 8
    assert all(c.startswith("#") and len(c) == 7 for c in p.series)


@pytest.mark.parametrize("p", [CLARA, OSCURA], ids=["clara", "oscura"])
def test_los_colores_se_asignan_en_orden_fijo(p: Paleta) -> None:
    """El color sigue a la entidad: la ranura 0 es siempre el mismo tono."""
    assert p.color(0) == p.series[0]
    assert p.color(3) == p.series[3]


@pytest.mark.parametrize("p", [CLARA, OSCURA], ids=["clara", "oscura"])
def test_agotar_las_ranuras_da_error_en_vez_de_inventar_un_tono(p: Paleta) -> None:
    """Una novena serie debe agruparse o separarse, nunca recibir un color nuevo."""
    with pytest.raises(IndexError, match="ranuras de color"):
        p.color(len(p.series))


@pytest.mark.parametrize("p", [CLARA, OSCURA], ids=["clara", "oscura"])
def test_los_colores_de_estado_no_coinciden_con_ninguna_serie(p: Paleta) -> None:
    """Un color de estado nunca debe poder confundirse con una serie."""
    estados = {p.bueno, p.aviso, p.grave, p.critico}
    assert not estados & set(p.series)


def test_los_dos_modos_son_paletas_distintas() -> None:
    """El modo oscuro se escalona para su fondo, no es una inversión automática."""
    assert CLARA.series != OSCURA.series
    assert CLARA.fondo != OSCURA.fondo
    # Los colores de estado sí son invariantes por diseño
    assert CLARA.critico == OSCURA.critico


def test_el_layout_usa_tinta_neutra_para_el_texto() -> None:
    """El texto nunca lleva el color de la serie: el color lo aporta la marca."""
    lay = layout_base(CLARA, titulo="Prueba")
    assert lay["font"]["color"] == CLARA.tinta_secundaria
    assert lay["title"]["font"]["color"] == CLARA.tinta
    assert lay["font"]["color"] not in CLARA.series


def test_el_layout_deja_la_rejilla_discreta() -> None:
    lay = layout_base(CLARA)
    assert lay["yaxis"]["gridcolor"] == CLARA.rejilla
    assert lay["xaxis"]["showgrid"] is False
    assert lay["plot_bgcolor"] == CLARA.fondo


def test_sin_titulo_no_se_reserva_espacio_para_el() -> None:
    assert layout_base(CLARA, titulo=None)["title"] is None
    assert layout_base(CLARA)["margin"]["t"] < layout_base(CLARA, titulo="X")["margin"]["t"]


# ---------------------------------------------------------------------------
# La regla del eje único
# ---------------------------------------------------------------------------
def _figuras(df: pd.DataFrame) -> dict:
    tabla = pd.DataFrame(
        {
            "energia_kwh": [100.0, 120.0, 130.0],
            "coste_eur": [15.0, 17.0, 16.5],
            "grados_hora": [400.0, 0.0, 0.0],
        },
        index=["reactivo_actual", "predictivo_ciego", "predictivo_precio"],
    )
    sens = pd.DataFrame({
        "k_envoltura": [0.12, 0.03, 0.12, 0.03],
        "constante_tiempo_h": [8.3, 33.3, 8.3, 33.3],
        "margen_preacondicionamiento": [0.0, 0.0, 3.0, 3.0],
        "ahorro_pct": [0.28, 5.68, -0.32, 9.54],
    })
    serie = df["consumo_total_kwh"]
    return {
        "serie_consumo": gr.serie_consumo(df, CLARA),
        "desglose": gr.desglose_componentes(df, CLARA),
        "perfil": gr.perfil_horario(df, CLARA),
        "prediccion": gr.prediccion_vs_real(serie, serie * 1.05, CLARA),
        "anomalias": gr.consumo_con_anomalias(
            df, pd.Series(False, index=df.index), CLARA
        ),
        "barras": gr.barras_por_categoria(["A", "B"], [10.0, 25.0], CLARA),
        "estrategias": gr.comparativa_estrategias(tabla, CLARA, "coste_eur", "Coste", " €"),
        "sensibilidad": gr.sensibilidad_envolvente(sens, CLARA, "Sensibilidad"),
    }


def test_ningun_grafico_tiene_doble_eje_vertical(df: pd.DataFrame) -> None:
    """La regla más importante: dos escalas permiten insinuar cualquier correlación."""
    for nombre, fig in _figuras(df).items():
        ejes = [k for k in fig.layout.to_plotly_json() if k.startswith("yaxis")]
        assert len(ejes) <= 1, f"{nombre} tiene {len(ejes)} ejes verticales"
        for traza in fig.data:
            assert getattr(traza, "yaxis", None) in (None, "y"), (
                f"{nombre} asigna una traza a un eje secundario"
            )


def test_todos_los_graficos_usan_el_fondo_del_tema(df: pd.DataFrame) -> None:
    for nombre, fig in _figuras(df).items():
        assert fig.layout.plot_bgcolor == CLARA.fondo, nombre


# ---------------------------------------------------------------------------
# Gráficos concretos
# ---------------------------------------------------------------------------
def test_el_desglose_apila_los_componentes(df: pd.DataFrame) -> None:
    fig = gr.desglose_componentes(df, CLARA)
    assert len(fig.data) == len(gr.COMPONENTES)
    assert all(t.stackgroup == "uno" for t in fig.data)
    # Cada componente conserva su ranura de color
    for i, traza in enumerate(fig.data):
        assert traza.line.color == CLARA.color(i)


def test_una_sola_serie_no_necesita_leyenda(df: pd.DataFrame) -> None:
    """El título ya la nombra; una leyenda de un elemento es ruido."""
    fig = gr.barras_por_categoria(["A", "B"], [1.0, 2.0], CLARA, titulo="Algo")
    assert len(fig.data) == 1


def test_varias_series_llevan_leyenda(df: pd.DataFrame) -> None:
    fig = gr.perfil_horario(df, CLARA)
    assert len(fig.data) >= 2
    assert all(t.name for t in fig.data), "Toda serie debe ir nombrada"


def test_los_rotulos_de_barras_no_se_recortan() -> None:
    """Regresión: Plotly recorta el rótulo de la barra mayor contra el borde."""
    valores = [24460.0, 17920.0]
    fig = gr.barras_por_categoria(["A", "B"], valores, CLARA)
    assert fig.data[0].cliponaxis is False
    assert fig.layout.xaxis.range[1] > max(valores), "Falta holgura para el rótulo"


def test_las_barras_llevan_el_valor_rotulado() -> None:
    fig = gr.barras_por_categoria(["A"], [1234.0], CLARA, unidad=" €")
    assert fig.data[0].text == ("1,234 €",)


def test_las_anomalias_usan_el_color_de_estado_critico(df: pd.DataFrame) -> None:
    """Y van acompañadas de texto: el aviso no se apoya solo en el color."""
    marcadas = pd.Series(False, index=df.index)
    marcadas.iloc[[5, 40]] = True
    fig = gr.consumo_con_anomalias(df, marcadas, CLARA)

    assert len(fig.data) == 2
    marcas = fig.data[1]
    assert marcas.marker.color == CLARA.critico
    assert "Anomalía" in marcas.name
    assert "Anomalía" in marcas.hovertemplate


def test_sin_anomalias_no_se_dibuja_la_capa(df: pd.DataFrame) -> None:
    fig = gr.consumo_con_anomalias(df, pd.Series(False, index=df.index), CLARA)
    assert len(fig.data) == 1


def test_la_prediccion_se_distingue_del_dato_real(df: pd.DataFrame) -> None:
    """Además del color, el trazo discontinuo separa previsión de observación."""
    serie = df["consumo_total_kwh"]
    fig = gr.prediccion_vs_real(serie, serie * 1.1, CLARA)
    real, prevision = fig.data
    assert real.line.color != prevision.line.color
    assert prevision.line.dash == "dash"


def test_las_estrategias_conservan_su_color(df: pd.DataFrame) -> None:
    """La misma estrategia debe verse igual en todos los gráficos del panel."""
    tabla = pd.DataFrame(
        {"coste_eur": [10.0, 12.0, 11.0], "grados_hora": [400.0, 0.0, 0.0]},
        index=["reactivo_actual", "predictivo_ciego", "predictivo_precio"],
    )
    a = gr.comparativa_estrategias(tabla, CLARA, "coste_eur", "Coste")
    b = gr.comparativa_estrategias(tabla, CLARA, "grados_hora", "Confort")
    assert list(a.data[0].marker.color) == list(b.data[0].marker.color)
    assert list(a.data[0].y) == list(b.data[0].y)


def test_la_sensibilidad_marca_el_punto_de_equilibrio() -> None:
    sens = pd.DataFrame({
        "constante_tiempo_h": [8.3, 33.3],
        "margen_preacondicionamiento": [3.0, 3.0],
        "ahorro_pct": [-0.32, 9.54],
    })
    fig = gr.sensibilidad_envolvente(sens, CLARA, "Sensibilidad")
    lineas = fig.layout.to_plotly_json().get("shapes", ())
    assert lineas, "Falta la referencia del cero, que separa ahorro de sobrecoste"


def test_el_perfil_separa_laborables_de_festivos(df: pd.DataFrame) -> None:
    fig = gr.perfil_horario(df, CLARA)
    nombres = {t.name for t in fig.data}
    assert "Laborables" in nombres
    assert any("festivos" in n for n in nombres)
    # Un perfil de 24 horas, no la serie completa
    assert all(len(t.x) <= 24 for t in fig.data)


def test_el_recall_por_tipo_ordena_de_mayor_a_menor() -> None:
    met = pd.DataFrame({
        "detector": ["compuesto_4canales", "isolation_forest"],
        "recall_HVAC_STUCK_ON": [0.95, 0.90],
        "recall_SENSOR_FROZEN": [0.73, 0.02],
        "recall_episodios": [0.80, 0.57],
    })
    fig = gr.recall_por_tipo(met, CLARA, "Por tipo")
    valores = list(fig.data[0].x)
    assert valores == sorted(valores, reverse=True)


def test_sin_detector_compuesto_devuelve_figura_vacia() -> None:
    met = pd.DataFrame({"detector": ["isolation_forest"], "recall_X": [0.5]})
    assert len(gr.recall_por_tipo(met, CLARA, "T").data) == 0
