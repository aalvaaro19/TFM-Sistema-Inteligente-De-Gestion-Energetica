from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from tfm_energia.dashboard.tema import Paleta, layout_base


# Componentes del consumo, en orden fijo de mayor a menor variabilidad
COMPONENTES = (
    ("consumo_hvac_kwh", "Climatización"),
    ("consumo_equipos_kwh", "Equipos"),
    ("consumo_iluminacion_kwh", "Iluminación"),
    ("consumo_base_kwh", "Carga base"),
)


def _linea(p: Paleta, x, y, nombre: str, color: str, ancho: float = 2.0, **kw) -> go.Scatter:
    return go.Scatter(
        x=x, y=y, name=nombre, mode="lines",
        line={"color": color, "width": ancho},
        hovertemplate="%{y:,.1f}<extra>" + nombre + "</extra>",
        **kw,
    )


def serie_consumo(df: pd.DataFrame, p: Paleta, titulo: str | None = None) -> go.Figure:
    """Evolución del consumo total en el periodo seleccionado."""
    fig = go.Figure(_linea(p, df.index, df["consumo_total_kwh"], "Consumo", p.color(0)))
    fig.update_layout(
        **layout_base(p, titulo=titulo),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="kWh por hora")
    return fig


def desglose_componentes(df: pd.DataFrame, p: Paleta, titulo: str | None = None) -> go.Figure:
    """Áreas apiladas con el reparto del consumo por componente.

    Se deja un hueco de 2 px del color del fondo entre segmentos para que los
    límites se lean sin depender del contraste entre tonos contiguos.
    """
    fig = go.Figure()
    for i, (col, etiqueta) in enumerate(COMPONENTES):
        if col not in df.columns:
            continue
        fig.add_trace(go.Scatter(
            x=df.index, y=df[col], name=etiqueta, mode="lines",
            stackgroup="uno",
            line={"color": p.color(i), "width": 2},
            fillcolor=p.color(i),
            hovertemplate="%{y:,.1f} kWh<extra>" + etiqueta + "</extra>",
        ))
    fig.update_layout(**layout_base(p, titulo=titulo), hovermode="x unified")
    fig.update_yaxes(title_text="kWh por hora")
    return fig


def perfil_horario(df: pd.DataFrame, p: Paleta, titulo: str | None = None) -> go.Figure:
    """Consumo medio por hora del día, separando laborables de no laborables.

    Es la vista que revela el patrón de oficina, invisible en la serie completa.
    """
    laborable = (df.index.dayofweek < 5) & ~df.get(
        "es_festivo", pd.Series(False, index=df.index)
    ).astype(bool)

    fig = go.Figure()
    for i, (mascara, etiqueta) in enumerate(
        ((laborable, "Laborables"), (~laborable, "Fines de semana y festivos"))
    ):
        g = df[mascara]
        if g.empty:
            continue
        medio = g.groupby(g.index.hour)["consumo_total_kwh"].mean()
        fig.add_trace(_linea(p, medio.index, medio.to_numpy(), etiqueta, p.color(i)))

    fig.update_layout(**layout_base(p, titulo=titulo), hovermode="x unified")
    fig.update_xaxes(title_text="Hora del día", dtick=3)
    fig.update_yaxes(title_text="kWh medios")
    return fig


def prediccion_vs_real(
    real: pd.Series, prevision: pd.Series, p: Paleta, titulo: str | None = None
) -> go.Figure:
    """Contraste entre el consumo registrado y la previsión del modelo."""
    fig = go.Figure()
    fig.add_trace(_linea(p, real.index, real.to_numpy(), "Real", p.color(0)))
    fig.add_trace(_linea(
        p, prevision.index, prevision.to_numpy(), "Previsión", p.color(1),
        line_dash="dash",
    ))
    fig.update_layout(**layout_base(p, titulo=titulo), hovermode="x unified")
    fig.update_yaxes(title_text="kWh por hora")
    return fig


def consumo_con_anomalias(
    df: pd.DataFrame, detecciones: pd.Series, p: Paleta, titulo: str | None = None
) -> go.Figure:
    """Serie de consumo con las detecciones marcadas.

    Las marcas usan el color de estado crítico, reservado y nunca empleado como
    serie, y van acompañadas del texto en la etiqueta emergente: el aviso no se
    apoya solo en el color.
    """
    fig = go.Figure(_linea(p, df.index, df["consumo_total_kwh"], "Consumo", p.color(0)))

    marcadas = df.loc[df.index.intersection(detecciones[detecciones].index)]
    if not marcadas.empty:
        fig.add_trace(go.Scatter(
            x=marcadas.index, y=marcadas["consumo_total_kwh"],
            name="Anomalía detectada", mode="markers",
            marker={
                "color": p.critico, "size": 9, "symbol": "x",
                "line": {"color": p.fondo, "width": 2},
            },
            hovertemplate="⚠ Anomalía · %{y:,.1f} kWh<extra></extra>",
        ))
    fig.update_layout(**layout_base(p, titulo=titulo), hovermode="closest")
    fig.update_yaxes(title_text="kWh por hora")
    return fig


def barras_por_categoria(
    etiquetas: list[str],
    valores: list[float],
    p: Paleta,
    titulo: str | None = None,
    formato: str = "{:,.0f}",
    unidad: str = "",
    color: str | None = None,
) -> go.Figure:
    """Barras horizontales con el valor rotulado en cada una.

    Una sola serie, así que no lleva leyenda: el título ya dice qué se mide. El
    rótulo directo evita que el lector tenga que estimar contra el eje.
    """
    fig = go.Figure(go.Bar(
        x=valores, y=etiquetas, orientation="h",
        marker={"color": color or p.color(0), "cornerradius": 4},
        text=[formato.format(v) + unidad for v in valores],
        textposition="outside",
        textfont={"color": p.tinta_secundaria, "size": 12},
        # Sin esto, Plotly recorta el rótulo de la barra más larga contra el
        # borde del área de trazado
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.1f}" + unidad + "<extra></extra>",
    ))
    fig.update_layout(**layout_base(p, alto=90 + 46 * len(etiquetas), titulo=titulo))
    # Holgura a la derecha para que quepa el rótulo de la barra mayor
    maximo = max(valores) if valores else 1.0
    fig.update_xaxes(
        showgrid=True, title_text=unidad.strip() or None,
        range=[0, maximo * 1.18],
    )
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def comparativa_estrategias(
    tabla: pd.DataFrame, p: Paleta, columna: str, titulo: str, unidad: str = ""
) -> go.Figure:
    """Compara una magnitud entre estrategias de control.

    Cada estrategia conserva su color en todos los gráficos del panel, de modo
    que el lector la reconoce sin volver a mirar la leyenda.
    """
    orden = ["reactivo_actual", "predictivo_ciego", "predictivo_precio"]
    nombres = {
        "reactivo_actual": "Control actual (reactivo)",
        "predictivo_ciego": "Predictivo sin precio",
        "predictivo_precio": "Predictivo con precio",
    }
    presentes = [e for e in orden if e in tabla.index]

    fig = go.Figure(go.Bar(
        x=[tabla.loc[e, columna] for e in presentes],
        y=[nombres[e] for e in presentes],
        orientation="h",
        marker={
            "color": [p.color(orden.index(e)) for e in presentes],
            "cornerradius": 4,
        },
        text=[f"{tabla.loc[e, columna]:,.2f}{unidad}" for e in presentes],
        textposition="outside",
        textfont={"color": p.tinta_secundaria, "size": 12},
        cliponaxis=False,
        hovertemplate="%{y}: %{x:,.2f}" + unidad + "<extra></extra>",
    ))
    fig.update_layout(**layout_base(p, alto=90 + 46 * len(presentes), titulo=titulo))
    valores = [float(tabla.loc[e, columna]) for e in presentes]
    tope = max(valores) if valores and max(valores) > 0 else 1.0
    fig.update_xaxes(range=[0, tope * 1.18])
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return fig


def sensibilidad_envolvente(sens: pd.DataFrame, p: Paleta, titulo: str) -> go.Figure:
    """Ahorro alcanzable según la calidad de la envolvente.

    Es el gráfico que explica el resultado del proyecto: el desplazamiento de
    carga solo compensa si el edificio conserva el calor que almacena.
    """
    fig = go.Figure()
    for i, (margen, g) in enumerate(sens.groupby("margen_preacondicionamiento")):
        g = g.sort_values("constante_tiempo_h")
        fig.add_trace(go.Scatter(
            x=g["constante_tiempo_h"], y=g["ahorro_pct"],
            name=f"Margen {margen:.0f} °C", mode="lines+markers",
            line={"color": p.color(i), "width": 2},
            marker={"size": 9, "line": {"color": p.fondo, "width": 2}},
            hovertemplate="τ %{x:.0f} h → %{y:+.2f}%<extra>Margen "
                          + f"{margen:.0f} °C</extra>",
        ))
    # Referencia visual del punto de equilibrio
    fig.add_hline(y=0, line={"color": p.eje, "width": 1, "dash": "dot"})
    fig.update_layout(**layout_base(p, titulo=titulo), hovermode="x unified")
    fig.update_xaxes(title_text="Constante de tiempo del edificio (horas)")
    fig.update_yaxes(title_text="Ahorro (%)")
    return fig


def recall_por_tipo(metricas: pd.DataFrame, p: Paleta, titulo: str) -> go.Figure:
    """Capacidad de detección por tipo de avería, para el detector compuesto."""
    fila = metricas[metricas["detector"].str.startswith("compuesto")]
    if fila.empty:
        return go.Figure()

    columnas = [c for c in fila.columns if c.startswith("recall_") and c != "recall_episodios"]
    medias = fila[columnas].mean()
    etiquetas = [c.removeprefix("recall_").replace("_", " ").title() for c in columnas]

    orden = np.argsort(medias.to_numpy())[::-1]
    return barras_por_categoria(
        [etiquetas[i] for i in orden],
        [float(medias.iloc[i]) for i in orden],
        p, titulo=titulo, formato="{:.0%}", color=p.color(2),
    )
