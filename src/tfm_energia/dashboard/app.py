"""Cuadro de mando del sistema de gestión energética.

Traduce los resultados de las fases previas en información útil para quien
gestiona las instalaciones: cuánto se consume, cuánto se prevé consumir, qué
comportamientos anómalos hay y cuánto se puede ahorrar.

Arranque:
    streamlit run src/tfm_energia/dashboard/app.py

Lee los artefactos de `data/processed/` a través del mismo repositorio que usa la
API. Se accede en local en lugar de por HTTP para que el despliegue en Streamlit
Cloud no necesite mantener un segundo servicio en pie; la API queda como interfaz
de integración para terceros y comparte exactamente la misma capa de acceso, así
que ambas sirven cifras idénticas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from tfm_energia.api import repositorio as repo
from tfm_energia.config import SEDES
from tfm_energia.dashboard import graficos as gr
from tfm_energia.dashboard.tema import CLARA, OSCURA, Paleta

st.set_page_config(
    page_title="Gestión Energética Inteligente",
    page_icon="⚡",
    layout="wide",
)

NOMBRES_ESTRATEGIA = {
    "reactivo_actual": "Control actual (reactivo)",
    "predictivo_ciego": "Predictivo sin precio",
    "predictivo_precio": "Predictivo con precio",
}


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def cargar_sede(sede: str) -> pd.DataFrame:
    return repo.datos_sede(sede)


@st.cache_data(show_spinner=False)
def cargar_artefacto(nombre: str) -> pd.DataFrame | None:
    """Lee un artefacto y devuelve None si su fase no se ha ejecutado."""
    lectores = {
        "metricas_modelos": repo.metricas_modelos,
        "anomalias": repo.anomalias,
        "optimizacion": repo.optimizacion,
        "optimizacion_horaria": repo.optimizacion_horaria,
    }
    try:
        return lectores[nombre]()
    except FileNotFoundError:
        return None


@st.cache_data(show_spinner=False)
def cargar_sensibilidad() -> pd.DataFrame | None:
    path = repo.PROCESSED_DIR / "optimizacion_sensibilidad.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_data(show_spinner=False)
def cargar_backtest(sede: str) -> pd.DataFrame | None:
    path = repo.ruta("backtest", sede)
    if not path.exists():
        return None
    bt = pd.read_parquet(path)
    return bt[bt["modelo"] == "gradient_boosting"] if "modelo" in bt.columns else bt


def aviso_fase_pendiente(fase: str, script: str) -> None:
    st.info(
        f"Todavía no hay resultados de {fase}. "
        f"Ejecuta `python scripts/{script}` para generarlos.",
        icon="⏳",
    )


# ---------------------------------------------------------------------------
# Piezas de interfaz
# ---------------------------------------------------------------------------
def indicador(col, etiqueta: str, valor: str, detalle: str | None = None) -> None:
    """Indicador destacado. El número manda; la etiqueta lo nombra."""
    with col:
        st.metric(etiqueta, valor, help=detalle)


def con_tabla(fig, datos: pd.DataFrame, titulo: str = "Ver los datos") -> None:
    """Publica un gráfico junto a su tabla desplegable.

    La tabla no es un extra: la paleta en modo claro tiene dos tonos por debajo
    de 3:1 de contraste sobre el fondo, y la norma de accesibilidad exige
    entonces una vía alternativa para leer los valores.
    """
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with st.expander(titulo):
        st.dataframe(datos, use_container_width=True)


# ---------------------------------------------------------------------------
# Pestañas
# ---------------------------------------------------------------------------
def vista_resumen(p: Paleta) -> None:
    st.subheader("Situación de las cuatro sedes")

    opt = cargar_artefacto("optimizacion")
    anom = cargar_artefacto("anomalias")

    filas = []
    for sede_id, meta in SEDES.items():
        d = cargar_sede(sede_id)
        anios = len(d) / 8760
        consumo = d["consumo_total_kwh"].sum() / anios
        coste = d["coste_eur"].sum() / anios
        filas.append({
            "Sede": meta["nombre"],
            "sede_id": sede_id,
            "Consumo anual (kWh)": consumo,
            "Coste anual (€)": coste,
            "Intensidad (kWh/m²)": consumo / meta["superficie_m2"],
            "Climatización (%)": 100 * d["consumo_hvac_kwh"].sum() / d["consumo_total_kwh"].sum(),
            "Anomalías": int((anom["sede"] == sede_id).sum()) if anom is not None else 0,
        })
    resumen = pd.DataFrame(filas)

    cols = st.columns(4)
    indicador(cols[0], "Coste anual conjunto",
              f"{resumen['Coste anual (€)'].sum():,.0f} €",
              "Suma de las cuatro sedes, promediada sobre el histórico de dos años")
    indicador(cols[1], "Consumo anual conjunto",
              f"{resumen['Consumo anual (kWh)'].sum()/1000:,.0f} MWh")
    indicador(cols[2], "Intensidad media",
              f"{resumen['Intensidad (kWh/m²)'].mean():,.0f} kWh/m²",
              "Referencia IDAE para oficinas: 120–200 kWh/m² al año")
    indicador(cols[3], "Anomalías detectadas", f"{resumen['Anomalías'].sum():,}")

    izq, der = st.columns(2)
    with izq:
        con_tabla(
            gr.barras_por_categoria(
                resumen["Sede"].tolist(), resumen["Coste anual (€)"].tolist(),
                p, titulo="Coste eléctrico anual por sede", unidad=" €",
            ),
            resumen[["Sede", "Coste anual (€)"]].round(0),
        )
    with der:
        con_tabla(
            gr.barras_por_categoria(
                resumen["Sede"].tolist(), resumen["Intensidad (kWh/m²)"].tolist(),
                p, titulo="Intensidad energética por sede", unidad=" kWh/m²",
                color=p.color(2),
            ),
            resumen[["Sede", "Intensidad (kWh/m²)", "Climatización (%)"]].round(1),
        )

    if opt is not None:
        st.subheader("Ahorro identificado")
        precio = opt[opt["estrategia"] == "predictivo_precio"]
        ahorro = float(precio["ahorro_eur"].sum())
        base = float(opt[opt["estrategia"] == "predictivo_ciego"]["coste_eur"].sum())
        c1, c2, c3 = st.columns(3)
        indicador(c1, "Ahorro por gestionar el precio", f"{ahorro:,.0f} €",
                  "Diferencia entre optimizar viendo los precios PVPC y no verlos, "
                  "con idéntico confort")
        indicador(c2, "Sobre el coste de climatización",
                  f"{100*ahorro/base:+.1f} %" if base else "—")
        confort_actual = float(opt[opt["estrategia"] == "reactivo_actual"]["grados_hora"].sum())
        confort_opt = float(precio["grados_hora"].sum())
        indicador(c3, "Confort recuperado",
                  f"{confort_actual - confort_opt:,.0f} °C·h",
                  "Grados-hora fuera de la banda de confort que el control actual "
                  "incumple y el predictivo evita")
    else:
        aviso_fase_pendiente("la optimización", "optimizar_costes.py")


def vista_consumo(sede: str, df: pd.DataFrame, p: Paleta) -> None:
    st.subheader(f"Consumo · {SEDES[sede]['nombre']}")

    c1, c2, c3, c4 = st.columns(4)
    indicador(c1, "Consumo del periodo", f"{df['consumo_total_kwh'].sum():,.0f} kWh")
    indicador(c2, "Coste del periodo", f"{df['coste_eur'].sum():,.0f} €")
    indicador(c3, "Punta horaria", f"{df['consumo_total_kwh'].max():,.1f} kWh")
    indicador(c4, "Precio medio", f"{df['precio_eur_kwh'].mean():.4f} €/kWh")

    con_tabla(
        gr.desglose_componentes(df, p, titulo="Reparto del consumo por componente"),
        df[[c for c, _ in gr.COMPONENTES if c in df.columns]].describe().round(2),
    )

    izq, der = st.columns(2)
    with izq:
        con_tabla(
            gr.perfil_horario(df, p, titulo="Perfil medio por hora del día"),
            df.groupby(df.index.hour)["consumo_total_kwh"].mean().round(2).to_frame("kWh medios"),
        )
    with der:
        reparto = df.groupby("franja_pvpc")["consumo_total_kwh"].sum().sort_values(ascending=False)
        con_tabla(
            gr.barras_por_categoria(
                reparto.index.tolist(), reparto.to_numpy().tolist(),
                p, titulo="Consumo por franja tarifaria", unidad=" kWh", color=p.color(1),
            ),
            df.groupby("franja_pvpc")
              .agg(kWh=("consumo_total_kwh", "sum"), coste=("coste_eur", "sum"),
                   precio_medio=("precio_eur_kwh", "mean"))
              .round(2),
        )


def vista_prediccion(sede: str, df: pd.DataFrame, p: Paleta) -> None:
    st.subheader(f"Previsión de demanda · {SEDES[sede]['nombre']}")

    bt = cargar_backtest(sede)
    if bt is None:
        aviso_fase_pendiente("el modelado predictivo", "train_predictivo.py")
        return

    origenes = sorted(bt["origen"].unique())
    origen = st.select_slider(
        "Momento desde el que se predice",
        options=origenes,
        value=origenes[-1],
        format_func=lambda t: pd.Timestamp(t).strftime("%d/%m/%Y %H:%M"),
    )
    tramo = bt[bt["origen"] == origen].sort_values("timestamp")

    mae = float(np.mean(np.abs(tramo["real"] - tramo["pred"])))
    c1, c2, c3 = st.columns(3)
    indicador(c1, "Horizonte", f"{len(tramo)} h")
    indicador(c2, "Error absoluto medio", f"{mae:,.2f} kWh")
    indicador(c3, "Consumo previsto", f"{tramo['pred'].sum():,.0f} kWh")

    con_tabla(
        gr.prediccion_vs_real(
            pd.Series(tramo["real"].to_numpy(), index=pd.DatetimeIndex(tramo["timestamp"])),
            pd.Series(tramo["pred"].to_numpy(), index=pd.DatetimeIndex(tramo["timestamp"])),
            p, titulo="Previsión a 48 horas frente al consumo registrado",
        ),
        tramo[["timestamp", "h", "real", "pred"]].round(2).set_index("timestamp"),
    )

    metricas = cargar_artefacto("metricas_modelos")
    if metricas is not None:
        st.markdown("**Comparativa de modelos**")
        m = metricas[metricas["sede"] == sede].sort_values("MAE")
        st.dataframe(
            m[["modelo", "MAE", "RMSE", "MAPE", "R2", "MBE"]].round(3).set_index("modelo"),
            use_container_width=True,
        )
        st.caption(
            "Validación por origen móvil: en cada punto de partida se reentrena con el "
            "histórico disponible y se predicen las 48 horas siguientes."
        )


def vista_anomalias(sede: str, df: pd.DataFrame, p: Paleta) -> None:
    st.subheader(f"Anomalías · {SEDES[sede]['nombre']}")

    anom = cargar_artefacto("anomalias")
    if anom is None:
        aviso_fase_pendiente("la detección de anomalías", "detectar_anomalias.py")
        return

    de_sede = anom[anom["sede"] == sede]
    columnas_det = [c for c in de_sede.columns if c.startswith("det_")]
    detectores = ["Cualquiera"] + [c.removeprefix("det_") for c in columnas_det]
    elegido = st.selectbox("Canal de detección", detectores)

    if elegido == "Cualquiera":
        marcadas = de_sede[columnas_det].any(axis=1)
    else:
        marcadas = de_sede[f"det_{elegido}"].astype(bool)
    marcadas = marcadas[marcadas]

    en_rango = marcadas.index.intersection(df.index)
    c1, c2, c3 = st.columns(3)
    indicador(c1, "Avisos en el periodo", f"{len(en_rango):,}")
    if "es_anomalia" in de_sede.columns:
        reales = de_sede.loc[en_rango, "es_anomalia"].sum() if len(en_rango) else 0
        indicador(c2, "Confirmados", f"{int(reales):,}",
                  "Coinciden con una avería etiquetada en el dataset de evaluación")
        indicador(c3, "Precisión", f"{reales/len(en_rango):.0%}" if len(en_rango) else "—")

    con_tabla(
        gr.consumo_con_anomalias(
            df, pd.Series(True, index=en_rango).reindex(df.index, fill_value=False),
            p, titulo="Consumo con las detecciones señaladas",
        ),
        de_sede.loc[en_rango, [c for c in de_sede.columns if not c.startswith("det_")]]
              .round(2) if len(en_rango) else pd.DataFrame(),
    )

    path = repo.PROCESSED_DIR / "anomalias_metricas.csv"
    if path.exists():
        met = pd.read_csv(path)
        izq, der = st.columns(2)
        with izq:
            st.plotly_chart(
                gr.recall_por_tipo(met, p, "Detección por tipo de avería"),
                use_container_width=True, config={"displaylogo": False},
            )
        with der:
            st.markdown("**Rendimiento de cada canal**")
            resumen = (met.groupby("detector")[["precision", "recall", "f1", "recall_episodios"]]
                          .mean().sort_values("f1", ascending=False).round(3))
            st.dataframe(resumen, use_container_width=True)
            st.caption(
                "Todos los canales comparten el mismo presupuesto de avisos, para que "
                "sus precisiones sean comparables."
            )


def vista_optimizacion(sede: str, p: Paleta) -> None:
    st.subheader(f"Optimización del coste · {SEDES[sede]['nombre']}")

    opt = cargar_artefacto("optimizacion")
    if opt is None:
        aviso_fase_pendiente("la optimización", "optimizar_costes.py")
        return

    tabla = opt[opt["sede"] == sede].set_index("estrategia")
    if tabla.empty:
        st.warning(f"Sin resultados de optimización para {SEDES[sede]['nombre']}.")
        return

    st.markdown(
        "Se comparan tres formas de gobernar la climatización sobre los mismos datos, "
        "la misma física y **la misma banda de confort**. La diferencia entre las dos "
        "estrategias predictivas aísla el valor de ver el precio de la electricidad."
    )

    if {"predictivo_ciego", "predictivo_precio"} <= set(tabla.index):
        base = tabla.loc["predictivo_ciego"]
        precio = tabla.loc["predictivo_precio"]
        c1, c2, c3 = st.columns(3)
        indicador(c1, "Ahorro por gestionar el precio",
                  f"{base['coste_eur'] - precio['coste_eur']:,.0f} €",
                  "Con confort idéntico en ambos casos")
        indicador(c2, "Precio medio pagado",
                  f"{precio['precio_medio']:.4f} €/kWh",
                  f"Sin gestionar el precio: {base['precio_medio']:.4f} €/kWh")
        indicador(c3, "Energía consumida",
                  f"{100*(precio['energia_kwh']/base['energia_kwh']-1):+.1f} %",
                  "El desplazamiento de carga exige almacenar calor, y la envolvente "
                  "pierde parte de él")

    izq, der = st.columns(2)
    with izq:
        st.plotly_chart(
            gr.comparativa_estrategias(tabla, p, "coste_eur", "Coste del periodo", " €"),
            use_container_width=True, config={"displaylogo": False},
        )
    with der:
        st.plotly_chart(
            gr.comparativa_estrategias(
                tabla, p, "grados_hora", "Confort fuera de banda", " °C·h"
            ),
            use_container_width=True, config={"displaylogo": False},
        )

    st.dataframe(
        tabla.rename(index=NOMBRES_ESTRATEGIA)[
            ["energia_kwh", "coste_eur", "precio_medio", "grados_hora", "ahorro_pct"]
        ].round(3),
        use_container_width=True,
    )
    st.caption(
        "El control actual resulta más barato porque **no alcanza la consigna**: acumula "
        "grados-hora fuera de banda que las estrategias predictivas eliminan."
    )

    sens = cargar_sensibilidad()
    if sens is not None:
        st.subheader("De qué depende el ahorro")
        con_tabla(
            gr.sensibilidad_envolvente(
                sens, p, "Ahorro según la inercia térmica del edificio"
            ),
            sens.round(3),
        )
        st.markdown(
            "Desplazar consumo a horas baratas obliga a almacenar calor en la masa del "
            "edificio, y la envolvente pierde parte de él cada hora. **El ahorro solo "
            "aparece cuando el edificio conserva ese calor el tiempo suficiente**: con "
            "una constante de tiempo de 8 horas el margen de precio se consume en "
            "pérdidas, mientras que a partir de unas 30 horas el desplazamiento sí "
            "resulta rentable. La conclusión práctica es que conviene aislar antes de "
            "automatizar."
        )


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
def main() -> None:
    st.title("⚡ Sistema Inteligente de Gestión Energética")
    st.caption(
        "Oficinas de Madrid, Sevilla, Barcelona y Oviedo · sensores IoT, "
        "meteorología de AEMET y precios PVPC de e·sios"
    )

    with st.sidebar:
        st.header("Selección")
        sede = st.selectbox(
            "Sede", list(SEDES), format_func=lambda s: SEDES[s]["nombre"]
        )
        modo_oscuro = st.toggle("Modo oscuro", value=False)
        p = OSCURA if modo_oscuro else CLARA

        df_completo = cargar_sede(sede)
        fecha_min = df_completo.index.min().date()
        fecha_max = df_completo.index.max().date()
        rango = st.date_input(
            "Periodo",
            value=(fecha_max - pd.Timedelta(days=30), fecha_max),
            min_value=fecha_min,
            max_value=fecha_max,
        )

        st.divider()
        estado = repo.disponibles()
        st.caption("**Fases calculadas**")
        for clave, listo in estado.items():
            st.caption(f"{'✅' if listo else '⏳'} {clave.replace('_', ' ')}")

    if isinstance(rango, tuple) and len(rango) == 2:
        desde, hasta = rango
        mascara = (df_completo.index.date >= desde) & (df_completo.index.date <= hasta)
        df = df_completo[mascara]
    else:
        df = df_completo

    if df.empty:
        st.warning("El periodo seleccionado no contiene datos.")
        return

    pestanas = st.tabs(
        ["Resumen", "Consumo", "Previsión", "Anomalías", "Optimización"]
    )
    with pestanas[0]:
        vista_resumen(p)
    with pestanas[1]:
        vista_consumo(sede, df, p)
    with pestanas[2]:
        vista_prediccion(sede, df, p)
    with pestanas[3]:
        vista_anomalias(sede, df, p)
    with pestanas[4]:
        vista_optimizacion(sede, p)


main()
