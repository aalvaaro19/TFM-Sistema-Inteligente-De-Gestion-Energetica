"""API REST del sistema de gestión energética.

Expone los resultados de las fases previas —previsión de demanda, anomalías
detectadas y plan de climatización optimizado— para que los consuma el cuadro de
mando o cualquier otro cliente.

La API **no recalcula**: sirve lo que dejaron los scripts de cada fase. Así
responde rápido, no arrastra el coste de entrenar modelos en cada petición y
sigue funcionando aunque MongoDB no esté disponible.

Arranque en desarrollo:
    uvicorn tfm_energia.api.main:app --reload

Documentación interactiva en http://localhost:8000/docs
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query

from tfm_energia.api import repositorio as repo
from tfm_energia.api.schemas import (
    Anomalia,
    Estado,
    KpisSede,
    MetricasModelo,
    Optimizacion,
    Prediccion,
    PuntoPrediccion,
    ResultadoEstrategia,
    Sede,
)
from tfm_energia.config import SEDES

VERSION = "0.1.0"

app = FastAPI(
    title="Sistema Inteligente de Gestión Energética",
    description=(
        "API del TFM: previsión de demanda eléctrica, detección de anomalías y "
        "optimización del coste de climatización por franjas horarias."
    ),
    version=VERSION,
)


def _sede_valida(sede: str) -> str:
    if sede not in SEDES:
        raise HTTPException(404, f"Sede desconocida: {sede}. Opciones: {list(SEDES)}")
    return sede


# ---------------------------------------------------------------------------
# Estado y catálogo
# ---------------------------------------------------------------------------
@app.get("/health", response_model=Estado, tags=["estado"])
def health() -> Estado:
    """Salud del servicio y disponibilidad de cada artefacto."""
    return Estado(
        version=VERSION,
        mongo_conectado=repo.mongo_conectado(),
        sedes=list(SEDES),
        artefactos=repo.disponibles(),
    )


@app.get("/sedes", response_model=list[Sede], tags=["catálogo"])
def listar_sedes() -> list[Sede]:
    """Catálogo de sedes con sus características."""
    return [
        Sede(
            id=sede_id,
            nombre=meta["nombre"],
            superficie_m2=meta["superficie_m2"],
            ocupacion_max=meta["ocupacion_max"],
            clima=meta["clima"],
            estacion_aemet=meta["aemet_station"],
        )
        for sede_id, meta in SEDES.items()
    ]


# ---------------------------------------------------------------------------
# Previsión de demanda
# ---------------------------------------------------------------------------
@app.get("/prediccion/{sede}", response_model=Prediccion, tags=["predicción"])
def prediccion(
    sede: str,
    horizonte: int = Query(48, ge=1, le=168, description="Horas a predecir"),
    desde: datetime | None = Query(None, description="Origen; por defecto el final del histórico"),
) -> Prediccion:
    """Previsión de consumo a partir del backtest del modelo ganador.

    Se sirven las predicciones ya calculadas en la fase de modelado en lugar de
    entrenar al vuelo: reentrenar en cada petición costaría más de un minuto y no
    aportaría nada, porque el modelo solo cambia al reentrenarlo.
    """
    _sede_valida(sede)
    path = repo.ruta("backtest", sede)
    if not path.exists():
        raise HTTPException(
            503, "No hay predicciones disponibles: ejecuta scripts/train_predictivo.py"
        )

    bt = pd.read_parquet(path)
    bt = bt[bt["modelo"] == "gradient_boosting"] if "modelo" in bt.columns else bt
    if bt.empty:
        raise HTTPException(503, "El backtest no contiene el modelo esperado")

    origen = pd.Timestamp(desde) if desde else bt["origen"].max()
    tramo = bt[bt["origen"] == origen].sort_values("timestamp").head(horizonte)
    if tramo.empty:
        disponibles = sorted(bt["origen"].unique())[-3:]
        raise HTTPException(404, f"Sin predicción para {origen}. Últimos orígenes: {disponibles}")

    datos = repo.datos_sede(sede)
    precios = datos["precio_eur_kwh"].reindex(pd.DatetimeIndex(tramo["timestamp"]))

    puntos = [
        PuntoPrediccion(
            timestamp=ts,
            consumo_previsto_kwh=round(float(pred), 3),
            precio_eur_kwh=None if pd.isna(pr) else round(float(pr), 5),
            coste_previsto_eur=None if pd.isna(pr) else round(float(pred) * float(pr), 4),
        )
        for ts, pred, pr in zip(tramo["timestamp"], tramo["pred"], precios.to_numpy())
    ]
    return Prediccion(
        sede=sede,
        modelo="gradient_boosting",
        horizonte_h=len(puntos),
        generada_en=datetime.now(timezone.utc),
        puntos=puntos,
    )


@app.get("/modelos/metricas", response_model=list[MetricasModelo], tags=["predicción"])
def metricas(sede: str | None = None) -> list[MetricasModelo]:
    """Métricas comparadas de los modelos predictivos."""
    try:
        df = repo.metricas_modelos()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    if sede:
        _sede_valida(sede)
        df = df[df["sede"] == sede]

    return [
        MetricasModelo(
            sede=f["sede"], modelo=f["modelo"],
            mae=round(float(f["MAE"]), 4), rmse=round(float(f["RMSE"]), 4),
            mape=round(float(f["MAPE"]), 2), r2=round(float(f["R2"]), 4),
        )
        for _, f in df.iterrows()
    ]


# ---------------------------------------------------------------------------
# Anomalías
# ---------------------------------------------------------------------------
@app.get("/anomalias/{sede}", response_model=list[Anomalia], tags=["anomalías"])
def anomalias(
    sede: str,
    limite: int = Query(100, ge=1, le=1000),
    desde: datetime | None = None,
) -> list[Anomalia]:
    """Anomalías detectadas, de la más reciente a la más antigua."""
    _sede_valida(sede)
    try:
        df = repo.anomalias()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    df = df[df["sede"] == sede]
    if desde is not None:
        df = df[df.index >= pd.Timestamp(desde)]

    columnas_det = [c for c in df.columns if c.startswith("det_")]

    def numero(fila, columna: str, decimales: int) -> float | None:
        """Redondea, o devuelve None si el dato no está."""
        valor = fila.get(columna)
        return None if valor is None or pd.isna(valor) else round(float(valor), decimales)

    def texto(fila, columna: str) -> str | None:
        """Texto opcional. `or None` no sirve: NaN es truthy y colaría al esquema."""
        valor = fila.get(columna)
        if valor is None or pd.isna(valor) or str(valor).strip() == "":
            return None
        return str(valor)

    salida = []
    for ts, fila in df.sort_index(ascending=False).head(limite).iterrows():
        salida.append(
            Anomalia(
                timestamp=ts,
                sede=sede,
                consumo_total_kwh=round(float(fila["consumo_total_kwh"]), 3),
                consumo_hvac_kwh=numero(fila, "consumo_hvac_kwh", 3),
                temperatura_interior_c=numero(fila, "temperatura_interior_c", 2),
                detectores=[c.removeprefix("det_") for c in columnas_det if bool(fila[c])],
                es_anomalia_real=bool(fila["es_anomalia"]) if "es_anomalia" in fila else None,
                tipo_real=texto(fila, "tipo_anomalia"),
            )
        )
    return salida


# ---------------------------------------------------------------------------
# Optimización
# ---------------------------------------------------------------------------
@app.get("/optimizacion/{sede}", response_model=Optimizacion, tags=["optimización"])
def optimizacion(sede: str) -> Optimizacion:
    """Comparación de estrategias de climatización para una sede."""
    _sede_valida(sede)
    try:
        df = repo.optimizacion()
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc

    df = df[df["sede"] == sede]
    if df.empty:
        raise HTTPException(404, f"Sin resultados de optimización para {sede}")

    return Optimizacion(
        sede=sede,
        referencia="predictivo_ciego",
        estrategias=[
            ResultadoEstrategia(
                estrategia=f["estrategia"],
                energia_kwh=round(float(f["energia_kwh"]), 1),
                coste_eur=round(float(f["coste_eur"]), 2),
                precio_medio_eur_kwh=round(float(f["precio_medio"]), 5),
                grados_hora_fuera_banda=round(float(f["grados_hora"]), 1),
                ahorro_eur=round(float(f["ahorro_eur"]), 2),
                ahorro_pct=round(float(f["ahorro_pct"]), 2),
            )
            for _, f in df.iterrows()
        ],
    )


# ---------------------------------------------------------------------------
# Indicadores para el cuadro de mando
# ---------------------------------------------------------------------------
@app.get("/kpis", response_model=list[KpisSede], tags=["cuadro de mando"])
def kpis() -> list[KpisSede]:
    """Indicadores agregados de todas las sedes."""
    try:
        opt = repo.optimizacion()
    except FileNotFoundError:
        opt = pd.DataFrame()

    try:
        anom = repo.anomalias()
    except FileNotFoundError:
        anom = pd.DataFrame()

    salida = []
    for sede_id, meta in SEDES.items():
        try:
            datos = repo.datos_sede(sede_id)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from exc

        anios = len(datos) / 8760
        consumo = float(datos["consumo_total_kwh"].sum() / anios)
        coste = float(datos["coste_eur"].sum() / anios)

        ahorro_eur = ahorro_pct = None
        if not opt.empty:
            fila = opt[(opt["sede"] == sede_id) & (opt["estrategia"] == "predictivo_precio")]
            if not fila.empty:
                ahorro_eur = round(float(fila["ahorro_eur"].iloc[0]), 2)
                ahorro_pct = round(float(fila["ahorro_pct"].iloc[0]), 2)

        salida.append(
            KpisSede(
                sede=sede_id,
                nombre=meta["nombre"],
                consumo_anual_kwh=round(consumo, 0),
                coste_anual_eur=round(coste, 0),
                intensidad_kwh_m2=round(consumo / meta["superficie_m2"], 1),
                porcentaje_hvac=round(
                    100 * float(datos["consumo_hvac_kwh"].sum() / datos["consumo_total_kwh"].sum()),
                    1,
                ),
                anomalias_detectadas=(
                    int((anom["sede"] == sede_id).sum()) if not anom.empty else 0
                ),
                ahorro_potencial_eur=ahorro_eur,
                ahorro_potencial_pct=ahorro_pct,
            )
        )
    return salida


@app.post("/cache/limpiar", tags=["estado"])
def limpiar_cache() -> dict[str, str]:
    """Fuerza la relectura de los artefactos tras reejecutar alguna fase."""
    repo.limpiar_cache()
    return {"estado": "cache vaciada"}
