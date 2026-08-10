"""Esquemas de entrada y salida de la API.

Se definen con Pydantic para que FastAPI valide las peticiones, serialice las
respuestas y genere la documentación OpenAPI automáticamente. Tener el contrato
explícito evita que el dashboard y la API se desincronicen en silencio.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Estado(BaseModel):
    """Salud del servicio y de sus dependencias."""

    servicio: str = "tfm-energia-api"
    version: str
    mongo_conectado: bool
    sedes: list[str]
    artefactos: dict[str, bool] = Field(
        description="Qué resultados de las fases previas están disponibles"
    )


class Sede(BaseModel):
    """Ficha de una sede."""

    id: str
    nombre: str
    superficie_m2: int
    ocupacion_max: int
    clima: str
    estacion_aemet: str


class PuntoPrediccion(BaseModel):
    """Consumo previsto para una hora."""

    timestamp: datetime
    consumo_previsto_kwh: float
    precio_eur_kwh: float | None = None
    coste_previsto_eur: float | None = None


class Prediccion(BaseModel):
    """Previsión de consumo a 48 horas."""

    sede: str
    modelo: str
    horizonte_h: int
    generada_en: datetime
    puntos: list[PuntoPrediccion]

    @property
    def consumo_total_kwh(self) -> float:
        return sum(p.consumo_previsto_kwh for p in self.puntos)


class Anomalia(BaseModel):
    """Detección de un comportamiento anómalo."""

    timestamp: datetime
    sede: str
    consumo_total_kwh: float
    consumo_hvac_kwh: float | None = None
    temperatura_interior_c: float | None = None
    detectores: list[str] = Field(description="Canales que dieron la alarma")
    es_anomalia_real: bool | None = Field(
        default=None, description="Etiqueta de referencia, solo en el dataset de evaluación"
    )
    tipo_real: str | None = None


class MetricasModelo(BaseModel):
    """Métricas de un modelo predictivo en una sede."""

    sede: str
    modelo: str
    mae: float
    rmse: float
    mape: float
    r2: float


class ResultadoEstrategia(BaseModel):
    """Coste y confort de una estrategia de control."""

    estrategia: str
    energia_kwh: float
    coste_eur: float
    precio_medio_eur_kwh: float
    grados_hora_fuera_banda: float
    ahorro_eur: float | None = None
    ahorro_pct: float | None = None


class Optimizacion(BaseModel):
    """Comparación de estrategias de climatización en una sede."""

    sede: str
    referencia: str
    estrategias: list[ResultadoEstrategia]


class KpisSede(BaseModel):
    """Indicadores agregados de una sede, para el cuadro de mando."""

    sede: str
    nombre: str
    consumo_anual_kwh: float
    coste_anual_eur: float
    intensidad_kwh_m2: float
    porcentaje_hvac: float
    anomalias_detectadas: int
    ahorro_potencial_eur: float | None = None
    ahorro_potencial_pct: float | None = None
