"""Repositorio MongoDB para persistencia de datos del proyecto.

Abstrae operaciones CRUD sobre las colecciones del proyecto, tanto contra
MongoDB local (Docker) como contra MongoDB Atlas.
"""
from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from loguru import logger
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from tfm_energia.config import settings


# Nombres de colecciones
COL_SENSORES = "sensores_iot"
COL_METEO = "meteo_aemet"
COL_PRECIOS = "precios_pvpc"
COL_PREDICCIONES = "predicciones"
COL_ANOMALIAS = "anomalias"
COL_RECOMENDACIONES = "recomendaciones"

# Campos que deben persistirse como fecha BSON, no como texto.
# El payload JSON de los sensores solo puede transportar cadenas ISO 8601
# (JSON no tiene tipo fecha), así que la conversión corresponde a la capa de
# ingesta. Es el equivalente Python de la etapa `Field Type Converter` del
# pipeline de StreamSets.
CAMPOS_FECHA = ("timestamp", "fecha", "fecha_local", "fecha_prediccion", "ingest_ts")


def parsear_fecha_iso(valor: Any) -> Any:
    """Convierte una cadena ISO 8601 en `datetime`; deja el resto intacto.

    Si el valor no es una fecha reconocible se devuelve tal cual, de modo que la
    función es idempotente y segura de aplicar dos veces.

    Los `pandas.Timestamp` se degradan a `datetime` nativo. Aunque `Timestamp`
    hereda de `datetime`, pymongo calcula mal su desfase horario durante la hora
    ambigua del cambio a horario de invierno: las 02:00+01:00 se persistían como
    02:00 UTC en lugar de 01:00 UTC, chocando con la hora siguiente y perdiendo
    un registro por cada cambio de hora.
    """
    if isinstance(valor, datetime):
        a_nativo = getattr(valor, "to_pydatetime", None)
        return a_nativo() if callable(a_nativo) else valor
    if not isinstance(valor, str):
        return valor
    try:
        # `fromisoformat` de Python 3.10 no admite el sufijo 'Z'
        return datetime.fromisoformat(valor.replace("Z", "+00:00"))
    except ValueError:
        return valor


def normalizar_fechas(doc: dict[str, Any]) -> dict[str, Any]:
    """Devuelve una copia del documento con sus campos de fecha tipados.

    MongoDB almacena las fechas en UTC. Al insertar un `datetime` con zona
    horaria, pymongo hace la conversión automáticamente y conserva el instante
    correcto pese a los cambios de hora (CET/CEST).
    """
    salida = dict(doc)
    for campo in CAMPOS_FECHA:
        if campo in salida:
            salida[campo] = parsear_fecha_iso(salida[campo])
    return salida


class MongoRepository:
    """Capa de acceso a MongoDB."""

    def __init__(
        self, uri: str | None = None, db_name: str | None = None
    ) -> None:
        self.uri = uri or settings.mongo_uri
        self.db_name = db_name or settings.mongo_db
        # `tz_aware=True` hace que las fechas se recuperen con zona horaria (UTC)
        # en vez de como datetime naive, lo que evita desfases de 1-2 h al
        # convertir a hora local según sea CET o CEST.
        self._client = MongoClient(self.uri, tz_aware=True, tzinfo=timezone.utc)
        self.db: Database = self._client[self.db_name]
        logger.info(f"Conectado a MongoDB: {self.db_name}")

    def close(self) -> None:
        self._client.close()

    # ---------- Setup ----------
    def crear_indices(self) -> None:
        """Crea índices óptimos para consultas habituales."""
        self.db[COL_SENSORES].create_index(
            [("sede", ASCENDING), ("timestamp", ASCENDING)],
            name="idx_sede_timestamp",
        )
        self.db[COL_SENSORES].create_index(
            [("tipo", ASCENDING)], name="idx_tipo_sensor"
        )
        self.db[COL_METEO].create_index(
            [("sede", ASCENDING), ("fecha", ASCENDING)], name="idx_meteo"
        )
        self.db[COL_PRECIOS].create_index(
            [("fecha_local", ASCENDING)], unique=True, name="idx_precios_ts"
        )
        self.db[COL_PREDICCIONES].create_index(
            [("sede", ASCENDING), ("modelo", ASCENDING), ("timestamp", ASCENDING)],
            name="idx_predicciones",
        )
        logger.info("Índices creados / asegurados.")

    # ---------- Inserts ----------
    def insertar_eventos_sensor(self, eventos: Iterable[dict[str, Any]]) -> int:
        """Inserta múltiples eventos de sensor (formato JSON IoT).

        Los eventos llegan con `timestamp` como cadena ISO (lo único que puede
        transportar un JSON); aquí se tipa a fecha antes de persistir.
        """
        col = self.db[COL_SENSORES]
        eventos = [normalizar_fechas(e) for e in eventos]
        if not eventos:
            return 0
        res = col.insert_many(eventos, ordered=False)
        return len(res.inserted_ids)

    def insertar_meteo(self, registros: Iterable[dict[str, Any]]) -> int:
        """Inserta observaciones diarias de AEMET, una por sede y fecha.

        Se hace *upsert* sobre (sede, fecha): reejecutar la carga actualiza los
        registros existentes en vez de duplicarlos, que es lo que hace falta
        cuando AEMET publica datos consolidados con retraso.
        """
        from pymongo import ReplaceOne

        col = self.db[COL_METEO]
        regs = [normalizar_fechas(r) for r in registros]
        if not regs:
            return 0

        operaciones = [
            ReplaceOne({"sede": r["sede"], "fecha": r["fecha"]}, r, upsert=True)
            for r in regs
        ]
        res = col.bulk_write(operaciones, ordered=False)
        return res.upserted_count + res.modified_count

    def insertar_precios(self, registros: Iterable[dict[str, Any]]) -> int:
        col = self.db[COL_PRECIOS]
        regs = [normalizar_fechas(r) for r in registros]
        if not regs:
            return 0
        # upsert para evitar duplicados
        bulk = []
        from pymongo import ReplaceOne

        for r in regs:
            bulk.append(ReplaceOne({"fecha_local": r["fecha_local"]}, r, upsert=True))
        if bulk:
            res = col.bulk_write(bulk, ordered=False)
            return res.upserted_count + res.modified_count
        return 0

    # ---------- Queries ----------
    def consumos_por_sede(self, sede: str, limit: int = 1000) -> list[dict]:
        return list(
            self.db[COL_SENSORES]
            .find({"sede": sede, "tipo": "consumo_electrico"})
            .sort("timestamp", ASCENDING)
            .limit(limit)
        )

    def precios_rango(self, ini: str | datetime, fin: str | datetime) -> list[dict]:
        """Precios PVPC entre dos instantes (admite cadena ISO o datetime).

        Los extremos se tipan a fecha para que la comparación sea temporal y no
        lexicográfica: dos cadenas ISO con distinto offset (+01:00 vs +02:00)
        ordenan mal aunque representen el mismo instante.
        """
        return list(
            self.db[COL_PRECIOS]
            .find({"fecha_local": {"$gte": parsear_fecha_iso(ini), "$lte": parsear_fecha_iso(fin)}})
            .sort("fecha_local", ASCENDING)
        )

    # ---------- Utilidades ----------
    def vaciar_coleccion(self, nombre: str) -> int:
        res = self.db[nombre].delete_many({})
        return res.deleted_count
