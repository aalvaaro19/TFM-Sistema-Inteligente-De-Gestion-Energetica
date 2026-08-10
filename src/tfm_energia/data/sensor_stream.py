"""Emisión de lecturas de sensores en formato de flujo (JSON Lines).

Los parquet del generador son un *dataset*, no un *flujo*. Un pipeline de
ingesta necesita lo segundo: ficheros de eventos que van apareciendo, tal como
los dejaría el gateway IoT de cada sede. Este módulo cubre ese hueco.

Formato elegido: **JSON Lines** (un objeto JSON por línea, sin array
envolvente). Es el estándar de los flujos de eventos porque permite escritura
en modo *append*, procesado línea a línea sin cargar el fichero completo y
lectura nativa desde el origen `Directory` de StreamSets.

Cada evento lleva además los metadatos que añadiría un gateway real:

  * ``gateway_id`` – qué equipo lo emitió.
  * ``ingest_ts`` – cuándo se emitió, que **no** es el instante de la lectura.
    La diferencia entre ambos es la latencia de la telemetría.

El emisor puede corromper una fracción de los eventos imitando fallos reales de
sensores. Sin datos defectuosos, la rama de rechazo del pipeline nunca se
ejercita y no hay forma de demostrar que la validación funciona.
"""
from __future__ import annotations

import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.data.synthetic_generator import to_iot_json_payload


# Rangos físicamente admisibles de cada magnitud. Los comparte la validación de
# la ingesta: es el contrato de calidad del dato.
RANGOS_VALIDOS: dict[str, tuple[float, float]] = {
    "temperatura_c": (-20.0, 55.0),
    "humedad_pct": (0.0, 100.0),
    "co2_ppm": (300.0, 5000.0),
    "consumo_total_kwh": (0.0, 1000.0),
    "consumo_hvac_kwh": (0.0, 1000.0),
    "consumo_iluminacion_kwh": (0.0, 1000.0),
    "consumo_equipos_kwh": (0.0, 1000.0),
}

# Campos que todo evento debe traer para ser procesable
CAMPOS_OBLIGATORIOS = ("sensor_id", "tipo", "timestamp", "sede")

# Averías simuladas, cada una con su equivalente en el mundo real
TIPOS_DEFECTO = (
    "campo_ausente",      # paquete truncado en la transmisión
    "valor_nulo",         # sensor desconectado
    "fuera_de_rango",     # valor centinela de sensor averiado (-999)
    "timestamp_invalido", # reloj del gateway mal sincronizado
    "tipo_incorrecto",    # número serializado como texto
)

VALOR_CENTINELA = -999.0


@dataclass
class ConfigEmisor:
    """Parámetros de la emisión."""

    gateway_prefijo: str = "gw"
    tasa_defectos: float = 0.01     # fracción de eventos corrompidos
    seed: int = 42
    particion: str = "dia"          # "dia" | "mes"


# ---------------------------------------------------------------------------
# Corrupción controlada
# ---------------------------------------------------------------------------
def corromper_evento(
    evento: dict[str, Any], tipo_defecto: str, rng: np.random.Generator
) -> dict[str, Any]:
    """Devuelve una copia del evento con un defecto concreto inyectado."""
    if tipo_defecto not in TIPOS_DEFECTO:
        raise ValueError(f"Defecto desconocido: {tipo_defecto}. Opciones: {TIPOS_DEFECTO}")

    ev = dict(evento)
    # Campos numéricos disponibles en este evento
    numericos = [c for c in RANGOS_VALIDOS if c in ev]

    if tipo_defecto == "campo_ausente":
        ev.pop(str(rng.choice(CAMPOS_OBLIGATORIOS)), None)
    elif tipo_defecto == "valor_nulo" and numericos:
        ev[str(rng.choice(numericos))] = None
    elif tipo_defecto == "fuera_de_rango" and numericos:
        ev[str(rng.choice(numericos))] = VALOR_CENTINELA
    elif tipo_defecto == "timestamp_invalido":
        ev["timestamp"] = "0000-00-00T99:99:99"
    elif tipo_defecto == "tipo_incorrecto" and numericos:
        campo = str(rng.choice(numericos))
        ev[campo] = str(ev[campo])

    return ev


def es_valido(evento: dict[str, Any]) -> tuple[bool, str]:
    """Valida un evento contra el contrato de calidad.

    Devuelve `(True, "")` si es correcto, o `(False, motivo)` si no lo es. El
    motivo se persiste en la rama de rechazo para poder auditar la calidad.
    """
    for campo in CAMPOS_OBLIGATORIOS:
        if campo not in evento:
            return False, f"campo_ausente:{campo}"
        if evento[campo] is None:
            return False, f"campo_nulo:{campo}"

    try:
        datetime.fromisoformat(str(evento["timestamp"]).replace("Z", "+00:00"))
    except ValueError:
        return False, "timestamp_invalido"

    for campo, (minimo, maximo) in RANGOS_VALIDOS.items():
        if campo not in evento:
            continue
        valor = evento[campo]
        if valor is None:
            return False, f"valor_nulo:{campo}"
        if not isinstance(valor, (int, float)) or isinstance(valor, bool):
            return False, f"tipo_incorrecto:{campo}"
        if not (minimo <= valor <= maximo):
            return False, f"fuera_de_rango:{campo}"

    return True, ""


# ---------------------------------------------------------------------------
# Emisor
# ---------------------------------------------------------------------------
class EmisorSensores:
    """Convierte el histórico de una sede en ficheros de eventos JSON Lines."""

    def __init__(self, sede_id: str, cfg: ConfigEmisor | None = None) -> None:
        self.sede_id = sede_id
        self.cfg = cfg or ConfigEmisor()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.gateway_id = f"{self.cfg.gateway_prefijo}-{sede_id}-01"

    # -- Construcción de eventos -------------------------------------------
    def eventos_desde_df(self, df: pd.DataFrame) -> Iterator[dict[str, Any]]:
        """Genera los eventos de un DataFrame, con metadatos de gateway."""
        emitido_en = datetime.now(timezone.utc).isoformat()
        for evento in to_iot_json_payload(df):
            evento["gateway_id"] = self.gateway_id
            evento["ingest_ts"] = emitido_en

            if self.rng.random() < self.cfg.tasa_defectos:
                defecto = str(self.rng.choice(TIPOS_DEFECTO))
                evento = corromper_evento(evento, defecto, self.rng)

            yield evento

    # -- Particionado -------------------------------------------------------
    def clave_particion(self, ts: pd.Timestamp) -> str:
        """Sufijo del fichero según la granularidad configurada."""
        if self.cfg.particion == "mes":
            return ts.strftime("%Y%m")
        if self.cfg.particion == "dia":
            return ts.strftime("%Y%m%d")
        raise ValueError(f"Partición no soportada: {self.cfg.particion}")

    def ruta_particion(self, base: Path, clave: str) -> Path:
        """`base/sede=madrid/lecturas_20240101.jsonl`.

        El directorio lleva el patrón `clave=valor` de Hive, que es el que
        entienden de forma nativa tanto Spark como el origen Directory de SDC.
        """
        return base / f"sede={self.sede_id}" / f"lecturas_{clave}.jsonl"

    # -- Escritura ----------------------------------------------------------
    @staticmethod
    def escribir_jsonl(eventos: list[dict[str, Any]], destino: Path) -> int:
        """Escribe los eventos como JSON Lines. Crea el directorio si falta."""
        destino.parent.mkdir(parents=True, exist_ok=True)
        with destino.open("w", encoding="utf-8") as f:
            for ev in eventos:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return len(eventos)

    def volcar_lote(self, df: pd.DataFrame, base: Path) -> dict[str, int]:
        """Vuelca el histórico completo particionado. Devuelve {fichero: nº eventos}."""
        df = df.copy()
        df["_clave"] = df["timestamp"].map(self.clave_particion)

        resumen: dict[str, int] = {}
        for clave, grupo in df.groupby("_clave", sort=True):
            eventos = list(self.eventos_desde_df(grupo.drop(columns="_clave")))
            destino = self.ruta_particion(base, str(clave))
            resumen[destino.name] = self.escribir_jsonl(eventos, destino)

        total = sum(resumen.values())
        logger.info(
            f"{self.sede_id}: {total:,} eventos en {len(resumen)} ficheros "
            f"({self.cfg.particion}) → {base / f'sede={self.sede_id}'}"
        )
        return resumen

    def emitir_stream(
        self,
        df: pd.DataFrame,
        base: Path,
        horas_por_lote: int = 1,
        intervalo_s: float = 1.0,
        max_lotes: int | None = None,
    ) -> int:
        """Emite en tiempo acelerado: un fichero cada `intervalo_s` segundos.

        Es el modo para la demostración en vivo del pipeline: se arranca
        StreamSets, se lanza esto y se ven subir los contadores.
        """
        df = df.sort_values("timestamp").reset_index(drop=True)
        total = 0
        lotes = 0

        for inicio in range(0, len(df), horas_por_lote):
            if max_lotes is not None and lotes >= max_lotes:
                break
            grupo = df.iloc[inicio : inicio + horas_por_lote]
            eventos = list(self.eventos_desde_df(grupo))

            marca = pd.Timestamp(grupo["timestamp"].iloc[0]).strftime("%Y%m%d%H%M%S")
            destino = self.ruta_particion(base, marca)
            total += self.escribir_jsonl(eventos, destino)
            lotes += 1

            logger.debug(f"  lote {lotes}: {len(eventos)} eventos → {destino.name}")
            if intervalo_s > 0:
                time.sleep(intervalo_s)

        logger.info(f"{self.sede_id}: emitidos {total:,} eventos en {lotes} lotes")
        return total


# ---------------------------------------------------------------------------
# Lectura (la usa la ingesta)
# ---------------------------------------------------------------------------
def leer_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Lee un fichero JSON Lines evento a evento.

    Una línea ilegible no aborta la lectura: se emite como evento defectuoso
    para que la ingesta la enrute a la rama de rechazo, igual que haría el
    pipeline ante un registro corrupto.
    """
    with path.open("r", encoding="utf-8") as f:
        for n, linea in enumerate(f, start=1):
            linea = linea.strip()
            if not linea:
                continue
            try:
                yield json.loads(linea)
            except json.JSONDecodeError as exc:
                yield {"_error_parseo": str(exc), "_linea": n, "_fichero": path.name}
