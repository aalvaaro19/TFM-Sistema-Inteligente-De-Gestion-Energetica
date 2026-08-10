"""Cliente para la API e·sios de Red Eléctrica de España.

Documentación: https://api.esios.ree.es/

Para precios PVPC horarios y otros indicadores del sistema eléctrico.
Requiere `ESIOS_API_TOKEN` (solicitar por mail a consultasios@ree.es).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from tfm_energia.config import settings


BASE_URL = "https://api.esios.ree.es"
TIMEOUT = 90.0   # e·sios es lento con históricos grandes

# Indicadores e·sios más útiles para el TFM
INDICADOR_PVPC_DEFAULT = 1001          # PVPC 2.0TD general
INDICADOR_PVPC_PEAJE_PUNTA = 1014      # peaje punta
INDICADOR_DEMANDA_REAL = 1293


class EsiosClient:
    """Cliente síncrono para la API de e·sios."""

    def __init__(self, token: str | None = None) -> None:
        self.token = token or settings.esios_api_token
        if not self.token:
            raise ValueError(
                "ESIOS_API_TOKEN no configurado. "
                "Solicita uno por mail a consultasios@ree.es "
                "y añádelo a tu archivo .env"
            )
        self._client = httpx.Client(
            timeout=TIMEOUT,
            headers={
                "Accept": "application/json; application/vnd.esios-api-v1+json",
                "Content-Type": "application/json",
                "x-api-key": self.token,
            },
        )

    def __enter__(self) -> "EsiosClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def indicador(
        self,
        indicador_id: int,
        start: datetime,
        end: datetime,
        geo_ids: list[int] | None = None,
    ) -> pd.DataFrame:
        """Descarga un indicador entre dos timestamps.

        Parseo de fechas en UTC para tolerar transiciones DST (marzo/octubre)
        donde la API devuelve offsets mixtos `+01:00` y `+02:00`.
        """
        params: dict[str, Any] = {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "time_trunc": "hour",
        }
        if geo_ids:
            params["geo_ids[]"] = geo_ids

        r = self._client.get(f"{BASE_URL}/indicators/{indicador_id}", params=params)
        r.raise_for_status()
        data = r.json()
        values = data.get("indicator", {}).get("values", [])
        if not values:
            return pd.DataFrame()

        df = pd.DataFrame(values)
        # utc=True absorbe offsets mixtos sin warnings ni errores en DST.
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["fecha_local"] = df["datetime"].dt.tz_convert("Europe/Madrid")
        return df

    def precio_pvpc(self, start: date, end: date) -> pd.DataFrame:
        """Descarga el PVPC entre dos fechas y devuelve serie horaria normalizada.

        Trocea por bloques mensuales para evitar timeouts en peticiones grandes.
        """
        frames = []
        cursor = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) + pd.Timedelta(days=1)

        while cursor < end_ts:
            bloque_fin = min(cursor + pd.DateOffset(months=1), end_ts)
            logger.info(f"  e·sios PVPC: {cursor.date()} → {bloque_fin.date()}")
            try:
                df_bloque = self.indicador(
                    INDICADOR_PVPC_DEFAULT,
                    cursor.to_pydatetime(),
                    bloque_fin.to_pydatetime(),
                )
                if not df_bloque.empty:
                    frames.append(df_bloque)
            except Exception as e:
                logger.error(f"    Falló bloque {cursor.date()}: {type(e).__name__}")
            cursor = bloque_fin

        if not frames:
            logger.warning("No se recibieron datos PVPC.")
            return pd.DataFrame()

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")
        df = df.rename(columns={"value": "precio_eur_mwh"})
        df["precio_eur_kwh"] = df["precio_eur_mwh"] / 1000.0
        df["franja_pvpc"] = df["fecha_local"].apply(clasificar_franja)
        return df[["fecha_local", "precio_eur_mwh", "precio_eur_kwh", "franja_pvpc"]]


def clasificar_franja(ts: pd.Timestamp) -> str:
    """Clasifica un timestamp en franja PVPC 2.0TD (punta / llano / valle).

    Tarifa 2.0TD vigente para consumidores < 15 kW:
      * Valle:  00-08 y todos los fines de semana y festivos
      * Llano:  08-10, 14-18, 22-24
      * Punta:  10-14 y 18-22
    """
    if ts.weekday() >= 5:
        return "valle"
    h = ts.hour
    if h < 8 or h >= 24:
        return "valle"
    if 10 <= h < 14 or 18 <= h < 22:
        return "punta"
    return "llano"
