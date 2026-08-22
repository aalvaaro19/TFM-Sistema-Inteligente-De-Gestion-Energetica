from __future__ import annotations

from datetime import date, datetime
from typing import Any

import httpx
import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from tfm_energia.config import settings


BASE_URL = "https://opendata.aemet.es/opendata/api"
TIMEOUT = 30.0


class AemetClient:
    """Cliente síncrono para la API de AEMET OpenData."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or settings.aemet_api_key
        if not self.api_key:
            raise ValueError(
                "AEMET_API_KEY no configurada. "
                "Solicita una en https://opendata.aemet.es/centrodedatos/altaUsuario "
                "y añádela a tu archivo .env"
            )
        self._client = httpx.Client(timeout=TIMEOUT, headers={"api_key": self.api_key})

    def __enter__(self) -> "AemetClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self._client.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _get(self, endpoint: str) -> dict:
        """AEMET responde con un JSON que contiene la URL real de los datos."""
        url = f"{BASE_URL}{endpoint}"
        r = self._client.get(url)
        r.raise_for_status()
        meta = r.json()
        if meta.get("estado") != 200:
            raise RuntimeError(f"AEMET error: {meta}")
        # Segunda llamada al endpoint real de datos
        data_url = meta["datos"]
        r2 = self._client.get(data_url)
        r2.raise_for_status()
        return r2.json()

    def climatologico_diario(
        self, station_id: str, fecha_ini: date, fecha_fin: date
    ) -> pd.DataFrame:
        """Datos climatológicos diarios por estación.

        Limitación AEMET: máximo ~6 meses por petición.
        """
        ini = fecha_ini.strftime("%Y-%m-%dT00:00:00UTC")
        fin = fecha_fin.strftime("%Y-%m-%dT23:59:59UTC")
        endpoint = (
            f"/valores/climatologicos/diarios/datos/fechaini/{ini}/"
            f"fechafin/{fin}/estacion/{station_id}"
        )
        data = self._get(endpoint)
        df = pd.DataFrame(data)
        # Limpieza típica: comas como decimales
        num_cols = ["tmed", "tmin", "tmax", "prec", "velmedia", "racha", "hrMedia"]
        for col in num_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(",", ".", regex=False)
                    .replace({"Ip": "0", "nan": None})
                )
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "fecha" in df.columns:
            df["fecha"] = pd.to_datetime(df["fecha"])
        return df

    def fetch_historico_por_lotes(
        self, station_id: str, start: date, end: date
    ) -> pd.DataFrame:
        """Descarga histórico fraccionando en bloques de 5 meses."""
        frames = []
        cursor = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        while cursor < end_ts:
            bloque_fin = min(cursor + pd.DateOffset(months=5), end_ts)
            logger.info(
                f"AEMET {station_id}: {cursor.date()} → {bloque_fin.date()}"
            )
            df = self.climatologico_diario(
                station_id, cursor.date(), bloque_fin.date()
            )
            frames.append(df)
            cursor = bloque_fin + pd.Timedelta(days=1)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
