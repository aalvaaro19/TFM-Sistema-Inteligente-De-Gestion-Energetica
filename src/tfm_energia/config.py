"""Configuración centralizada del proyecto.

Carga variables de entorno desde `.env` y expone un objeto `settings` tipado.
"""
from datetime import date
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB
    mongo_uri: str = "mongodb://tfmuser:tfmpass@localhost:27017/"
    mongo_db: str = "energia"

    # APIs
    aemet_api_key: str = ""
    esios_api_token: str = ""

    # Generación sintética
    synthetic_start_date: date = date(2024, 1, 1)
    synthetic_end_date: date = date(2025, 12, 31)
    synthetic_seed: int = 42

    # Logging
    log_level: str = "INFO"


settings = Settings()


# Catálogo de sedes con coordenadas y estación AEMET más cercana
SEDES = {
    "madrid": {
        "nombre": "Madrid",
        "lat": 40.4168,
        "lon": -3.7038,
        "aemet_station": "3195",       # Madrid Retiro
        "superficie_m2": 1200,
        "ocupacion_max": 80,
        "clima": "continental",
    },
    "sevilla": {
        "nombre": "Sevilla",
        "lat": 37.3891,
        "lon": -5.9845,
        "aemet_station": "5783",       # Sevilla Aeropuerto
        "superficie_m2": 900,
        "ocupacion_max": 60,
        "clima": "mediterraneo_calido",
    },
    "barcelona": {
        "nombre": "Barcelona",
        "lat": 41.3851,
        "lon": 2.1734,
        "aemet_station": "0076",       # Barcelona Aeropuerto
        "superficie_m2": 1500,
        "ocupacion_max": 100,
        "clima": "mediterraneo",
    },
    "oviedo": {
        "nombre": "Oviedo",
        "lat": 43.3614,
        "lon": -5.8593,
        "aemet_station": "1212E",      # Asturias Aeropuerto (Avilés) - mejor cobertura histórica que 1249I
        "superficie_m2": 750,
        "ocupacion_max": 50,
        "clima": "oceanico",
    },
}
