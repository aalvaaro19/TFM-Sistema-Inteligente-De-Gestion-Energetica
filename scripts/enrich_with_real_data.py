from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from tfm_energia.config import PROCESSED_DIR, RAW_DIR, SEDES, SYNTHETIC_DIR


def cargar_aemet_horario(sede_id: str) -> pd.DataFrame:
    """Lee CSV AEMET diario y lo interpola a horario.

    AEMET sólo provee diaria gratis. Interpolamos linealmente entre días para
    obtener una aproximación horaria de la temperatura media. Para la modulación
    diaria (curva sinusoidal entre tmin y tmax) usamos un perfil paramétrico.
    """
    path = RAW_DIR / "aemet" / f"meteo_{sede_id}.csv"
    if not path.exists():
        logger.warning(f"No existe {path.name} — sede {sede_id} usará T sintética.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"])
    df = df.sort_values("fecha").reset_index(drop=True)

    # Columnas relevantes
    cols_num = ["tmed", "tmin", "tmax", "hrMedia"]
    for c in cols_num:
        if c not in df.columns:
            df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Interpolar lineal para huecos
    df[cols_num] = df[cols_num].interpolate(method="linear", limit_direction="both")
    df["tmin"] = df["tmin"].fillna(df["tmed"] - 5)
    df["tmax"] = df["tmax"].fillna(df["tmed"] + 5)
    df["hrMedia"] = df["hrMedia"].fillna(60.0)

    # Construir índice horario tz-aware con pd.date_range, que respeta DST nativamente:
    # genera 23h el último domingo de marzo (skip 02:00) y 25h el de octubre (02:00 ×2).
    start = df["fecha"].min()
    end = df["fecha"].max() + pd.Timedelta(hours=23)
    idx = pd.date_range(start=start, end=end, freq="h", tz="Europe/Madrid")

    # Para cada hora del índice, buscamos su día y aplicamos la curva sinusoidal
    df_h = pd.DataFrame({"timestamp": idx})
    df_h["fecha"] = df_h["timestamp"].dt.tz_localize(None).dt.normalize()
    df_diario = df.set_index("fecha")[["tmin", "tmax", "hrMedia"]]
    df_h = df_h.merge(df_diario, left_on="fecha", right_index=True, how="left")

    # Curva: mínimo a las 6h, máximo a las 16h
    h = df_h["timestamp"].dt.hour.values
    factor = np.where(
        (h >= 6) & (h <= 26),
        np.sin(np.pi * (h - 6) / 20.0),
        -0.3,
    )
    df_h["temperatura_real_c"] = (df_h["tmin"] + df_h["tmax"]) / 2 + (
        df_h["tmax"] - df_h["tmin"]
    ) / 2 * factor
    df_h["humedad_real_pct"] = df_h["hrMedia"]

    return df_h[["timestamp", "temperatura_real_c", "humedad_real_pct"]]


def cargar_pvpc() -> pd.DataFrame:
    """Lee el CSV de precios PVPC y lo prepara para el merge."""
    path = RAW_DIR / "esios" / "pvpc_horario.csv"
    if not path.exists():
        logger.warning(f"No existe {path.name} — el dataset no llevará precios reales.")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["fecha_local"] = pd.to_datetime(df["fecha_local"], utc=True).dt.tz_convert(
        "Europe/Madrid"
    )
    df = df.rename(columns={"fecha_local": "timestamp"})
    return df[["timestamp", "precio_eur_mwh", "precio_eur_kwh", "franja_pvpc"]]


def enriquecer_sede(sede_id: str, df_pvpc: pd.DataFrame) -> pd.DataFrame:
    """Combina sintético + AEMET + PVPC para una sede."""
    df_sint = pd.read_parquet(SYNTHETIC_DIR / f"sede_{sede_id}.parquet")
    df_sint["timestamp"] = pd.to_datetime(df_sint["timestamp"])
    if df_sint["timestamp"].dt.tz is None:
        df_sint["timestamp"] = df_sint["timestamp"].dt.tz_localize(
            "Europe/Madrid",
            nonexistent="shift_forward",
            ambiguous="infer",
        )

    logger.info(f"  Simulado: {len(df_sint):,} filas")
    df = df_sint.copy()

    # La meteorología de AEMET ya NO se sustituye aquí: se inyecta en el
    # generador, de modo que la temperatura interior y el consumo derivan de
    # ella. Sustituirla en este punto —con la física ya calculada— dejaba cada
    # fila declarando una temperatura exterior distinta de la que había
    # producido su propio consumo, un desajuste medio de 3,8 °C que hacía
    # inservible cualquier modelo térmico ajustado sobre el dataset.
    if "temperatura_exterior_c_sintetica" in df.columns:
        desvio = (
            df["temperatura_exterior_c"] - df["temperatura_exterior_c_sintetica"]
        ).abs().mean()
        logger.info(f"  Meteorología real aplicada en origen (desvío medio vs sintética: {desvio:.2f} °C)")
    else:
        logger.warning("  Sin columnas sintéticas: el dataset se generó sin meteorología real.")

    # PVPC
    if not df_pvpc.empty:
        df = df.merge(df_pvpc, on="timestamp", how="left")
        cobertura_precio = df["precio_eur_kwh"].notna().mean() * 100
        logger.info(f"  Tras PVPC: cobertura precio = {cobertura_precio:.1f}%")
        # Para huecos de precio, rellenar con la media horaria
        df["precio_eur_kwh"] = df.groupby(df["timestamp"].dt.hour)[
            "precio_eur_kwh"
        ].transform(lambda s: s.fillna(s.mean()))
        df["franja_pvpc"] = df["franja_pvpc"].fillna("llano")

        # Coste por registro
        df["coste_eur"] = df["consumo_total_kwh"] * df["precio_eur_kwh"]
    else:
        df["precio_eur_kwh"] = np.nan
        df["franja_pvpc"] = "llano"
        df["coste_eur"] = np.nan

    return df


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_pvpc = cargar_pvpc()
    if not df_pvpc.empty:
        logger.info(f"PVPC cargado: {len(df_pvpc):,} horas, "
                    f"precio medio {df_pvpc['precio_eur_kwh'].mean():.4f} €/kWh")

    frames = []
    for sede_id in SEDES.keys():
        logger.info(f"=== Enriqueciendo {sede_id} ===")
        df = enriquecer_sede(sede_id, df_pvpc)
        out = PROCESSED_DIR / f"enriquecido_{sede_id}.parquet"
        df.to_parquet(out, index=False)
        logger.info(f"  → {out.name} ({len(df):,} filas, {len(df.columns)} cols)")
        frames.append(df)

    consolidado = pd.concat(frames, ignore_index=True)
    consolidado.to_parquet(PROCESSED_DIR / "enriquecido_consolidado.parquet", index=False)
    logger.info(f"Consolidado enriquecido: {len(consolidado):,} filas")

    # Sanity check del coste anual
    if "coste_eur" in consolidado.columns and consolidado["coste_eur"].notna().any():
        coste_anual = consolidado.groupby("sede")["coste_eur"].sum() / 2  # 2 años
        logger.info("Coste anual estimado por sede (€):")
        for sede, coste in coste_anual.items():
            logger.info(f"  {sede:12s}: {coste:>10,.0f} €")


if __name__ == "__main__":
    main()
