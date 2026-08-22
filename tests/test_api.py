
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tfm_energia.api import repositorio as repo
from tfm_energia.api.main import app
from tfm_energia.config import SEDES


@pytest.fixture(scope="module")
def cliente() -> TestClient:
    return TestClient(app)


def _hay(clave: str) -> bool:
    return repo.disponibles().get(clave, False)


# ---------------------------------------------------------------------------
# Estado y catálogo
# ---------------------------------------------------------------------------
def test_health_responde_siempre(cliente: TestClient) -> None:
    """Debe responder aunque falten artefactos o MongoDB esté caído."""
    r = cliente.get("/health")
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["servicio"] == "tfm-energia-api"
    assert set(cuerpo["sedes"]) == set(SEDES)
    assert isinstance(cuerpo["mongo_conectado"], bool)
    assert "enriquecido" in cuerpo["artefactos"]


def test_catalogo_de_sedes(cliente: TestClient) -> None:
    r = cliente.get("/sedes")
    assert r.status_code == 200
    sedes = r.json()
    assert len(sedes) == len(SEDES)
    madrid = next(s for s in sedes if s["id"] == "madrid")
    assert madrid["superficie_m2"] == SEDES["madrid"]["superficie_m2"]
    assert madrid["estacion_aemet"] == SEDES["madrid"]["aemet_station"]


def test_sede_desconocida_da_404(cliente: TestClient) -> None:
    for ruta in ("/prediccion/paris", "/anomalias/paris", "/optimizacion/paris"):
        assert cliente.get(ruta).status_code == 404


# ---------------------------------------------------------------------------
# Predicción
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _hay("backtest"), reason="Requiere scripts/train_predictivo.py")
def test_prediccion_devuelve_el_horizonte_pedido(cliente: TestClient) -> None:
    r = cliente.get("/prediccion/madrid", params={"horizonte": 24})
    assert r.status_code == 200
    cuerpo = r.json()
    assert cuerpo["sede"] == "madrid"
    assert cuerpo["horizonte_h"] == 24
    assert len(cuerpo["puntos"]) == 24
    assert all(p["consumo_previsto_kwh"] >= 0 for p in cuerpo["puntos"])


@pytest.mark.skipif(not _hay("backtest"), reason="Requiere scripts/train_predictivo.py")
def test_prediccion_incluye_coste(cliente: TestClient) -> None:
    """El coste debe ser el producto del consumo por el precio de esa hora."""
    puntos = cliente.get("/prediccion/madrid", params={"horizonte": 12}).json()["puntos"]
    con_precio = [p for p in puntos if p["precio_eur_kwh"] is not None]
    assert con_precio, "Debería haber precio para las horas del histórico"
    for p in con_precio:
        esperado = p["consumo_previsto_kwh"] * p["precio_eur_kwh"]
        assert p["coste_previsto_eur"] == pytest.approx(esperado, abs=1e-3)


@pytest.mark.skipif(not _hay("backtest"), reason="Requiere scripts/train_predictivo.py")
def test_prediccion_ordenada_en_el_tiempo(cliente: TestClient) -> None:
    puntos = cliente.get("/prediccion/madrid", params={"horizonte": 48}).json()["puntos"]
    marcas = [p["timestamp"] for p in puntos]
    assert marcas == sorted(marcas)


def test_horizonte_fuera_de_rango_se_rechaza(cliente: TestClient) -> None:
    assert cliente.get("/prediccion/madrid", params={"horizonte": 0}).status_code == 422
    assert cliente.get("/prediccion/madrid", params={"horizonte": 500}).status_code == 422


@pytest.mark.skipif(not _hay("metricas_modelos"), reason="Requiere las métricas")
def test_metricas_de_modelos(cliente: TestClient) -> None:
    r = cliente.get("/modelos/metricas", params={"sede": "madrid"})
    assert r.status_code == 200
    filas = r.json()
    assert filas
    assert all(f["sede"] == "madrid" for f in filas)
    # El gradient boosting debe ser el de menor MAE
    mejor = min(filas, key=lambda f: f["mae"])
    assert mejor["modelo"] == "gradient_boosting"


# ---------------------------------------------------------------------------
# Anomalías
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _hay("anomalias_detectadas"), reason="Requiere la fase 6")
def test_anomalias_respeta_el_limite(cliente: TestClient) -> None:
    r = cliente.get("/anomalias/madrid", params={"limite": 5})
    assert r.status_code == 200
    assert len(r.json()) <= 5


@pytest.mark.skipif(not _hay("anomalias_detectadas"), reason="Requiere la fase 6")
def test_anomalias_indican_que_detector_salto(cliente: TestClient) -> None:
    """Sin saber qué canal avisó, la alerta no es accionable."""
    filas = cliente.get("/anomalias/madrid", params={"limite": 20}).json()
    assert filas
    assert all(f["sede"] == "madrid" for f in filas)
    assert any(f["detectores"] for f in filas)


@pytest.mark.skipif(not _hay("anomalias_detectadas"), reason="Requiere la fase 6")
def test_anomalias_de_la_mas_reciente_a_la_mas_antigua(cliente: TestClient) -> None:
    marcas = [f["timestamp"] for f in cliente.get("/anomalias/madrid").json()]
    assert marcas == sorted(marcas, reverse=True)


# ---------------------------------------------------------------------------
# Optimización
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _hay("optimizacion_resumen"), reason="Requiere la fase 7")
def test_optimizacion_devuelve_las_tres_estrategias(cliente: TestClient) -> None:
    r = cliente.get("/optimizacion/madrid")
    assert r.status_code == 200
    cuerpo = r.json()
    nombres = {e["estrategia"] for e in cuerpo["estrategias"]}
    assert {"reactivo_actual", "predictivo_ciego", "predictivo_precio"} <= nombres


@pytest.mark.skipif(not _hay("optimizacion_resumen"), reason="Requiere la fase 7")
def test_el_predictivo_cumple_el_confort(cliente: TestClient) -> None:
    """Hallazgo del proyecto: el reactivo es barato porque incumple la banda."""
    estrategias = {
        e["estrategia"]: e for e in cliente.get("/optimizacion/madrid").json()["estrategias"]
    }
    assert estrategias["predictivo_precio"]["grados_hora_fuera_banda"] < (
        estrategias["reactivo_actual"]["grados_hora_fuera_banda"]
    )


def test_optimizacion_sin_artefacto_da_503(cliente: TestClient, monkeypatch) -> None:
    """Si falta la fase, la API debe decirlo con claridad, no romperse."""
    def sin_datos():
        raise FileNotFoundError("Falta la optimización")

    monkeypatch.setattr(repo, "optimizacion", sin_datos)
    r = cliente.get("/optimizacion/madrid")
    assert r.status_code == 503
    assert "optimizaci" in r.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Cuadro de mando
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _hay("enriquecido"), reason="Requiere el dataset enriquecido")
def test_kpis_de_todas_las_sedes(cliente: TestClient) -> None:
    r = cliente.get("/kpis")
    assert r.status_code == 200
    kpis = r.json()
    assert len(kpis) == len(SEDES)
    for k in kpis:
        assert k["consumo_anual_kwh"] > 0
        assert k["coste_anual_eur"] > 0
        assert 0 < k["intensidad_kwh_m2"] < 500
        assert 0 <= k["porcentaje_hvac"] <= 100


@pytest.mark.skipif(not _hay("enriquecido"), reason="Requiere el dataset enriquecido")
def test_intensidad_coherente_con_la_superficie(cliente: TestClient) -> None:
    for k in cliente.get("/kpis").json():
        esperada = k["consumo_anual_kwh"] / SEDES[k["sede"]]["superficie_m2"]
        assert k["intensidad_kwh_m2"] == pytest.approx(esperada, abs=0.2)


def test_limpiar_cache(cliente: TestClient) -> None:
    r = cliente.post("/cache/limpiar")
    assert r.status_code == 200
    assert "cache" in r.json()["estado"]
