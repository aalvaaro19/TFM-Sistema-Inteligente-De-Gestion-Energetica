# ⚡ Sistema Inteligente de Gestión Energética

> Sistema basado en Inteligencia Artificial para la predicción del consumo energético y la optimización de costes en oficinas mediante IoT, Machine Learning y análisis de datos.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Tests](https://img.shields.io/badge/tests-274%20passing-brightgreen)
![Estado](https://img.shields.io/badge/TFM-Desarrollo%20completado-success)

---

# 📖 Descripción

Este repositorio contiene el desarrollo del **Trabajo Fin de Máster** del Máster en Inteligencia Artificial, cuyo objetivo es diseñar e implementar un sistema capaz de **monitorizar, predecir y optimizar el consumo energético** de un parque de oficinas mediante técnicas de Inteligencia Artificial y Big Data.

El sistema integra las lecturas de sensores IoT con datos externos —la observación meteorológica de AEMET y el precio horario de la electricidad de e·sios— para anticipar la demanda y decidir el funcionamiento óptimo de la climatización.

Se aplica a cuatro sedes con climas distintos: **Madrid** (continental), **Sevilla** (mediterráneo cálido), **Barcelona** (mediterráneo) y **Oviedo** (oceánico). Esa diversidad es deliberada: permite comprobar si las conclusiones se sostienen fuera de un único escenario.

---

# 🎯 Objetivos

- 📡 Integrar datos procedentes de sensores IoT.
- 🌦️ Incorporar la observación meteorológica real de AEMET.
- ⚡ Obtener los precios horarios del mercado eléctrico español.
- 🗄️ Diseñar una arquitectura escalable basada en MongoDB.
- 🤖 Desarrollar modelos predictivos de demanda energética.
- 🔍 Detectar anomalías y consumos ineficientes.
- 💰 Optimizar el coste aprovechando las franjas horarias más económicas.
- 📊 Desarrollar un cuadro de mando para la toma de decisiones.

---

# 🏗 Arquitectura del sistema

```text
 AEMET OpenData ──┐
                  ├─► Simulación del edificio ─► Flujo JSON Lines ─► Ingesta ─► MongoDB
 e·sios (PVPC) ───┘        (física real)          (gateway IoT)     (validación)    │
                                                                                   ▼
                        ┌──────────────────────────────────────────────────────────┤
                        ▼                        ▼                       ▼          │
                 Predicción 48 h        Detección de anomalías    Optimización LP    │
                 (gradient boosting)    (4 canales)               (PuLP + modelo    │
                        │                        │                 térmico)         │
                        └────────────┬───────────┴───────────┬─────────┘            │
                                     ▼                       ▼                      │
                             API REST (FastAPI) ◄────────────┴──────────────────────┘
                                     │
                                     ▼
                           Cuadro de mando (Streamlit)
```

La meteorología observada **gobierna la física del edificio**: la temperatura interior y el consumo de climatización se derivan de ella, no se le añaden después.

---

# 🧠 Inteligencia Artificial

### 📈 Predicción de demanda

Se comparan cuatro familias de modelos con el **mismo protocolo de backtesting de origen móvil**: en cada punto de partida se reentrena con el histórico disponible y se predicen las 48 horas siguientes. Sin ese protocolo común, la comparación no sería legítima.

| Modelo | Papel |
|---|---|
| Referencias ingenuas (5) | Suelo que cualquier modelo debe superar |
| Media de perfil semanal | Referencia estadística |
| SARIMAX con términos de Fourier | Doble estacionalidad, diaria y semanal, con exógenas |
| **Gradient boosting** | **Modelo seleccionado** |

El modelo elegido usa un **enfoque directo multi-paso**: un único regresor válido para todo el horizonte que solo emplea variables observables en el momento de predecir, evitando la acumulación de error de los métodos recursivos.

### 🔍 Detección de anomalías

Detector **no supervisado** de cuatro canales especializados, cada uno vigilando la señal donde su avería deja huella. Las etiquetas del dataset se reservan exclusivamente para evaluar.

La combinación es necesaria: Isolation Forest rinde mal cuando la anomalía es extrema en una sola dimensión, como un sensor bloqueado cuyo valor es normal pero deja de variar.

### 💰 Optimización energética

Programación lineal sobre un **modelo térmico del edificio** validado contra los datos, sujeta a la banda de confort. El desplazamiento de carga no consiste en mover kWh de una hora a otra: usa la masa del edificio como batería térmica, y la envolvente pierde parte de lo almacenado.

La referencia de comparación es **el mismo optimizador con tarifa plana**: misma física y mismo confort, sin señal de precio. Así el ahorro queda atribuido sin ambigüedad al control consciente del precio.

---

# 📈 Resultados obtenidos

### Predicción de demanda

| Sede | MAE (kWh) | R² | Mejora sobre la mejor referencia |
|---|---|---|---|
| Sevilla | 1,16 | **0,981** | 71 % |
| Oviedo | 1,20 | **0,969** | 67 % |
| Madrid | 2,76 | **0,968** | 61 % |
| Barcelona | 3,22 | **0,849** | 51 % |

### Detección de anomalías

F1 de **0,579** con un presupuesto de aviso del 2 % de las horas, **AUC 0,935** y el **80 % de los episodios** detectados. Por tipo de avería: fuga de equipos 0,97 · climatización atascada 0,95 · sensor bloqueado 0,73 · pico de consumo 0,09.

### Optimización de costes

El arbitraje de precios funciona: el control consciente del precio paga un **7,9 % menos por kWh**. Pero desplazar carga obliga a almacenar calor, y con la envolvente de estas oficinas las pérdidas cancelan la ventaja, dejando el **ahorro en torno al 0 %**.

El análisis de sensibilidad explica el resultado y lo convierte en una recomendación accionable:

| Constante de tiempo del edificio | Ahorro |
|---|---|
| 8 h (envolvente actual) | −0,3 % |
| 12 h | +1,8 % |
| 20 h | +1,9 % |
| **33 h (bien aislado)** | **+9,5 %** |

**El desplazamiento de carga solo resulta rentable si la envolvente conserva el calor almacenado**: conviene aislar antes de automatizar. El objetivo inicial del 15 % no es alcanzable por esta vía, ya que el techo teórico del arbitraje es del 7,5 % dado que la climatización representa entre el 22 y el 32 % de la factura.

Además, el control predictivo **elimina por completo los 15.443 °C·h de incumplimiento de confort** que acumula el control reactivo a lo largo del año: este resulta más barato precisamente porque no alcanza la consigna.

---

# 📂 Estructura del proyecto

```text
.
├── data/
│   ├── raw/           Descargas de AEMET y e·sios
│   ├── synthetic/     Simulación de las sedes
│   ├── stream/        Flujo de eventos IoT (JSON Lines)
│   └── processed/     Dataset enriquecido y resultados de cada fase
│
├── src/tfm_energia/
│   ├── data/          Clientes de API, simulador, ingesta, repositorio Mongo
│   ├── features/      Construcción de variables
│   ├── models/        Métricas, referencias, SARIMAX, gradient boosting, anomalías
│   ├── optimization/  Modelo térmico y optimizador lineal
│   ├── api/           API REST
│   ├── dashboard/     Cuadro de mando
│   └── pipeline.py    Grafo de etapas y dependencias
│
├── scripts/           Ejecutables de cada etapa + orquestador y verificador
├── streamsets/        Diseño del pipeline de ingesta y su imagen Docker
├── notebooks/         Exploración y prototipado
├── docs/              Documentación y borradores de la memoria
└── tests/             Suite de pruebas (274)
```

---

# 🚀 Puesta en marcha

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env      # editar con los tokens y la URI de MongoDB
docker compose up -d mongodb mongo-express   # alternativa a Atlas
```

## Reproducir el proyecto completo

Las etapas tienen dependencias entre sí y ejecutarlas en orden incorrecto no siempre falla: puede producir resultados calculados sobre datos obsoletos. El orquestador declara ese grafo y lo respeta.

```powershell
python scripts\ejecutar_pipeline.py --estado     # qué está hecho y qué falta
python scripts\ejecutar_pipeline.py --simular    # el plan, sin ejecutarlo
python scripts\ejecutar_pipeline.py             # todo desde cero (~1 h)
python scripts\ejecutar_pipeline.py --reanudar  # solo lo que falte
```

Al terminar, comprobar la coherencia del resultado:

```powershell
python scripts\verificar_integridad.py
```

Cada una de sus diez comprobaciones corresponde a un fallo real detectado durante el desarrollo que **no lanzaba ninguna excepción**: producía resultados plausibles calculados sobre datos incoherentes. Una semilla que no reproducía entre ejecuciones, una temperatura exterior que no era la que había generado su propio consumo, fechas guardadas como texto que rompían las consultas en los cambios de hora, o capas de datos desincronizadas tras regenerar solo una parte.

## Etapas del proceso

| Etapa | Script | Produce |
|---|---|---|
| `descargar_aemet` | `download_aemet.py` | Observación diaria por sede |
| `descargar_esios` | `download_esios.py` | Precios PVPC horarios |
| `generar` | `generate_synthetic.py` | Simulación de las 4 sedes |
| `enriquecer` | `enrich_with_real_data.py` | Dataset con precios y coste |
| `simular_sensores` | `simulate_sensors.py` | Flujo de eventos JSON Lines |
| `ingerir` | `ingest_stream.py` | Colección `sensores_iot` |
| `cargar_apis` | `load_apis_to_mongo.py` | Colecciones `meteo_aemet` y `precios_pvpc` |
| `entrenar` | `train_predictivo.py` | Métricas y predicciones de los modelos |
| `anomalias` | `detectar_anomalias.py` | Detecciones y métricas por tipo |
| `optimizar` | `optimizar_costes.py` | Comparativa de estrategias y sensibilidad |

## Servicios

```powershell
uvicorn tfm_energia.api.main:app --reload          # API en :8000/docs
streamlit run src\tfm_energia\dashboard\app.py     # cuadro de mando en :8501
```

## Pruebas

```powershell
pytest                       # suite completa con informe de cobertura
pytest tests\test_api.py -q  # un módulo concreto
```

---

# 🛠 Tecnologías utilizadas

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.10 |
| Base de datos | MongoDB Atlas · MongoDB 7 en local |
| Ingesta de datos | Implementación propia en Python · StreamSets Data Collector (diseño) |
| Machine Learning | Scikit-Learn |
| Series temporales | statsmodels (SARIMAX) |
| Optimización | PuLP con solver CBC |
| Procesamiento de datos | Pandas · NumPy · PyArrow |
| API | FastAPI · Pydantic |
| Visualización | Streamlit · Plotly |
| Infraestructura | Docker Compose |
| Pruebas | pytest · pytest-cov |

> **Sobre StreamSets**: el pipeline quedó diseñado y documentado, con su imagen Docker preparada, pero no llegó a materializarse en Data Collector. Las versiones 5.7.0 y 3.22.3 exigen un código de activación que, tras la adquisición de StreamSets por IBM, solo se concede a cuentas empresariales. La ingesta operativa es la implementación en Python, con equivalencia demostrada etapa por etapa y cubierta por pruebas automáticas. Los detalles están en [`streamsets/README.md`](streamsets/README.md).

---

# 📊 Fuentes de datos

- 🏢 Lecturas de sensores IoT de las cuatro oficinas.
- 🌦️ **AEMET OpenData**: observación meteorológica diaria por estación.
- ⚡ **e·sios (Red Eléctrica de España)**: precio PVPC horario.

Las lecturas de los sensores proceden de un modelo de simulación calibrado con referencias del sector, gobernado por la meteorología real. Es la limitación principal del trabajo y está documentada como tal.

---

# 🔮 Líneas futuras

- Instrumentación física real en sustitución del modelo de simulación.
- Predicción por zonas térmicas en lugar de tratar cada sede como un único espacio.
- Optimización estocástica que incorpore la incertidumbre de la previsión.
- Incorporación de autoconsumo fotovoltaico y almacenamiento en baterías.
- Integración bidireccional con el sistema de gestión del edificio (BACnet, KNX).
- Reentrenamiento periódico y detección automática de deriva del modelo.
- Criterio ambiental explícito, usando la intensidad de carbono horaria del mix eléctrico.

---

# 👨‍💻 Autor

**Álvaro Bermejo Urgel**

Trabajo Fin de Máster — Máster en Inteligencia Artificial
Tutor: Daniel Rubia Yagüe

---

# ⭐ Agradecimientos

Este proyecto ha sido desarrollado como Trabajo Fin de Máster y representa la integración de conocimientos en Inteligencia Artificial, Big Data, IoT y Ciencia de Datos aplicados a un problema real de eficiencia energética.

Si este proyecto te resulta interesante, no dudes en dejar una ⭐ al repositorio.
