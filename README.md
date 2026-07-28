# ⚡ Sistema Inteligente de Gestión Energética

> Sistema basado en Inteligencia Artificial para la predicción del consumo energético y la optimización de costes en edificios inteligentes mediante IoT, Machine Learning y análisis de datos.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Estado](https://img.shields.io/badge/TFM-En%20desarrollo-yellow)

---

# 📖 Descripción

Este repositorio contiene el desarrollo del **Trabajo Fin de Máster (TFM)** del Máster en Inteligencia Artificial, cuyo objetivo es diseñar e implementar un sistema inteligente capaz de **monitorizar, predecir y optimizar el consumo energético** de oficinas mediante técnicas de Inteligencia Artificial y Big Data.

El sistema integra información procedente de sensores IoT con datos externos, como la previsión meteorológica y el precio horario de la electricidad, para anticipar la demanda energética y recomendar el funcionamiento óptimo de los sistemas de climatización.

La finalidad es conseguir un edificio más eficiente, sostenible y con un menor coste energético sin comprometer el confort de los trabajadores.

---

# 🎯 Objetivos

- 📡 Integrar datos procedentes de sensores IoT.
- 🌦️ Incorporar información meteorológica en tiempo real.
- ⚡ Obtener los precios horarios del mercado eléctrico español.
- 🗄️ Diseñar una arquitectura escalable basada en MongoDB.
- 🤖 Desarrollar modelos predictivos de demanda energética.
- 🔍 Detectar anomalías y consumos ineficientes mediante IA.
- 💰 Optimizar el coste energético aprovechando las franjas horarias más económicas.
- 📊 Desarrollar un dashboard interactivo para la visualización y toma de decisiones.

---

# 🏗 Arquitectura del sistema

```text
                     Sensores IoT
                          │
                          ▼
                 StreamSets Data Collector
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
 OpenWeatherMap API                 API ESIOS
        │                                   │
        └─────────────────┬─────────────────┘
                          ▼
                    MongoDB Atlas
                          │
            ┌─────────────┴─────────────┐
            │                           │
      Predicción IA             Detección de anomalías
            │                           │
            └─────────────┬─────────────┘
                          ▼
               Motor de optimización
                          │
                          ▼
                 Dashboard Streamlit
```

---

# 🚀 Funcionalidades

- Ingesta automática de datos IoT.
- Integración de APIs externas.
- Almacenamiento de datos históricos.
- Predicción del consumo energético.
- Detección de anomalías.
- Optimización del uso de la climatización.
- Recomendaciones basadas en el precio de la electricidad.
- Visualización mediante dashboard interactivo.

---

# 🧠 Inteligencia Artificial

El proyecto combina diferentes técnicas de Machine Learning para obtener un sistema inteligente de gestión energética.

### 📈 Predicción de demanda

Se entrenan modelos de series temporales capaces de estimar el consumo energético futuro teniendo en cuenta:

- Temperatura
- Humedad
- Calidad del aire
- Hora del día
- Predicción meteorológica
- Históricos de consumo

---

### 🔍 Detección de anomalías

Se emplean algoritmos de aprendizaje no supervisado para detectar automáticamente:

- Consumos anómalos.
- Equipos de climatización ineficientes.
- Posibles averías.
- Consumo durante periodos de inactividad.

---

### 💰 Optimización energética

El sistema combina las predicciones obtenidas con el precio horario de la electricidad para decidir el momento óptimo de funcionamiento de los sistemas HVAC, reduciendo el gasto energético.

---

# 🛠 Tecnologías utilizadas

| Categoría | Tecnología |
|-----------|------------|
| Lenguaje | Python |
| Base de datos | MongoDB Atlas |
| Ingesta de datos | StreamSets Data Collector |
| Machine Learning | Scikit-Learn |
| Series temporales | Prophet |
| Procesamiento de datos | Pandas · NumPy |
| Visualización | Streamlit |
| APIs | OpenWeatherMap · ESIOS |
| Control de versiones | Git · GitHub |

---

# 📂 Estructura del proyecto

```text
📦 sistema-gestion-energetica
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│
├── src/
│   ├── ingestion/
│   ├── preprocessing/
│   ├── forecasting/
│   ├── anomaly_detection/
│   ├── optimization/
│   └── dashboard/
│
├── models/
│
├── config/
│
├── requirements.txt
│
└── README.md
```

---

# 📊 Fuentes de datos

El sistema trabaja con información procedente de distintas fuentes:

- 🏢 Sensores IoT de oficinas.
- 🌦️ API OpenWeatherMap.
- ⚡ API ESIOS (Red Eléctrica de España).
- 📈 Datos históricos de consumo energético.

---

# 📈 Resultados esperados

Con la implementación del sistema se pretende conseguir:

- ✅ Predicción precisa del consumo energético.
- ✅ Optimización automática de la climatización.
- ✅ Reducción estimada del **15%** del coste energético.
- ✅ Disminución de la huella de carbono.
- ✅ Mejora en la toma de decisiones mediante visualización interactiva.

---

# 🔮 Líneas futuras

- Implementación de modelos LSTM y Transformers.
- Aprendizaje por refuerzo para el control automático de climatización.
- Despliegue mediante Docker.
- Escalado en Kubernetes.
- Integración con MQTT para datos en tiempo real.
- Soporte para múltiples edificios inteligentes.

---

# 👨‍💻 Autor

**Álvaro Bermejo Urgel**

Trabajo Fin de Máster — Máster en Inteligencia Artificial

---

# ⭐ Agradecimientos

Este proyecto ha sido desarrollado como Trabajo Fin de Máster y representa la integración de conocimientos en Inteligencia Artificial, Big Data, IoT y Ciencia de Datos aplicados a un problema real de eficiencia energética.

Si este proyecto te resulta interesante, no dudes en dejar una ⭐ al repositorio.
