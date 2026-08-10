# Memoria Técnica — Sistema Inteligente de Gestión Energética

**Autor:** Álvaro Bermejo Urgel
**Tutor:** Daniel Rubia Yagüe
**Máster:** Inteligencia Artificial Online
**Convocatoria:** Junio 2026

---

## Índice propuesto

### Resumen ejecutivo
- Abstract (castellano e inglés)
- Palabras clave

### 1. Introducción
- 1.1 Contexto y motivación
- 1.2 Problemática del consumo energético en oficinas
- 1.3 Hipótesis de partida
- 1.4 Estructura del documento

### 2. Objetivos
- 2.1 Objetivo general
- 2.2 Objetivos específicos
- 2.3 Alcance y limitaciones

### 3. Estado del arte
- 3.1 Sistemas BMS (Building Management Systems) tradicionales
- 3.2 Aplicación de IA al sector energético
- 3.3 Modelos predictivos de demanda eléctrica
- 3.4 Optimización de carga: load shifting
- 3.5 Plataformas IoT en edificios
- 3.6 Regulación: tarifa 2.0TD y horas valle en España

### 4. Tecnologías y herramientas
- 4.1 Lenguajes y entorno (Python 3.10+)
- 4.2 Big Data: StreamSets, Kafka conceptual, MongoDB
- 4.3 Machine Learning: scikit-learn, statsmodels, Prophet, TensorFlow/Keras
- 4.4 Optimización: PuLP, programación lineal
- 4.5 Producto: FastAPI, Streamlit, Docker
- 4.6 APIs externas: AEMET OpenData, e·sios REE
- 4.7 Justificación de cada elección

### 5. Metodología
- 5.1 CRISP-DM aplicado al proyecto
- 5.2 Planificación y fases
- 5.3 Gestión del repositorio y reproducibilidad

### 6. Diseño del sistema
- 6.1 Arquitectura general (diagrama)
- 6.2 Capa de ingesta (StreamSets)
- 6.3 Capa de almacenamiento (MongoDB modelo documental)
- 6.4 Capa de modelado (ML)
- 6.5 Capa de optimización
- 6.6 Capa de presentación (API + Streamlit)

### 7. Datos
- 7.1 Fuentes de datos
  - 7.1.1 Sensores IoT simulados (justificación del modelo paramétrico)
  - 7.1.2 AEMET OpenData (meteorología histórica)
  - 7.1.3 e·sios / REE (precios PVPC)
- 7.2 Modelo de generación sintético: bases físicas y calibración
- 7.3 Esquema de las colecciones MongoDB
- 7.4 Pipeline de ingesta StreamSets

### 8. Análisis exploratorio (EDA)
- 8.1 Caracterización por sede
- 8.2 Estacionalidad y patrones horarios
- 8.3 Correlación consumo / temperatura / ocupación
- 8.4 Análisis de precios PVPC y franjas

### 9. Modelado predictivo
- 9.1 Definición del problema (series temporales multivariantes)
- 9.2 Modelo baseline (naïve estacional)
- 9.3 SARIMAX con variables exógenas
- 9.4 Prophet con regresores
- 9.5 LSTM
- 9.6 Comparativa: MAE, RMSE, MAPE, tiempo de entrenamiento
- 9.7 Selección del modelo final

### 10. Detección de anomalías
- 10.1 Tipología de anomalías esperables
- 10.2 Isolation Forest
- 10.3 Métricas de evaluación contra anomalías inyectadas
- 10.4 Estrategia de alertado

### 11. Optimización de costes
- 11.1 Formulación matemática (programación lineal)
- 11.2 Restricciones de confort térmico
- 11.3 Función objetivo: minimización de coste
- 11.4 Implementación con PuLP
- 11.5 Resultados: ahorro estimado por sede

### 12. Sistema en producción
- 12.1 API REST (FastAPI): endpoints y contrato
- 12.2 Dashboard Streamlit
- 12.3 Despliegue: Streamlit Community Cloud + MongoDB Atlas
- 12.4 Monitorización y reentrenamiento

### 13. Evaluación de resultados
- 13.1 Cumplimiento de objetivos
- 13.2 KPIs alcanzados (% de ahorro, RMSE, latencia, precisión anomalías)
- 13.3 Validación del 15% de ahorro objetivo

### 14. Aspectos éticos, sociales y de sostenibilidad
- 14.1 RGPD y datos de ocupación
- 14.2 Impacto en eficiencia energética y CO2
- 14.3 Sesgos potenciales del modelo

### 15. Conclusiones y líneas futuras
- 15.1 Conclusiones
- 15.2 Lecciones aprendidas
- 15.3 Trabajo futuro (reinforcement learning, modelos por edificio, etc.)

### Bibliografía

### Anexos
- A. Glosario
- B. Manual de despliegue
- C. Estructura del repositorio
- D. Capturas del dashboard
- E. Diagramas de arquitectura completos
