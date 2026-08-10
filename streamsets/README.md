# Pipeline de ingesta — StreamSets Data Collector

Este directorio contiene el pipeline de ingesta de eventos IoT exportado desde
StreamSets Data Collector, junto con las instrucciones para reproducirlo.

## Arquitectura del pipeline

```
                                          ┌─────────────────────┐
                                    ┌────►│  MongoDB            │
                                    │     │  energia.sensores_iot│
┌───────────┐   ┌──────────────┐   ┌┴───────────────┐           │
│ Directory │──►│  Expression  │──►│ Stream Selector│           │
│  (JSON)   │   │  Evaluator   │   │  (validación)  │           │
└───────────┘   └──────────────┘   └┬───────────────┘           │
                                    │     ┌─────────────────────┴┐
                                    └────►│  Local FS            │
                                          │  (rama de rechazo)   │
                                          └──────────────────────┘
```

El campo `timestamp` viaja como texto ISO 8601 desde el gateway, porque JSON no
tiene tipo fecha. Un **Field Type Converter** en la rama válida lo convierte a
fecha antes de escribir en MongoDB.

## Equivalencia con la implementación en Python

Ambas implementaciones hacen exactamente lo mismo. La de Python
(`scripts/ingest_stream.py`) es la que está cubierta por tests automáticos y la
que garantiza que el sistema es reproducible sin depender de SDC.

| Etapa en SDC | Equivalente en Python | Función |
|---|---|---|
| Directory (origin) | `sensor_stream.leer_jsonl()` | Lee ficheros JSON Lines |
| Expression Evaluator | `PipelineIngesta.enriquecer()` | Añade `_origen` y `_procesado_ts` |
| Stream Selector | `PipelineIngesta.clasificar()` | Enruta válidos / rechazados |
| Field Type Converter | `mongo_repository.normalizar_fechas()` | Tipa las fechas |
| MongoDB (destination) | `MongoRepository.insertar_eventos_sensor()` | Persiste |
| Local FS (error) | `PipelineIngesta.volcar_rechazados()` | Archiva con el motivo |

## Rutas dentro del contenedor

El `docker-compose.yml` monta tres directorios del proyecto:

| En el host | En el contenedor | Uso |
|---|---|---|
| `data/stream/` | `/tfm/stream` (solo lectura) | Eventos de entrada |
| `data/stream_rechazados_sdc/` | `/tfm/rechazados` | Rama de error del pipeline |
| `streamsets/` | `/tfm/pipelines` | Pipelines exportados |

La rama de rechazo de SDC escribe en un directorio distinto al de la ingesta de
Python (`data/stream_rechazados/`) para poder comparar los resultados de ambas
implementaciones y comprobar que coinciden.

## Configuración de las etapas

**Directory (origin)**
- Files Directory: `/tfm/stream`
- File Name Pattern: `*.jsonl`
- Pattern Mode: Glob
- Read Order: Last Modified Timestamp
- Include Subdirectories: sí (los eventos están en `sede=*/`)
- Data Format: JSON
- JSON Content: Multiple JSON objects (un objeto por línea)

**Expression Evaluator**
- `/_origen` → `${record:attribute('filename')}`
- `/_procesado_ts` → `${time:now()}`
- `/sede` → `${str:toLower(str:trim(record:value('/sede')))}`

**Stream Selector** — la primera condición que se cumple gana; la última es la
rama por defecto (registros válidos).

Condición de rechazo (versión resumida; el detalle está en el JSON exportado):
```
${record:value('/timestamp') == null or
  record:value('/sede') == null or
  record:value('/sensor_id') == null or
  record:value('/tipo') == null}
```

**Field Type Converter** (solo en la rama válida)
- Campo `/timestamp` → tipo Datetime, formato `yyyy-MM-dd'T'HH:mm:ssXXX`

**MongoDB (destination)**
- Connection String: `mongodb://mongodb:27017` (nombre del servicio en la red de Compose)
- Database: `energia`
- Collection: `sensores_iot`
- Authentication: Username/Password (`tfmuser` / el valor de `MONGO_PASSWORD`)
- Write Mode: Insert

**Local FS (rama de error)**
- Directory Template: `/tfm/rechazados`
- Data Format: JSON
- Max Records in File: 5000

## Estado: diseño documentado, no materializado en SDC

**El pipeline no llegó a construirse dentro de StreamSets Data Collector, y la
razón es una dependencia externa que no se puede resolver desde el proyecto.**

Se comprobó empíricamente con dos versiones:

| Versión | Resultado |
|---|---|
| 5.7.0 (última) | Exige activación: `enabled: true, valid: false` |
| 3.22.3 (última de la serie 3) | **También** la exige, con idéntico comportamiento |

En ambas, el usuario administrador queda restringido al rol `admin-activation`,
que solo permite activar el producto: cualquier intento de crear un pipeline por
la API devuelve `403`. No existe opción de configuración para desactivarlo; va
firmado con RSA dentro de la imagen.

La pantalla inicial de la versión 5.7.0 indica que el código de activación se
solicita a través del portal de soporte y que está reservado a **cuentas
empresariales**. Tras la adquisición de StreamSets por IBM, la vía gratuita para
la versión autogestionada dejó de estar disponible para usuarios individuales; la
alternativa que ofrece el producto es su plataforma en la nube, que cambiaría la
arquitectura del sistema y depende de una cuenta con caducidad.

Se descartó esa vía y el pipeline se implementó en Python con equivalencia
demostrada etapa por etapa (ver la tabla anterior), que es lo que permite que el
proyecto sea reproducible y demostrable en su totalidad. La implementación en
Python tiene además una ventaja sobre SDC en este contexto: **está cubierta por
25 pruebas automáticas**, incluidas las que verifican la idempotencia de la
ingesta y que ningún evento se pierde.

## Qué sí quedó preparado

La infraestructura es funcional y reproducible, de modo que el pipeline podría
montarse en cuanto se dispusiera de una activación:

* `Dockerfile` con la librería de MongoDB ya instalada. La imagen oficial solo
  incluye `basic-lib`, y hacerlo a mano dentro del contenedor no sirve: las
  librerías se instalan en el sistema de ficheros de la imagen y se pierden al
  recrearlo. Fijado en la imagen, `docker compose up` deja el entorno listo.
* Los tres volúmenes montados y verificados: el contenedor ve los eventos en
  `/tfm/stream`, escribe los rechazos en `/tfm/rechazados` y exporta a
  `/tfm/pipelines`.
* El diseño completo de las etapas y su configuración, documentado arriba.

```bash
docker compose up -d streamsets   # la imagen ronda los 2,5 GB
docker logs -f tfm_streamsets     # aquí aparece el aviso de activación
```

Un detalle útil si se retoma: **SDC 3.x usa autenticación por formulario**, no
básica, así que para usar su API REST hay que arrancarlo con
`SDC_CONF_HTTP_AUTHENTICATION=basic`. Sin eso, todas las llamadas se redirigen al
formulario de acceso y devuelven respuestas vacías que parecen un problema de
permisos.
