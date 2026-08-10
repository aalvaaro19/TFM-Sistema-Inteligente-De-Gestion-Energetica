# Guía de puesta en marcha

Este documento describe los pasos necesarios para preparar el entorno de desarrollo del TFM.

## Requisitos previos

| Herramienta | Versión | Notas |
|---|---|---|
| Python | 3.10 - 3.11 | Probado con 3.10 |
| Docker Desktop | reciente | Solo si quieres MongoDB local |
| Git | reciente | Para versionado |
| Cuenta MongoDB Atlas | gratuita | https://www.mongodb.com/atlas (free tier 512MB) |
| Token AEMET | gratuito | https://opendata.aemet.es/centrodedatos/altaUsuario |
| Token e·sios | gratuito | mail a consultasios@ree.es |

---

## 1. Crear y activar entorno virtual

```powershell
cd C:\Users\alvar\TFM_ÁLVARO_BERMEJO_URGEL
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

## 2. Instalar dependencias

```powershell
pip install --upgrade pip
pip install -e ".[dev]"
```

Esto instala todas las librerías de `pyproject.toml` y el propio paquete en modo editable, de modo que cualquier cambio en `src/` se refleja sin reinstalar.

## 3. Configurar variables de entorno

```powershell
copy .env.example .env
notepad .env
```

Rellena al menos:

- `MONGO_URI`: dejar local por ahora o pegar la URI de Atlas
- `AEMET_API_KEY`: cuando lo tengas
- `ESIOS_API_TOKEN`: cuando lo tengas (1-2 días)

## 4. Generar dataset sintético

Esto **no necesita tokens ni base de datos**, se puede ejecutar ya mismo:

```powershell
python scripts\generate_synthetic.py
```

Verás logs por cada sede y se generarán archivos en `data/synthetic/`:

- `sede_madrid.csv` / `.parquet`
- `sede_sevilla.csv` / `.parquet`
- `sede_barcelona.csv` / `.parquet`
- `sede_oviedo.csv` / `.parquet`
- `consolidado.parquet` (todas las sedes juntas)

Cada archivo contiene ~17.500 registros horarios (2 años × 24h).

## 5. Validar con tests

```powershell
pytest -v
```

Deberían pasar todos los tests del generador sintético.

## 6. (Opcional) Levantar MongoDB local con Docker

Alternativa a MongoDB Atlas para trabajar sin conexión:

```powershell
docker compose up -d mongodb mongo-express
```

Esto arranca MongoDB en `localhost:27017` y su interfaz web en
http://localhost:8081 (admin/admin). Después, apunta `MONGO_URI` del `.env` a
`mongodb://tfmuser:tfmpass@localhost:27017/`.

El servicio `streamsets` del `docker-compose.yml` **queda fuera del flujo
habitual**: Data Collector exige un código de activación que ya solo se concede a
cuentas empresariales. Los motivos y el diseño del pipeline están en
[streamsets/README.md](streamsets/README.md); la ingesta operativa es la
implementación en Python.

## 7. Ejecutar el proyecto

El orquestador conoce las dependencias entre etapas y el orden correcto:

```powershell
# Qué está hecho y qué falta
python scripts\ejecutar_pipeline.py --estado

# Ver el plan antes de lanzarlo
python scripts\ejecutar_pipeline.py --simular

# Todo desde cero (~1 h)
python scripts\ejecutar_pipeline.py

# Solo lo que falte
python scripts\ejecutar_pipeline.py --reanudar

# Sin MongoDB
python scripts\ejecutar_pipeline.py --sin-mongo
```

Las etapas y lo que produce cada una están en la tabla del
[README](README.md#etapas-del-proceso).

## 8. Comprobar la integridad del resultado

```powershell
python scripts\verificar_integridad.py
```

Diez comprobaciones que verifican que las capas de datos están sincronizadas y
que los resultados se calcularon sobre el dataset actual. Cada una corresponde a
un fallo real detectado durante el desarrollo que **no lanzaba ninguna
excepción**: producía resultados plausibles sobre datos incoherentes.

Ejecútalo siempre después de regenerar datos. La causa más frecuente de que falle
es haber recalculado una capa y no las que dependen de ella.

## 9. Levantar los servicios

```powershell
# API REST — documentación en http://localhost:8000/docs
uvicorn tfm_energia.api.main:app --reload

# Cuadro de mando — http://localhost:8501
streamlit run src\tfm_energia\dashboard\app.py
```

---

## Comprobación final

- [ ] Tokens de AEMET y e·sios en el `.env`
- [ ] MongoDB accesible (Atlas o local con Docker)
- [ ] `pytest` en verde
- [ ] `python scripts\ejecutar_pipeline.py --estado` con todas las etapas completas
- [ ] `python scripts\verificar_integridad.py` con las diez comprobaciones correctas
- [ ] API y cuadro de mando arrancan sin error
