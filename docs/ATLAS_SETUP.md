# Configuración de MongoDB Atlas

Guía rápida para conectar el proyecto al cluster gratuito de MongoDB Atlas.

## 1. Crear el cluster (si aún no lo has hecho)

1. Entra en https://www.mongodb.com/atlas
2. Crear nuevo proyecto: `TFM-Energia`
3. Crear nuevo cluster: **M0 Free Tier** (512 MB, suficiente)
4. Región: la más cercana (Frankfurt o Ireland)

## 2. Crear usuario de base de datos

1. Atlas → **Security → Database Access → Add New Database User**
2. Authentication Method: **Password**
3. Username: `tfm_user`
4. Password: genera uno seguro y **guárdalo** (no podrás recuperarlo)
5. Database User Privileges: **Read and write to any database**
6. Add User

## 3. Configurar acceso de red

1. Atlas → **Security → Network Access → Add IP Address**
2. Para desarrollo, lo más cómodo: **Allow access from anywhere** (`0.0.0.0/0`)
3. (En producción se restringiría a una IP fija)
4. Confirm

## 4. Obtener la URI de conexión

1. Atlas → cluster → **Connect → Drivers**
2. Selecciona **Python**, versión 3.12 or later
3. Copia el connection string. Tendrá la forma:

```
mongodb+srv://tfm_user:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
```

4. Sustituye `<password>` por tu contraseña real (URL-encoded si tiene caracteres especiales)
5. Añade el nombre de la base de datos antes del `?`:

```
mongodb+srv://tfm_user:miPassword123@cluster0.xxxxx.mongodb.net/energia?retryWrites=true&w=majority&appName=Cluster0
```

## 5. Configurar en el proyecto

Edita el archivo `.env` (creándolo a partir de `.env.example` si no existe):

```bash
MONGO_URI=mongodb+srv://tfm_user:miPassword123@cluster0.xxxxx.mongodb.net/energia?retryWrites=true&w=majority
MONGO_DB=energia
```

## 6. Probar la conexión y cargar datos

```powershell
.\.venv\Scripts\Activate.ps1

# Smoke test: solo conexión + crear índices
python -c "from tfm_energia.data.mongo_repository import MongoRepository; r = MongoRepository(); r.crear_indices(); print('OK'); r.close()"

# Carga completa de los datos sintéticos
python scripts\load_to_mongo.py
```

Resultados esperados:

- ~210.000 documentos en la colección `sensores_iot` (3 sensores × 70.176 registros)
- Tiempo estimado: 1-3 minutos según latencia con Atlas

## 7. Inspeccionar los datos en Atlas

1. Atlas → cluster → **Browse Collections**
2. Verás:
   - Base `energia`
   - Colección `sensores_iot`
   - Índices `idx_sede_timestamp` e `idx_tipo_sensor`

Ejemplo de query desde la UI de Atlas:

```javascript
{ "sede": "madrid", "tipo": "consumo_electrico" }
```

## Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `ServerSelectionTimeoutError` | IP no whitelisted | Network Access → permite tu IP o `0.0.0.0/0` |
| `Authentication failed` | Password incorrecto o no URL-encoded | Regenera password sin caracteres raros |
| `SSL: CERTIFICATE_VERIFY_FAILED` | Sistema sin certs CA | `pip install --upgrade certifi` |
| Lentitud al cargar | Latencia con Atlas | Normal; carga local es 10× más rápida |
