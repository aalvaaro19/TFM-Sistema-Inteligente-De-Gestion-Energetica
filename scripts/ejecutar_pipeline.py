from __future__ import annotations

import argparse
import sys

from loguru import logger

from tfm_energia.config import settings
from tfm_energia.pipeline import (
    ETAPAS,
    ResultadoEtapa,
    ejecutar_etapa,
    resumen_estado,
    seleccionar,
)


def mostrar_estado() -> None:
    print("\n=== ESTADO DE LAS ETAPAS ===")
    print(f"{'':2} {'etapa':<20} {'artefactos':<12} descripción")
    print("-" * 92)
    for etapa, hecha in resumen_estado():
        if not etapa.artefactos:
            # Escriben en MongoDB, así que su estado no se puede leer en disco
            marca, estado = "?", "en MongoDB"
        elif hecha:
            marca, estado = "✔", "completos"
        else:
            marca, estado = "·", f"faltan {len(etapa.artefactos_faltantes())}"
        print(f"{marca:2} {etapa.nombre:<20} {estado:<12} {etapa.descripcion}")

    print("\nRequisitos externos:")
    for etapa in ETAPAS:
        marcas = []
        if etapa.requiere_token:
            marcas.append("token de API")
        if etapa.requiere_mongo:
            marcas.append("MongoDB")
        if marcas:
            print(f"  {etapa.nombre:<20} {', '.join(marcas)}")


def comprobar_requisitos(etapas: list) -> list[str]:
    """Avisos sobre requisitos externos que podrían faltar."""
    avisos = []
    if any(e.requiere_token for e in etapas):
        if not settings.aemet_api_key:
            avisos.append("AEMET_API_KEY no está definida en .env")
        if not settings.esios_api_token:
            avisos.append("ESIOS_API_TOKEN no está definido en .env")
    if any(e.requiere_mongo for e in etapas):
        try:
            from tfm_energia.data.mongo_repository import MongoRepository

            repo = MongoRepository()
            repo.db.command("ping")
            repo.close()
        except Exception as exc:  # noqa: BLE001
            avisos.append(f"MongoDB no responde ({type(exc).__name__})")
    return avisos


def main() -> int:
    ap = argparse.ArgumentParser(description="Orquestador del proyecto")
    ap.add_argument("--estado", action="store_true", help="Muestra qué hay hecho y sale")
    ap.add_argument("--simular", action="store_true", help="Muestra el plan sin ejecutar")
    ap.add_argument("--reanudar", action="store_true", help="Omite las etapas ya completadas")
    ap.add_argument("--desde", default=None, help="Empieza en esta etapa")
    ap.add_argument("--solo", nargs="*", default=None, help="Solo estas etapas")
    ap.add_argument("--saltar", nargs="*", default=None, help="Omite estas etapas")
    ap.add_argument("--sin-mongo", action="store_true", help="Omite las etapas que usan MongoDB")
    ap.add_argument("--sin-descargas", action="store_true", help="Omite las descargas de APIs")
    ap.add_argument(
        "--continuar-si-falla", action="store_true",
        help="No detiene la ejecución cuando una etapa falla",
    )
    args = ap.parse_args()

    if args.estado:
        mostrar_estado()
        return 0

    saltar = list(args.saltar or [])
    if args.sin_mongo:
        saltar += [e.nombre for e in ETAPAS if e.requiere_mongo]
    if args.sin_descargas:
        saltar += [e.nombre for e in ETAPAS if e.requiere_token]

    try:
        etapas = seleccionar(solo=args.solo, desde=args.desde, saltar=saltar)
    except ValueError as exc:
        logger.error(str(exc))
        return 2

    if args.reanudar:
        etapas = [e for e in etapas if not e.completada()]
        if not etapas:
            logger.success("Todas las etapas seleccionadas están ya completas.")
            return 0

    minutos = sum(e.minutos_estimados for e in etapas)
    print(f"\n=== PLAN: {len(etapas)} etapas, ~{minutos:.0f} min ===")
    for i, e in enumerate(etapas, 1):
        extras = []
        if e.requiere_token:
            extras.append("token")
        if e.requiere_mongo:
            extras.append("mongo")
        sufijo = f"  [{', '.join(extras)}]" if extras else ""
        print(f"  {i}. {e.nombre:<20} ~{e.minutos_estimados:>4.0f} min  {e.descripcion}{sufijo}")

    for aviso in comprobar_requisitos(etapas):
        logger.warning(f"Requisito pendiente: {aviso}")

    if args.simular:
        print("\n(--simular: no se ha ejecutado nada)")
        return 0

    print()
    resultados: list[ResultadoEtapa] = []
    for etapa in etapas:
        resultado = ejecutar_etapa(etapa)
        resultados.append(resultado)
        if resultado.estado == "fallida" and not args.continuar_si_falla:
            logger.error(
                f"Se detiene en '{etapa.nombre}'. Usa --continuar-si-falla para seguir, "
                f"o --desde {etapa.nombre} para retomar aquí una vez resuelto."
            )
            break

    print("\n=== RESUMEN ===")
    for r in resultados:
        marca = {"ejecutada": "✔", "fallida": "✖", "omitida": "·"}.get(r.estado, "?")
        print(f"  {marca} {r.etapa:<20} {r.estado:<10} {r.segundos:>7.0f}s  {r.detalle}")

    fallidas = [r for r in resultados if r.estado == "fallida"]
    total = sum(r.segundos for r in resultados)
    print(f"\n  {len(resultados)} etapas en {total/60:.1f} min · {len(fallidas)} fallidas")

    if not fallidas:
        print("\n  Siguiente paso recomendado: python scripts/verificar_integridad.py")
    return 1 if fallidas else 0


if __name__ == "__main__":
    sys.exit(main())
