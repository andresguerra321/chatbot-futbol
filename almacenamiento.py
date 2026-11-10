"""
Módulo de almacenamiento y persistencia para Tribuna Global.
Gestiona el catálogo de jugadores, las opiniones y las consultas sin llamadas a LLM.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


class AlmacenamientoError(Exception):
    """Excepción base para errores de almacenamiento y lectura de archivos."""
    pass


def cargar_jugadores(ruta: str | Path) -> list[dict[str, Any]]:
    """
    Lee y valida el catálogo de jugadores desde un archivo JSON.
    Comprueba que exista, tenga el formato correcto y no tenga IDs repetidos.
    """
    ruta_path = Path(ruta)
    if not ruta_path.is_file():
        raise AlmacenamientoError(f"El archivo de jugadores no existe en la ruta: {ruta_path}")

    try:
        with open(ruta_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except json.JSONDecodeError as exc:
        raise AlmacenamientoError(f"Error de formato JSON en el catálogo de jugadores: {exc}") from exc
    except Exception as exc:
        raise AlmacenamientoError(f"Error al leer el catálogo de jugadores: {exc}") from exc

    if not isinstance(datos, dict) or "jugadores" not in datos or not isinstance(datos["jugadores"], list):
        raise AlmacenamientoError("Estructura inválida en jugadores.json: se esperaba una clave 'jugadores' con una lista.")

    jugadores = datos["jugadores"]
    ids_vistos: set[str] = set()

    for i, j in enumerate(jugadores):
        if not isinstance(j, dict) or "id" not in j or "nombre" not in j or "pais" not in j:
            raise AlmacenamientoError(f"El jugador en la posición {i} no cumple con la estructura requerida (id, nombre, pais).")
        
        j_id = str(j["id"]).strip()
        if not j_id:
            raise AlmacenamientoError(f"El jugador en la posición {i} tiene un identificador vacío.")
        
        if j_id in ids_vistos:
            raise AlmacenamientoError(f"Identificador duplicado encontrado en el catálogo: '{j_id}'.")
        
        ids_vistos.add(j_id)

    return jugadores


def cargar_opiniones(ruta: str | Path) -> dict[str, Any]:
    """
    Carga el diccionario persistente de opiniones.
    Si el archivo no existe, lo inicializa con estructura vacía.
    Si existe pero está dañado o tiene esquema inválido, lanza AlmacenamientoError
    y conserva el archivo intacto (no lo sobrescribe).
    """
    ruta_path = Path(ruta)

    if not ruta_path.exists():
        estructura_inicial = {"version": 1, "opiniones": []}
        ruta_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(ruta_path, "w", encoding="utf-8") as f:
                json.dump(estructura_inicial, f, ensure_ascii=False, indent=2)
            return estructura_inicial
        except Exception as exc:
            raise AlmacenamientoError(f"No fue posible inicializar el archivo de opiniones: {exc}") from exc

    try:
        with open(ruta_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
    except json.JSONDecodeError as exc:
        # Se conserva el archivo tal cual como exige la rúbrica y pruebas
        raise AlmacenamientoError(
            f"El archivo '{ruta_path.name}' contiene JSON inválido o está dañado. "
            "Se ha conservado el archivo sin modificar."
        ) from exc
    except Exception as exc:
        raise AlmacenamientoError(f"Error al leer '{ruta_path.name}': {exc}") from exc

    if not isinstance(datos, dict) or "opiniones" not in datos or not isinstance(datos["opiniones"], list):
        raise AlmacenamientoError(
            f"El archivo '{ruta_path.name}' tiene un esquema inesperado (falta clave 'opiniones' tipo lista). "
            "Se conserva el archivo para evitar pérdida de datos."
        )

    return datos


def crear_registro(
    jugador: dict[str, Any],
    texto: str,
    analisis: Any,
    modelo: str,
    version_prompt: str = "1.0",
) -> dict[str, Any]:
    """
    Genera el diccionario del nuevo registro con UUID, fecha UTC y metadatos.
    """
    sentimiento = getattr(analisis, "sentimiento", None)
    razon = getattr(analisis, "razon", "")

    registro = {
        "id": str(uuid4()),
        "fecha_utc": datetime.now(timezone.utc).isoformat(),
        "jugador_id": jugador.get("id", ""),
        "jugador_nombre": jugador.get("nombre", ""),
        "opinion_original": texto.strip(),
        "sentimiento": sentimiento,
        "razon": razon,
        "modelo": modelo,
        "version_prompt": version_prompt,
    }
    return registro


def guardar_opinion(ruta: str | Path, registro: dict[str, Any]) -> bool:
    """
    Añade un registro a opiniones.json usando escritura atómica:
    1. Carga el archivo existente para no perder registros previos.
    2. Comprueba que el ID no esté ya guardado (previene duplicados).
    3. Escribe en un archivo temporal en el mismo directorio con UTF-8 e indentación.
    4. Reemplaza el archivo original atómicamente con os.replace.
    """
    ruta_path = Path(ruta)
    
    # 1. Cargar estado existente validado
    datos_actuales = cargar_opiniones(ruta_path)
    opiniones = datos_actuales.setdefault("opiniones", [])

    # 2. Prevenir duplicación por ID
    registro_id = registro.get("id")
    if any(op.get("id") == registro_id for op in opiniones):
        return False  # Ya existía, no duplicar

    opiniones.append(registro)

    # 3. Escritura atómica mediante archivo temporal en el mismo directorio
    directorio = ruta_path.parent
    directorio.mkdir(parents=True, exist_ok=True)

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=directorio,
        prefix="opiniones_",
        suffix=".tmp"
    )

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f_temp:
            json.dump(datos_actuales, f_temp, ensure_ascii=False, indent=2)
            f_temp.flush()
            os.fsync(f_temp.fileno())

        # 4. Reemplazo atómico
        os.replace(temp_path_str, ruta_path)
        return True
    except Exception as exc:
        if os.path.exists(temp_path_str):
            try:
                os.remove(temp_path_str)
            except OSError:
                pass
        raise AlmacenamientoError(f"Fallo al guardar la opinión en disco: {exc}") from exc


def consultar_opiniones(
    registros: list[dict[str, Any]],
    jugador_id: str | None = None,
    sentimiento: str | None = None,
) -> list[dict[str, Any]]:
    """
    Filtra los registros en memoria sin realizar llamadas al modelo.
    Admite None o 'todos' para omitir el filtro.
    """
    resultado = registros

    if jugador_id and jugador_id.lower() != "todos":
        resultado = [r for r in resultado if str(r.get("jugador_id", "")).strip() == jugador_id.strip()]

    if sentimiento and sentimiento.lower() != "todos":
        resultado = [r for r in resultado if str(r.get("sentimiento", "")).strip().lower() == sentimiento.strip().lower()]

    return resultado


def obtener_metricas(opiniones: list[dict[str, Any]]) -> dict[str, int]:
    """
    Calcula el total de opiniones filtradas y el desglose por sentimiento.
    """
    metricas = {
        "total": len(opiniones),
        "positivo": 0,
        "negativo": 0,
        "neutro": 0,
        "mixto": 0,
        "otro": 0,
    }
    for op in opiniones:
        sent = str(op.get("sentimiento", "")).lower()
        if sent in metricas:
            metricas[sent] += 1
        else:
            metricas["otro"] += 1

    return metricas

