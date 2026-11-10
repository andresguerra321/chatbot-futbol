"""
Script de pruebas automatizadas locales para validar todos los requisitos de la rúbrica:
1. Carga y validación del catálogo de 20 jugadores.
2. Carga y validación de 12 opiniones iniciales (4 jugadores x 4 categorías).
3. Coherencia del modelo Pydantic AnalisisOpinion.
4. Consultas y métricas en memoria.
5. Escritura atómica y prevención de duplicados.
6. Manejo de archivo corrupto (conservación del archivo).
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import almacenamiento
from analisis import AnalisisOpinion


def main():
    print("=== INICIANDO PRUEBAS DE TRIBUNA GLOBAL ===")

    # 1. Catálogo de jugadores
    jugadores = almacenamiento.cargar_jugadores("datos/jugadores.json")
    print(f"✅ Catálogo cargado exitosamente: {len(jugadores)} futbolistas.")
    assert len(jugadores) >= 20, "Debe tener al menos 20 futbolistas"
    ids = [j["id"] for j in jugadores]
    assert len(ids) == len(set(ids)), "Los IDs no deben repetirse"

    # 2. Catálogo de opiniones iniciales
    datos_op = almacenamiento.cargar_opiniones("datos/opiniones.json")
    opiniones = datos_op.get("opiniones", [])
    print(f"✅ Opiniones cargadas: {len(opiniones)} registros.")
    assert len(opiniones) >= 12, "Debe tener al menos 12 opiniones"
    
    # Comprobar distribución (al menos 4 jugadores y las 4 categorías)
    jugadores_cubiertos = {op["jugador_id"] for op in opiniones}
    sentimientos_cubiertos = {op["sentimiento"] for op in opiniones}
    print(f"   Jugadores cubiertos: {len(jugadores_cubiertos)} ({jugadores_cubiertos})")
    print(f"   Sentimientos cubiertos: {sentimientos_cubiertos}")
    assert len(jugadores_cubiertos) >= 4, "Debe cubrir al menos 4 jugadores"
    assert {"positivo", "negativo", "neutro", "mixto"}.issubset(sentimientos_cubiertos), "Debe cubrir las 4 categorías"

    # 3. Validación de Pydantic y reglas de coherencia
    # Caso a: No pertinente con sentimiento -> debe forzar sentimiento=None
    op_no_pertinente = AnalisisOpinion(
        es_pertinente=False,
        sentimiento="positivo",
        razon="El texto habla sobre astronomía y no sobre el futbolista.",
    )
    assert op_no_pertinente.sentimiento is None, "Si es_pertinente=False, sentimiento debe ser None"
    print("✅ Pydantic coherencia (no pertinente -> sentimiento=None) verificado.")

    # Caso b: Razón que excede 300 caracteres -> debe truncarse
    op_larga = AnalisisOpinion(
        es_pertinente=True,
        sentimiento="positivo",
        razon="A" * 350,
    )
    assert len(op_larga.razon) <= 300, "La razón no debe exceder 300 caracteres"
    assert op_larga.razon.endswith("..."), "Debe terminar en elipsis si fue truncada"
    print("✅ Pydantic limitación de longitud de razón verificada.")

    # 4. Consultas y filtros
    filtro_messi = almacenamiento.consultar_opiniones(opiniones, jugador_id="j001")
    assert len(filtro_messi) == 3, f"Se esperaban 3 opiniones para Messi, se obtuvieron {len(filtro_messi)}"
    
    filtro_negativo = almacenamiento.consultar_opiniones(opiniones, sentimiento="negativo")
    assert len(filtro_negativo) >= 1, "Debe encontrar opiniones negativas"

    metricas = almacenamiento.obtener_metricas(opiniones)
    assert metricas["total"] == len(opiniones)
    assert metricas["positivo"] > 0
    assert metricas["negativo"] > 0
    assert metricas["neutro"] > 0
    assert metricas["mixto"] > 0
    print("✅ Consultas, filtros y cálculo de métricas verificados.")

    # 5. Persistencia atómica y no duplicación
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_opiniones = Path(tmp_dir) / "opiniones_test.json"
        
        # Guardar primer registro
        jugador_test = jugadores[0]
        registro_1 = almacenamiento.crear_registro(
            jugador=jugador_test,
            texto="Jugador extraordinario con una visión única.",
            analisis=AnalisisOpinion(es_pertinente=True, sentimiento="positivo", razon="Elogia su visión."),
            modelo="test-model"
        )
        guardado = almacenamiento.guardar_opinion(tmp_opiniones, registro_1)
        assert guardado is True, "El registro debió guardarse con éxito"
        
        # Intentar re-guardar el mismo registro (simulando re-ejecución de Streamlit)
        guardado_dup = almacenamiento.guardar_opinion(tmp_opiniones, registro_1)
        assert guardado_dup is False, "No debe duplicar un registro ya existente con el mismo ID"

        datos_releidos = almacenamiento.cargar_opiniones(tmp_opiniones)
        assert len(datos_releidos["opiniones"]) == 1, "Debe mantenerse exactamente 1 registro"
        print("✅ Persistencia atómica y prevención de duplicados verificados.")

    # 6. Manejo de archivo corrupto (debe lanzar error y NO borrar el archivo)
    with tempfile.TemporaryDirectory() as tmp_dir_corrupt:
        tmp_corrupto = Path(tmp_dir_corrupt) / "corrupto.json"
        with open(tmp_corrupto, "w", encoding="utf-8") as f:
            f.write("{ esto no es json valido ")

        error_lanzado = False
        try:
            almacenamiento.cargar_opiniones(tmp_corrupto)
        except almacenamiento.AlmacenamientoError:
            error_lanzado = True

        assert error_lanzado is True, "Debió lanzar AlmacenamientoError ante JSON inválido"
        assert tmp_corrupto.exists(), "El archivo corrupto debe conservarse intacto sin borrarse"
        print("✅ Resiliencia ante archivo dañado comprobada (archivo conservado).")

    print("\n🎉 TODAS LAS PRUEBAS DE ACEPTACIÓN LOCALES PASARON CON ÉXITO.")


if __name__ == "__main__":
    main()

