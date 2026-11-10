"""
Aplicación Principal: Tribuna Global
Chatbot para análisis de percepción de futbolistas con LangChain y Streamlit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st

load_dotenv(override=True)

from almacenamiento import (
    AlmacenamientoError,
    cargar_jugadores,
    cargar_opiniones,
    consultar_opiniones,
    crear_registro,
    guardar_opinion,
    obtener_metricas,
)
from analisis import AnalisisOpinion, analizar_opinion, obtener_cliente_llm

# Configuración de página
st.set_page_config(
    page_title="Tribuna Global - Percepción de Futbolistas",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Definición de rutas absolutas basadas en la ubicación del archivo
BASE_DIR = Path(__file__).resolve().parent
RUTA_JUGADORES = BASE_DIR / "datos" / "jugadores.json"
RUTA_OPINIONES = BASE_DIR / "datos" / "opiniones.json"

# Estilos e insignias para sentimientos
BADGES_SENTIMIENTO = {
    "positivo": "🟢 Positivo",
    "negativo": "🔴 Negativo",
    "neutro": "⚪ Neutro",
    "mixto": "🟡 Mixto",
}


def inicializar_estado():
    """Inicializa las variables de sesión requeridas por Streamlit."""
    if "mensajes_chat" not in st.session_state:
        st.session_state.mensajes_chat = []
    if "ultimo_error" not in st.session_state:
        st.session_state.ultimo_error = None


def limpiar_conversacion():
    """Limpia el historial visual de la sesión sin alterar opiniones.json."""
    st.session_state.mensajes_chat = []
    st.session_state.ultimo_error = None
    st.toast("Chat de sesión reiniciado. Los datos guardados se mantienen intactos.", icon="🧹")


def main():
    inicializar_estado()

    # --- BARRA LATERAL ---
    st.sidebar.title("⚽ Tribuna Global")
    st.sidebar.caption("Análisis de sentimiento con LangChain")

    # Configuración de API Key (permite ingresar en UI o leer de .env)
    groq_env = os.getenv("GROQ_API_KEY", "")
    api_key_sidebar = st.sidebar.text_input(
        "Groq API Key (opcional si está en .env):",
        value=groq_env if not groq_env.startswith("gsk_tu_clave") else "",
        type="password",
        help="Introduce tu clave de Groq (gsk_...) para clasificar opiniones.",
    )
    if api_key_sidebar:
        os.environ["GROQ_API_KEY"] = api_key_sidebar.strip()

    # Selector de modelo de Groq activo
    modelo_actual = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    modelos_opciones = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
    indice_modelo = modelos_opciones.index(modelo_actual) if modelo_actual in modelos_opciones else 0
    modelo_elegido = st.sidebar.selectbox(
        "Modelo de IA (Groq):",
        options=modelos_opciones,
        index=indice_modelo,
        help="Modelos activos y verificados en tu cuenta de Groq."
    )
    os.environ["GROQ_MODEL"] = modelo_elegido

    # Modo de prueba offline/demo
    modo_demo = st.sidebar.checkbox(
        "🧪 Modo prueba local (sin API)",
        value=False,
        help="Permite probar la interfaz, guardado en JSON y consultas completas sin consumir la API de Groq."
    )

    st.sidebar.divider()

    # Carga del catálogo de jugadores
    try:
        catalogo_jugadores = cargar_jugadores(RUTA_JUGADORES)
    except AlmacenamientoError as exc:
        st.sidebar.error(f"Error en catálogo: {exc}")
        st.error(f"No fue posible cargar el catálogo de jugadores: {exc}")
        return

    # Selector de jugador (Actividad 1: muestra nombre y país, usa ID internamente)
    opciones_jugador = {
        f"{j['nombre']} ({j['pais']})": j for j in catalogo_jugadores
    }
    etiqueta_seleccionada = st.sidebar.selectbox(
        "Selecciona un futbolista:",
        options=list(opciones_jugador.keys()),
        index=0,
        help="El análisis se vinculará estrictamente al jugador seleccionado al momento de enviar la opinión."
    )
    jugador_activo = opciones_jugador[etiqueta_seleccionada]

    st.sidebar.info(
        f"**Jugador activo:** {jugador_activo['nombre']}\n\n"
        f"**País:** {jugador_activo['pais']}\n\n"
        f"**ID:** `{jugador_activo['id']}`"
    )

    st.sidebar.divider()
    if st.sidebar.button("🗑️ Nueva conversación", on_click=limpiar_conversacion, use_container_width=True):
        pass

    # --- SECCIÓN PRINCIPAL: PESTAÑAS ---
    tab_emitir, tab_consultar = st.tabs(["💬 Emitir opinión", "📊 Consultar opiniones"])

    # ==========================================
    # PESTAÑA 1: EMITIR OPINIÓN
    # ==========================================
    with tab_emitir:
        st.subheader("Emitir opinión sobre un futbolista")
        st.caption(
            f"Escribe tu opinión honesta sobre **{jugador_activo['nombre']}** (entre 5 y 1000 caracteres). "
            "El modelo evaluará pertinencia, sentimiento (positivo, negativo, neutro o mixto) y justificación."
        )

        # Mostrar historial visual de la sesión actual
        for msg in st.session_state.mensajes_chat:
            with st.chat_message(msg["rol"]):
                if msg["rol"] == "user":
                    st.markdown(f"**[{msg['jugador_nombre']}]**: {msg['contenido']}")
                else:
                    if msg.get("es_pertinente"):
                        badge = BADGES_SENTIMIENTO.get(msg.get("sentimiento"), "⚪")
                        st.markdown(f"**Clasificación:** {badge}")
                        st.markdown(f"**Razón:** {msg.get('razon')}")
                        if msg.get("guardado"):
                            st.caption(f"💾 *Guardado en disco con ID: `{msg.get('registro_id', '')[:8]}...`*")
                    else:
                        st.warning(f"⚠️ **Mensaje no pertinente:** {msg.get('razon')}")
                        st.caption("ℹ️ *Esta opinión no fue guardada en el catálogo porque no corresponde al jugador.*")

        # Captura de nueva opinión
        opinion_input = st.chat_input("Escribe tu opinión sobre el jugador seleccionado...")

        if opinion_input is not None:
            opinion_limpia = opinion_input.strip()

            # Validación de entrada (Actividad 4 y casos de prueba)
            if not opinion_limpia:
                st.warning("⚠️ La opinión no puede estar vacía ni contener solo espacios en blanco.")
            elif len(opinion_limpia) < 5 or len(opinion_limpia) > 1000:
                st.warning(
                    f"⚠️ La opinión debe tener entre 5 y 1000 caracteres. (Longitud actual: {len(opinion_limpia)})."
                )
            else:
                # 1. Registrar mensaje de usuario en sesión
                st.session_state.mensajes_chat.append({
                    "rol": "user",
                    "contenido": opinion_limpia,
                    "jugador_id": jugador_activo["id"],
                    "jugador_nombre": jugador_activo["nombre"],
                })

                # Mostrar inmediatamente el mensaje de usuario
                with st.chat_message("user"):
                    st.markdown(f"**[{jugador_activo['nombre']}]**: {opinion_limpia}")

                # 2. Análisis con IA (LangChain + Groq)
                with st.chat_message("assistant"):
                    with st.spinner(f"Analizando opinión sobre {jugador_activo['nombre']}..."):
                        try:
                            analisis, modelo_utilizado = analizar_opinion(
                                jugador=jugador_activo,
                                texto=opinion_limpia,
                                nombre_modelo=modelo_elegido,
                                modo_simulacion=modo_demo,
                            )
                        except Exception as exc:
                            mensaje_err = str(exc)
                            if "401" in mensaje_err or "invalid_api_key" in mensaje_err.lower():
                                st.error("❌ **Error 401 de Groq: Tu API Key no es válida o fue rechazada.**")
                                st.warning(
                                    "La clave configurada (`gsk_...`) no existe o fue revocada en los servidores de Groq.\n\n"
                                    "👉 **¿Cómo continuar ya mismo?**\n"
                                    "- **Opción 1:** En la barra lateral izquierda, activa la casilla **`🧪 Modo prueba local (sin API)`** para usar el sistema inmediatamente sin depender de Groq.\n"
                                    "- **Opción 2:** Entra a [console.groq.com/keys](https://console.groq.com/keys), crea una clave nueva y pégala en `.env` o en la barra lateral."
                                )
                            else:
                                st.error(f"❌ Fallo al clasificar la opinión: {exc}")
                            st.info("💡 Tu mensaje se conserva en pantalla para que no lo pierdas.")
                            return

                        # 3. Procesar resultado
                        if not analisis.es_pertinente:
                            st.warning(f"⚠️ **Mensaje no pertinente:** {analisis.razon}")
                            st.caption("ℹ️ No se guardó en el archivo JSON por no referirse al jugador seleccionado.")

                            st.session_state.mensajes_chat.append({
                                "rol": "assistant",
                                "es_pertinente": False,
                                "sentimiento": None,
                                "razon": analisis.razon,
                                "guardado": False,
                            })
                        else:
                            badge = BADGES_SENTIMIENTO.get(analisis.sentimiento, "⚪ Neutro")
                            st.markdown(f"**Clasificación:** {badge}")
                            st.markdown(f"**Razón:** {analisis.razon}")

                            # 4. Guardar en disco mediante persistencia atómica
                            try:
                                registro = crear_registro(
                                    jugador=jugador_activo,
                                    texto=opinion_limpia,
                                    analisis=analisis,
                                    modelo=modelo_utilizado,
                                )
                                exito = guardar_opinion(RUTA_OPINIONES, registro)
                                if exito:
                                    st.success("✅ Opinión analizada y guardada exitosamente en `opiniones.json`.")
                                    st.caption(f"Registro persistido: `{registro['id']}`")
                                else:
                                    st.info("ℹ️ Registro ya existente; no se duplicó.")

                                st.session_state.mensajes_chat.append({
                                    "rol": "assistant",
                                    "es_pertinente": True,
                                    "sentimiento": analisis.sentimiento,
                                    "razon": analisis.razon,
                                    "guardado": exito,
                                    "registro_id": registro["id"],
                                })
                            except AlmacenamientoError as exc_guardado:
                                st.error(f"❌ Error al guardar el registro en disco: {exc_guardado}")
                                # No mostrar confirmación si falló la escritura
                                st.session_state.mensajes_chat.append({
                                    "rol": "assistant",
                                    "es_pertinente": True,
                                    "sentimiento": analisis.sentimiento,
                                    "razon": analisis.razon,
                                    "guardado": False,
                                })

    # ==========================================
    # PESTAÑA 2: CONSULTAR OPINIONES
    # ==========================================
    with tab_consultar:
        st.subheader("Consultas y estadísticas del registro persistente")
        st.caption(
            "Consulta las opiniones guardadas en `opiniones.json` filtrando por jugador y sentimiento. "
            "Esta vista no realiza ninguna llamada al modelo de IA."
        )

        try:
            datos_persistidos = cargar_opiniones(RUTA_OPINIONES)
            todas_opiniones = datos_persistidos.get("opiniones", [])
        except AlmacenamientoError as exc:
            st.error(f"Error al cargar las opiniones guardadas: {exc}")
            return

        # Filtros de consulta
        col_f1, col_f2, col_f3 = st.columns([2, 2, 1])

        with col_f1:
            opciones_filtro_jugador = ["Todos"] + [f"{j['nombre']} ({j['id']})" for j in catalogo_jugadores]
            seleccion_f_jugador = st.selectbox("Filtrar por jugador:", opciones_filtro_jugador)
            jugador_id_filtro = None
            if seleccion_f_jugador != "Todos":
                # Extraer id entre paréntesis
                jugador_id_filtro = seleccion_f_jugador.split("(")[-1].replace(")", "").strip()

        with col_f2:
            opciones_filtro_sentimiento = ["Todos", "positivo", "negativo", "neutro", "mixto"]
            seleccion_f_sentimiento = st.selectbox("Filtrar por sentimiento:", opciones_filtro_sentimiento)
            sentimiento_filtro = None if seleccion_f_sentimiento == "Todos" else seleccion_f_sentimiento

        with col_f3:
            st.write("")
            st.write("")
            if st.button("🔄 Actualizar", help="Vuelve a leer el archivo opiniones.json desde disco", use_container_width=True):
                st.rerun()

        # Aplicar filtros
        opiniones_filtradas = consultar_opiniones(
            registros=todas_opiniones,
            jugador_id=jugador_id_filtro,
            sentimiento=sentimiento_filtro,
        )

        # Ordenar por defecto de más reciente a más antigua para que las nuevas aparezcan arriba
        opiniones_a_mostrar = sorted(
            opiniones_filtradas,
            key=lambda x: str(x.get("fecha_utc", "")),
            reverse=True,
        )

        # Métricas resumidas
        metricas = obtener_metricas(opiniones_filtradas)

        m_col0, m_col1, m_col2, m_col3, m_col4 = st.columns(5)
        m_col0.metric("Total registradas", metricas["total"])
        m_col1.metric("🟢 Positivas", metricas["positivo"])
        m_col2.metric("🔴 Negativas", metricas["negativo"])
        m_col3.metric("⚪ Neutras", metricas["neutro"])
        m_col4.metric("🟡 Mixtas", metricas["mixto"])

        st.divider()

        # Visualización de resultados
        if not opiniones_a_mostrar:
            st.info("ℹ️ No se encontraron opiniones registradas con los filtros seleccionados.")
        else:
            st.caption(f"Mostrando **{len(opiniones_a_mostrar)}** opiniones (ordenadas de más reciente a más antigua):")
            # Preparar datos formateados para st.dataframe
            tabla_datos = []
            for op in opiniones_a_mostrar:
                tabla_datos.append({
                    "Fecha (UTC)": op.get("fecha_utc", "")[:19].replace("T", " "),
                    "Jugador": op.get("jugador_nombre", ""),
                    "Sentimiento": BADGES_SENTIMIENTO.get(op.get("sentimiento"), op.get("sentimiento")),
                    "Opinión original": op.get("opinion_original", ""),
                    "Justificación": op.get("razon", ""),
                    "Modelo": op.get("modelo", ""),
                })

            st.dataframe(
                tabla_datos,
                use_container_width=True,
                hide_index=True,
                height=450,
            )

            # Botón para descargar resultados filtrados en JSON (Actividad 6)
            json_exportable = json.dumps(
                {"total": len(opiniones_filtradas), "opiniones": opiniones_filtradas},
                ensure_ascii=False,
                indent=2,
            )

            st.download_button(
                label="📥 Descargar resultados filtrados (JSON)",
                data=json_exportable,
                file_name="opiniones_filtradas.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()

