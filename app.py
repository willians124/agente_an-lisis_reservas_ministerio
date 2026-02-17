import streamlit as st
import duckdb
import pandas as pd
import os
from openai import OpenAI
import re

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(
    page_title="Data Intelligence",
    layout="wide"
)

st.title("📊 Data Intelligence Copilot")
st.caption("Explorador Analítico de Reservas Turísticas")

# -----------------------
# OPENAI CLIENT
# -----------------------

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY no configurada en Secrets.")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------
# LOAD DATA
# -----------------------

@st.cache_resource
def load_data():
    df = pd.read_excel("reservas.xlsx")
    df["fechaVisita"] = pd.to_datetime(df["fechaVisita"], errors="coerce")
    return df

df = load_data()

con = duckdb.connect()
con.register("data", df)

# -----------------------
# SESSION STATE
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "analysis_summary" not in st.session_state:
    st.session_state.analysis_summary = ""

# -----------------------
# MOSTRAR HISTORIAL
# -----------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# -----------------------
# INPUT
# -----------------------

if prompt := st.chat_input("Pregunta algo sobre los datos..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # -----------------------
    # STEP 1: GENERAR SQL
    # -----------------------

    schema_description = """
    Tabla: data

    Columnas disponibles:
    - estado_r (estado de reserva)
    - ruta
    - razon_social
    - totalvisitante
    - cant_bajas
    - fechaVisita
    - nidLugar
    """

    try:
        sql_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=250,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Genera únicamente SQL válido para DuckDB.
                    Usa la tabla 'data'.
                    Devuelve solo una consulta SELECT.
                    No uses markdown.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    {schema_description}

                    Pregunta:
                    {prompt}
                    """
                }
            ]
        )

        sql_query = sql_response.choices[0].message.content.strip()

        sql_query = re.sub(r"```.*?\n", "", sql_query)
        sql_query = sql_query.replace("```", "").strip()

        if not sql_query.lower().startswith("select"):
            raise Exception("SQL inválido")

    except:
        with st.chat_message("assistant"):
            st.write("No pude interpretar esa consulta.")
        st.stop()

    # -----------------------
    # STEP 2: EJECUTAR SQL
    # -----------------------

    try:
        result = con.execute(sql_query).df()
    except:
        with st.chat_message("assistant"):
            st.write("No pude ejecutar el análisis sobre los datos.")
        st.stop()

    if result.empty:
        with st.chat_message("assistant"):
            st.write("La consulta no devolvió resultados relevantes.")
        st.stop()

    # -----------------------
    # STEP 3: ANALISIS DEL DATO
    # -----------------------

    try:
        analysis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=600,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un analista de datos explorando un dataset turístico.

                    Usa el contexto previo si existe.
                    No des recomendaciones.
                    No sugieras acciones.
                    Solo analiza comportamiento del dato.

                    Enfócate en:
                    - Patrones
                    - Concentraciones
                    - Distribuciones
                    - Variaciones relevantes
                    - Relaciones entre variables
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    CONTEXTO PREVIO:
                    {st.session_state.analysis_summary}

                    NUEVA PREGUNTA:
                    {prompt}

                    RESULTADO OBTENIDO:
                    {result.head(25).to_string()}

                    Analiza estos resultados.
                    """
                }
            ]
        )

        analysis_text = analysis_response.choices[0].message.content

    except:
        with st.chat_message("assistant"):
            st.write("Ocurrió un error generando el análisis.")
        st.stop()

    # Mostrar respuesta
    with st.chat_message("assistant"):
        st.write(analysis_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": analysis_text}
    )

    # Actualizar memoria resumida (limitada para bajo costo)
    st.session_state.analysis_summary += "\n" + analysis_text[:400]
