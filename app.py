import streamlit as st
import duckdb
import pandas as pd
import os
from openai import OpenAI
import re

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(page_title="Data Exploration Copilot", layout="wide")
st.title("🧠 Data Exploration Copilot")

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY no configurada.")
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
# CHAT STATE
# -----------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar historial
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Input usuario
if prompt := st.chat_input("Haz una pregunta sobre los datos..."):

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.write(prompt)

    # -----------------------
    # PASO 1: GENERAR SQL INTERNO
    # -----------------------

    schema_description = """
    Tabla: data

    Columnas principales:
    - estado_r
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
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Genera únicamente SQL válido para DuckDB.
                    Usa la tabla 'data'.
                    No uses markdown.
                    Solo devuelve la consulta SELECT.
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
            raise Exception("SQL inválido generado.")

    except Exception as e:
        with st.chat_message("assistant"):
            st.write("No pude interpretar esa consulta.")
        st.stop()

    # -----------------------
    # PASO 2: EJECUTAR SQL
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
    # PASO 3: ANALISIS DEL DATO
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

                    Analiza el comportamiento de los datos de manera clara.

                    - Identifica patrones
                    - Señala concentraciones relevantes
                    - Detecta variaciones importantes
                    - Describe tendencias si existen
                    - Interpreta relaciones si se observan

                    No des recomendaciones.
                    No sugieras acciones.
                    Solo analiza los datos.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Pregunta original:
                    {prompt}

                    Resultado obtenido:
                    {result.head(25).to_string()}

                    Analiza estos resultados.
                    """
                }
            ]
        )

        analysis_text = analysis_response.choices[0].message.content

    except Exception as e:
        with st.chat_message("assistant"):
            st.write("Ocurrió un error generando el análisis.")
        st.stop()

    with st.chat_message("assistant"):
        st.write(analysis_text)

    st.session_state.messages.append(
        {"role": "assistant", "content": analysis_text}
    )
