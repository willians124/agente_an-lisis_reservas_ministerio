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
    page_title="Data Intelligence Copilot",
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
# MOSTRAR HISTORIAL CHAT
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
    # SCHEMA DETALLADO
    # -----------------------

    schema_description = """
    Tabla: data

    Columnas:

    - nidreserva (id reserva)
    - scodigo_reserva (codigo único)
    - estado_r (estado de la reserva)
        Valores posibles:
        - Reservado
        - Pagado
        - Anulado
        - Vencido
        - Cerrado
        - Fusionado
        - Penalizado

    - ruta (nombre de ruta turística)

    - razon_social (agencia o empresa)

    - nguia (cantidad de guías asignados)
    - npa_cocinero (cantidad de cocineros)
    - npa_ayudante (cantidad de ayudantes)
    - npa_porteador (cantidad de porteadores)

    - totalvisitante (total de visitantes en la reserva)
    - cant_bajas (cantidad de bajas/cancelaciones parciales)

    - campamentos
    - guias

    - fechaVisita (fecha programada de visita)

    - nidLugar (lugar turístico)
        Valores posibles:
        - 1 = Llaqta Machupicchu
        - 2 = Red de Camino Inka
    """

    # -----------------------
    # GENERAR SQL
    # -----------------------

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
    # EJECUTAR SQL
    # -----------------------

    try:
        result = con.execute(sql_query).df()
    except:
        with st.chat_message("assistant"):
            st.write("No pude ejecutar el análisis.")
        st.stop()

    if result.empty:
        with st.chat_message("assistant"):
            st.write("La consulta no devolvió resultados.")
        st.stop()

    # -----------------------
    # MOSTRAR RESULTADOS
    # -----------------------

    st.divider()

    st.subheader("🧾 Consulta Ejecutada")
    with st.expander("Ver SQL generado"):
        st.code(sql_query, language="sql")

    st.subheader("📋 Resultado del Análisis")
    st.dataframe(result, use_container_width=True)

    # -----------------------
    # ANALISIS DEL DATO
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
                    Eres un analista de datos.
                    Interpreta únicamente el comportamiento numérico.
                    No des recomendaciones.
                    No sugieras acciones.
                    Solo describe patrones, concentraciones, variaciones y relaciones.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Pregunta:
                    {prompt}

                    Resultado:
                    {result.head(25).to_string()}

                    Analiza los datos.
                    """
                }
            ]
        )

        analysis_text = analysis_response.choices[0].message.content

    except:
        with st.chat_message("assistant"):
            st.write("Error generando el análisis.")
        st.stop()

    st.subheader("🧠 Análisis Interpretativo")

    st.markdown(
        f"""
        <div style="
            padding:20px;
            border-radius:10px;
            background-color:#f4f6f8;
            border-left:6px solid #2E86C1;
        ">
        {analysis_text}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.analysis_summary += "\n" + analysis_text[:400]

    st.session_state.messages.append(
        {"role": "assistant", "content": analysis_text}
    )
