import streamlit as st
import duckdb
import pandas as pd
import os
import matplotlib.pyplot as plt
from openai import OpenAI
import re

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(
    page_title="Executive Reservation Copilot",
    layout="wide"
)

st.title("📊 Executive Reservation Intelligence Copilot")

# -----------------------
# OPENAI CLIENT
# -----------------------

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

# -----------------------
# DUCKDB
# -----------------------

con = duckdb.connect()
con.register("data", df)

# -----------------------
# METRICS
# -----------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reservas", f"{df.shape[0]:,}")
col2.metric("Total Visitantes", f"{int(df['totalvisitante'].sum()):,}")
col3.metric("Reservas Pagadas", f"{int((df['estado_r']=='Pagado').sum()):,}")
col4.metric("Reservas Anuladas", f"{int((df['estado_r']=='Anulado').sum()):,}")

st.divider()

# -----------------------
# USER INPUT
# -----------------------

st.subheader("💬 Consulta")

user_prompt = st.text_area("Haz tu pregunta:")

if st.button("Analizar"):

    if not user_prompt.strip():
        st.warning("Escribe una pregunta.")
        st.stop()

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

    # -----------------------
    # STEP 1: GENERAR SQL
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
                    No uses markdown.
                    Solo devuelve la consulta.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    {schema_description}

                    Pregunta:
                    {user_prompt}
                    """
                }
            ]
        )

        sql_query = sql_response.choices[0].message.content.strip()

    except Exception as e:
        st.error(f"Error generando SQL: {e}")
        st.stop()

    # Limpiar posibles bloques markdown
    sql_query = re.sub(r"```.*?\n", "", sql_query)
    sql_query = sql_query.replace("```", "").strip()

    if not sql_query.lower().startswith("select"):
        st.error("La consulta generada no es válida.")
        st.stop()

    st.code(sql_query, language="sql")

    # -----------------------
    # STEP 2: EJECUTAR SQL
    # -----------------------

    try:
        result = con.execute(sql_query).df()
    except Exception as e:
        st.error(f"Error ejecutando SQL: {e}")
        st.stop()

    if result.empty:
        st.warning("No hay resultados.")
        st.stop()

    st.subheader("📋 Resultado")
    st.dataframe(result, use_container_width=True)

    # -----------------------
    # GRÁFICO INTELIGENTE
    # -----------------------

    if result.shape[1] == 2:

        st.subheader("📈 Visualización")

        col_x = result.columns[0]
        col_y = result.columns[1]

        result_sorted = result.sort_values(by=col_y, ascending=False)

        fig_width = max(8, len(result_sorted) * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, 5))

        ax.bar(
            result_sorted[col_x].astype(str),
            result_sorted[col_y]
        )

        if len(result_sorted) > 6:
            plt.xticks(rotation=60, ha="right")
        else:
            plt.xticks(rotation=0)

        plt.tight_layout()
        st.pyplot(fig)

    # -----------------------
    # STEP 3: ANALISIS DEL DATO
    # -----------------------

    try:
        analysis_response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un analista de datos.
                    Interpreta patrones numéricos.
                    No des recomendaciones.
                    No sugieras acciones.
                    Solo analiza comportamiento del dato.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Pregunta:
                    {user_prompt}

                    Resultado:
                    {result.head(20).to_string()}

                    Analiza los datos.
                    """
                }
            ]
        )

        analysis_text = analysis_response.choices[0].message.content

    except Exception as e:
        st.error(f"Error generando análisis: {e}")
        st.stop()

    st.subheader("🧠 Análisis del Dato")
    st.write(analysis_text)
