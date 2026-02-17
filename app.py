import streamlit as st
import duckdb
import pandas as pd
import os
from openai import OpenAI
import re
import matplotlib.pyplot as plt

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(page_title="Data Intelligence Copilot", layout="wide")

st.title("📊 Data Intelligence Copilot")
st.caption("Explorador Analítico de Reservas Turísticas")

# -----------------------
# OPENAI
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

con = duckdb.connect()
con.register("data", df)

# -----------------------
# PANORAMA AUTOMÁTICO
# -----------------------

if "auto_overview" not in st.session_state:
    with st.spinner("Analizando estructura general del dataset..."):
        resumen = df.groupby("estado_r")["nidreserva"].count().reset_index()

        overview = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=350,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un analista senior.
                    Describe el panorama general del dataset.
                    No des recomendaciones.
                    Solo interpreta patrones visibles.
                    """
                },
                {
                    "role": "user",
                    "content": f"""
                    Distribución por estado:
                    {resumen.to_string()}
                    """
                }
            ]
        )

        st.subheader("📊 Panorama General")
        st.write(overview.choices[0].message.content)

        st.session_state.auto_overview = True

st.divider()

# -----------------------
# SUGERENCIAS
# -----------------------

st.markdown("### 🔎 Sugerencias de Análisis")

col1, col2, col3 = st.columns(3)

if col1.button("Distribución por Estado"):
    st.session_state.prefill = "Analiza la distribución de reservas por estado."

if col2.button("Top Agencias"):
    st.session_state.prefill = "Analiza la concentración de visitantes por agencia."

if col3.button("Tendencia Temporal"):
    st.session_state.prefill = "Evalúa el comportamiento mensual de visitantes."

# -----------------------
# INPUT
# -----------------------

prompt = st.chat_input("Pregunta algo sobre los datos...")

if "prefill" in st.session_state:
    prompt = st.session_state.prefill
    del st.session_state.prefill

if prompt:

    st.divider()
    st.markdown("## 🧠 Consulta")

    # -----------------------
    # SCHEMA
    # -----------------------

    schema_description = """
    Tabla: data

    Columnas:
    - estado_r
    - ruta
    - razon_social
    - totalvisitante
    - cant_bajas
    - fechaVisita
    - nidLugar
    """

    # -----------------------
    # GENERAR SQL
    # -----------------------

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
                Devuelve solo SELECT.
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
        st.error("No se pudo generar una consulta válida.")
        st.stop()

    # -----------------------
    # EJECUTAR SQL
    # -----------------------

    try:
        result = con.execute(sql_query).df()
    except Exception as e:
        st.error(f"Error ejecutando SQL: {e}")
        st.stop()

    if result.empty:
        st.warning("La consulta no devolvió resultados.")
        st.stop()

    # -----------------------
    # MOSTRAR SQL
    # -----------------------

    with st.expander("🧾 Ver SQL generado"):
        st.code(sql_query, language="sql")

    # -----------------------
    # MOSTRAR TABLA
    # -----------------------

    st.subheader("📋 Resultado")
    st.dataframe(result, use_container_width=True)

    # -----------------------
    # GRAFICACIÓN INTELIGENTE
    # -----------------------

    if result.shape[1] == 2:

        st.subheader("📈 Visualización Automática")

        col_x = result.columns[0]
        col_y = result.columns[1]

        # Si fecha → línea
        if pd.api.types.is_datetime64_any_dtype(result[col_x]):
            fig, ax = plt.subplots(figsize=(10,5))
            ax.plot(result[col_x], result[col_y])
            ax.set_title("Tendencia Temporal")
            plt.xticks(rotation=45)
            st.pyplot(fig)

        else:
            result_sorted = result.sort_values(by=col_y, ascending=False)
            fig_width = max(8, len(result_sorted) * 0.6)
            fig, ax = plt.subplots(figsize=(fig_width,5))

            ax.bar(result_sorted[col_x].astype(str),
                   result_sorted[col_y])

            if len(result_sorted) > 6:
                plt.xticks(rotation=60, ha="right")
            else:
                plt.xticks(rotation=0)

            st.pyplot(fig)

    # -----------------------
    # ANALISIS PROFUNDO
    # -----------------------

    analysis_response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=600,
        messages=[
            {
                "role": "system",
                "content": """
                Eres un analista senior de datos.

                Analiza profundamente el resultado considerando:
                1. Patrón dominante
                2. Concentración o dispersión
                3. Magnitudes relativas
                4. Variaciones relevantes
                5. Posibles anomalías visibles

                No des recomendaciones.
                Solo análisis técnico interpretativo.
                """
            },
            {
                "role": "user",
                "content": f"""
                Pregunta:
                {prompt}

                Resultado:
                {result.head(25).to_string()}

                Analiza.
                """
            }
        ]
    )

    st.subheader("🧠 Análisis Interpretativo")
    st.write(analysis_response.choices[0].message.content)
