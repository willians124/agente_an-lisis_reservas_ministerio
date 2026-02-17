import streamlit as st
import duckdb
import pandas as pd
import os
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------

st.set_page_config(page_title="Executive Reservation Copilot",
                   layout="wide")

st.title("📊 Executive Reservation Intelligence Copilot")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# LOAD DATA (CACHED)
# -----------------------

@st.cache_resource
def load_data():
    df = pd.read_excel("reservas.xlsx")
    df["fechaVisita"] = pd.to_datetime(df["fechaVisita"])
    return df

df = load_data()

# -----------------------
# DUCKDB
# -----------------------

con = duckdb.connect()
con.register("data", df)

# -----------------------
# EXECUTIVE DASHBOARD HEADER
# -----------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reservas", f"{df.shape[0]:,}")
col2.metric("Total Visitantes", f"{df['totalvisitante'].sum():,}")
col3.metric("Reservas Pagadas", f"{(df['estado_r']=='Pagado').sum():,}")
col4.metric("Reservas Anuladas", f"{(df['estado_r']=='Anulado').sum():,}")

st.divider()

# -----------------------
# CHAT SECTION
# -----------------------

st.subheader("💬 Consulta Ejecutiva")

user_prompt = st.text_area(
    "Haz una pregunta estratégica sobre las reservas:",
    placeholder="Ej: ¿Qué rutas generan más volumen? ¿Hay anomalías en visitantes? ¿Qué agencias tienen más anulaciones?"
)

if st.button("Analizar"):

    schema_description = """
    Tabla: data

    Columnas:
    - nidreserva (id reserva)
    - scodigo_reserva (codigo)
    - estado_r (Pagado / Anulado)
    - ruta (nombre de ruta)
    - razon_social (agencia)
    - nguia (cantidad guias)
    - npa_cocinero
    - npa_ayudante
    - npa_porteador
    - totalvisitante
    - cant_bajas
    - campamentos
    - guias
    - fechaVisita (fecha)
    - nidLugar
    """

    # -----------------------
    # STEP 1: GENERATE SQL
    # -----------------------

    sql_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
                Eres un experto analista de datos.
                Genera SOLO SQL compatible con DuckDB.
                Usa la tabla llamada data.
                No expliques.
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

    st.code(sql_query, language="sql")

    # -----------------------
    # STEP 2: EXECUTE SQL
    # -----------------------

    try:
        result = con.execute(sql_query).df()

        if result.shape[0] == 0:
            st.warning("La consulta no devolvió resultados.")
        else:
            st.subheader("📋 Resultado")
            st.dataframe(result, use_container_width=True)

            # -----------------------
            # AUTO CHART
            # -----------------------

            if result.shape[1] == 2:
                st.subheader("📈 Visualización")

                fig, ax = plt.subplots()
                ax.bar(result.iloc[:, 0].astype(str),
                       result.iloc[:, 1])
                plt.xticks(rotation=45)
                st.pyplot(fig)

            # -----------------------
            # STEP 3: EXECUTIVE INSIGHT
            # -----------------------

            explanation = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """
                        Eres un asesor estratégico.
                        Explica resultados para un gerente.
                        Sé claro, ejecutivo y enfocado en impacto.
                        Incluye:
                        - Insight principal
                        - Riesgo o oportunidad
                        - Recomendación concreta
                        """
                    },
                    {
                        "role": "user",
                        "content": f"""
                        Resultado:
                        {result.head(20).to_string()}

                        Pregunta original:
                        {user_prompt}
                        """
                    }
                ]
            )

            st.subheader("🧠 Insight Ejecutivo")
            st.write(explanation.choices[0].message.content)

    except Exception as e:
        st.error(f"Error ejecutando SQL: {e}")
