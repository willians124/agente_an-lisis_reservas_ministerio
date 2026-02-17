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

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY no configurada en Secrets.")
    st.stop()

# -----------------------
# LOAD DATA (CACHED)
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
# EXECUTIVE METRICS
# -----------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Reservas", f"{df.shape[0]:,}")
col2.metric("Total Visitantes", f"{int(df['totalvisitante'].sum()):,}")
col3.metric("Reservas Pagadas", f"{int((df['estado_r']=='Pagado').sum()):,}")
col4.metric("Reservas Anuladas", f"{int((df['estado_r']=='Anulado').sum()):,}")

st.divider()

# -----------------------
# CHAT SECTION
# -----------------------

st.subheader("💬 Consulta Ejecutiva")

user_prompt = st.text_area(
    "Haz una pregunta estratégica sobre las reservas:",
    placeholder="Ej: ¿Qué rutas generan más volumen? ¿Qué agencias tienen más anulaciones? ¿Hay ineficiencia operativa?"
)

if st.button("Analizar"):

    if not user_prompt.strip():
        st.warning("Escribe una pregunta primero.")
        st.stop()

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
    # GENERATE SQL + ANALYSIS (1 CALL)
    # -----------------------

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=700,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": """
                    Eres un analista experto.
                    Genera SQL puro compatible con DuckDB.
                    No uses markdown ni ```.

                    Luego escribe un análisis interpetrativo.

                    Formato exacto:

                    SQL:
                    <query>

                    Interpretación del análisis:
                    <explicacion>
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

        full_response = response.choices[0].message.content

    except Exception as e:
        st.error(f"Error llamando al modelo: {e}")
        st.stop()

    # -----------------------
    # PARSE RESPONSE
    # -----------------------

    def clean_sql(text):
        text = text.replace("SQL:", "")
        text = re.sub(r"```.*?\n", "", text)
        text = text.replace("```", "")
        return text.strip()

    parts = re.split(r"ANALISIS:", full_response, flags=re.IGNORECASE)

    
    def extract_sql_and_analysis(text):
        sql_match = re.search(r"SQL:\s*(.*?)\s*ANALISIS:", text, re.DOTALL | re.IGNORECASE)
        analysis_match = re.search(r"ANALISIS:\s*(.*)", text, re.DOTALL | re.IGNORECASE)
    
        if not sql_match or not analysis_match:
            return None, None
    
        sql = sql_match.group(1).strip()
        analysis = analysis_match.group(1).strip()
    
        return sql, analysis
    
    
    sql_query, analysis_text = extract_sql_and_analysis(full_response)
    
    if not sql_query or not analysis_text:
        st.error("El modelo no devolvió el formato esperado.")
        st.write(full_response)  # 👈 Esto te ayuda a debuggear en demo
        st.stop()
        
    sql_part = parts[0]
    analysis_text = parts[1].strip()

    sql_query = clean_sql(sql_part)

    # Validación mínima
    if not sql_query.lower().startswith("select"):
        st.error("La consulta generada no es válida.")
        st.stop()

    st.code(sql_query, language="sql")

    # -----------------------
    # EXECUTE SQL
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
    # SHOW RESULT
    # -----------------------

    st.subheader("📋 Resultado")
    st.dataframe(result, use_container_width=True)

    # -----------------------
    # AUTO CHART
    # -----------------------

    if result.shape[1] == 2:
    
        st.subheader("📈 Visualización")
    
        col_x = result.columns[0]
        col_y = result.columns[1]
    
        # Ordenar automáticamente por valor descendente
        result_sorted = result.sort_values(by=col_y, ascending=False)
    
        fig_width = max(8, len(result_sorted) * 0.6)  # ancho dinámico
        fig, ax = plt.subplots(figsize=(fig_width, 5))
    
        ax.bar(
            result_sorted[col_x].astype(str),
            result_sorted[col_y]
        )
    
        # Rotación inteligente según cantidad de categorías
        if len(result_sorted) > 6:
            plt.xticks(rotation=60, ha="right")
        else:
            plt.xticks(rotation=0)
    
        ax.set_xlabel("")
        ax.set_ylabel(col_y)
    
        plt.tight_layout()
        st.pyplot(fig)

    # -----------------------
    # EXECUTIVE INSIGHT
    # -----------------------

    st.subheader("🧠 Insight Ejecutivo")
    st.write(analysis_text)
