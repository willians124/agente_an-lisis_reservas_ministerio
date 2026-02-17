import streamlit as st
import duckdb
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from openai import OpenAI

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(page_title="Data Intelligence Dashboard", layout="wide")

st.title("📊 Data Intelligence Dashboard")
st.caption("Panel Analítico de Reservas Turísticas + Copiloto Conversacional")

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

@st.cache_resource
def get_con(df_):
    con_ = duckdb.connect()
    con_.register("data", df_)
    return con_

con = get_con(df)

# -----------------------
# HELPERS
# -----------------------
def extract_sql(text):
    text = re.sub(r"```.*?\n", "", text)
    text = text.replace("```", "").strip()

    match = re.search(r"(select .*?;)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    match = re.search(r"(select .*?$)", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    return None

def wants_histogram(text):
    return any(x in text.lower() for x in ["histograma", "histogram", "distribución", "distribucion"])

def is_date_like(s):
    return pd.to_datetime(s, errors="coerce").notna().mean() > 0.8

def plot_result(result, prompt):
    if result.empty:
        return

    num_cols = [c for c in result.columns if pd.api.types.is_numeric_dtype(result[c])]
    non_num_cols = [c for c in result.columns if c not in num_cols]

    if wants_histogram(prompt) and len(num_cols) > 0:
        col = num_cols[0]
        st.markdown("### 📈 Histograma")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.hist(result[col].dropna(), bins=20)
        ax.set_title(f"Distribución de {col}")
        st.pyplot(fig)
        return

    if result.shape[1] == 2:
        x, y = result.columns

        if is_date_like(result[x]):
            st.markdown("### 📈 Tendencia")
            fig, ax = plt.subplots(figsize=(8,4))
            tmp = result.copy()
            tmp[x] = pd.to_datetime(tmp[x])
            tmp = tmp.sort_values(x)
            ax.plot(tmp[x], tmp[y])
            plt.xticks(rotation=45)
            st.pyplot(fig)
        else:
            st.markdown("### 📊 Gráfico")
            tmp = result.sort_values(y, ascending=False)
            fig, ax = plt.subplots(figsize=(8,4))
            ax.bar(tmp[x].astype(str), tmp[y])
            plt.xticks(rotation=60, ha="right")
            st.pyplot(fig)
        return

    if len(non_num_cols) >= 1 and len(num_cols) >= 2:
        cat = non_num_cols[0]
        m1, m2 = num_cols[:2]
        tmp = result.sort_values(m1, ascending=False)

        st.markdown(f"### 📊 {m1} por {cat}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(tmp[cat].astype(str), tmp[m1])
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

        st.markdown(f"### 📊 {m2} por {cat}")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(tmp[cat].astype(str), tmp[m2])
        plt.xticks(rotation=60, ha="right")
        st.pyplot(fig)

# -----------------------
# KPIs
# -----------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Reservas", f"{df.shape[0]:,}")
k2.metric("Total Visitantes", f"{int(df['totalvisitante'].sum()):,}")
k3.metric("Anuladas", f"{int((df['estado_r']=='Anulado').sum()):,}")
k4.metric("Prom. Visitantes/Reserva", f"{df['totalvisitante'].mean():.2f}")

st.divider()

# -----------------------
# DASHBOARD AUTOMÁTICO
# -----------------------
st.subheader("📊 Panorama General")

estado_df = df.groupby("estado_r")["nidreserva"].count().reset_index()
rutas_df = df.groupby("ruta")["totalvisitante"].sum().sort_values(ascending=False).head(8)

df["mes"] = df["fechaVisita"].dt.to_period("M").astype(str)
tendencia_df = df.groupby("mes")["totalvisitante"].sum().reset_index()

colA, colB = st.columns(2)

with colA:
    st.markdown("### Distribución por Estado")
    fig, ax = plt.subplots()
    ax.bar(estado_df["estado_r"], estado_df["nidreserva"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

with colB:
    st.markdown("### Tendencia Mensual")
    fig, ax = plt.subplots()
    ax.plot(tendencia_df["mes"], tendencia_df["totalvisitante"])
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.divider()

# -----------------------
# CHAT CON MEMORIA
# -----------------------
st.subheader("💬 Copiloto Analítico")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Haz una consulta...")

if prompt:

    st.session_state.chat_history.append({"role": "user", "content": prompt})

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
    - npa_cocinero
    - npa_ayudante
    - npa_porteador

    - totalvisitante
    - cant_bajas

    - campamentos
    - guias

    - fechaVisita

    - nidLugar
        Valores posibles:
        - 1 = Llaqta Machupicchu
        - 2 = Red de Camino Inka
    """

    sql_gen = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=800,
        messages=[
            {
    "role": "system",
    "content": """
Eres un experto en SQL analítico.

Tu tarea es traducir preguntas de negocio a SQL válido para DuckDB.

Reglas:
- Siempre genera una consulta SELECT completa.
- Si la pregunta es compleja, usa GROUP BY.
- Si requiere comparación temporal, usa fechaVisita.
- Si requiere ranking, usa ORDER BY y LIMIT 10.
- Si compara estados (Anulado, Cerrado, Pagado), usa SUM con CASE WHEN.
- Nunca expliques.
- Nunca escribas texto adicional.
- Devuelve solo SQL.
"""
},
            {"role": "user", "content": f"{schema_description}\nPregunta: {prompt}"}
        ]
    )

    raw_sql = sql_gen.choices[0].message.content
    sql_query = extract_sql(raw_sql)

    if not sql_query:
        with st.chat_message("assistant"):
            st.error("No se pudo generar SQL válido.")
        st.stop()

    try:
        result = con.execute(sql_query).df()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error ejecutando SQL: {e}")
        st.stop()

    with st.chat_message("assistant"):

        with st.expander("🧾 Ver SQL generado"):
            st.code(sql_query, language="sql")

        st.markdown("### 📋 Resultado")
        st.dataframe(result, use_container_width=True)

        plot_result(result, prompt)

        analysis = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=600,
            messages=[
                {"role": "system", "content": """
                Eres un analista senior.
                Analiza profundamente los datos.
                No des recomendaciones.
                Describe patrones, concentración y variaciones.
                """},
                {"role": "user", "content": result.head(25).to_string()}
            ]
        )

        analysis_text = analysis.choices[0].message.content

        st.markdown("### 🧠 Análisis Interpretativo")
        st.write(analysis_text)

    st.session_state.chat_history.append({"role": "assistant", "content": analysis_text})
