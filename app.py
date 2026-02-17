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
st.caption("Panel Analítico de Reservas Turísticas + Agente Conversacional")

# -----------------------
# OPENAI
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

# DuckDB (reutilizable)
@st.cache_resource
def get_con(df_):
    con_ = duckdb.connect()
    con_.register("data", df_)
    return con_

con = get_con(df)

# -----------------------
# HELPERS
# -----------------------
def clean_sql(sql_text: str) -> str:
    sql_text = sql_text.strip()
    sql_text = re.sub(r"```.*?\n", "", sql_text)
    sql_text = sql_text.replace("```", "").strip()
    return sql_text

def run_sql(sql_query: str) -> pd.DataFrame:
    return con.execute(sql_query).df()

def safe_bar(ax, x, y):
    ax.bar(x, y)
    ax.tick_params(axis="x", labelrotation=45)

def safe_line(ax, x, y):
    ax.plot(x, y)
    ax.tick_params(axis="x", labelrotation=45)

# -----------------------
# KPI HEADER
# -----------------------
k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Reservas", f"{df.shape[0]:,}")
k2.metric("Total Visitantes", f"{int(df['totalvisitante'].sum()):,}")
k3.metric("Anuladas", f"{int((df['estado_r'] == 'Anulado').sum()):,}")
k4.metric("Prom. Visitantes/Reserva", f"{df['totalvisitante'].mean():.2f}")

st.divider()

# -----------------------
# DASHBOARD AUTOMÁTICO
# -----------------------
st.subheader("📊 Panorama General (Automático)")

# 1) Distribución por estado
estado_df = (
    df.groupby("estado_r")["nidreserva"]
    .count()
    .reset_index()
    .rename(columns={"nidreserva": "reservas"})
    .sort_values("reservas", ascending=False)
)

# 2) Top rutas por visitantes
rutas_df = (
    df.groupby("ruta")["totalvisitante"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# 3) Top agencias por visitantes
agencias_df = (
    df.groupby("razon_social")["totalvisitante"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

# 4) Tendencia mensual de visitantes
df_dash = df.copy()
df_dash["mes"] = df_dash["fechaVisita"].dt.to_period("M").astype(str)
tendencia_df = (
    df_dash.dropna(subset=["mes"])
    .groupby("mes")["totalvisitante"]
    .sum()
    .reset_index()
    .sort_values("mes")
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### Distribución por Estado")
    fig, ax = plt.subplots(figsize=(7, 4))
    safe_bar(ax, estado_df["estado_r"].astype(str), estado_df["reservas"])
    plt.tight_layout()
    st.pyplot(fig)

with c2:
    st.markdown("### Tendencia Mensual de Visitantes")
    fig, ax = plt.subplots(figsize=(7, 4))
    safe_line(ax, tendencia_df["mes"], tendencia_df["totalvisitante"])
    plt.tight_layout()
    st.pyplot(fig)

c3, c4 = st.columns(2)

with c3:
    st.markdown("### Top 10 Rutas por Visitantes")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(rutas_df.index.astype(str), rutas_df.values)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

with c4:
    st.markdown("### Top 10 Agencias por Visitantes")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(agencias_df.index.astype(str), agencias_df.values)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# -----------------------
# INSIGHT AUTOMÁTICO (texto)
# -----------------------
st.subheader("🧠 Interpretación del Panorama")

if "overview_text" not in st.session_state:
    with st.spinner("Generando interpretación automática..."):
        overview = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            max_tokens=450,
            messages=[
                {
                    "role": "system",
                    "content": """
Eres un analista senior.
Interpreta el panorama general del dataset usando SOLO la información entregada.
No des recomendaciones ni acciones.
Haz un análisis descriptivo: patrones, concentración, variaciones, distribución, estacionalidad si se observa.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Distribución por estado:
{estado_df.to_string(index=False)}

Top rutas por visitantes:
{rutas_df.to_string()}

Top agencias por visitantes:
{agencias_df.to_string()}

Tendencia mensual (primeros 12):
{tendencia_df.head(12).to_string(index=False)}
"""
                }
            ]
        )
        st.session_state.overview_text = overview.choices[0].message.content

st.write(st.session_state.overview_text)

st.divider()

# -----------------------
# SUGERENCIAS (botones)
# -----------------------
st.markdown("### 🔎 Sugerencias de exploración (1 clic)")

b1, b2, b3, b4 = st.columns(4)

if b1.button("Anulaciones por Ruta"):
    st.session_state.prefill = "Analiza la proporción de anulaciones por ruta (conteo y %)."

if b2.button("Concentración por Agencia"):
    st.session_state.prefill = "Analiza la concentración de visitantes por agencia (top 15 y % acumulado)."

if b3.button("Llaqta vs Camino Inka"):
    st.session_state.prefill = "Compara nidLugar 1 vs 2 en visitantes totales y reservas por estado."

if b4.button("Bajas vs Visitantes"):
    st.session_state.prefill = "Analiza la relación entre cant_bajas y totalvisitante (agrupa por rangos si aplica)."

st.divider()

# -----------------------
# CHAT / CONSULTA AVANZADA
# -----------------------
st.subheader("💬 Agente Conversacional (consulta + resultado + gráfico + análisis)")

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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Mostrar historial
for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
        st.write(m["content"])

prompt = st.chat_input("Escribe tu consulta...")

if "prefill" in st.session_state:
    prompt = st.session_state.prefill
    del st.session_state.prefill

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # 1) Generar SQL
    try:
        with st.spinner("Generando consulta..."):
            sql_resp = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=320,
                messages=[
                    {
                        "role": "system",
                        "content": """
Genera únicamente SQL válido para DuckDB.
Usa la tabla 'data'.
Devuelve solo una consulta SELECT (sin markdown).
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
        sql_query = clean_sql(sql_resp.choices[0].message.content)
        if not sql_query.lower().startswith("select"):
            raise ValueError("SQL inválido generado.")
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"No pude generar SQL: {e}")
        st.stop()

    # 2) Ejecutar
    try:
        result = run_sql(sql_query)
        if result.empty:
            with st.chat_message("assistant"):
                st.write("La consulta no devolvió resultados.")
            st.stop()
    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error ejecutando SQL: {e}")
        st.stop()

    # 3) Mostrar SQL + Tabla
    with st.chat_message("assistant"):
        st.markdown("**🧾 SQL ejecutado (expandible):**")
        with st.expander("Ver SQL"):
            st.code(sql_query, language="sql")

        st.markdown("**📋 Resultado:**")
        st.dataframe(result, use_container_width=True)

        # 4) Graficación automática para 2 columnas
        if result.shape[1] == 2:
            st.markdown("**📈 Gráfico automático:**")
            xcol, ycol = result.columns[0], result.columns[1]

            # si x es fecha o parece fecha
            try:
                x_as_dt = pd.to_datetime(result[xcol], errors="coerce")
                is_date_like = x_as_dt.notna().mean() > 0.8
            except:
                is_date_like = False

            if is_date_like:
                # ordenar por fecha
                tmp = result.copy()
                tmp[xcol] = pd.to_datetime(tmp[xcol], errors="coerce")
                tmp = tmp.dropna(subset=[xcol]).sort_values(xcol)

                fig, ax = plt.subplots(figsize=(9, 4))
                ax.plot(tmp[xcol], tmp[ycol])
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                tmp = result.copy().sort_values(ycol, ascending=False)
                fig_width = max(8, len(tmp) * 0.55)
                fig, ax = plt.subplots(figsize=(fig_width, 4))
                ax.bar(tmp[xcol].astype(str), tmp[ycol])
                if len(tmp) > 6:
                    plt.xticks(rotation=60, ha="right")
                plt.tight_layout()
                st.pyplot(fig)

        # 5) Análisis interpretativo (texto)
        with st.spinner("Generando análisis interpretativo..."):
            try:
                analysis_resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.25,
                    max_tokens=550,
                    messages=[
                        {
                            "role": "system",
                            "content": """
Eres un analista senior de datos.
Interpreta únicamente los datos del resultado.
No des recomendaciones ni acciones.
Haz análisis descriptivo: patrón dominante, concentración/dispersión, magnitudes relativas, variaciones, posibles anomalías.
"""
                        },
                        {
                            "role": "user",
                            "content": f"""
Pregunta:
{prompt}

Resultado (hasta 25 filas):
{result.head(25).to_string()}
"""
                        }
                    ]
                )
                analysis_text = analysis_resp.choices[0].message.content
            except Exception as e:
                analysis_text = f"No pude generar el análisis: {e}"

        st.markdown("**🧠 Análisis interpretativo:**")
        st.write(analysis_text)

    st.session_state.chat_history.append({"role": "assistant", "content": analysis_text})
