%%writefile app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Análisis de Churn Telco",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/telco_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    df['Churn_bin'] = (df['Churn'] == 'Yes').astype(int)
    return df

df = load_data()

# ─── SIDEBAR ───────────────────────────────────────────────
st.sidebar.title("📡 Telco Churn")
st.sidebar.markdown("**Análisis de cancelación de clientes ISP**")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔍 Exploración", "📊 Visualizaciones",
     "🔎 Análisis", "🎯 Predictor"]
)

st.sidebar.divider()
contrato_filtro = st.sidebar.multiselect(
    "Filtrar por contrato",
    options=df['Contract'].unique(),
    default=list(df['Contract'].unique())
)

if not contrato_filtro:
    st.sidebar.warning("Selecciona al menos un tipo de contrato.")
    df_f = df.copy()
else:
    df_f = df[df['Contract'].isin(contrato_filtro)]

# ─── INICIO ────────────────────────────────────────────────
if pagina == "🏠 Inicio":
    st.title("📡 Análisis de Churn en Telecomunicaciones")
    st.markdown(
        "Esta aplicación analiza el dataset **Telco Customer Churn** "
        "para identificar patrones de cancelación de servicio en clientes ISP."
    )
    st.divider()

    total = len(df_f)
    churned = int(df_f['Churn'].value_counts().get('Yes', 0))
    avg_charge = float(df_f['MonthlyCharges'].mean())
    avg_tenure = float(df_f['tenure'].mean())
    churn_pct = round(churned / total * 100, 1) if total > 0 else 0

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("👥 Total clientes", f"{total:,}")
    col2.metric("❌ Cancelaciones", f"{churned:,}", f"{churn_pct}%")
    col3.metric("💲 Cargo mensual prom.", f"${avg_charge:.2f}")
    col4.metric("📅 Antigüedad prom.", f"{avg_tenure:.1f} meses")

    st.divider()
    st.subheader("Vista previa del dataset")
    st.dataframe(df_f.head(20), use_container_width=True, height=350)

# ─── EXPLORACIÓN ───────────────────────────────────────────
elif pagina == "🔍 Exploración":
    st.title("🔍 Exploración del dataset")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dimensiones")
        st.write(f"**Filas:** {df_f.shape[0]:,}  |  **Columnas:** {df_f.shape[1]}")
        st.subheader("Tipos de datos y nulos")
        tipos = pd.DataFrame({
            'Tipo': df_f.dtypes.astype(str),
            'Nulos': df_f.isnull().sum()
        })
        st.dataframe(tipos, use_container_width=True)
    with col2:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df_f.describe().round(2), use_container_width=True)

    st.divider()
    st.subheader("Distribución de variable numérica")
    num_cols = df_f.select_dtypes('number').columns.tolist()
    col_sel = st.selectbox("Elegir columna", num_cols)
    fig = px.histogram(
        df_f, x=col_sel, color='Churn', barmode='overlay', nbins=40,
        color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'},
        title=f'Distribución de {col_sel} por churn'
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── VISUALIZACIONES ───────────────────────────────────────
elif pagina == "📊 Visualizaciones":
    st.title("📊 Visualizaciones principales")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(
            df_f, names='Churn',
            title='Distribución de churn',
            color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'},
            hole=0.4
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_c = (df_f.groupby(['Contract', 'Churn'])
                   .size().reset_index(name='count'))
        fig = px.bar(
            churn_c, x='Contract', y='count', color='Churn',
            barmode='group', title='Churn por tipo de contrato',
            color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'}
        )
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(
            df_f, x='Churn', y='MonthlyCharges', color='Churn',
            title='Cargos mensuales vs churn',
            color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        sample = df_f.sample(min(500, len(df_f)), random_state=42)
        fig = px.scatter(
            sample, x='tenure', y='TotalCharges', color='Churn',
            title='Antigüedad vs cargos totales',
            color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'},
            opacity=0.6
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn por servicio de internet")
    inet = (df_f.groupby(['InternetService', 'Churn'])
            .size().reset_index(name='n'))
    fig = px.bar(
        inet, x='InternetService', y='n', color='Churn',
        barmode='group',
        color_discrete_map={'No': '#378ADD', 'Yes': '#E24B4A'}
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── ANÁLISIS ──────────────────────────────────────────────
elif pagina == "🔎 Análisis":
    st.title("🔎 Análisis de factores de churn")

    st.subheader("Tasa de churn por variable categórica")
    var_cat = st.selectbox("Variable", [
        'Contract', 'InternetService', 'PaymentMethod',
        'TechSupport', 'OnlineSecurity', 'SeniorCitizen'
    ])
    resumen = (df_f.groupby(var_cat)['Churn_bin']
               .agg(['mean', 'count']).reset_index())
    resumen.columns = [var_cat, 'Tasa churn (%)', 'Total clientes']
    resumen['Tasa churn (%)'] = (resumen['Tasa churn (%)'] * 100).round(1)
    resumen = resumen.sort_values('Tasa churn (%)', ascending=False)

    fig = px.bar(
        resumen, x=var_cat, y='Tasa churn (%)',
        text='Tasa churn (%)',
        title=f'Tasa de churn por {var_cat} (%)',
        color='Tasa churn (%)',
        color_continuous_scale='RdYlBu_r'
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(resumen, use_container_width=True)

    st.divider()
    st.subheader("Correlaciones numéricas")
    num_df = df_f.select_dtypes('number').drop(columns=['Churn_bin'], errors='ignore')
    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr, text_auto=True,
        color_continuous_scale='RdBu_r',
        title='Matriz de correlación'
    )
    st.plotly_chart(fig, use_container_width=True)

# ─── PREDICTOR ─────────────────────────────────────────────
elif pagina == "🎯 Predictor":
    st.title("🎯 Estimador de riesgo de churn")
    st.info("Ajusta las características del cliente para estimar su probabilidad de cancelación.")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
        monthly = st.slider("Cargo mensual ($)", 18, 120, 65)
        contrato = st.selectbox(
            "Tipo de contrato",
            ["Month-to-month", "One year", "Two year"]
        )
        internet = st.selectbox(
            "Servicio de internet",
            ["Fiber optic", "DSL", "No"]
        )
    with col2:
        tech_support = st.selectbox(
            "Soporte técnico",
            ["No", "Yes", "No internet service"]
        )
        paperless = st.radio("Factura electrónica", ["Yes", "No"])
        senior = st.radio("Cliente senior (mayor de 65)", [0, 1])

    # Cálculo del score
    score = 0
    if contrato == "Month-to-month":
        score += 35
    elif contrato == "One year":
        score += 15
    if internet == "Fiber optic":
        score += 20
    if tech_support == "No":
        score += 15
    if tenure < 6:
        score += 20
    elif tenure < 12:
        score += 10
    if monthly > 85:
        score += 10
    if paperless == "Yes":
        score += 5
    if senior == 1:
        score += 5
    score = min(score, 99)

    st.divider()
    st.subheader("Resultado del estimador")

    # Componentes nativos — sin unsafe_allow_html dinámico
    col_r1, col_r2 = st.columns([1, 2])
    with col_r1:
        st.metric("Probabilidad estimada de churn", f"{score}%")
        if score > 50:
            st.error("⚠️ RIESGO ALTO — cliente propenso a cancelar")
        elif score > 30:
            st.warning("⚡ RIESGO MEDIO — monitorear de cerca")
        else:
            st.success("✅ RIESGO BAJO — cliente estable")

        st.progress(score / 100)

    with col_r2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Índice de riesgo"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#E24B4A' if score > 50
                        else '#EF9F27' if score > 30 else '#1D9E75'},
                'steps': [
                    {'range': [0, 30], 'color': '#E1F5EE'},
                    {'range': [30, 60], 'color': '#FAEEDA'},
                    {'range': [60, 100], 'color': '#FCEBEB'}
                ]
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Factores de riesgo detectados")
    factores = []
    if contrato == "Month-to-month":
        factores.append("Contrato mensual (+35 pts) — el factor de mayor impacto")
    if internet == "Fiber optic":
        factores.append("Servicio de fibra óptica (+20 pts)")
    if tech_support == "No":
        factores.append("Sin soporte técnico (+15 pts)")
    if tenure < 6:
        factores.append("Cliente muy nuevo, menos de 6 meses (+20 pts)")
    elif tenure < 12:
        factores.append("Cliente reciente, menos de 12 meses (+10 pts)")
    if monthly > 85:
        factores.append("Cargo mensual elevado (>$85) (+10 pts)")

    if factores:
        for f in factores:
            st.markdown(f"- {f}")
    else:
        st.markdown("Sin factores de riesgo significativos detectados.")

# ─── FOOTER ────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("📡 Proyecto Final · Ciencia de Datos · 2025")
