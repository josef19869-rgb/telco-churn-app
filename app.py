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
st.sidebar.image("https://img.icons8.com/fluency/96/signal-tower.png", width=60)
st.sidebar.title("📡 Telco Churn")
st.sidebar.markdown("**Análisis de cancelación de clientes ISP**")
st.sidebar.divider()

pagina = st.sidebar.radio(
    "Navegación",
    ["🏠 Inicio", "🔍 Exploración", "📊 Visualizaciones", "🔎 Análisis", "🎯 Predictor"]
)

st.sidebar.divider()
contrato_filtro = st.sidebar.multiselect(
    "Filtrar por contrato",
    options=df['Contract'].unique(),
    default=df['Contract'].unique()
)
df_f = df[df['Contract'].isin(contrato_filtro)]

# ─── PÁGINA: INICIO ────────────────────────────────────────
if pagina == "🏠 Inicio":
    st.title("📡 Análisis de Churn en Telecomunicaciones")
    st.markdown("""
    Esta aplicación analiza el dataset **Telco Customer Churn** para identificar 
    patrones de cancelación de servicio en clientes de un operador ISP.
    """)
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    total = len(df_f)
    churned = df_f['Churn'].value_counts().get('Yes', 0)
    avg_charge = df_f['MonthlyCharges'].mean()
    avg_tenure = df_f['tenure'].mean()
    col1.metric("👥 Total clientes", f"{total:,}")
    col2.metric("❌ Cancelaciones", f"{churned:,}", f"{churned/total*100:.1f}%")
    col3.metric("💲 Cargo mensual prom.", f"${avg_charge:.2f}")
    col4.metric("📅 Antigüedad prom.", f"{avg_tenure:.1f} meses")

    st.divider()
    st.subheader("Vista previa del dataset")
    st.dataframe(df_f.head(20), use_container_width=True, height=350)

# ─── PÁGINA: EXPLORACIÓN ───────────────────────────────────
elif pagina == "🔍 Exploración":
    st.title("🔍 Exploración del dataset")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dimensiones")
        st.write(f"**Filas:** {df_f.shape[0]:,}  |  **Columnas:** {df_f.shape[1]}")
        st.subheader("Tipos de datos")
        tipos = pd.DataFrame({'Tipo': df_f.dtypes.astype(str), 'Nulos': df_f.isnull().sum()})
        st.dataframe(tipos, use_container_width=True)
    with col2:
        st.subheader("Estadísticas descriptivas")
        st.dataframe(df_f.describe().round(2), use_container_width=True)

    st.divider()
    st.subheader("Distribución de variable seleccionada")
    col_sel = st.selectbox("Elegir columna", df_f.select_dtypes('number').columns.tolist())
    fig = px.histogram(df_f, x=col_sel, color='Churn',
                       barmode='overlay', nbins=40,
                       color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'},
                       title=f'Distribución de {col_sel} por churn')
    st.plotly_chart(fig, use_container_width=True)

# ─── PÁGINA: VISUALIZACIONES ───────────────────────────────
elif pagina == "📊 Visualizaciones":
    st.title("📊 Visualizaciones principales")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(df_f, names='Churn', title='Distribución de churn',
                     color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'}, hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        churn_contrato = df_f.groupby(['Contract','Churn']).size().reset_index(name='count')
        fig = px.bar(churn_contrato, x='Contract', y='count', color='Churn',
                     barmode='group', title='Churn por tipo de contrato',
                     color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'})
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig = px.box(df_f, x='Churn', y='MonthlyCharges', color='Churn',
                     title='Cargos mensuales vs churn',
                     color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'})
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = px.scatter(df_f.sample(min(500, len(df_f))),
                         x='tenure', y='TotalCharges', color='Churn',
                         title='Antigüedad vs cargos totales',
                         color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'},
                         opacity=0.6)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Churn por servicio de internet")
    fig = px.bar(df_f.groupby(['InternetService','Churn']).size().reset_index(name='n'),
                 x='InternetService', y='n', color='Churn', barmode='group',
                 color_discrete_map={'No':'#378ADD','Yes':'#E24B4A'})
    st.plotly_chart(fig, use_container_width=True)

# ─── PÁGINA: ANÁLISIS ──────────────────────────────────────
elif pagina == "🔎 Análisis":
    st.title("🔎 Análisis de factores de churn")

    st.subheader("Tasa de churn por variable categórica")
    var_cat = st.selectbox("Variable", ['Contract','InternetService','PaymentMethod',
                                         'TechSupport','OnlineSecurity','SeniorCitizen'])
    resumen = df_f.groupby(var_cat)['Churn_bin'].agg(['mean','count']).reset_index()
    resumen.columns = [var_cat, 'Tasa churn', 'Total']
    resumen['Tasa churn'] = (resumen['Tasa churn']*100).round(1)
    resumen = resumen.sort_values('Tasa churn', ascending=False)

    fig = px.bar(resumen, x=var_cat, y='Tasa churn', text='Tasa churn',
                 title=f'Tasa de churn por {var_cat} (%)',
                 color='Tasa churn', color_continuous_scale='RdYlBu_r')
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(resumen, use_container_width=True)

    st.divider()
    st.subheader("Correlaciones numéricas")
    num_cols = df_f.select_dtypes('number').drop(columns=['Churn_bin'])
    corr = num_cols.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Matriz de correlación')
    st.plotly_chart(fig, use_container_width=True)

# ─── PÁGINA: PREDICTOR ─────────────────────────────────────
elif pagina == "🎯 Predictor":
    st.title("🎯 Estimador de riesgo de churn")
    st.info("Ajusta las características del cliente para estimar su probabilidad de cancelación.")

    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Antigüedad (meses)", 0, 72, 12)
        monthly = st.slider("Cargo mensual ($)", 18, 120, 65)
        contrato = st.selectbox("Tipo de contrato", ["Month-to-month","One year","Two year"])
        internet = st.selectbox("Servicio de internet", ["Fiber optic","DSL","No"])
    with col2:
        tech_support = st.selectbox("Soporte técnico", ["No","Yes","No internet service"])
        paperless = st.radio("Factura electrónica", ["Yes","No"])
        senior = st.radio("Cliente senior", [0, 1])

    score = 0
    if contrato == "Month-to-month": score += 35
    elif contrato == "One year": score += 15
    if internet == "Fiber optic": score += 20
    if tech_support == "No": score += 15
    if tenure < 6: score += 20
    elif tenure < 12: score += 10
    if monthly > 85: score += 10
    if paperless == "Yes": score += 5
    if senior == 1: score += 5
    score = min(score, 99)

    st.divider()
    color = "#E24B4A" if score > 50 else "#EF9F27" if score > 30 else "#1D9E75"
    st.markdown(f"""
    
    {score}%
    {"⚠️ RIESGO ALTO" if score>50 else "⚡ RIESGO MEDIO" if score>30 else "✅ RIESGO BAJO"}
    
    """, unsafe_allow_html=True)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x':[0,1],'y':[0,1]},
        gauge={'axis':{'range':[0,100]},
               'bar':{'color':color},
               'steps':[{'range':[0,30],'color':'#E1F5EE'},
                        {'range':[30,60],'color':'#FAEEDA'},
                        {'range':[60,100],'color':'#FCEBEB'}]}))
    st.plotly_chart(fig, use_container_width=True)

# ─── FOOTER ────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.caption("📡 Proyecto Final · Ciencia de Datos · 2024")
