import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Configuración de página ancha para mejor visualización
st.set_page_config(page_title="Dashboard Data Marketing", page_icon="📈", layout="wide")

# Cargar el modelo en caché
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_marketing.pkl')
    except FileNotFoundError:
        return None

# Cargar el dataset original para el EDA
@st.cache_data
def cargar_datos():
    try:
        # Asegúrate de que este archivo también esté subido a tu GitHub
        return pd.read_excel('DataMarketing.xlsx') 
    except FileNotFoundError:
        return None

modelo = cargar_modelo()
df = cargar_datos()

st.title("🚀 Dashboard Avanzado: Data Marketing y Predicción de Ventas")
st.markdown("Plataforma interactiva para el análisis de ROI publicitario y simulación de escenarios predictivos basados en Machine Learning.")

# Creación de Pestañas para organizar el Dashboard
tab1, tab2, tab3 = st.tabs(["📊 Análisis Exploratorio (EDA)", "🔮 Simulador y Línea de Proyección", "⚖️ Comparador de Escenarios"])

# ==========================================
# PESTAÑA 1: EDA Y CORRELACIÓN
# ==========================================
with tab1:
    st.header("Análisis Exploratorio de Datos (EDA)")
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Vista Previa del Dataset")
            st.dataframe(df.head(8), use_container_width=True)
            
        with col2:
            st.subheader("Estadísticas Descriptivas")
            st.dataframe(df.describe(), use_container_width=True)
            
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Matriz de Correlación")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
            st.pyplot(fig)
            st.info("💡 **Insight:** El mapa de calor confirma matemáticamente que la Inversión en TV (0.78) tiene la correlación más fuerte con el volumen de ventas.")
            
        with col4:
            st.subheader("Dispersión: Inversión vs Ventas")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='TV', y='Ventas', color='blue', label='TV')
            sns.scatterplot(data=df, x='Radio', y='Ventas', color='orange', label='Radio')
            ax2.set_xlabel("Inversión (Miles de $)")
            ax2.set_ylabel("Ventas (Unidades)")
            st.pyplot(fig2)
    else:
        st.warning("⚠️ No se encontró el archivo 'DataMarketing.xlsx'. Por favor súbelo al repositorio junto con la app para visualizar el EDA.")

# ==========================================
# PESTAÑA 2: SIMULADOR PRINCIPAL
# ==========================================
with tab2:
    st.header("Simulador de Inversión y Proyección Continua")
    if modelo is not None:
        col_tv, col_rad = st.columns(2)
        with col_tv:
            inv_tv = st.slider("📺 Inversión en TV ($)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
        with col_rad:
            inv_rad = st.slider("📻 Inversión en Radio ($)", min_value=0.0, max_value=200.0, value=30.0, step=2.0)
            
        # Predicción actual
        pred_ventas = modelo.predict(pd.DataFrame({'TV': [inv_tv], 'Radio': [inv_rad]}))[0]
        st.metric(label="📊 Ventas Estimadas Proyectadas", value=f"{pred_ventas:,.2f} Unidades")
        
        st.divider()
        
        # Simulación de crecimiento (Curva)
        st.subheader("📈 Simulación de Escalamiento")
        st.write("¿Cómo aumentarían las ventas si escalamos el presupuesto de TV en los próximos niveles, manteniendo la Radio fija?")
        
        # Crear un rango de +0 hasta +200 de inversión en TV para simular
        tvs_simulados = np.linspace(inv_tv, inv_tv + 200, 10)
        ventas_simuladas = [modelo.predict(pd.DataFrame({'TV': [t], 'Radio': [inv_rad]}))[0] for t in tvs_simulados]
        
        fig3, ax3 = plt.subplots(figsize=(10, 3))
        ax3.plot(tvs_simulados, ventas_simuladas, marker='o', linestyle='-', color='#2e7b5b', linewidth=2)
        ax3.set_title(f"Proyección de Ventas al escalar inversión en TV (Radio fija en {inv_rad})")
        ax3.set_xlabel("Inversión Escalonada en TV")
        ax3.set_ylabel("Ventas Estimadas")
        ax3.grid(True, linestyle='--', alpha=0.7)
        st.pyplot(fig3)
        
    else:
        st.error("⚠️ El modelo predictivo no se encuentra cargado.")

# ==========================================
# PESTAÑA 3: COMPARADOR A/B
# ==========================================
with tab3:
    st.header("Comparación de Escenarios (A/B Testing)")
    st.write("Utiliza esta herramienta para evaluar si un aumento presupuestal justifica el retorno en ventas proyectado.")
    
    if modelo is not None:
        colA, colB = st.columns(2)
        
        with colA:
            st.success("🔵 Escenario A: Presupuesto Base")
            tv_a = st.number_input("Inversión TV (A)", min_value=0.0, value=100.0, step=10.0)
            rad_a = st.number_input("Inversión Radio (A)", min_value=0.0, value=20.0, step=5.0)
            pred_a = modelo.predict(pd.DataFrame({'TV': [tv_a], 'Radio': [rad_a]}))[0]
            st.metric("Ventas Escenario A", f"{pred_a:,.2f} Unds.")
            
        with colB:
            st.info("🟢 Escenario B: Nuevo Presupuesto Propuesto")
            tv_b = st.number_input("Inversión TV (B)", min_value=0.0, value=180.0, step=10.0)
            rad_b = st.number_input("Inversión Radio (B)", min_value=0.0, value=40.0, step=5.0)
            pred_b = modelo.predict(pd.DataFrame({'TV': [tv_b], 'Radio': [rad_b]}))[0]
            
            # El parámetro delta dibuja una flecha verde/roja indicando la diferencia
            st.metric("Ventas Escenario B", f"{pred_b:,.2f} Unds.", delta=f"{pred_b - pred_a:,.2f} Unidades extra vs Escenario A")
