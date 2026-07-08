import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Configuración de página
st.set_page_config(page_title="Dashboard Data Marketing", page_icon="📈", layout="wide")

# Cargar el modelo en caché
@st.cache_resource
def cargar_modelo():
    try:
        return joblib.load('modelo_marketing.pkl')
    except FileNotFoundError:
        return None

# Cargar el dataset
@st.cache_data
def cargar_datos():
    try:
        return pd.read_excel('DataMarketing.xlsx')
    except FileNotFoundError:
        return None

modelo = cargar_modelo()
df = cargar_datos()

st.title("🚀 Dashboard Avanzado: Data Marketing y Predicción de Ventas")
st.markdown("Plataforma interactiva para el análisis de ROI publicitario y simulación de escenarios predictivos basados en Machine Learning.")

# Creación de 5 Pestañas
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 EDA", 
    "🤖 Comparación de Modelos", 
    "📉 Validación", 
    "🔮 Simulador", 
    "⚖️ Comparador A/B"
])

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
            
        with col4:
            st.subheader("Dispersión: Inversión vs Ventas")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(data=df, x='TV', y='Ventas', color='blue', label='TV')
            sns.scatterplot(data=df, x='Radio', y='Ventas', color='orange', label='Radio')
            ax2.set_xlabel("Inversión (Miles de $)")
            ax2.set_ylabel("Ventas (Unidades)")
            st.pyplot(fig2)
    else:
        st.warning("⚠️ No se encontró 'DataMarketing.xlsx'.")

# ==========================================
# PESTAÑA 2: COMPARACIÓN DE MODELOS
# ==========================================
with tab2:
    st.header("Evaluación y Selección del Mejor Modelo")
    st.write("Entrenamiento en tiempo real de 3 configuraciones de regresión lineal para justificar la exclusión del canal Diario.")
    
    if df is not None:
        y_all = df['Ventas']
        escenarios = {
            'Modelo 1 (Solo TV)': ['TV'],
            'Modelo 2 (TV + Radio)': ['TV', 'Radio'],
            'Modelo 3 (TV + Radio + Diario)': ['TV', 'Radio', 'Diario']
        }
        
        nombres = []
        r2_scores = []
        
        for nom, vars_ in escenarios.items():
            X_all = df[vars_]
            Xt, Xte, yt, yte = train_test_split(X_all, y_all, test_size=0.20, random_state=42)
            m_temp = LinearRegression().fit(Xt, yt)
            r2_scores.append(r2_score(yte, m_temp.predict(Xte)))
            nombres.append(nom.replace(" (", "\n("))
            
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        grafico = sns.barplot(x=nombres, y=r2_scores, palette='viridis', ax=ax_bar)
        ax_bar.set_ylabel('Score R²')
        ax_bar.set_ylim(0, 1)
        for i, v in enumerate(r2_scores):
            grafico.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold', fontsize=12)
        st.pyplot(fig_bar)
        st.info("🎯 **Conclusión:** El Modelo 2 (TV + Radio) es el óptimo. Añadir el Diario (Modelo 3) no aporta ninguna mejora predictiva.")

# ==========================================
# PESTAÑA 3: VALIDACIÓN DEL MODELO GANADOR
# ==========================================
with tab3:
    st.header("Validación del Modelo Elegido (TV + Radio)")
    if df is not None and modelo is not None:
        X = df[['TV', 'Radio']]
        y = df['Ventas']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
        
        y_pred = modelo.predict(X_test)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("R² (Precisión Global)", f"{r2:.4f}", "Óptimo")
        col_m2.metric("RMSE (Error Promedio)", f"S/. {rmse:.2f} Millones", "Margen", delta_color="inverse")
        
        st.divider()
        st.subheader("Ventas Reales vs. Predicciones")
        fig_disp, ax_disp = plt.subplots(figsize=(8, 4))
        ax_disp.scatter(y_test, y_pred, color='dodgerblue', alpha=0.8, edgecolor='k', s=60, label='Validación')
        ax_disp.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2, label='Predicción Perfecta')
        ax_disp.set_xlabel('Ventas Reales')
        ax_disp.set_ylabel('Ventas Esperadas')
        ax_disp.legend()
        st.pyplot(fig_disp)

# ==========================================
# PESTAÑA 4: SIMULADOR PRINCIPAL
# ==========================================
with tab4:
    st.header("Simulador de Crecimiento Escalonado")
    if modelo is not None:
        col_tv, col_rad = st.columns(2)
        with col_tv:
            inv_tv = st.slider("📺 Inversión en TV (Millones de $)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
        with col_rad:
            inv_rad = st.slider("📻 Inversión en Radio (Millones de $)", min_value=0.0, max_value=200.0, value=30.0, step=2.0)

        pred_ventas = modelo.predict(pd.DataFrame({'TV': [inv_tv], 'Radio': [inv_rad]}))[0]
        st.metric(label="📊 Ventas Estimadas Proyectadas", value=f"S/. {pred_ventas:,.2f} Millones")
        
        st.divider()
        st.write("¿Cómo aumentarían las ventas si escalamos progresivamente el presupuesto de TV, manteniendo la Radio fija?")
        
        tvs_sim = np.linspace(inv_tv, inv_tv + 200, 10)
        ventas_sim = [modelo.predict(pd.DataFrame({'TV': [t], 'Radio': [inv_rad]}))[0] for t in tvs_sim]
        
        fig_line, ax_line = plt.subplots(figsize=(10, 3))
        ax_line.plot(tvs_sim, ventas_sim, marker='o', linestyle='-', color='#2e7b5b', linewidth=2)
        ax_line.set_title(f"Proyección de Ventas al escalar TV (Radio fija en {inv_rad})")
        ax_line.set_xlabel("Inversión Escalonada en TV")
        ax_line.set_ylabel("Ventas Estimadas")
        st.pyplot(fig_line)

# ==========================================
# PESTAÑA 5: COMPARADOR A/B
# ==========================================
with tab5:
    st.header("Comparación de Escenarios (A/B Testing)")
    if modelo is not None:
        colA, colB = st.columns(2)
        with colA:
            st.success("🔵 Escenario A: Presupuesto Base")
            tv_a = st.number_input("Inversión TV (A)", min_value=0.0, value=100.0, step=10.0)
            rad_a = st.number_input("Inversión Radio (A)", min_value=0.0, value=20.0, step=5.0)
            pred_a = modelo.predict(pd.DataFrame({'TV': [tv_a], 'Radio': [rad_a]}))[0]
            st.metric("Ventas Escenario A", f"S/. {pred_a:,.2f} M.")
            
        with colB:
            st.info("🟢 Escenario B: Nuevo Presupuesto Propuesto")
            tv_b = st.number_input("Inversión TV (B)", min_value=0.0, value=200.0, step=10.0)
            rad_b = st.number_input("Inversión Radio (B)", min_value=0.0, value=30.0, step=5.0)
            pred_b = modelo.predict(pd.DataFrame({'TV': [tv_b], 'Radio': [rad_b]}))[0]
            st.metric("Ventas Escenario B", f"S/. {pred_b:,.2f} M.", delta=f"+ S/. {pred_b - pred_a:,.2f} Millones vs A")
