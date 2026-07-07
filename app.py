import streamlit as st
import pandas as pd
import joblib

# Configuración de la página
st.set_page_config(page_title="Simulador de Ventas | Data Marketing", page_icon="📈", layout="centered")

# Cargar el modelo en caché para optimizar el rendimiento
@st.cache_resource
def cargar_modelo():
    # Intenta cargar el modelo, manejando el error si el archivo no existe aún
    try:
        return joblib.load('modelo_marketing.pkl')
    except FileNotFoundError:
        return None

modelo = cargar_modelo()

# Encabezado de la aplicación
st.title("📈 Simulador de Ventas Multi-Canal")
st.markdown("""
Ajuste el presupuesto de inversión en las campañas de **Televisión** y **Radio** para predecir el impacto en las ventas totales. 
*El modelo excluye la inversión en Diarios por su baja correlación estadística.*
""")

# Mostrar advertencia si el modelo no está disponible
if modelo is None:
    st.warning("⚠️ El modelo 'modelo_marketing.pkl' no se ha encontrado. Asegúrate de ejecutar la fase de entrenamiento en Colab y colocar el archivo `.pkl` en el mismo directorio que esta aplicación.")
else:
    # Interfaz de entrada de datos (solo si el modelo está cargado)
    st.subheader("Presupuesto Publicitario (en Miles de $)")

    col1, col2 = st.columns(2)

    with col1:
        inversion_tv = st.number_input("📺 Inversión en TV", min_value=0.0, max_value=500.0, value=150.0, step=10.0)

    with col2:
        inversion_radio = st.number_input("📻 Inversión en Radio", min_value=0.0, max_value=200.0, value=30.0, step=5.0)

    st.divider()

    # Botón de cálculo
    if st.button("Generar Proyección de Ventas 🚀", use_container_width=True):
        
        # Organizar los datos tal como los espera el modelo
        datos_entrada = pd.DataFrame({
            'TV': [inversion_tv],
            'Radio': [inversion_radio]
        })

        # Ejecutar la predicción
        prediccion_ventas = modelo.predict(datos_entrada)[0]

        # Mostrar el resultado final
        st.success("Cálculo realizado con éxito basándose en el modelo de Regresión Lineal Múltiple.")
        
        st.metric(
            label="Ventas Estimadas Proyectadas", 
            value=f"{prediccion_ventas:,.2f} Unidades"
        )
        
        st.info("💡 Consejo Estratégico: Ajuste los valores para encontrar la combinación óptima que maximice el retorno de inversión (ROI).")
