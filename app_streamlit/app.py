import streamlit as st
import joblib
import pandas as pd

model = joblib.load('models/trained_model.pkl')

st.title("Sistema de Predicción: Riesgo de Stockout")

# --- INPUTS ---
col1, col2 = st.columns(2)
with col1:
    current_stock = st.number_input("Stock Actual", value=100)
    daily_demand = st.number_input("Demanda Diaria", value=10)
    lead_time_days = st.number_input("Lead Time (días)", value=5)
with col2:
    
    promotion_active = st.selectbox("Promoción Activa", [0, 1])
    weather_impact = st.selectbox("Impacto Clima", ["High", "Low", "Medium"])
    supplier_reliability_score = st.number_input("Fiabilidad Proveedor", value=0.8)

if st.button("Analizar Riesgo"):
    # Crear el DataFrame con los nombres originales
    input_data = pd.DataFrame([[
        current_stock, daily_demand, lead_time_days, 
        promotion_active, weather_impact, supplier_reliability_score
    ]], columns=[
        'current_stock', 'daily_demand', 'lead_time_days', 
        'promotion_active', 'weather_impact', 'supplier_reliability_score'
    ])
    
    # Aplicar get_dummies para que la estructura sea idéntica al entrenamiento
    
    input_data = pd.get_dummies(input_data, drop_first=True)
    
    # RELLENAR COLUMNAS FALTANTES
    # Si el usuario elige "Low", no aparecerá la columna "High". 
    # Debemos rellenar con ceros las columnas que el modelo espera pero que no están en el input actual.
    
    # Obtenemos las columnas que el modelo vio durante el entrenamiento 
    model_features = model.feature_names_in_
    
    # Añadimos las columnas que faltan rellenándolas con 0
    for feature in model_features:
        if feature not in input_data.columns:
            input_data[feature] = 0
            
    # Reordenamos para asegurar el orden correcto
    input_data = input_data[model_features]
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("Riesgo Detectado.")
    else:
        st.success("Stock correcto.")


st.divider()
st.info("Nota: Este sistema es una herramienta de apoyo basada en un modelo de Machine Learning. No detecta bien las probabilidades de Stockout. Se probarán otros modelos")