import pandas as pd
import joblib
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def entrenar_modelo():
    # Cargar datos
    ruta_datos = Path(__file__).resolve().parent.parent / 'data' / 'processed' / 'inventory_ready_for_model.csv'
    df = pd.read_csv(ruta_datos)
    
    target = 'stockout_risk'
    
    # Preparación (Separación)
    y = df[target]
    X = df.drop(columns=[target])
    
    # Convertimos categóricas a numéricas
    X = pd.get_dummies(X, drop_first=True)
    
    # Train/Test
    # Reservamos un 20% de los datos para validar que el modelo funciona
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entrenamiento
    model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    
    # Guardar modelo
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/trained_model.pkl')
    
    print("Modelo entrenado con éxito usando train_test_split.")

if __name__ == "__main__":
    entrenar_modelo()