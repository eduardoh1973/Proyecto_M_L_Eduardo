import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def evaluar_modelo():
    # 1. Cargar el modelo y los datos
    model = joblib.load('models/trained_model.pkl')
    df = pd.read_csv('data/processed/inventory_ready_for_model.csv')
    
    # 2. Preparar los datos igual que en el entrenamiento
    target = 'stockout_risk'
    X = df.drop(columns=[target])
    y = df[target]
    X = pd.get_dummies(X, drop_first=True)
    
    # 3. Dividir los datos (Misma semilla que en training.py)
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Evaluar SOLO sobre el conjunto de prueba
    predicciones = model.predict(X_test)
    
    print("--- EVALUACIÓN EN CONJUNTO DE PRUEBA (TEST SET) ---")
    print("\nMATRIZ DE CONFUSIÓN:")
    print(confusion_matrix(y_test, predicciones))
    
    print("\nREPORTE DE CLASIFICACIÓN:")
    print(classification_report(y_test, predicciones, zero_division=0))

if __name__ == "__main__":
    evaluar_modelo()