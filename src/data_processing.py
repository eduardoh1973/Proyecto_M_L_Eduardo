import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

def procesar_datos(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # Codificación de variables (basado en el análisis previo)
    le = LabelEncoder()
    df['stockout_risk'] = le.fit_transform(df['stockout_risk'])
    df['promotion_active'] = le.fit_transform(df['promotion_active'])
    df['weather_impact'] = df['weather_impact'].map({'Low': 0, 'Medium': 1, 'High': 2})
    
    # Guardar en data/processed
    df.to_csv(output_path, index=False)
    print(f"Datos procesados guardados en {output_path}")

if __name__ == "__main__":
    procesar_datos('data/raw/dataset.csv', 'data/processed/processed_data.csv')