import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib


# Rutas
RUTA_DATASET_MASTER ="data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv"
RUTA_MODELO_PKL = "data/model_memory/cerebro_meteo_temperatura.pkl"
RUTA_COLS_PKL = "data/model_memory/columnas_modelo_temperatura.pkl"


def entrenar_modelo_temperatura():
    print("\n INICIANDO PROCESO DE RE-ENTRENAMIENTO SEMANAL...")
    
    # 1. Cargar el Dataset Maestro (que ya contiene los datos nuevos de la semana)
    if not os.path.exists(RUTA_DATASET_MASTER):
        print(f" Error: No encuentro {RUTA_DATASET_MASTER}")
        return

    dt = pd.read_csv(RUTA_DATASET_MASTER)

    # 2. Limpieza de Columnas Prohibidas (Igual que tenías)
    cols_a_borrar_de_X = [
        "Fecha", 
        "TARGET_Temp_Manana",       
        "TARGET_Lluvia_Manana", # Si la tienes, quítala de X
        "Dia_Del_Ano",              
        "Viento_Direccion_Grados",  
        "Precip_Total_mm"           
    ]
    
    # Filtrar solo las que existen
    cols_a_borrar = [c for c in cols_a_borrar_de_X if c in dt.columns]
    
    X = dt.drop(columns=cols_a_borrar)
    y = dt["TARGET_Temp_Manana"]

    # 3. Split (Entrenamiento / Test)
    # Usamos random_state fijo para reproducibilidad, o quítalo para variedad
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    # 4. Entrenar el Modelo
    print(f"    Entrenando Random Forest con {len(X_train)} registros...")
    modelo = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=40)
    modelo.fit(X_train, y_train)

    # 5. Validación rápida (opcional, para ver si va bien)
    val_pred = modelo.predict(X_test)
    val_error = mean_absolute_error(y_test, val_pred)
    print(f"    Error Medio (MAE) del nuevo modelo: {val_error:.4f} °C")

    # 6. Guardar el Cerebro
    joblib.dump(modelo, RUTA_MODELO_PKL)
    
    # Guardar las columnas exactas (Vital para que la app no falle)
    cols_entrenamiento = list(X.columns)
    joblib.dump(cols_entrenamiento, RUTA_COLS_PKL)

    print("✅ RE-ENTRENAMIENTO FINALIZADO. Modelo actualizado guardado.")
    
if __name__ == "__main__":
    entrenar_modelo_nuevo()





