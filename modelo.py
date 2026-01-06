import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import requests
import random
from datetime import datetime, timedelta, date
import joblib


# Leer el csv
dt = pd.read_csv("data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv")


# Limpiar tabla
# Sin Temp_Media_C_Media_3dias se equivoca menos a la hora de predecir la temperatura
# Estas columnas confunden al modelo o son la respuesta
cols_a_borrar_de_X = [
    "Fecha", 
    "TARGET_Temp_Manana",       # La respuesta
    "Dia_Del_Ano",              # Redundante (tenemos Sin/Cos)
    "Viento_Direccion_Grados",  # Redundante (tenemos Sin/Cos)
    "Precip_Total_mm"           # Redundante (tenemos Lluvia_Binaria)
]

# Separar los datos en x y
# Borramos las columnas prohibidas solo si existen en el CSV
X = dt.drop(columns=[c for c in cols_a_borrar_de_X if c in dt.columns])
y = dt["TARGET_Temp_Manana"]

# Separar los datos de test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=40
)

# Separar datos de test y validacion
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42
)

# Crear el modelo
modelo = RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    random_state=40,
    n_jobs=-1
)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Validar el modelo
val_pred = modelo.predict(X_val)
val_error = mean_absolute_error(y_val, val_pred)

print("Error en validation:", val_error)
print("\n💾 Guardando el modelo entrenado...")

# Guardamos el Modelo
joblib.dump(modelo, 'data/model_memory/cerebro_meteo.pkl')

# Guardamos el orden exacto de las columnas
columnas_entrenamiento = list(X.columns)
joblib.dump(columnas_entrenamiento, 'data/model_memory/columnas_modelo.pkl')

print("✅ Modelo guardado como 'cerebro_meteo.pkl'")
print("✅ Columnas guardadas como 'columnas_modelo.pkl'")
print("   Ya puedes usar 'app_prediccion.py'")



