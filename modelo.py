import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import requests
import random


# Leer el csv
dt = pd.read_csv("dataset_entrenamiento_final.csv")

# Limpiar tabla
# Sin Temp_Media_C_Media_3dias se equivoca menos a la hora de predecir la temperatura
dt = dt.drop(columns=["Fecha", "Mes", "Dia_Del_Ano", "Temp_Media_C_Media_3dias"])

# Separar los datos en x y
X = dt.drop("TARGET_Temp_Manana", axis=1)
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





# Funcion de predecir datos
def predecir_temperatura(datos_hoy):
    pred = modelo.predict(datos_hoy)
    pred_valor = pred[0]
    print(f"La prediccion es: {pred_valor:.2f}")
    


# Predecir prueba

randomfila = random.randrange(1, 100)
datos_hoy = X_test.iloc[[randomfila]].copy()

predecir_temperatura(datos_hoy)
print(f"El dato real es:  {y_test.iloc[randomfila]}")





# Obtener tiempo de hoy

url = (
    "https://api.open-meteo.com/v1/forecast"
    "?latitude=41.3874"
    "&longitude=2.1686"
    "&current=temperature_2m,relative_humidity_2m,wind_speed_10m"
    "&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean"
    "&timezone=Europe/Madrid"
)

data = requests.get(url).json()