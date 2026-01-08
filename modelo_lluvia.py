import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    "TARGET_Temp_Manana",     
    "Dia_Del_Ano",              
    "Viento_Direccion_Grados", 
    "Precip_Total_mm" ,
    "TARGET_Lluvia_Manana"         
]

# Separar los datos en x y
# Borramos las columnas prohibidas solo si existen en el CSV
X = dt.drop(columns=[c for c in cols_a_borrar_de_X if c in dt.columns])
y = dt["TARGET_Lluvia_Manana"]

# Separar los datos de test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.30,
    random_state=40,
    stratify=y

)

# Separar datos de test y validacion
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp

)

# Crear el modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=40,
    n_jobs=-1,
    class_weight="balanced"
)

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Validar el modelo
val_pred = modelo.predict(X_val)
val_error = mean_absolute_error(y_val, val_pred)

print("Error en validation:", val_error)
print("\n💾 Guardando el modelo entrenado...")

# Guardamos el Modelo
joblib.dump(modelo, 'data/model_memory/cerebro_meteo_lluvia.pkl')

# Guardamos el orden exacto de las columnas
columnas_entrenamiento = list(X.columns)
joblib.dump(columnas_entrenamiento, 'data/model_memory/columnas_modelo_lluvia.pkl')

print("✅ Modelo guardado como 'cerebro_meteo_temperatura.pkl'")
print("✅ Columnas guardadas como 'columnas_modelo_temperatura.pkl'")
print("   Ya puedes usar 'app_prediccion.py'")





from sklearn.metrics import accuracy_score, classification_report, confusion_matrix





# ================================
# TEST RANDOM DE LLUVIA (DEBUG)
# ================================

idx = random.randint(0, len(dt) - 1)
fila_random = dt.iloc[[idx]].copy()

lluvia_real = int(fila_random["TARGET_Lluvia_Manana"].values[0])

X_test_random = pd.DataFrame(index=fila_random.index)
for col in columnas_entrenamiento:
    X_test_random[col] = fila_random[col] if col in fila_random.columns else 0

# Predicción binaria (0/1)
lluvia_predicha = int(modelo.predict(X_test_random)[0])

# Probabilidad (más informativo)
proba_lluvia = float(modelo.predict_proba(X_test_random)[0][1])

print("\n" + "=" * 60)
print(f"🧪 TEST FILA RANDOM (idx={idx})")
if "Fecha" in fila_random.columns:
    print(f"📅 Fecha: {fila_random['Fecha'].values[0]}")
print("-" * 60)
print(f"🌧️ REAL:        {lluvia_real}  ({'Sí' if lluvia_real==1 else 'No'})")
print(f"🌧️ PREDICHA:    {lluvia_predicha}  ({'Sí' if lluvia_predicha==1 else 'No'})")
print(f"🌧️ Prob lluvia: {proba_lluvia*100:.1f}%")
print("=" * 60)




# ================================
# TEST 100 FILAS RANDOM (DEBUG)
# ================================

import random
from sklearn.metrics import confusion_matrix

N = 100
indices = random.sample(range(len(dt)), k=min(N, len(dt)))

y_true = []
y_pred = []

for idx in indices:
    fila = dt.iloc[[idx]].copy()

    real = int(fila["TARGET_Lluvia_Manana"].values[0])

    X_row = pd.DataFrame(index=fila.index)
    for col in columnas_entrenamiento:
        X_row[col] = fila[col] if col in fila.columns else 0

    pred = int(modelo.predict(X_row)[0])

    y_true.append(real)
    y_pred.append(pred)

# Matriz de confusión
# [[TN, FP],
#  [FN, TP]]
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

# Métricas separadas
accuracy_global = (tp + tn) / (tp + tn + fp + fn)

acierto_llueve = tp / (tp + fn) if (tp + fn) > 0 else 0
acierto_no_llueve = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n" + "=" * 60)
print(f"🧪 TEST {len(indices)} FILAS RANDOM (LLUVIA)")
print("-" * 60)
print(f"🎯 Accuracy global: {accuracy_global*100:.2f}%")
print()
print(f"🌧️ Acierto CUANDO LLUEVE:     {acierto_llueve*100:.2f}%")
print(f"☀️ Acierto CUANDO NO LLUEVE:  {acierto_no_llueve*100:.2f}%")
print()
print("Matriz de confusión:")
print(cm)
print("=" * 60)
