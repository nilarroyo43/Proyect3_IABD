import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error
import joblib
import os
import random

# CONFIGURACIÓN DE RUTAS
RUTA_DATASET_MASTER = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv"
RUTA_MODELO_LLUVIA_PKL = "data/model_memory/cerebro_meteo_lluvia.pkl"
RUTA_COLS_LLUVIA_PKL = "data/model_memory/columnas_modelo_lluvia.pkl"

def entrenar_modelo_lluvia():
    """
    Función PRINCIPAL: Carga datos, entrena y guarda el .pkl
    Esta es la única parte que le interesa a la App automática.
    """
    print("\n☔ INICIANDO RE-ENTRENAMIENTO MODELO LLUVIA...")
    
    if not os.path.exists(RUTA_DATASET_MASTER):
        print(f"❌ Error: No encuentro {RUTA_DATASET_MASTER}")
        return None, None, None # Devolvemos None si falla

    dt = pd.read_csv(RUTA_DATASET_MASTER)
    dt = dt.dropna(subset=["TARGET_Lluvia_Manana"])

    # 1. Limpieza
    cols_a_borrar_de_X = [
        "Fecha", 
        "TARGET_Temp_Manana",     
        "TARGET_Lluvia_Manana", 
        "Dia_Del_Ano",              
        "Viento_Direccion_Grados", 
        "Precip_Total_mm" 
    ]
    cols_borrar = [c for c in cols_a_borrar_de_X if c in dt.columns]
    X = dt.drop(columns=cols_borrar)
    
    # Limpieza de Nulos en Target
    dt_clean = dt.dropna(subset=["TARGET_Lluvia_Manana"])
    X = X.loc[dt_clean.index]
    y = dt_clean["TARGET_Lluvia_Manana"]

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=40, stratify=y
    )

    # 3. Entrenamiento
    print(f"   🧠 Entrenando Clasificador con {len(X_train)} registros...")
    modelo = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=40
    )
    modelo.fit(X_train, y_train)

    # 4. Guardado
    os.makedirs(os.path.dirname(RUTA_MODELO_LLUVIA_PKL), exist_ok=True)
    joblib.dump(modelo, RUTA_MODELO_LLUVIA_PKL)
    
    cols_entrenamiento = list(X.columns)
    joblib.dump(cols_entrenamiento, RUTA_COLS_LLUVIA_PKL)

    print("✅ RE-ENTRENAMIENTO LLUVIA FINALIZADO.")
    
    # Devolvemos los datos para poder hacer tests manuales si queremos
    return modelo, X_test, y_test, dt_clean, cols_entrenamiento

def ejecutar_pruebas_visuales(modelo, dt_completo, cols_entrenamiento):
    """
    Tus pruebas originales. Solo se ejecutan si tú quieres.
    """
    print("\n" + "="*60)
    print("🔬 MODO DEBUG: EJECUTANDO PRUEBAS VISUALES")
    print("="*60)
    
    # --- TEST 1: FILA RANDOM ---
    idx = random.randint(0, len(dt_completo) - 1)
    fila_random = dt_completo.iloc[[idx]].copy()
    lluvia_real = int(fila_random["TARGET_Lluvia_Manana"].values[0])

    X_test_random = pd.DataFrame(index=fila_random.index)
    for col in cols_entrenamiento:
        X_test_random[col] = fila_random[col] if col in fila_random.columns else 0

    lluvia_predicha = int(modelo.predict(X_test_random)[0])
    proba_lluvia = float(modelo.predict_proba(X_test_random)[0][1])

    print(f"\n🧪 TEST FILA RANDOM (Fecha: {fila_random['Fecha'].values[0]})")
    print(f"   Real: {lluvia_real} | Predicha: {lluvia_predicha} | Prob: {proba_lluvia:.2f}")

    # --- TEST 2: 100 FILAS ---
    print(f"\n🧪 TEST ESTADÍSTICO (100 Muestras)...")
    indices = random.sample(range(len(dt_completo)), k=min(100, len(dt_completo)))
    y_true, y_pred = [], []

    for idx in indices:
        fila = dt_completo.iloc[[idx]].copy()
        real = int(fila["TARGET_Lluvia_Manana"].values[0])
        
        X_row = pd.DataFrame(index=fila.index)
        for col in cols_entrenamiento:
            X_row[col] = fila[col] if col in fila.columns else 0
            
        pred = int(modelo.predict(X_row)[0])
        y_true.append(real)
        y_pred.append(pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Manejo de división por cero
    acc_lluvia = tp / (tp + fn) if (tp + fn) > 0 else 0
    acc_no_lluvia = tn / (tn + fp) if (tn + fp) > 0 else 0

    print(f"   Matriz Confusión: {cm.tolist()}")
    print(f"   Acierto Lluvia: {acc_lluvia*100:.1f}% | Acierto Sol: {acc_no_lluvia*100:.1f}%")
    print("="*60)

# ==============================================================================
# PUNTO DE ENTRADA INTELIGENTE
# ==============================================================================
if __name__ == "__main__":
    # ESTO SOLO SE EJECUTA SI TÚ LE DAS AL PLAY A ESTE ARCHIVO
    # (La App NO ejecutará esto, solo importará la función de arriba)
    mod, _, _, dt_clean, cols = entrenar_modelo_lluvia()
    
    if mod is not None:
        ejecutar_pruebas_visuales(mod, dt_clean, cols)