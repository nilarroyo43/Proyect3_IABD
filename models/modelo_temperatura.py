import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib


# ========================
# RUTAS
# ========================
RUTA_DATASET_MASTER = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv"

RUTA_MODELO_MAX = "data/model_memory/cerebro_meteo_temp_maxima.pkl"
RUTA_MODELO_MEDIA = "data/model_memory/cerebro_meteo_temp_media.pkl"
RUTA_MODELO_MIN = "data/model_memory/cerebro_meteo_temp_minima.pkl"

RUTA_COLS_PKL = "data/model_memory/columnas_modelo_temperatura.pkl"


def entrenar_modelos_temperatura():
    print("\nINICIANDO PROCESO DE RE-ENTRENAMIENTO SEMANAL (3 MODELOS)...")

    # 1. Cargar Dataset
    if not os.path.exists(RUTA_DATASET_MASTER):
        print(f"Error: No encuentro {RUTA_DATASET_MASTER}")
        return

    dt = pd.read_csv(RUTA_DATASET_MASTER)

    # 2. Crear targets next-day
    dt["TARGET_Temp_Maxima_Manana"] = dt["Temp_Maxima_C"].shift(-1)
    dt["TARGET_Temp_Media_Manana"]  = dt["Temp_Media_C"].shift(-1)
    dt["TARGET_Temp_Minima_Manana"] = dt["Temp_Minima_C"].shift(-1)

    dt = dt.dropna(subset=[
        "TARGET_Temp_Maxima_Manana",
        "TARGET_Temp_Media_Manana",
        "TARGET_Temp_Minima_Manana"
    ])

    if dt.empty:
        print("⚠️ Dataset vacío tras limpiar NaNs. Abortando.")
        return

    # 3. Columnas prohibidas (idéntico criterio que tu modelo actual)
    cols_a_borrar_de_X = [
        "Fecha",
        "TARGET_Temp_Maxima_Manana",
        "TARGET_Temp_Media_Manana",
        "TARGET_Temp_Minima_Manana",
        "TARGET_Temp_Manana",       # por si existe de modelo anterior
        "TARGET_Lluvia_Manana",
        "Dia_Del_Ano",
        "Viento_Direccion_Grados",
        "Precip_Total_mm"
    ]

    cols_a_borrar = [c for c in cols_a_borrar_de_X if c in dt.columns]

    X = dt.drop(columns=cols_a_borrar)

    # Guardamos columnas una sola vez
    joblib.dump(list(X.columns), RUTA_COLS_PKL)

    # 4. Targets y rutas de guardado
    targets = {
        "MAXIMA": (dt["TARGET_Temp_Maxima_Manana"], RUTA_MODELO_MAX),
        "MEDIA":  (dt["TARGET_Temp_Media_Manana"],  RUTA_MODELO_MEDIA),
        "MINIMA": (dt["TARGET_Temp_Minima_Manana"], RUTA_MODELO_MIN)
    }

    # 5. Entrenamiento de los tres modelos
    for nombre, (y, ruta_guardado) in targets.items():

        print(f"\n🔥 Entrenando modelo TEMPERATURA {nombre}...")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.30, random_state=42
        )

        print(f"    Entrenando Random Forest con {len(X_train)} registros...")

        modelo = RandomForestRegressor(
            n_estimators=200,
            n_jobs=-1,
            random_state=40
        )

        modelo.fit(X_train, y_train)

        # Validación
        y_pred = modelo.predict(X_test)

        mae  = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)

        print(f"    MAE : {mae:.4f} °C")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R²  : {r2:.4f}")

        # Guardar modelo
        joblib.dump(modelo, ruta_guardado)
        print(f"    Modelo guardado en: {ruta_guardado}")

    print("\n✅ RE-ENTRENAMIENTO FINALIZADO. 3 MODELOS ACTUALIZADOS.")


if __name__ == "__main__":
    entrenar_modelos_temperatura()
