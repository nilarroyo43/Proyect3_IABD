import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from data.scraper_prediccion import obtener_media_barcelona 

# === CONFIGURACIÓN ===
RUTA_MODELO = "data/model_memory/cerebro_meteo.pkl"
RUTA_COLUMNAS = "data/model_memory/columnas_modelo.pkl"
RUTA_HISTORICO = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv" 

def predecir_tiempo(fecha_target_str):
    """
    fecha_target_str: La fecha que el usuario QUIERE SABER (El Futuro).
    Ej: Input "2025-12-19".
    
    Lógica interna:
    1. Calcula el día anterior (18).
    2. Baja datos reales del 18.
    3. Predice el 19.
    """
    
    # 1. CÁLCULO DE FECHAS (UX MAGIA)
    try:
        fecha_target_obj = datetime.strptime(fecha_target_str, "%Y-%m-%d")
        # Restamos 1 día para saber qué datos buscar en Meteocat
        fecha_datos_obj = fecha_target_obj - timedelta(days=1)
        fecha_datos_str = fecha_datos_obj.strftime("%Y-%m-%d")
    except ValueError:
        print("❌ Error: Formato de fecha incorrecto. Usa AAAA-MM-DD (ej: 2025-12-19)")
        return

    print(f"\n🔮 Objetivo: Predecir el tiempo para el {fecha_target_str}")
    print(f"🕵️  Estrategia: Buscando datos reales de AYER ({fecha_datos_str})...")

    # 2. CARGAR CEREBRO
    if not os.path.exists(RUTA_MODELO):
        print("❌ Error: Falta 'cerebro_meteo.pkl'. Ejecuta entrenar_modelo.py")
        return

    modelo = joblib.load(RUTA_MODELO)
    cols_entrenamiento = joblib.load(RUTA_COLUMNAS)
    
    # Cargar histórico para las medias móviles
    df_hist = pd.read_csv(RUTA_HISTORICO)
    df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
    
    # 3. SCRAPING (Usamos la fecha RESTADA)
    df_hoy_raw = obtener_media_barcelona(fecha_datos_str)
    
    if df_hoy_raw is None:
        print(f"⚠️ No se encontraron datos para el día {fecha_datos_str}.")
        print("   (Recuerda: Para predecir el 19, Meteocat debe tener publicados ya los datos del 18).")
        return

    df_hoy_raw['Fecha'] = pd.to_datetime(df_hoy_raw['Fecha'])

    # 4. UNIR CON HISTORIA (Ingeniería de datos)
    cols_raw = [c for c in df_hoy_raw.columns if c in df_hist.columns]
    contexto = df_hist[cols_raw].tail(14).copy()
    
    df_full = pd.concat([contexto, df_hoy_raw], ignore_index=True)
    df_full = df_full.sort_values('Fecha').set_index('Fecha')
    
    # 5. RE-CALCULAR FEATURES
    # A) Ciclos
    df_full['Dia_Del_Ano'] = df_full.index.dayofyear
    df_full['Dia_Sin'] = np.sin(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)
    df_full['Dia_Cos'] = np.cos(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)
    
    if 'Viento_Direccion_Grados' in df_full.columns:
        df_full['Viento_Direccion_Grados'] = df_full['Viento_Direccion_Grados'].ffill()
        rads = np.deg2rad(df_full['Viento_Direccion_Grados'])
        df_full['Viento_Dir_Sin'] = np.sin(rads)
        df_full['Viento_Dir_Cos'] = np.cos(rads)

    # B) Lluvia
    if 'Precip_Total_mm' in df_full.columns:
        df_full['Precip_Total_mm'] = df_full['Precip_Total_mm'].fillna(0)
        df_full['Llovio_Hoy'] = (df_full['Precip_Total_mm'] > 0.1).astype(int)

    # C) Medias Móviles y Deltas
    ventanas = [3, 7]
    for col in ['Temp_Media_C', 'Presion_Media_hPa', 'Viento_Maximo_kmh']:
        if col in df_full.columns:
            df_full[col] = df_full[col].interpolate()
            for v in ventanas:
                df_full[f'{col}_Media_{v}dias'] = df_full[col].rolling(window=v).mean()
            df_full[f'{col}_Delta'] = df_full[col].diff().fillna(0)

    # 6. FILTRAR FILA PARA EL MODELO
    fila_prediccion = df_full.iloc[[-1]].copy()
    X_final = pd.DataFrame(index=fila_prediccion.index)
    
    for col in cols_entrenamiento:
        if col in fila_prediccion.columns:
            X_final[col] = fila_prediccion[col]
        else:
            X_final[col] = 0 
            
    # 7. PREDICCIÓN FINAL
    temp_predicha = modelo.predict(X_final)[0]
    
    print("\n" + "="*50)
    print(f"📅 PARA LA FECHA:         {fecha_target_str}")
    print("-" * 50)
    print(f"🌡️  Temperatura Prevista:  {temp_predicha:.2f} °C")
    print("="*50 + "\n")
    
    print(f"(Basado en los datos reales del día anterior: {fila_prediccion['Temp_Media_C'].values[0]:.1f}°C)")

# === EJECUCIÓN ===
if __name__ == "__main__":
    print("--- 🌦️  PREDICTOR METEOCAT BCN ---")
    fecha = input("¿Qué día quieres predecir? (YYYY-MM-DD): ")
    # Ejemplo: El usuario pone 2025-12-19
    predecir_tiempo(fecha)