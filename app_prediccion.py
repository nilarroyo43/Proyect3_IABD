import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# IMPORTAMOS TUS HERRAMIENTAS
from data.scraper_prediccion import obtener_media_barcelona
from modelo_temperatura import entrenar_modelo_temperatura 
from modelo_lluvia import entrenar_modelo_lluvia

# === CONFIGURACIÓN ===
# Rutas unificadas para ambos modelos
RUTA_MODELO_TEMPERATURA = "data/model_memory/cerebro_meteo_temperatura.pkl"
RUTA_COLUMNAS_TEMPERATURA = "data/model_memory/columnas_modelo_temperatura.pkl"

RUTA_MODELO_LLUVIA = "data/model_memory/cerebro_meteo_lluvia.pkl"
RUTA_COLUMNAS_LLUVIA = "data/model_memory/columnas_modelo_lluvia.pkl"

RUTA_HISTORICO = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv" 

def pipeline_completo():
    print("🏭 INICIANDO PIPELINE DE PRODUCCIÓN (TEMP + LLUVIA)")
    print("===================================================")
    
    # ==============================================================================
    # FASE 1: ACTUALIZACIÓN DE DATOS (ETL + FEATURE ENGINEERING)
    # ==============================================================================
    if not os.path.exists(RUTA_HISTORICO):
        print("❌ Error crítico: No existe el dataset maestro.")
        return

    df_hist = pd.read_csv(RUTA_HISTORICO)
    df_hist['Fecha'] = pd.to_datetime(df_hist['Fecha'])
    df_hist = df_hist.sort_values('Fecha')
    
    ultima_fecha = df_hist['Fecha'].iloc[-1]
    # Buscamos datos de AYER (porque hoy aún no ha terminado)
    fecha_ayer = pd.to_datetime(datetime.now().date() - timedelta(days=1))
    
    nuevo_dato_agregado = False

    # A) DESCARGA Y UNIÓN (Lógica Automática)
    if ultima_fecha < fecha_ayer:
        fecha_obj_str = fecha_ayer.strftime("%Y-%m-%d")
        print(f"📥 Descargando datos nuevos del {fecha_obj_str}...")
        
        df_nuevo_raw = obtener_media_barcelona(fecha_obj_str)
        
        if df_nuevo_raw is not None:
            df_nuevo_raw['Fecha'] = pd.to_datetime(df_nuevo_raw['Fecha'])
            
            cols_raw = [
                'Fecha', 'Temp_Media_C', 'Temp_Maxima_C', 'Temp_Minima_C', 
                'Humedad_Media_Pct', 'Precip_Total_mm', 'Viento_Maximo_kmh', 
                'Viento_Direccion_Grados', 'Presion_Media_hPa', 'Irrad_Solar_MJm2'
            ]
            
            cols_existentes_raw = [c for c in cols_raw if c in df_hist.columns]
            df_hist_raw = df_hist[cols_existentes_raw].copy()
            
            df_full = pd.concat([df_hist_raw, df_nuevo_raw], ignore_index=True)
            df_full = df_full.sort_values('Fecha').set_index('Fecha')
            nuevo_dato_agregado = True
        else:
            print("⚠️ No se pudieron descargar datos. Usando histórico existente.")
            df_full = df_hist.set_index('Fecha')
    else:
        print("✅ Datos al día.")
        df_full = df_hist.set_index('Fecha')

    # B) FEATURE ENGINEERING
    print("⚙️ Recalculando variables matemáticas...")
    
    df_full['Dia_Del_Ano'] = df_full.index.dayofyear
    df_full['Dia_Sin'] = np.sin(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)
    df_full['Dia_Cos'] = np.cos(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)

    if 'Viento_Direccion_Grados' in df_full.columns:
        df_full['Viento_Direccion_Grados'] = df_full['Viento_Direccion_Grados'].ffill()
        rads = np.deg2rad(df_full['Viento_Direccion_Grados'])
        df_full['Viento_Dir_Sin'] = np.sin(rads)
        df_full['Viento_Dir_Cos'] = np.cos(rads)

    if 'Precip_Total_mm' in df_full.columns:
        df_full['Precip_Total_mm'] = df_full['Precip_Total_mm'].fillna(0)
        df_full['Lluvia_Binaria'] = (df_full['Precip_Total_mm'] > 0.1).astype(int)

    cols_tendencia = ['Temp_Media_C', 'Presion_Media_hPa', 'Viento_Maximo_kmh']
    ventanas = [3, 7]

    for col in cols_tendencia:
        if col in df_full.columns:
            df_full[col] = df_full[col].interpolate(limit_direction='both')
            for v in ventanas:
                df_full[f'{col}_Media_{v}dias'] = df_full[col].rolling(window=v, min_periods=1).mean()
            df_full[f'{col}_Delta'] = df_full[col].diff().fillna(0)

    # Targets (Solo para guardado)
    df_full['TARGET_Temp_Manana'] = df_full['Temp_Media_C'].shift(-1)
    if 'Lluvia_Binaria' in df_full.columns:
        df_full['TARGET_Lluvia_Manana'] = df_full['Lluvia_Binaria'].shift(-1)

    # C) GUARDADO
    if nuevo_dato_agregado:
        df_to_save = df_full.reset_index()
        df_to_save.to_csv(RUTA_HISTORICO, index=False)
        print(f"💾 Dataset Maestro actualizado y guardado.")

    # ==============================================================================
    # FASE 2: RE-ENTRENAMIENTO (LUNES)
    # ==============================================================================
    hoy = datetime.now()
    if hoy.weekday() == 0:
        print("\n📅 Es Lunes: Ejecutando Re-entrenamiento Semanal...")
        # NOTA: Asegúrate de que 'entrenar_modelo_nuevo' también sepa entrenar
        # el modelo de lluvia, o llama aquí a una función extra para la lluvia.
        entrenar_modelo_temperatura()
        entrenar_modelo_lluvia()
    else:
        print(f"\n📅 Hoy es {hoy.strftime('%A')}. No toca re-entrenar.")

    # ==============================================================================
    # FASE 3: PREDICCIÓN DUAL (TEMP + LLUVIA)
    # ==============================================================================
    print("\n🔮 GENERANDO PREDICCIONES...")
    
    # 1. Cargar Modelos
    if not os.path.exists(RUTA_MODELO_TEMPERATURA) or not os.path.exists(RUTA_MODELO_LLUVIA):
        print("⚠️ Faltan modelos (.pkl). Asegúrate de haberlos entrenado.")
        # Aquí podrías forzar el entreno si quieres
    
    try:
        mod_temp = joblib.load(RUTA_MODELO_TEMPERATURA)
        cols_temp = joblib.load(RUTA_COLUMNAS_TEMPERATURA)
        
        mod_lluvia = joblib.load(RUTA_MODELO_LLUVIA)
        cols_lluvia = joblib.load(RUTA_COLUMNAS_LLUVIA)
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return

    # 2. Preparar Datos (Última fila)
    fila_prediccion = df_full.iloc[[-1]].copy()
    
    # Preparar X para Temperatura
    X_temp = pd.DataFrame(index=fila_prediccion.index)
    for col in cols_temp:
        X_temp[col] = fila_prediccion[col] if col in fila_prediccion else 0
        
    # Preparar X para Lluvia (Pueden ser columnas distintas)
    X_lluvia = pd.DataFrame(index=fila_prediccion.index)
    for col in cols_lluvia:
        X_lluvia[col] = fila_prediccion[col] if col in fila_prediccion else 0

    # 3. Predecir
    pred_temp = mod_temp.predict(X_temp)[0]
    
    # Lluvia: Usamos probabilidad para ser más precisos
    try:
        prob_lluvia = mod_lluvia.predict_proba(X_lluvia)[0][1] # Probabilidad de clase 1 (Sí)
    except:
        prob_lluvia = mod_lluvia.predict(X_lluvia)[0] # Fallback si no es clasificador
        
    umbral_lluvia = 0.30 # Si hay más de 30% de probabilidad, avisamos
    es_lluvia = prob_lluvia > umbral_lluvia
    
    fecha_target = fila_prediccion.index[0] + timedelta(days=1)
    
    print("-" * 50)
    print(f"📅 BASADO EN DATOS DE:      {fila_prediccion.index[0].date()}")
    print("-" * 50)
    print(f"🌡️  Temperatura Ayer:    {fila_prediccion['Temp_Media_C'].values[0]:.2f} °C")
    print("-" * 50)
    print(f"🚀  PREDICCIÓN PARA HOY/MAÑANA ({fecha_target.date()}):")
    print(f"    🌡️  Temperatura:  {pred_temp:.2f} °C")
    print(f"    💧  Lluvia:       {prob_lluvia*100:.1f}% de probabilidad")
    
    if es_lluvia:
        print("    ☔  AVISO: ¡Coge el paraguas!")
    else:
        print("    ☀️  Tranquilo, probablemente no llueva.")
    print("=" * 50)

if __name__ == "__main__":
    pipeline_completo()