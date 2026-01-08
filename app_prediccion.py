import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# IMPORTAMOS TUS HERRAMIENTAS
from data.scraper_prediccion import obtener_media_barcelona
from modelo_temperatura import entrenar_modelo_nuevo 

# === CONFIGURACIÓN ===
RUTA_MODELO = "data/model_memory/cerebro_meteo_temperatura.pkl"
RUTA_COLUMNAS = "data/model_memory/columnas_modelo_temperatura.pkl"
RUTA_HISTORICO = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv"


def pipeline_completo():
    print("🏭 INICIANDO PIPELINE DE PRODUCCIÓN")
    print("===================================")
    
    # ==============================================================================
    # FASE 1: ACTUALIZACIÓN DE DATOS (ETL + FEATURE ENGINEERING EXACTO)
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

    # A) DESCARGA Y UNIÓN
    if ultima_fecha < fecha_ayer:
        fecha_obj_str = fecha_ayer.strftime("%Y-%m-%d")
        print(f"📥 Descargando datos nuevos del {fecha_obj_str}...")
        
        df_nuevo_raw = obtener_media_barcelona(fecha_obj_str)
        
        if df_nuevo_raw is not None:
            df_nuevo_raw['Fecha'] = pd.to_datetime(df_nuevo_raw['Fecha'])
            
            # Identificamos columnas base (las que trae el scraper, no las calculadas)
            cols_raw = [
                'Fecha', 'Temp_Media_C', 'Temp_Maxima_C', 'Temp_Minima_C', 
                'Humedad_Media_Pct', 'Precip_Total_mm', 'Viento_Maximo_kmh', 
                'Viento_Direccion_Grados', 'Presion_Media_hPa', 'Irrad_Solar_MJm2'
            ]
            
            # Filtramos solo las raw del histórico para evitar duplicar columnas calculadas viejas
            # (Las recalcularemos todas ahora para asegurar consistencia)
            cols_existentes_raw = [c for c in cols_raw if c in df_hist.columns]
            df_hist_raw = df_hist[cols_existentes_raw].copy()
            
            # Unimos
            df_full = pd.concat([df_hist_raw, df_nuevo_raw], ignore_index=True)
            df_full = df_full.sort_values('Fecha').set_index('Fecha')
            nuevo_dato_agregado = True
        else:
            print("⚠️ No se pudieron descargar datos. Usando histórico existente.")
            df_full = df_hist.set_index('Fecha')
    else:
        print("✅ Datos al día.")
        df_full = df_hist.set_index('Fecha')

    # B) FEATURE ENGINEERING (COPIA EXACTA DE TU ARCHIVO GLOBAL)
    # -------------------------------------------------------------------------
    print("⚙️ Ejecutando Ingeniería de Características (Exact Match)...")
    
    # 1. Fechas Cíclicas
    df_full['Dia_Del_Ano'] = df_full.index.dayofyear
    df_full['Dia_Sin'] = np.sin(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)
    df_full['Dia_Cos'] = np.cos(2 * np.pi * df_full['Dia_Del_Ano'] / 365.0)

    # 2. Viento Cíclico
    if 'Viento_Direccion_Grados' in df_full.columns:
        # Rellenar nulos temporalmente para el cálculo
        df_full['Viento_Direccion_Grados'] = df_full['Viento_Direccion_Grados'].ffill()
        rads = np.deg2rad(df_full['Viento_Direccion_Grados'])
        df_full['Viento_Dir_Sin'] = np.sin(rads)
        df_full['Viento_Dir_Cos'] = np.cos(rads)

    # 3. Lluvia Binaria
    if 'Precip_Total_mm' in df_full.columns:
        df_full['Precip_Total_mm'] = df_full['Precip_Total_mm'].fillna(0)
        df_full['Lluvia_Binaria'] = (df_full['Precip_Total_mm'] > 0.1).astype(int)

    # 4. Inercia y Tendencias (Medias Móviles y Deltas)
    cols_tendencia = ['Temp_Media_C', 'Presion_Media_hPa', 'Viento_Maximo_kmh']
    ventanas = [3, 7]

    for col in cols_tendencia:
        if col in df_full.columns:
            # Interpolación suave para huecos pequeños antes de calcular medias
            df_full[col] = df_full[col].interpolate(limit_direction='both')
            
            for v in ventanas:
                # Rolling window (min_periods=1 para no perder datos al inicio)
                df_full[f'{col}_Media_{v}dias'] = df_full[col].rolling(window=v, min_periods=1).mean()
            
            # Delta
            df_full[f'{col}_Delta'] = df_full[col].diff().fillna(0)

    # 5. TARGETS (Para guardar en el histórico y usar en futuros entrenamientos)
    # Shift -1: El target de HOY es la temperatura de MAÑANA
    df_full['TARGET_Temp_Manana'] = df_full['Temp_Media_C'].shift(-1)
    if 'Lluvia_Binaria' in df_full.columns:
        df_full['TARGET_Lluvia_Manana'] = df_full['Lluvia_Binaria'].shift(-1)

    # -------------------------------------------------------------------------
    
    # C) GUARDADO
    if nuevo_dato_agregado:
        # Guardamos todo. La última fila tendrá TARGET = NaN, pero eso es correcto
        # (porque no sabemos el futuro de la última fila hasta mañana)
        df_to_save = df_full.reset_index()
        df_to_save.to_csv(RUTA_HISTORICO, index=False)
        print(f"💾 Dataset Maestro actualizado y guardado.")

    # ==============================================================================
    # FASE 2: RE-ENTRENAMIENTO (LUNES)
    # ==============================================================================
    hoy = datetime.now()
    if hoy.weekday() == 0:  # 0 = Lunes
        print("\n📅 Es Lunes: Ejecutando Re-entrenamiento Semanal...")
        entrenar_modelo_nuevo()
    else:
        print(f"\n📅 Hoy es {hoy.strftime('%A')}. No toca re-entrenar.")

    # ==============================================================================
    # FASE 3: PREDICCIÓN
    # ==============================================================================
    print("\n🔮 GENERANDO PREDICCIÓN...")
    
    if not os.path.exists(RUTA_MODELO):
        print("⚠️ No hay modelo. Entrenando por primera vez...")
        entrenar_modelo_nuevo()

    try:
        modelo = joblib.load(RUTA_MODELO)
        cols_modelo = joblib.load(RUTA_COLUMNAS)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    # Usamos la última fila del dataframe que acabamos de procesar
    fila_prediccion = df_full.iloc[[-1]].copy()
    
    # Alinear columnas con el modelo (Rellenar con 0 si falta algo raro)
    X_final = pd.DataFrame(index=fila_prediccion.index)
    for col in cols_modelo:
        if col in fila_prediccion.columns:
            X_final[col] = fila_prediccion[col]
        else:
            X_final[col] = 0
            
    # Predecir
    prediccion = modelo.predict(X_final)[0]
    
    # Fecha para la que predecimos (Día de los datos + 1)
    fecha_target = fila_prediccion.index[0] + timedelta(days=1)
    
    print("-" * 50)
    print(f"📅 DATOS BASE DEL:      {fila_prediccion.index[0].date()}")
    print("-" * 50)
    print(f"🌡️  Temperatura Ayer:    {fila_prediccion['Temp_Media_C'].values[0]:.2f} °C")
    print(f"🌧️  Llovió ayer?:        {'SÍ' if fila_prediccion['Lluvia_Binaria'].values[0] == 1 else 'NO'}")
    print("-" * 50)
    print(f"🚀  PREDICCIÓN PARA HOY/MAÑANA ({fecha_target.date()}):")
    print(f"    👉 {prediccion:.2f} °C")
    print("=" * 50)

if __name__ == "__main__":
    pipeline_completo()