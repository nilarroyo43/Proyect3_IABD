from pathlib import Path
import pandas as pd
import numpy as np
import glob
HERE = Path(__file__).resolve().parent

# =================================================================
# CONFIGURACIÓN
# =================================================================
PATRON_ARCHIVOS = HERE / "clean_datasets" / "clean_meteocat_*.csv" # Busca automáticamente los archivos limpios
ARCHIVO_FINAL = HERE / "training_datasets" / "dataset_entrenamiento_barcelona_MASTER.csv"

print("INICIANDO INGENIERÍA DE CARACTERÍSTICAS (FEATURE ENGINEERING)")
print("==============================================================")

# -----------------------------------------------------------
# PASO 1: FUSIÓN INTELIGENTE (MEDIA DE BARCELONA)
# -----------------------------------------------------------
archivos = glob.glob(str(PATRON_ARCHIVOS))
if not archivos:
    print("No encuentro archivos 'clean_...'.")
    exit()

print(f" Fusionando {len(archivos)} estaciones: {archivos}")

lista_dfs = []
for a in archivos:
    df = pd.read_csv(a)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    lista_dfs.append(df)

df_total = pd.concat(lista_dfs)

# Pre-procesamiento de Viento Vectorial (Antes de hacer la media)
if 'Viento_Direccion_Grados' in df_total.columns:
    rads = np.deg2rad(df_total['Viento_Direccion_Grados'])
    df_total['v_sin'] = np.sin(rads)
    df_total['v_cos'] = np.cos(rads)

# FUSIÓN: Agrupar por fecha y calcular media de todo
df_media = df_total.groupby('Fecha').mean()

# Recuperar Grados del Viento (Desde vectores medios)
if 'v_sin' in df_media.columns:
    angulo = np.arctan2(df_media['v_sin'], df_media['v_cos'])
    df_media['Viento_Direccion_Grados'] = (np.rad2deg(angulo) + 360) % 360
    df_media = df_media.drop(columns=['v_sin', 'v_cos'])

# -----------------------------------------------------------
# PASO 2: CREACIÓN DE FEATURES (VARIABLES PARA IA)
# -----------------------------------------------------------
print(" Generando variables predictivas...")

# A) Fechas Cíclicas (Calendario Circular)
df_media['Dia_Del_Ano'] = df_media.index.dayofyear
df_media['Dia_Sin'] = np.sin(2 * np.pi * df_media['Dia_Del_Ano'] / 365.0)
df_media['Dia_Cos'] = np.cos(2 * np.pi * df_media['Dia_Del_Ano'] / 365.0)

# B) Viento Cíclico (Para que el modelo entienda la dirección)
if 'Viento_Direccion_Grados' in df_media.columns:
    rads = np.deg2rad(df_media['Viento_Direccion_Grados'])
    df_media['Viento_Dir_Sin'] = np.sin(rads)
    df_media['Viento_Dir_Cos'] = np.cos(rads)

# C) Lluvia Binaria (¿Llovió? 1=Si, 0=No)
if 'Precip_Total_mm' in df_media.columns:
    df_media['Lluvia_Binaria'] = (df_media['Precip_Total_mm'] > 0.1).astype(int)

# D) Inercia y Tendencias (Medias Móviles y Deltas)
cols_tendencia = ['Temp_Media_C', 'Presion_Media_hPa', 'Viento_Maximo_kmh']
ventanas = [3, 7]

for col in cols_tendencia:
    if col in df_media.columns:
        # Medias móviles (Rolling Window)
        for v in ventanas:
            df_media[f'{col}_Media_{v}dias'] = df_media[col].rolling(window=v, min_periods=1).mean()
        
        # Delta (Cambio diario: Hoy - Ayer)
        df_media[f'{col}_Delta'] = df_media[col].diff().fillna(0)

# -----------------------------------------------------------
# PASO 3: TARGETS (EL FUTURO A PREDECIR)
# -----------------------------------------------------------
print(" Generando Targets (Futuro)...")

# Target 1: Temperatura de Mañana
df_media['TARGET_Temp_Manana'] = df_media['Temp_Media_C'].shift(-1)

# Target 2: Lluvia de Mañana (Binario 0/1) <-- NUEVO
if 'Lluvia_Binaria' in df_media.columns:
    df_media['TARGET_Lluvia_Manana'] = df_media['Lluvia_Binaria'].shift(-1)

# LIMPIEZA FINAL: Borrar la última fila (no tiene futuro conocido)
# Nos aseguramos de borrar si falta CUALQUIERA de los targets importantes
cols_targets = ['TARGET_Temp_Manana']
if 'TARGET_Lluvia_Manana' in df_media.columns:
    cols_targets.append('TARGET_Lluvia_Manana')

df_media = df_media.dropna(subset=cols_targets)

# -----------------------------------------------------------
# PASO 4: GUARDADO
# -----------------------------------------------------------
df_media.to_csv(ARCHIVO_FINAL)

print(f"\n EXCELENTE. Dataset Maestro guardado en: {ARCHIVO_FINAL}")
print(f"   - Dimensiones finales: {df_media.shape}")
print(f"   - Listo para entrenar Random Forest.")
print(f"   - IMPORTANTE: En el entrenamiento, ELIMINA de X estas columnas:")
print(f"     ['Fecha, TARGET_Temp_Manana, Dia_Del_Ano, Viento_Direccion_Grados, Precip_Total_mm, TARGET_Lluvia_Manana']")