import pandas as pd
import numpy as np


# 1. CARGA Y PREPARACIÓN BÁSICA
ARCHIVO_ENTRADA = "meteocat_D5_resumen_historico.csv" 
ARCHIVO_SALIDA = "dataset_entrenamiento_final.csv"

print("🔄 Cargando dataset...")
df = pd.read_csv(ARCHIVO_ENTRADA)

# Convertir Fecha y ordenar
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values('Fecha')
df = df.set_index('Fecha')

print(f"   Filas iniciales: {len(df)}")
print(f"   Nulos iniciales:\n{df.isnull().sum()}")


# 2. IMPUTACIÓN DE NULOS (ESTRATEGIA: MEDIA MENSUAL)
print("\n🛠️ Rellenando valores nulos...")

# Creamos columnas auxiliares para agrupar
df['Año'] = df.index.year
df['Mes'] = df.index.month

# Lista de columnas numéricas a limpiar (excluyendo las de fecha)
cols_meteo = [c for c in df.columns if c not in ['Año', 'Mes']]

for col in cols_meteo:
    # 1. Rellenar con la media del mes ESPECÍFICO de ese AÑO (ej: Enero 2020)
    # "Si falta un dato en Enero 2020, usa el promedio de Enero 2020"
    df[col] = df[col].fillna(df.groupby(['Año', 'Mes'])[col].transform('mean'))
    
    # 2. (Respaldo) Si todo el mes estaba vacío, rellenar con la media del mes HISTÓRICO
    # "Si falta todo Enero 2020, usa el promedio de todos los Eneros"
    df[col] = df[col].fillna(df.groupby(['Mes'])[col].transform('mean'))

# 3. (Respaldo final) Si queda algún hueco suelto, usar el día anterior (ffill)
df = df.ffill().bfill()

print("   ✅ Nulos eliminados.")


# 3. INGENIERÍA DE CARACTERÍSTICAS (FEATURES)
print("\n🧠 Generando variables inteligentes para el modelo...")

# A) Estacionalidad (El modelo necesita saber en qué punto del año está)
df['Dia_Del_Ano'] = df.index.dayofyear
# Usamos Seno/Coseno para que el modelo entienda que el día 365 está cerca del día 1 (Cíclico)
df['Dia_Sin'] = np.sin(2 * np.pi * df['Dia_Del_Ano'] / 365.0)
df['Dia_Cos'] = np.cos(2 * np.pi * df['Dia_Del_Ano'] / 365.0)

# B) Tendencias (Rolling Means) - ¿Venimos de una semana de calor?
# Media de los últimos 7 días (excluyendo el actual para no ensuciar, aunque al predecir mañana da igual)
ventanas = [3, 7] # 3 días y 1 semana
for col in ['Temp_Media_C', 'Presion_Media_hPa']:
    for ventana in ventanas:
        # Calculamos la media de los últimos X días
        df[f'{col}_Media_{ventana}dias'] = df[col].rolling(window=ventana).mean()

# C) Variación Diaria (Diferencia hoy - ayer)
# Ayuda a detectar cambios bruscos
df['Delta_Temp'] = df['Temp_Media_C'].diff()


# 4. CREAR EL TARGET (EL FUTURO)
print("\n🎯 Creando columna objetivo (Target)...")

# Queremos predecir la Temperatura Media de MAÑANA
df['TARGET_Temp_Manana'] = df['Temp_Media_C'].shift(-1)


# 5. LIMPIEZA FINAL Y GUARDADO
# Eliminar columnas auxiliares que ya no necesitamos para entrenar
# (Mantenemos Mes y Dia_Del_Ano si quieres, o usas Sin/Cos)
df = df.drop(columns=['Año']) 
df = df.drop(columns=['Viento_Maximo_kmh']) 

# Eliminar filas con NaNs generados por el Rolling (al principio) o el Shift (al final)
df = df.dropna()


print(f"\n📊 Dimensiones finales: {df.shape}")
print(f"   Variables listas para entrenar: {list(df.columns)}")

# Guardar
df.to_csv(ARCHIVO_SALIDA)
print(f"\n💾 Archivo guardado correctamente: {ARCHIVO_SALIDA}")
print("   Listo para entrenar tu Random Forest.")