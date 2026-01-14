import pandas as pd
import numpy as np
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent


# =================================================================
# CONFIGURACIÓN
# =================================================================
ARCHIVOS_ENTRADA = [
    HERE / "raw_datasets"/"meteocat_D5_resumen_historico.csv",
    HERE / "raw_datasets"/"meteocat_X4_resumen_historico.csv",
    HERE / "raw_datasets"/"meteocat_X8_resumen_historico.csv"
]


print(" INICIANDO AUDITORÍA Y LIMPIEZA DE DATOS")
print("===========================================")

for archivo in ARCHIVOS_ENTRADA:
    if not os.path.exists(archivo):
        print(f"  Saltando {archivo} (No existe)")
        continue

    print(f"\nPROCESANDO: {archivo}")
    df = pd.read_csv(archivo)
    
    # Preparar Fecha (Esencial para ordenar)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    df = df.sort_values('Fecha').set_index('Fecha')

    # -----------------------------------------------------------
    # PASO 1: AUDITORÍA DE CALIDAD 
    # -----------------------------------------------------------
    total_celdas = df.size
    total_nulos = df.isnull().sum().sum()
    pct_inventado = (total_nulos / total_celdas) * 100
    
    print(f"   ---------------------------------------------")
    print(f"     REPORTE DE CALIDAD :")
    print(f"   - Filas totales: {len(df)}")
    print(f"   - Datos faltantes totales: {total_nulos} celdas")
    print(f"   - Porcentaje de datos a imputar: {pct_inventado:.4f}%")
    
    if total_nulos > 0:
        print(f"   - Columnas más afectadas:")
        print(df.isnull().sum()[df.isnull().sum() > 0])
    else:
        print("   - Estado: PERFECTO ")
    print(f"   ---------------------------------------------")

    # -----------------------------------------------------------
    # PASO 2: LIMPIEZA E IMPUTACIÓN (RELLENAR HUECOS)
    # -----------------------------------------------------------
    
    # A) Lluvia: Si falta, es 0 (Asunción segura)
    if 'Precip_Total_mm' in df.columns:
        df['Precip_Total_mm'] = df['Precip_Total_mm'].fillna(0)

    # B) Viento Dirección: Copiar el anterior (ffill)
    # No se debe interpolar grados numéricamente
    if 'Viento_Direccion_Grados' in df.columns:
        df['Viento_Direccion_Grados'] = df['Viento_Direccion_Grados'].ffill().bfill()

    # C) Resto de variables: Interpolación temporal 
    cols_continuas = [c for c in df.select_dtypes(include=np.number).columns 
                      if 'Direccion' not in c and 'Precip' not in c]
    df[cols_continuas] = df[cols_continuas].interpolate(method='time')

    # D) Red de Seguridad: Medias Mensuales
    # Si interpolación falla, usar media del mes
    df['Mes_Aux'] = df.index.month
    for col in df.select_dtypes(include=np.number).columns:
        if col != 'Mes_Aux':
            media_mensual = df.groupby('Mes_Aux')[col].transform('mean')
            df[col] = df[col].fillna(media_mensual)
    df = df.drop(columns=['Mes_Aux'])

    # -----------------------------------------------------------
    # PASO 3: GUARDADO
    # -----------------------------------------------------------
    nombre_limpio = f"clean_{archivo.name}"
    ruta_carpeta = HERE / "clean_datasets"
    ruta_completa = ruta_carpeta / nombre_limpio
    df.to_csv(ruta_completa)
    print(f"    Archivo limpio guardado: {nombre_limpio}")

print("\n FASE 1 COMPLETADA.")
