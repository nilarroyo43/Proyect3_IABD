import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm 
import warnings
import numpy as np # Necesario para usar np.nan

# Suprimir advertencias de read_html sobre la transposición
warnings.filterwarnings("ignore", category=UserWarning)

# 1. CONFIGURACIÓN INICIAL Y MAPEO
URL_BASE = "https://www.meteo.cat/observacions/xema/dades?"
ESTACION_ID = "D5" # Barcelona - Av. Lluís Companys

# --- AJUSTE CRÍTICO DEL ÍNDICE ---
TABLA_INDEX_RESUMEN = 0 

# Mapeo para renombrar y limpiar las columnas
# La columna 0 de la tabla de Meteocat -> Columna final del CSV
MAPEO_COLUMNAS = {
    'Temperatura mitjana': 'Temp_Media_C',
    'Temperatura màxima': 'Temp_Maxima_C',
    'Temperatura mínima': 'Temp_Minima_C',
    'Humitat relativa mitjana': 'Humedad_Media_Pct',
    'Precipitació acumulada': 'Precip_Total_mm',
    'Ratxa màxima del vent (10 m)': 'Viento_Maximo_kmh', 
    'Pressió atmosfèrica mitjana': 'Presion_Media_hPa',
    'Irradiació solar global': 'Irrad_Solar_MJm2'
}

# PRUEBA PILOTO (2 días)
#FECHA_INICIO = datetime(2024, 1, 1)
#FECHA_FIN = datetime(2024, 1, 2).date() # Asegura un rango pequeño

# EJECUCIÓN COMPLETA
FECHA_INICIO = datetime(2000, 1, 1)
FECHA_FIN = datetime.now().date() - timedelta(days=1) 

TIEMPO_ESPERA = 1.5 # Pausa ética entre peticiones (en segundos)
datos_diarios = []
fechas_a_scrapear = pd.date_range(start=FECHA_INICIO, end=FECHA_FIN, freq='D')

print(f"Iniciando extracción de {len(fechas_a_scrapear)} resúmenes diarios.")

# 2. BUCLE DE EXTRACCIÓN Y LIMPIEZA
for fecha_dt in tqdm(fechas_a_scrapear):
    
    fecha_str = fecha_dt.strftime('%Y-%m-%d')
    url_a_scrapear = (
        f"{URL_BASE}codi={ESTACION_ID}&dia={fecha_str}T00:00Z"
    )
    
    try:
        time.sleep(TIEMPO_ESPERA)  
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url_a_scrapear, headers=headers, timeout=10)
        response.raise_for_status() 
        
        tablas_encontradas = pd.read_html(response.text)

        if len(tablas_encontradas) > TABLA_INDEX_RESUMEN:
            
            df_resumen = tablas_encontradas[TABLA_INDEX_RESUMEN].copy() 
            
            # 1. Selección y Transposición: Usamos columnas 0 (Variable) y 1 (Valor)
            df_resumen = df_resumen[[0, 1]]
            df_resumen.columns = ['Variable', 'Valor']
            df_pivotado = df_resumen.set_index('Variable').T 
            
            # 2. Renombrado y Limpieza de unidades
        df_fila = pd.DataFrame()

        for col_cat, col_final in MAPEO_COLUMNAS.items():
            if col_cat in df_pivotado.columns:
                
                # Obtenemos el valor como un objeto Series de Pandas para poder usar .str
                valor_serie = df_pivotado[col_cat].iloc[0] 
                
                # --- GESTIÓN DE NULOS: REEMPLAZAR "sense dades" ---
                if str(valor_serie).strip().lower() == "sense dades":
                    limpio = np.nan
                else:
                    # 1. Aplicar la limpieza de strings a la serie temporal
                    # Creamos una serie temporal para asegurar el uso de .str.replace
                    limpieza_temporal = pd.Series(str(valor_serie))
                    
                    # 2. Limpiamos unidades y texto (se aplica a la serie temporal)
                    limpio_str = (
                        limpieza_temporal
                        .str.replace(r'[^\d\.\-]+', '', regex=True) # Conserva dígitos, punto decimal y signo menos
                        .str.strip()
                    )
                    
                    # 3. Convertir a flotante (si hay error en la conversión, se convierte en NaN)
                    try:
                        # El valor limpio es el primer (y único) elemento de la serie resultante
                        limpio = float(limpio_str.iloc[0]) 
                    except ValueError:
                        # Si tras la limpieza queda una cadena vacía o inválida (ej. solo '-'), convertir a NaN
                        limpio = np.nan 
                        
                # Asignar el valor (limpio o NaN) al DataFrame final de la fila
                df_fila[col_final] = [limpio]
                
        # 3. Añadir la fecha y almacenar
        df_fila['Fecha'] = fecha_dt.strftime('%Y-%m-%d')
        datos_diarios.append(df_fila)
            
        # Nota: Los días sin datos pueden generar una tabla vacía o no tener la tabla resumen
        # Si el día no está en el CSV, podemos asumirlo como un día completamente nulo.
        
    except requests.exceptions.RequestException as e:
        tqdm.write(f"  -> Error de petición para {fecha_str}: {e}")
    except ValueError:
        tqdm.write(f"  -> Error de Pandas (no se encontró tabla) para {fecha_str}.")
    except Exception as e:
        tqdm.write(f"  -> Error inesperado en el bucle para {fecha_str}: {e}")

print("\n--- Extracción del Bucle Finalizada ---")

# 3. CONSOLIDACIÓN Y GUARDADO
if datos_diarios:
    dataset_final = pd.concat(datos_diarios, ignore_index=True)
    
    # Ordenar por fecha y asegurar que la columna Fecha es tipo datetime
    dataset_final['Fecha'] = pd.to_datetime(dataset_final['Fecha'])
    dataset_final = dataset_final.sort_values(by='Fecha').reset_index(drop=True)
    
    nombre_archivo = f'meteocat_{ESTACION_ID}_resumen_historico.csv'
    dataset_final.to_csv(nombre_archivo, index=False)
    
    print(f"\n✅ Dataset de Resumen Diario guardado con éxito como '{nombre_archivo}'.")
    print(f"Total de días extraídos: {len(dataset_final)}")
    
else:
    print("\n⚠️ No se pudo extraer ningún dato.")