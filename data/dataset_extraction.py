import pandas as pd
import requests
import time
from datetime import datetime, timedelta
from tqdm import tqdm 
import warnings
import numpy as np

# Suprimir advertencias
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. CONFIGURACIÓN ---
URL_BASE = "https://www.meteo.cat/observacions/xema/dades?"
ESTACION_ID = "D5" # Barcelona - Observatori Fabra
# ESTACION_ID = "X4" # Barcelona - Raval
# ESTACION_ID = "X8" # Barcelona - Zona Universitaria

TABLA_INDEX_RESUMEN = 0 

MAPEO_COLUMNAS = {
    'Temperatura mitjana': 'Temp_Media_C',
    'Temperatura màxima': 'Temp_Maxima_C',
    'Temperatura mínima': 'Temp_Minima_C',
    'Humitat relativa mitjana': 'Humedad_Media_Pct',
    'Precipitació acumulada': 'Precip_Total_mm',
    'Ratxa màxima del vent ': 'Viento_Maximo_kmh', 
    'Pressió atmosfèrica mitjana': 'Presion_Media_hPa',
    'Irradiació solar global': 'Irrad_Solar_MJm2'
}

# FECHAS
FECHA_INICIO = datetime(2025, 12, 12)
FECHA_FIN = datetime.now().date() - timedelta(days=1) 
TIEMPO_ESPERA = 1.5 

datos_diarios = []
fechas_a_scrapear = pd.date_range(start=FECHA_INICIO, end=FECHA_FIN, freq='D')

print(f"Iniciando extracción de {len(fechas_a_scrapear)} días para la estación {ESTACION_ID}...")

# --- 2. BUCLE PRINCIPAL ---
for fecha_dt in tqdm(fechas_a_scrapear):
    
    fecha_str = fecha_dt.strftime('%Y-%m-%d')
    url_a_scrapear = f"{URL_BASE}codi={ESTACION_ID}&dia={fecha_str}T00:00Z"
    
    # Estructura base de la fila (Diccionario)
    fila_datos = {'Fecha': fecha_dt} # Iniciamos con la fecha
    
    # Inicializamos la columna de dirección del viento a NaN por defecto para asegurar que exista
    fila_datos['Viento_Direccion_Grados'] = np.nan

    try:
        time.sleep(TIEMPO_ESPERA)  
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url_a_scrapear, headers=headers, timeout=10)
        response.raise_for_status() 
        
        tablas_encontradas = pd.read_html(response.text)

        if len(tablas_encontradas) > TABLA_INDEX_RESUMEN:
            
            # Procesar tabla
            df_resumen = tablas_encontradas[TABLA_INDEX_RESUMEN].copy() 
            df_resumen = df_resumen[[0, 1]]
            df_resumen.columns = ['Variable', 'Valor']
            df_pivotado = df_resumen.set_index('Variable').T 
            columnas_reales = df_pivotado.columns.tolist()

            # Iterar sobre las columnas que queremos extraer
            for col_cat, col_final in MAPEO_COLUMNAS.items():
                
                # Búsqueda flexible (contiene texto)
                col_encontrada = next((c for c in columnas_reales if col_cat in c), None)

                val_final = np.nan # Valor por defecto

                if col_encontrada:
                    valor_serie = df_pivotado[col_encontrada].iloc[0] 
                    texto_bruto = str(valor_serie).strip()
                    
                    if texto_bruto.lower() != "sense dades":
                        # Separar valor principal de extras (ej: viento - dirección)
                        partes = texto_bruto.split('-')
                        parte_valor = partes[0]
                        
                        # Limpieza numérica básica
                        limpio_str = (
                            pd.Series(parte_valor)
                            .str.replace(r'[^\d\.\-]+', '', regex=True)
                            .str.strip()
                        )
                        try:
                            val_final = float(limpio_str.iloc[0])
                        except (ValueError, IndexError):
                            val_final = np.nan
                        
                        # --- LÓGICA ESPECIAL: DIRECCIÓN DEL VIENTO ---
                        if "Ratxa" in col_cat and len(partes) > 1:
                            try:
                                # Parte derecha del guión: " 194º "
                                dir_str = partes[1].replace('º', '').strip()
                                fila_datos['Viento_Direccion_Grados'] = float(dir_str)
                            except ValueError:
                                pass # Se queda como NaN (valor inicial)

                # Guardamos el valor principal en el diccionario
                fila_datos[col_final] = val_final

            # Convertimos el diccionario a DataFrame y lo guardamos
            datos_diarios.append(pd.DataFrame([fila_datos]))
        
        else:
            # Si no hay tabla, guardamos la fila solo con la fecha (y el resto NaNs)
            datos_diarios.append(pd.DataFrame([fila_datos]))
            
    except Exception as e:
        tqdm.write(f"  -> Error en {fecha_str}: {e}")


print("\n--- Procesamiento finalizado ---")

# --- 3. CONSOLIDACIÓN, ORDENACIÓN Y GUARDADO ---
if datos_diarios:
    dataset_final = pd.concat(datos_diarios, ignore_index=True)
    
    # 1. Aseguramos formato fecha
    dataset_final['Fecha'] = pd.to_datetime(dataset_final['Fecha'])
    dataset_final = dataset_final.sort_values(by='Fecha').reset_index(drop=True)
    
    # 2. DEFINIMOS EL ORDEN EXACTO DE LAS COLUMNAS
    # Aquí es donde decides quién va al lado de quién
    orden_deseado = [
        'Fecha',
        'Temp_Media_C',
        'Temp_Maxima_C',
        'Temp_Minima_C',
        'Humedad_Media_Pct',
        'Precip_Total_mm',
        'Viento_Maximo_kmh',       # <--- Velocidad
        'Viento_Direccion_Grados', # <--- Dirección (Justo al lado)
        'Presion_Media_hPa',
        'Irrad_Solar_MJm2'
    ]
    
    # Aplicamos el orden. 
    # Usamos reindex para evitar errores si alguna columna faltara (rellenaría con NaN)
    dataset_final = dataset_final.reindex(columns=orden_deseado)

    # --- CONFIGURACIÓN DE LA RUTA (Con pathlib) ---
    from pathlib import Path # Por si no lo has importado arriba
    HERE = Path(__file__).resolve().parent
    ruta_carpeta = HERE / "raw_datasets"
    ruta_carpeta.mkdir(parents=True, exist_ok=True)
    
    nombre_archivo = f'meteocat_{ESTACION_ID}_resumen_historico.csv'
    ruta_completa = ruta_carpeta / nombre_archivo
    
    # Guardamos
    dataset_final.to_csv(ruta_completa, index=False)
    
    print(f"\n✅ Guardado con éxito en:")
    print(f"📂 {ruta_completa}")
    print(f"📊 Filas totales: {len(dataset_final)}")

else:
    print("\n⚠️ No se han extraído datos.")