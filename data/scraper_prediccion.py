import pandas as pd
import requests
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# --- CONFIGURACIÓN ---
# Las 3 estaciones: Fabra (D5), Raval (X4), Zona Universitaria (X8)
ESTACIONES_ID = ["D5", "X4", "X8"] 

# URL Base idéntica al script que funciona
URL_BASE = "https://www.meteo.cat/observacions/xema/dades?"

# Mapeo idéntico al script que funciona
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

def obtener_media_barcelona(fecha_str):
    """
    Entrada: "2025-12-19" (String YYYY-MM-DD)
    Salida: DataFrame de 1 fila con la MEDIA de las 3 estaciones.
    """
    print(f"🌍 Conectando a Meteocat para el día {fecha_str}...")
    
    dfs_estaciones = []

    # 1. BUCLE DE EXTRACCIÓN (3 Estaciones)
    for codigo in ESTACIONES_ID:
        # Usamos la MISMA estructura de URL que en dataset_extraction.py
        # El servidor espera: dia=YYYY-MM-DDT00:00Z
        url = f"{URL_BASE}codi={codigo}&dia={fecha_str}T00:00Z"
        
        try:
            # Headers para parecer un navegador
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                print(f"   ⚠️ Error bajando estación {codigo} (Status {response.status_code})")
                continue

            tablas_encontradas = pd.read_html(response.text)
            
            # Si encontramos tablas (usamos índice 0 como en tu script original)
            if len(tablas_encontradas) > 0:
                df_resumen = tablas_encontradas[0].copy()
                
                # --- APLICAMOS LA LÓGICA DEL SCRIPT QUE FUNCIONA ---
                # 1. Seleccionar columnas 0 y 1 y transponer
                df_resumen = df_resumen[[0, 1]]
                df_resumen.columns = ['Variable', 'Valor']
                df_pivotado = df_resumen.set_index('Variable').T 
                columnas_reales = df_pivotado.columns.tolist()

                # Diccionario para guardar los datos de ESTA estación
                fila_datos = {}
                # Inicializar dirección viento en NaN
                fila_datos['Viento_Direccion_Grados'] = np.nan 

                # 2. Iterar y limpiar
                for col_cat, col_final in MAPEO_COLUMNAS.items():
                    # Buscar columna que contenga el texto (ej: "Temperatura mitjana")
                    col_encontrada = next((c for c in columnas_reales if col_cat in c), None)
                    val_final = np.nan

                    if col_encontrada:
                        valor_serie = df_pivotado[col_encontrada].iloc[0]
                        texto_bruto = str(valor_serie).strip()

                        if texto_bruto.lower() != "sense dades":
                            # Separar valor de extras (ej: "50.5 - 180º")
                            partes = texto_bruto.split('-')
                            parte_valor = partes[0]

                            # Limpieza Regex (Vital para quitar '°C', '%', etc.)
                            limpio_str = (
                                pd.Series(parte_valor)
                                .str.replace(r'[^\d\.\-]+', '', regex=True)
                                .str.strip()
                            )
                            try:
                                val_final = float(limpio_str.iloc[0])
                            except (ValueError, IndexError):
                                val_final = np.nan

                            # Lógica especial Dirección Viento (Parte derecha del guion)
                            if "Ratxa" in col_cat and len(partes) > 1:
                                try:
                                    dir_str = partes[1].replace('º', '').strip()
                                    fila_datos['Viento_Direccion_Grados'] = float(dir_str)
                                except ValueError:
                                    pass 

                    fila_datos[col_final] = val_final
                
                # Convertimos a DataFrame temporal y añadimos a la lista
                df_temp = pd.DataFrame([fila_datos])
                dfs_estaciones.append(df_temp)
                # print(f"   ✅ Datos {codigo} procesados correctamente.")

        except Exception as e:
            print(f"   ❌ Error procesando {codigo}: {e}")

    if not dfs_estaciones:
        print("❌ CRÍTICO: No se pudo bajar información de ninguna estación.")
        return None

    # 2. FUSIÓN Y MEDIA (Matemática Vectorial)
    df_total = pd.concat(dfs_estaciones, ignore_index=True)
    
    # Truco Vectorial para el Viento (Antes de hacer la media)
    # Convertimos a seno/coseno para promediar ángulos correctamente
    if 'Viento_Direccion_Grados' in df_total.columns:
        # Filtrar solo valores válidos para no romper np.deg2rad
        df_viento = df_total.dropna(subset=['Viento_Direccion_Grados'])
        if not df_viento.empty:
            rads = np.deg2rad(df_viento['Viento_Direccion_Grados'])
            df_total.loc[df_viento.index, 'v_sin'] = np.sin(rads)
            df_total.loc[df_viento.index, 'v_cos'] = np.cos(rads)

    # Calculamos la MEDIA de todo
    df_media = df_total.mean(numeric_only=True).to_frame().T
    
    # Recuperar Grados del Viento desde la media de vectores
    if 'v_sin' in df_media.columns and 'v_cos' in df_media.columns:
        angulo = np.arctan2(df_media['v_sin'], df_media['v_cos'])
        grados = np.rad2deg(angulo)
        df_media['Viento_Direccion_Grados'] = (grados + 360) % 360
        df_media = df_media.drop(columns=['v_sin', 'v_cos'])

    # Añadir la fecha para referencia y asegurar columnas
    df_media['Fecha'] = fecha_str
    
    return df_media

# Bloque de prueba
if __name__ == "__main__":
    # Prueba con ayer para asegurar que hay datos publicados
    from datetime import timedelta
    ayer = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Probando con fecha: {ayer}")
    
    datos = obtener_media_barcelona(ayer)
    if datos is not None:
        print("\n📊 RESULTADO FINAL (MEDIA BARCELONA):")
        print(datos.T)