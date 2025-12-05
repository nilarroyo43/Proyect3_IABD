import pandas as pd
import requests
from datetime import datetime

# --- CONFIGURACIÓN DE PRUEBA ---
URL_BASE = "https://www.meteo.cat/observacions/xema/dades?"
ESTACION_ID = "D5" 
FECHA_PRUEBA = "2024-01-01" # Día fácil para verificar
TIEMPO_ESPERA = 1 

url_a_scrapear = (
    f"{URL_BASE}codi={ESTACION_ID}&dia={FECHA_PRUEBA}T00:00Z"
)

print(f"Probando URL: {url_a_scrapear}")

try:
    # 1. Realizar la petición
    response = requests.get(url_a_scrapear)
    response.raise_for_status() 

    # 2. Encontrar TODAS las tablas
    tablas_encontradas = pd.read_html(response.text)

    print("\n--- DIAGNÓSTICO DE TABLAS ---")
    print(f"Total de tablas encontradas: {len(tablas_encontradas)}\n")

    # 3. Mostrar el encabezado de cada tabla para identificar el resumen
    for i, tabla in enumerate(tablas_encontradas):
        print(f"==================================================")
        print(f"Índice de Tabla: {i}")
        print(f"Forma (filas, columnas): {tabla.shape}")
        
        # Mostrar las primeras filas y las columnas para ver el contenido
        print(tabla.head(10))
        print("==================================================")

except requests.exceptions.RequestException as e:
    print(f"Error de conexión: {e}")
except ValueError:
    print("Error: Pandas no pudo encontrar ninguna tabla en la página.")