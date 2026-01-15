import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

# IMPORTAMOS TUS HERRAMIENTAS
from data.scraper_prediccion import obtener_media_barcelona
from models.modelo_temperatura import entrenar_modelo_temperatura 
from models.modelo_lluvia import entrenar_modelo_lluvia

# === CONFIGURACIÓN ===
RUTA_HISTORICO = "data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv" 

def pipeline_mantenimiento():
    """
    Pipeline Secuencial:
    1. Lee el histórico para ver dónde nos quedamos.
    2. Decide QUÉ fecha pedir (siguiente día o actualizar hoy).
    3. Llama al scraper con esa fecha específica.
    """
    print(" INICIANDO PIPELINE DE MANTENIMIENTO")
    print("========================================")
    
    # -------------------------------------------------------------------------
    # 1. LEER EL HISTÓRICO (Para saber qué fecha pedir)
    # -------------------------------------------------------------------------
    if not os.path.exists(RUTA_HISTORICO):
        print(" Error crítico: No existe el dataset maestro.")
        return

    df_historico = pd.read_csv(RUTA_HISTORICO)
    df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
    df_historico = df_historico.sort_values('Fecha')
    
    ultima_fecha = df_historico['Fecha'].iloc[-1].date() # Solo la fecha, sin hora
    hoy = datetime.now().date()
    
    print(f" Última fecha en CSV: {ultima_fecha}")
    print(f" Fecha real de hoy:   {hoy}")

    # -------------------------------------------------------------------------
    # 2. CALCULAR LA FECHA OBJETIVO (String)
    # -------------------------------------------------------------------------
    # Lógica:
    # - Si la última fecha es menor que hoy (ayer o antes) -> Toca pedir el SIGUIENTE día.
    # - Si la última fecha es HOY -> Toca pedir HOY otra vez (para actualizar dato).
    
    if ultima_fecha < hoy:
        # Caso normal: Vamos a por el día siguiente
        fecha_target = ultima_fecha + timedelta(days=1)
    else:
        # Caso actualización: Vamos a refrescar el dato de hoy
        fecha_target = ultima_fecha

    # Convertimos a String 'YYYY-MM-DD' que es lo que pide tu función
    fecha_str = fecha_target.strftime('%Y-%m-%d')
    
    print(f" Fecha calculada para scrapear: {fecha_str}")

    # -------------------------------------------------------------------------
    # 3. LLAMAR AL SCRAPER (Ahora sí, con el argumento)
    # -------------------------------------------------------------------------
    try:
        print(f" Llamando a obtener_media_barcelona('{fecha_str}')...")
        
        # AQUÍ ESTABA EL ERROR: Ahora le pasamos el argumento obligatorio
        nuevos_datos = obtener_media_barcelona(fecha_str)
        
        datos_guardados = False

        if nuevos_datos is not None and not nuevos_datos.empty:
            # Aseguramos formato fecha en el dato recibido
            nuevos_datos['Fecha'] = pd.to_datetime(nuevos_datos['Fecha'])
            fecha_recibida = nuevos_datos['Fecha'].iloc[0].date()
            
            print(f" Dato recibido correctamente para: {fecha_recibida}")

            # --- Lógica de Guardado ---
            if fecha_recibida > ultima_fecha:
                print(" ES UN DÍA NUEVO. Añadiendo al final...")
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                datos_guardados = True
                
            elif fecha_recibida == ultima_fecha:
                print(" ES EL MISMO DÍA. Actualizando/Sobreescribiendo...")
                # Borramos la fila vieja y ponemos la nueva
                df_historico = df_historico[df_historico['Fecha'].dt.date != fecha_recibida]
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                datos_guardados = True
        else:
            print(f" El scraper funcionó, pero Meteocat no tiene datos para {fecha_str} todavía.")

    except Exception as e:
        print(f" Error crítico durante el scraping: {e}")
        datos_guardados = False

    # -------------------------------------------------------------------------
    # 4. RE-ENTRENAMIENTO (Si hubo cambios)
    # -------------------------------------------------------------------------
    es_lunes = datetime.today().weekday() == 0
    
    if  es_lunes:
        print("\n Es lunes, Actualizando modelos...")
        try:
            entrenar_modelo_temperatura()
            entrenar_modelo_lluvia()
            print(" Modelos re-entrenados.")
        except Exception as e:
            print(f" Error re-entrenando: {e}")
    else:
        print("\n No se requiere re-entrenamiento.")

if __name__ == "__main__":
    pipeline_mantenimiento()