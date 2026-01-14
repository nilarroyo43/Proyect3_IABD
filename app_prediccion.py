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
    Pipeline de MLOps:
    1. Descarga datos nuevos (ETL).
    2. GESTIÓN INTELIGENTE: 
       - Si el dato es nuevo -> Lo añade.
       - Si el dato ya existe (mismo día) -> Lo SOBREESCRIBE (para corregir datos provisionales con finales).
    3. Si hay cambios, re-entrena los modelos.
    """
    print(" INICIANDO PIPELINE DE MANTENIMIENTO (ETL + RE-ENTRENO)")
    print("==========================================================")
    
    # ==============================================================================
    # FASE 1: ACTUALIZACIÓN DE DATOS (ETL + FEATURE ENGINEERING)
    # ==============================================================================
    if not os.path.exists(RUTA_HISTORICO):
        print(" Error crítico: No existe el dataset maestro.")
        return

    # 1. Cargar Histórico
    df_historico = pd.read_csv(RUTA_HISTORICO)
    df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
    df_historico = df_historico.sort_values('Fecha')
    
    ultima_fecha = df_historico['Fecha'].iloc[-1]
    print(f" Última fecha registrada en CSV: {ultima_fecha.date()}")

    # 2. Buscar nuevos datos (Scraping)
    print(" Conectando con Meteocat...")
    try:
        nuevos_datos = obtener_media_barcelona() # Devuelve DataFrame
        datos_guardados = False

        if nuevos_datos is not None and not nuevos_datos.empty:
            nuevos_datos['Fecha'] = pd.to_datetime(nuevos_datos['Fecha'])
            fecha_nueva = nuevos_datos['Fecha'].iloc[0]
            
            print(f"🔍 Dato descargado de Meteocat: {fecha_nueva.date()}")

            # --- CASO A: DATO TOTALMENTE NUEVO (Normalmente a las 16:00) ---
            if fecha_nueva > ultima_fecha:
                print(f" FECHA NUEVA DETECTADA. Añadiendo al histórico...")
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                df_actualizado = df_actualizado.drop_duplicates(subset='Fecha', keep='last')
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                
                print(" Dataset maestro actualizado (Nueva fila añadida).")
                datos_guardados = True
                
            # --- CASO B: DATO YA EXISTENTE -> SUSTITUCIÓN (Normalmente a las 09:00) ---
            elif fecha_nueva == ultima_fecha:
                print(" ACTUALIZACIÓN DE CALIDAD (El dato ya existía).")
                print("   -> Acción: Borrar dato provisional anterior y guardar el nuevo (Oficial).")
                
                # 1. Borramos la fila vieja del histórico
                df_historico = df_historico[df_historico['Fecha'] != fecha_nueva]
                
                # 2. Añadimos la nueva fila descargada
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                
                print(" Dato corregido y guardado exitosamente.")
                datos_guardados = True
                
            else:
                print(" El dato descargado es más antiguo que el histórico. Se ignora.")
        else:
            print(" No se han encontrado datos en Meteocat.")

    except Exception as e:
        print(f" Error en ETL: {e}")
        datos_guardados = False

    # ==============================================================================
    # FASE 2: RE-ENTRENAMIENTO (MLOPS)
    # ==============================================================================
    es_lunes = datetime.today().weekday() == 0
    
    if datos_guardados or es_lunes:
        print("\n DETECTADA NECESIDAD DE RE-ENTRENAMIENTO...")
        if datos_guardados: print("   -> Motivo: Datos actualizados (Nuevos o Corregidos).")
        if es_lunes: print("   -> Motivo: Mantenimiento semanal (Lunes).")

        try:
            print("    Re-entrenando Modelo Temperatura...")
            entrenar_modelo_temperatura()
            
            print("    Re-entrenando Modelo Lluvia...")
            entrenar_modelo_lluvia()
            
            print(" Modelos actualizados correctamente.")
        except Exception as e:
            print(f" Error crítico entrenando modelos: {e}")
    else:
        print("\n No hay cambios en los datos ni es lunes. No se entrena.")

    print("\n FIN DEL PROCESO DE MANTENIMIENTO.")

if __name__ == "__main__":
    pipeline_mantenimiento()