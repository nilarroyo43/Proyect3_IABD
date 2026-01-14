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
    2. Si hay datos nuevos o es lunes, re-entrena los modelos.
    NOTA: No realiza predicciones (eso se hace en la interfaz web).
    """
    print("🏭 INICIANDO PIPELINE DE MANTENIMIENTO (ETL + RE-ENTRENO)")
    print("==========================================================")
    
    # ==============================================================================
    # FASE 1: ACTUALIZACIÓN DE DATOS (ETL + FEATURE ENGINEERING)
    # ==============================================================================
    if not os.path.exists(RUTA_HISTORICO):
        print("❌ Error crítico: No existe el dataset maestro.")
        return

    # 1. Cargar Histórico
    df_historico = pd.read_csv(RUTA_HISTORICO)
    df_historico['Fecha'] = pd.to_datetime(df_historico['Fecha'])
    df_historico = df_historico.sort_values('Fecha')
    
    ultima_fecha = df_historico['Fecha'].iloc[-1]
    print(f"📂 Última fecha registrada: {ultima_fecha.date()}")

    # 2. Buscar nuevos datos (Scraping)
    print("📡 Conectando con Meteocat...")
    try:
        nuevos_datos = obtener_media_barcelona() # Devuelve DataFrame con la fila de hoy/ayer o None
        datos_guardados = False

        if nuevos_datos is not None and not nuevos_datos.empty:
            nuevos_datos['Fecha'] = pd.to_datetime(nuevos_datos['Fecha'])
            fecha_nueva = nuevos_datos['Fecha'].iloc[0]

            if fecha_nueva > ultima_fecha:
                print(f"✅ DATO NUEVO ENCONTRADO: {fecha_nueva.date()}")
                # Concatenar
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                # Limpieza extra por si acaso
                df_actualizado = df_actualizado.drop_duplicates(subset='Fecha', keep='last')
                # Guardar
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                print("💾 Dataset maestro actualizado.")
                datos_guardados = True
                
            elif fecha_nueva == ultima_fecha:
                print("ℹ️ Actualización intra-día (el dato ya existía, se sobreescribe por si ha variado).")
                df_historico = df_historico[df_historico['Fecha'] != fecha_nueva]
                df_actualizado = pd.concat([df_historico, nuevos_datos], ignore_index=True)
                df_actualizado.to_csv(RUTA_HISTORICO, index=False)
                print("💾 Dato actualizado.")
                datos_guardados = True
            else:
                print("⚠️ El dato descargado es antiguo. No se guarda.")
        else:
            print("⚠️ No se han encontrado datos nuevos disponibles.")

    except Exception as e:
        print(f"❌ Error en ETL: {e}")
        datos_guardados = False

    # ==============================================================================
    # FASE 2: RE-ENTRENAMIENTO (MLOPS)
    # ==============================================================================
    # Regla: Se re-entrena si hay datos nuevos O si es Lunes (para refrescar la lógica temporal)
    es_lunes = datetime.today().weekday() == 0
    
    if datos_guardados or es_lunes:
        print("\n🧠 DETECTADA NECESIDAD DE RE-ENTRENAMIENTO...")
        if datos_guardados: print("   -> Motivo: Nuevos datos ingresados.")
        if es_lunes: print("   -> Motivo: Mantenimiento semanal (Lunes).")

        try:
            print("   🌡️ Re-entrenando Modelo Temperatura...")
            entrenar_modelo_temperatura()
            
            print("   ☔ Re-entrenando Modelo Lluvia...")
            entrenar_modelo_lluvia()
            
            print("✅ Modelos actualizados correctamente en 'data/model_memory/'")
        except Exception as e:
            print(f"❌ Error crítico entrenando modelos: {e}")
    else:
        print("\n💤 No se requiere re-entrenamiento hoy.")

    print("\n🏁 FIN DEL PROCESO DE MANTENIMIENTO.")

if __name__ == "__main__":
    pipeline_mantenimiento()