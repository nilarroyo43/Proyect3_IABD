import streamlit as st
import pandas as pd
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import timedelta

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(
    page_title="MeteoBCN AI Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# --- RUTAS ---
# Ajustamos las rutas asumiendo que ejecutas desde la carpeta raíz del proyecto
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Carpeta src/
ROOT_DIR = os.path.dirname(BASE_DIR) # Carpeta proyecto/

RUTA_DATASET = os.path.join(ROOT_DIR, "data", "training_datasets", "dataset_entrenamiento_barcelona_MASTER.csv")
RUTA_MODELO_TEMP = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_temperatura.pkl")
RUTA_COLS_TEMP = os.path.join(ROOT_DIR, "data", "model_memory", "columnas_modelo_temperatura.pkl")
RUTA_MODELO_LLUVIA = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_lluvia.pkl")
RUTA_COLS_LLUVIA = os.path.join(ROOT_DIR, "data", "model_memory", "columnas_modelo_lluvia.pkl")

# --- FUNCIONES DE CARGA (Con Caché para velocidad) ---
@st.cache_data
def cargar_datos():
    if not os.path.exists(RUTA_DATASET):
        return None
    df = pd.read_csv(RUTA_DATASET)
    df['Fecha'] = pd.to_datetime(df['Fecha'])
    return df.sort_values('Fecha')

def cargar_modelos():
    try:
        mod_temp = joblib.load(RUTA_MODELO_TEMP)
        cols_temp = joblib.load(RUTA_COLS_TEMP)
        mod_lluvia = joblib.load(RUTA_MODELO_LLUVIA)
        cols_lluvia = joblib.load(RUTA_COLS_LLUVIA)
        return mod_temp, cols_temp, mod_lluvia, cols_lluvia
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, None

# --- INTERFAZ PRINCIPAL ---
def interface():
    st.title("🌦️ MeteoBCN: Sistema Predictivo Inteligente")
    st.markdown("Dashboard de control para el modelo de predicción meteorológica de Barcelona.")

    # Crear Pestañas
    tab1, tab2 = st.tabs(["🔮 Predicción en Vivo", "📊 Dashboard Analítico"])

    # ==========================================================================
    # PESTAÑA 1: PREDICCIÓN
    # ==========================================================================
    with tab1:
        st.header("Predicción para Mañana")
        
        df = cargar_datos()
        mod_temp, cols_temp, mod_lluvia, cols_lluvia = cargar_modelos()

        if df is None:
            st.error("❌ No se encuentra el Dataset Maestro. Ejecuta el pipeline primero.")
        elif mod_temp is None:
            st.error("❌ No se encuentran los modelos (.pkl).")
        else:
            # Cogemos la última fila (Datos de Ayer/Hoy)
            ultima_fila = df.iloc[[-1]].copy()
            fecha_datos = ultima_fila['Fecha'].values[0]
            fecha_prediccion = pd.to_datetime(fecha_datos) + timedelta(days=1)

            st.info(f" El modelo está usando los datos registrados el **{pd.to_datetime(fecha_datos).date()}** para predecir el tiempo del **{fecha_prediccion.date()}**.")

            col_btn, col_info = st.columns([1, 4])
            with col_btn:
                boton_predecir = st.button(" GENERAR PREDICCIÓN", type="primary", use_container_width=True)

            if boton_predecir:
                with st.spinner('Analizando patrones climáticos...'):
                    # 1. Preparar X para Temperatura
                    X_temp = pd.DataFrame(index=ultima_fila.index)
                    for col in cols_temp:
                        X_temp[col] = ultima_fila[col] if col in ultima_fila else 0
                    
                    # 2. Preparar X para Lluvia
                    X_lluvia = pd.DataFrame(index=ultima_fila.index)
                    for col in cols_lluvia:
                        X_lluvia[col] = ultima_fila[col] if col in ultima_fila else 0

                    # 3. Predecir
                    pred_temp = mod_temp.predict(X_temp)[0]
                    
                    # Intentar predecir probabilidad si el modelo lo soporta
                    try:
                        prob_lluvia = mod_lluvia.predict_proba(X_lluvia)[0][1]
                    except:
                        prob_lluvia = mod_lluvia.predict(X_lluvia)[0] # Fallback
                    
                    es_lluvia = prob_lluvia > 0.35 # Umbral personalizable

                # --- MOSTRAR RESULTADOS ---
                st.markdown("---")
                col_res1, col_res2, col_res3 = st.columns(3)

                with col_res1:
                    st.metric(label="🌡️ Temperatura Esperada", value=f"{pred_temp:.1f} °C")
                
                with col_res2:
                    st.metric(label="💧 Probabilidad de Lluvia", value=f"{prob_lluvia*100:.1f} %")

                with col_res3:
                    if es_lluvia:
                        st.markdown("# 🌧️")
                        st.warning("Se espera lluvia. ¡Coge paraguas!")
                    else:
                        st.markdown("# ☀️")
                        st.success("Cielo despejado o poca probabilidad de lluvia.")

                # Datos técnicos expandibles
                with st.expander("Ver datos técnicos de entrada (Input del modelo)"):
                    st.dataframe(ultima_fila)

    # ==========================================================================
    # PESTAÑA 2: GRÁFICOS
    # ==========================================================================
    with tab2:
        st.header("Análisis Histórico y Correlaciones")
        
        df = cargar_datos()
        if df is not None:
            # Filtros laterales dentro de la pestaña
            col_filt1, col_filt2 = st.columns(2)
            with col_filt1:
                fecha_min = df['Fecha'].min()
                fecha_max = df['Fecha'].max()
                
                # Default: Último año
                fecha_inicio = st.date_input("Fecha Inicio", value=fecha_max - timedelta(days=365), min_value=fecha_min, max_value=fecha_max)
            with col_filt2:
                fecha_fin = st.date_input("Fecha Fin", value=fecha_max, min_value=fecha_min, max_value=fecha_max)

            # Filtrar DF
            mask = (df['Fecha'] >= pd.to_datetime(fecha_inicio)) & (df['Fecha'] <= pd.to_datetime(fecha_fin))
            df_filtered = df.loc[mask]

            st.write(f"Mostrando **{len(df_filtered)}** registros.")

            # --- GRÁFICO 1: EVOLUCIÓN TEMP ---
            st.subheader(" Evolución de la Temperatura")
            st.line_chart(df_filtered.set_index("Fecha")[["Temp_Media_C", "Temp_Maxima_C", "Temp_Minima_C"]])

            # --- GRÁFICO 2: CORRELACIONES ---
            st.subheader(" Mapa de Correlaciones (Heatmap)")
            st.markdown("Este gráfico muestra qué variables influyen más en la temperatura y la lluvia.")
            
            # Seleccionamos solo columnas numéricas relevantes para no saturar
            cols_corr = [
                'Temp_Media_C', 'Temp_Media_C_Media_3dias', 
                'Presion_Media_hPa', 'Humedad_Media_Pct', 
                'Precip_Total_mm', 'Viento_Maximo_kmh',
                'Dia_Sin', 'Dia_Cos'
            ]
            # Asegurar que existen en el df
            cols_corr = [c for c in cols_corr if c in df_filtered.columns]
            
            if len(df_filtered) > 1:
                fig, ax = plt.subplots(figsize=(6, 4))
                corr_matrix = df_filtered[cols_corr].corr()
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax,annot_kws={"size": 8})
                st.pyplot(fig, use_container_width=False)
                
            else:
                st.warning("No hay suficientes datos para calcular correlaciones.")
                
            # ... (Aquí terminaba el código del Heatmap que te pasé antes) ...
            
            st.markdown("---")
            
            # DIVISIÓN EN DOS COLUMNAS PARA LOS NUEVOS GRÁFICOS
            col_graf3, col_graf4 = st.columns(2)

            # --- GRÁFICO 3: LA PRUEBA TRIGONOMÉTRICA (VECTOR VIENTO) ---
            with col_graf3:
                st.subheader("🌀 Vectorización del Viento(Scatter)")
                st.caption("Visualización de la direccion del viento en grados trasnformada a Sin/Cos. Debe formar un círculo.")
                
                if 'Viento_Dir_Cos' in df_filtered.columns and 'Viento_Dir_Sin' in df_filtered.columns:
                    fig_vec, ax_vec = plt.subplots(figsize=(5, 5))
                    
                    # Scatter plot con transparencia para ver densidad
                    ax_vec.scatter(
                        df_filtered['Viento_Dir_Cos'], 
                        df_filtered['Viento_Dir_Sin'], 
                        alpha=0.1, 
                        c='purple',
                        s=10
                    )
                    ax_vec.set_xlabel("Componente Coseno (Norte-Sur)")
                    ax_vec.set_ylabel("Componente Seno (Este-Oeste)")
                    ax_vec.set_title("Espacio Vectorial del Viento")
                    ax_vec.grid(True, linestyle='--', alpha=0.5)
                    
                    # Dibujar círculo unitario de referencia
                    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle=':')
                    ax_vec.add_artist(circle)
                    
                    # Fijar límites para que se vea cuadrado
                    ax_vec.set_xlim(-1.1, 1.1)
                    ax_vec.set_ylim(-1.1, 1.1)
                    
                    st.pyplot(fig_vec, use_container_width=False)
                else:
                    st.warning("Faltan las columnas vectoriales de viento.")

            # --- GRÁFICO 4: DISTRIBUCIÓN MENSUAL (BOXPLOT) ---
            with col_graf4:
                st.subheader("📅 Distribución Mensual")
                st.caption("Variabilidad de temperatura por mes (Detecta outliers).")
                
                # Extraemos el mes para agrupar
                df_filtered['Mes'] = df_filtered['Fecha'].dt.month_name()
                # Ordenar meses correctamente (no alfabético)
                orden_meses = ['January', 'February', 'March', 'April', 'May', 'June', 
                               'July', 'August', 'September', 'October', 'November', 'December']
                
                fig_box, ax_box = plt.subplots(figsize=(6, 5))
                
                sns.boxplot(
                    data=df_filtered, 
                    x='Mes', 
                    y='Temp_Media_C', 
                    order=orden_meses,
                    palette="Spectral",
                    ax=ax_box,
                    linewidth=1
                )
                
                ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax_box.set_title("Temperatura por Mes")
                ax_box.set_xlabel("")
                
                st.pyplot(fig_box, use_container_width=False)

