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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

RUTA_DATASET = os.path.join(ROOT_DIR, "data", "training_datasets", "dataset_entrenamiento_barcelona_MASTER.csv")

RUTA_MODELO_TEMP_MAX = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_temp_maxima.pkl")
RUTA_MODELO_TEMP_MEDIA = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_temp_media.pkl")
RUTA_MODELO_TEMP_MIN = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_temp_minima.pkl")
RUTA_COLS_TEMP = os.path.join(ROOT_DIR, "data", "model_memory", "columnas_modelo_temperatura.pkl")

RUTA_MODELO_LLUVIA = os.path.join(ROOT_DIR, "data", "model_memory", "cerebro_meteo_lluvia.pkl")
RUTA_COLS_LLUVIA = os.path.join(ROOT_DIR, "data", "model_memory", "columnas_modelo_lluvia.pkl")

# --- FUNCIONES DE CARGA ---
@st.cache_data
def cargar_datos():
    if not os.path.exists(RUTA_DATASET):
        return None
    df = pd.read_csv(RUTA_DATASET)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    return df.sort_values("Fecha")

@st.cache_resource
def cargar_modelos():
    try:
        mod_temp_max = joblib.load(RUTA_MODELO_TEMP_MAX)
        mod_temp_med = joblib.load(RUTA_MODELO_TEMP_MEDIA)
        mod_temp_min = joblib.load(RUTA_MODELO_TEMP_MIN)
        cols_temp = joblib.load(RUTA_COLS_TEMP)

        mod_lluvia = joblib.load(RUTA_MODELO_LLUVIA)
        cols_lluvia = joblib.load(RUTA_COLS_LLUVIA)

        return mod_temp_max, mod_temp_med, mod_temp_min, cols_temp, mod_lluvia, cols_lluvia
    except Exception as e:
        st.error(f"Error cargando modelos: {e}")
        return None, None, None, None, None, None

# --- INTERFAZ PRINCIPAL ---
def interface():
    st.title("🌦️ MeteoBCN: Sistema Predictivo Inteligente")
    st.markdown("Dashboard de control para el modelo de predicción meteorológica de Barcelona.")

    tab1, tab2 = st.tabs(["🔮 Predicción en Vivo", "📊 Dashboard Analítico"])

    # ==========================================================================
    # PESTAÑA 1: PREDICCIÓN
    # ==========================================================================
    with tab1:
        st.header("Predicción para Mañana")

        df = cargar_datos()
        mod_max, mod_med, mod_min, cols_temp, mod_lluvia, cols_lluvia = cargar_modelos()

        if df is None:
            st.error("❌ No se encuentra el Dataset Maestro.")
            return

        if mod_max is None:
            st.error("❌ No se encuentran los modelos entrenados.")
            return

        ultima_fila = df.iloc[[-1]].copy()
        fecha_datos = ultima_fila["Fecha"].values[0]
        fecha_pred = pd.to_datetime(fecha_datos) + timedelta(days=1)

        st.info(f"Usando datos del **{pd.to_datetime(fecha_datos).date()}** para predecir el **{fecha_pred.date()}**.")

        col_btn, col_info = st.columns([1, 4])
        with col_btn:
            boton_predecir = st.button("GENERAR PREDICCIÓN", type="primary", use_container_width=True)

        if boton_predecir:
            with st.spinner("Analizando patrones climáticos..."):

                # --- Preparar input temperatura ---
                X_temp = pd.DataFrame(index=ultima_fila.index)
                for col in cols_temp:
                    X_temp[col] = ultima_fila[col] if col in ultima_fila else 0

                # --- Preparar input lluvia ---
                X_lluvia = pd.DataFrame(index=ultima_fila.index)
                for col in cols_lluvia:
                    X_lluvia[col] = ultima_fila[col] if col in ultima_fila else 0

                # --- Predicciones temperatura ---
                pred_max = mod_max.predict(X_temp)[0]
                pred_med = mod_med.predict(X_temp)[0]
                pred_min = mod_min.predict(X_temp)[0]

                # --- Predicción lluvia ---
                try:
                    prob_lluvia = mod_lluvia.predict_proba(X_lluvia)[0][1]
                except:
                    prob_lluvia = mod_lluvia.predict(X_lluvia)[0]

                es_lluvia = prob_lluvia > 0.35

            # --- RESULTADOS ---
            st.markdown("---")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🌡️ Temp. Máxima", f"{pred_max:.1f} °C")
            col2.metric("🌡️ Temp. Media", f"{pred_med:.1f} °C")
            col3.metric("🌡️ Temp. Mínima", f"{pred_min:.1f} °C")
            col4.metric("💧 Prob. Lluvia", f"{prob_lluvia*100:.1f} %")

            if es_lluvia:
                st.warning("🌧️ Se espera lluvia. ¡Coge paraguas!")
            else:
                st.success("☀️ Cielo despejado o baja probabilidad de lluvia.")

            with st.expander("Ver datos técnicos de entrada"):
                st.dataframe(ultima_fila)

    # ==========================================================================
    # PESTAÑA 2: DASHBOARD ANALÍTICO
    # ==========================================================================
    with tab2:
        st.header("Análisis Histórico y Correlaciones")

        df = cargar_datos()
        if df is None:
            st.error("No hay datos disponibles.")
            return

        col_f1, col_f2 = st.columns(2)
        fecha_min = df["Fecha"].min()
        fecha_max = df["Fecha"].max()

        with col_f1:
            fecha_inicio = st.date_input("Fecha Inicio", value=fecha_max - timedelta(days=365),
                                          min_value=fecha_min, max_value=fecha_max)
        with col_f2:
            fecha_fin = st.date_input("Fecha Fin", value=fecha_max,
                                       min_value=fecha_min, max_value=fecha_max)

        mask = (df["Fecha"] >= pd.to_datetime(fecha_inicio)) & (df["Fecha"] <= pd.to_datetime(fecha_fin))
        df_filtered = df.loc[mask].copy()

        st.write(f"Mostrando **{len(df_filtered)}** registros.")

        # --- Evolución temperaturas ---
        st.subheader("Evolución de Temperatura")
        st.line_chart(df_filtered.set_index("Fecha")[["Temp_Media_C", "Temp_Maxima_C", "Temp_Minima_C"]])

        # --- Heatmap correlaciones ---
        st.subheader("Mapa de Correlaciones")

        cols_corr = [
            'Temp_Media_C', 'Temp_Media_C_Media_3dias',
            'Presion_Media_hPa', 'Humedad_Media_Pct',
            'Precip_Total_mm', 'Viento_Maximo_kmh',
            'Dia_Sin', 'Dia_Cos'
        ]
        cols_corr = [c for c in cols_corr if c in df_filtered.columns]

        if len(df_filtered) > 1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df_filtered[cols_corr].corr(), annot=True, cmap="coolwarm",
                        fmt=".2f", linewidths=0.5, ax=ax, annot_kws={"size":8})
            st.pyplot(fig, use_container_width=False)
        else:
            st.warning("No hay suficientes datos para correlaciones.")

        st.markdown("---")

        col_g1, col_g2 = st.columns(2)

        # --- Vector viento ---
        with col_g1:
            st.subheader("🌀 Vectorización del Viento")
            if "Viento_Dir_Cos" in df_filtered and "Viento_Dir_Sin" in df_filtered:
                fig_vec, ax_vec = plt.subplots(figsize=(5,5))
                ax_vec.scatter(df_filtered["Viento_Dir_Cos"], df_filtered["Viento_Dir_Sin"],
                               alpha=0.1, s=10)
                circle = plt.Circle((0,0),1, fill=False, linestyle=":")
                ax_vec.add_artist(circle)
                ax_vec.set_xlim(-1.1,1.1)
                ax_vec.set_ylim(-1.1,1.1)
                ax_vec.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_vec, use_container_width=False)
            else:
                st.warning("Faltan columnas de viento.")

        # --- Boxplot mensual ---
        with col_g2:
            st.subheader("📅 Distribución Mensual")
            df_filtered["Mes"] = df_filtered["Fecha"].dt.month_name()
            orden = ['January','February','March','April','May','June',
                     'July','August','September','October','November','December']

            fig_box, ax_box = plt.subplots(figsize=(6,5))
            sns.boxplot(data=df_filtered, x="Mes", y="Temp_Media_C",
                        order=orden, palette="Spectral", linewidth=1, ax=ax_box)
            ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45, ha="right", fontsize=8)
            ax_box.set_title("Temperatura por Mes")
            ax_box.set_xlabel("")
            st.pyplot(fig_box, use_container_width=False)


# --- EJECUCIÓN ---
if __name__ == "__main__":
    interface()
