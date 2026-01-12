import pandas as pd
import streamlit as st
from datetime import timedelta

def interface():
    st.title("🌦 Weather Data Explorer – Proyecto IABD")

    # Load data
    dt = pd.read_csv("data/training_datasets/dataset_entrenamiento_barcelona_MASTER.csv")

    # Ensure date column is datetime
    if "Fecha" in dt.columns:
        dt["Fecha"] = pd.to_datetime(dt["Fecha"])
    else:
        st.error("❌ No 'Fecha' column found.")
        return

    # Sort by date
    dt = dt.sort_values("Fecha")

    st.subheader("📄 Vista previa del dataset")
    st.dataframe(dt.head())

    # --- Timeline Selector ---
    st.subheader("⏳ Selecciona rango de años para visualizar")

    max_date = dt["Fecha"].max()
    min_date = dt["Fecha"].min()

    default_start = max_date - pd.DateOffset(years=3)   # ⬅️ Default = last 3 years

    start_date = st.date_input(
        "Fecha inicial:",
        value=default_start,
        min_value=min_date,
        max_value=max_date
    )

    end_date = st.date_input(
        "Fecha final:",
        value=max_date,
        min_value=min_date,
        max_value=max_date
    )

    # Filter dataset
    mask = (dt["Fecha"] >= pd.to_datetime(start_date)) & (dt["Fecha"] <= pd.to_datetime(end_date))
    dt_filtered = dt[mask]

    st.info(f"Mostrando datos desde **{start_date}** hasta **{end_date}** "
            f"({len(dt_filtered)} registros).")

    # List numeric columns
    numeric_cols = dt.select_dtypes(include=["float64", "int64"]).columns.tolist()

    st.subheader("📊 Gráfico histórico")
    selected = st.multiselect(
        "Selecciona variables para visualizar:",
        numeric_cols,
        default=["Temp_Media_C", "Temp_Maxima_C"]
    )

    if selected:
        st.line_chart(dt_filtered.set_index("Fecha")[selected])
    else:
        st.info("Selecciona al menos una variable para mostrar el gráfico.")

    # Additional useful charts
    st.subheader("🌡 Temperatura: Media, Máxima y Mínima")
    temp_cols = [c for c in ["Temp_Media_C", "Temp_Maxima_C", "Temp_Minima_C"] if c in dt.columns]
    if temp_cols:
        st.line_chart(dt_filtered.set_index("Fecha")[temp_cols])

    st.subheader("Precipitación")
    if "Precipitacion_mm" in dt.columns:
        st.line_chart(dt_filtered.set_index("Fecha")["Precipitacion_mm"])

    st.subheader("Humedad")
    humidity_cols = [c for c in dt.columns if "Humedad" in c]
    if humidity_cols:
        st.line_chart(dt_filtered.set_index("Fecha")[humidity_cols])