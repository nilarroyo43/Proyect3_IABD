
# 🌦️ MeteoBCN: Sistema Predictivo Inteligente (End-to-End)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-RandomForest-orange?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Data-Pandas-150458?style=for-the-badge&logo=pandas)
![Status](https://img.shields.io/badge/Status-Production-green?style=for-the-badge)

> **Sistema de predicción meteorológica hiper-local para Barcelona basado en aprendizaje continuo y metodología CRISP-DM.**

---

## 📖 Descripción del Proyecto

**MeteoBCN** no es solo un modelo predictivo, es un **Pipeline de Datos completo** diseñado para operar de forma autónoma. El sistema monitoriza tres puntos estratégicos de la ciudad de Barcelona (Montaña, Centro y Entrada Sur) para generar un "Dato Maestro" consolidado.

El objetivo es resolver la falta de precisión de los modelos generalistas en microclimas locales, proporcionando predicciones de temperatura y lluvia a 24 horas vista con una precisión ajustada al histórico real de la ciudad (2009-Actualidad).

### ⚙️ Arquitectura del Sistema
El proyecto sigue una arquitectura modular dividida en tres fases automatizadas:

1.  **ETL (Extract, Transform, Load):** Extracción diaria de datos crudos, limpieza de nulos e interpolación.
2.  **Feature Engineering Avanzado:** Transformación de variables temporales y vectoriales para maximizar la interpretabilidad del modelo.
3.  **MLOps (Continuous Training):** Detección automática del ciclo semanal para re-entrenar modelos y evitar el *Model Drift*.

---

## 📂 Estructura del Repositorio

A continuación se detalla la organización técnica del proyecto:

```text
proyect3_IABD/
│
├── 📜 app_prediccion.py    # [ENTRY POINT] Orquestador principal. Ejecuta el pipeline diario.
│      
├── 📂 models/
│   ├── 📜 modelo_temperatura.py      # Módulo de entrenamiento (Regresor Random Forest).
│   ├── 📜 modelo_lluvia.py           # Módulo de entrenamiento (Clasificador Random Forest).
│
├── 📂 data/                      # Gestión de Datos y Modelos
│   ├── 📜 scraper_prediccion.py  # Herramienta de Web Scraping (Meteocat).
│   │
│   ├── 📂 raw_datasets/          # Datos crudos (Staging Area)
│   │   ├── meteocat_D5_resumen_historico.csv  # Estación Fabra
│   │   ├── meteocat_X4_resumen_historico.csv  # Estación Raval
│   │   └── meteocat_X8_resumen_historico.csv  # Estación Zona Univ.
│   │
│   ├── 📂 training_datasets/     # Datos procesados
│   │   └── dataset_entrenamiento_barcelona_MASTER.csv  # Dataset consolidado para ML
│   │
│   └── 📂 model_memory/          # Persistencia (Artifacts)
│       ├── cerebro_meteo_temperatura.pkl
│       ├── cerebro_meteo_lluvia.pkl
│       └── *.pkl (Metadatos de columnas)
│
├── 📂 data/  
│   ├── 📜 scraper_prediccion.py  
│
│
└── 📜 requirements.txt           # Dependencias del proyecto
```

## 🧠 Metodología y Tecnologías

El proyecto se adhiere a la metodología **CRISP-DM**, destacando en las fases de Preparación y Modelado:

### 1. Ingeniería de Características (Feature Engineering)
Para que los modelos de Machine Learning entiendan la naturaleza cíclica del clima, hemos aplicado transformaciones matemáticas avanzadas:

* **Codificación Cíclica Temporal:** El día del año (1-365) se transforma en coordenadas `Seno` y `Coseno`. Esto permite al modelo entender que el 31 de diciembre está matemáticamente "al lado" del 1 de enero.
* **Vectorización del Viento:** La dirección del viento (0°-360°) se descompone en vectores `u` (seno) y `v` (coseno) para evitar discontinuidades numéricas.
* **Inercia Térmica:** Cálculo de medias móviles (*Rolling Windows*) de 3 y 7 días para capturar tendencias (olas de calor o frentes fríos).

### 2. Estrategia de Modelado (Machine Learning)
Se utiliza un enfoque de **Ensemble Learning** mediante **Random Forest**:

* **Predicción de Temperatura:** `RandomForestRegressor` con 200 estimadores. Optimizado para minimizar el error en grados centígrados.
* **Predicción de Lluvia:** `RandomForestClassifier` con ponderación de clases (`class_weight='balanced'`). Esto es crucial para corregir el desbalanceo natural de los datos (hay muchos más días de sol que de lluvia en Barcelona).

### 3. Automatización (Pipeline Diario)
El script `app_prediccion.py` actúa como un agente inteligente:

* **Detección de Estado:** Verifica la fecha del último registro. Si falta el día de ayer, lanza el scraper automáticamente.
* **Re-entrenamiento Semanal:** Cada lunes, el sistema dispara el proceso de re-entrenamiento, generando nuevos archivos `.pkl` que incorporan la información de la última semana.

---

## 🚀 Instalación y Despliegue

Sigue estos pasos para ejecutar el sistema en local:

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/nilarroyo43/Proyect3_IABD.git](https://github.com/nilarroyo43/Proyect3_IABD.git)
   cd MeteoBCN
   ```
2. **Crear entorno virtual (Recomendado):**

    ```bash

    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

    ```bash
    Instalar dependencias:
    pip install -r requirements.txt
    ```

    ```bash
    Ejecutar el Pipeline:
    python app_prediccion.py
    El sistema detectará automáticamente si faltan datos de ayer, los descargará y generará la predicción.
    ```
---

## 📊 Resultados y Evaluación
El modelo ha sido validado utilizando un split temporal (Train/Test) para evitar fugas de datos (Data Leakage):

* Temperatura: El modelo es capaz de predecir la temperatura media del día siguiente con un margen de error (MAE) competitivo, aprovechando la fuerte correlación con la inercia térmica de los últimos 3 días.

* Lluvia: El clasificador ofrece una probabilidad de precipitación, permitiendo establecer umbrales de alerta personalizados (ej. Alerta si prob > 30%).

## 👥  Autores
Proyecto desarrollado para el Máster en IABD.
Nil Arroyo
Pol Panyella
Ronald Intriago
