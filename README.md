# Análisis de Rentabilidad para la Petrolera OilyGiant

## Resumen del Proyecto

Este proyecto analiza datos geológicos de tres regiones diferentes para determinar cuál es la más prometedora para la extracción de petróleo. El objetivo es construir un modelo de machine learning que prediga el volumen de reservas en nuevos pozos.

Utilizando estas predicciones, se aplica una simulación de bootstrapping para evaluar la rentabilidad y los riesgos asociados a cada región, permitiendo a la empresa OilyGiant tomar una decisión de inversión informada.

## Estructura del Repositorio

- `OilyGiant.py`: Script principal de Python que contiene todo el análisis, desde la carga de datos hasta la conclusión final.
- `datasets/`: Carpeta que contiene los archivos de datos para cada región (`geo_data_0.csv`, `geo_data_1.csv`, `geo_data_2.csv`).
- `requirements.txt`: Archivo que lista las dependencias de Python necesarias para ejecutar el proyecto.
- `README.md`: Este archivo.

## Metodología

El análisis se divide en tres fases principales:

1.  **Análisis Exploratorio de Datos (EDA):** Se cargan y examinan los datos de las tres regiones. Se visualiza la distribución del volumen de reservas (`product`) para entender las características iniciales de cada zona.

2.  **Entrenamiento del Modelo:** Para cada región, se entrena un modelo de **Regresión Lineal**. Las características (`f0`, `f1`, `f2`) se utilizan para predecir el volumen de reservas. El rendimiento del modelo se evalúa mediante el Error Cuadrático Medio (RMSE).

3.  **Análisis de Rentabilidad con Bootstrapping:**
    - Se definen las condiciones económicas: un presupuesto de **100M$** para desarrollar los **200 mejores pozos** y un ingreso de **4500$** por unidad de producto (1000 barriles).
    - Se aplica una técnica de bootstrapping con 1000 iteraciones para simular el proceso de selección. En cada iteración, se eligen los 200 mejores pozos (según las predicciones del modelo) de una muestra aleatoria de 500.
    - Se calcula el beneficio utilizando el volumen **real** de los pozos seleccionados.
    - Finalmente, se calcula el beneficio promedio, el intervalo de confianza del 95% y el riesgo de pérdidas para cada región.

## Resultados Clave

El análisis de bootstrapping arrojó los siguientes resultados consolidados:

| Métrica                           | Región 0          | Región 1         | Región 2          |
| :-------------------------------- | :---------------- | :--------------- | :---------------- |
| **Beneficio Promedio (M$)**       | 42.59             | **51.52**        | 43.50             |
| **Intervalo de Confianza 95% (M$)** | (-10.21, 94.79)   | (6.89, 93.15)    | (-12.88, 96.97)   |
| **Riesgo de Pérdidas (%)**        | 6.00%             | **1.00%**        | 6.40%             |

## Conclusión y Recomendación

El criterio de decisión se basa en dos puntos:
1.  **Riesgo de Pérdidas:** Se descartan las regiones con un riesgo superior al 2.5%. Las regiones 0 y 2 no cumplen este criterio.
2.  **Rentabilidad:** Entre las regiones que cumplen el criterio de riesgo, se selecciona la que tiene el mayor beneficio promedio.

La **Región 1** es la única que cumple con el umbral de riesgo (1.0%) y, además, presenta el mayor beneficio promedio estimado (51.52 M$).

### Recomendación Final: Se recomienda proceder con el desarrollo de pozos en la **REGIÓN 1**.

## Cómo Ejecutar el Proyecto

1.  **Clonar el repositorio**
    ```bash
    git clone <URL_DEL_REPOSITORIO>
    cd Sprint11-OilyGiant
    ```

2.  **Crear y activar un entorno virtual**
    ```bash
    # Crear el entorno
    python -m venv venv
    # Activar (Windows)
    venv\Scripts\activate
    # Activar (macOS/Linux)
    source venv/bin/activate
    ```

3.  **Instalar dependencias**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ejecutar el script de análisis**
    ```bash
    python OilyGiant.py
    ```