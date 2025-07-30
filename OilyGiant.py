# ==============================================================================
# SECCIÓN 0: IMPORTACIÓN DE LIBRERÍAS
# ==============================================================================
# Importaciones estándar para el manejo de datos y operaciones numéricas
import pandas as pd
import numpy as np

# Importaciones para el modelado y la evaluación en scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Importaciones para la visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración para mejorar la presentación de los gráficos
sns.set_style('whitegrid')
# Ignorar advertencias que no afectan el resultado
import warnings
warnings.filterwarnings('ignore')


# ==============================================================================
# SECCIÓN 1: CARGA Y ANÁLISIS EXPLORATORIO DE DATOS
# ==============================================================================
print("--- 1. Carga y Análisis Exploratorio de Datos ---")
# Se cargan los datos de las tres regiones.
# Se incluye un bloque try-except para manejar diferentes entornos de ejecución.
try:
    df0 = pd.read_csv('./datasets/geo_data_0.csv')
    df1 = pd.read_csv('./datasets/geo_data_1.csv')
    df2 = pd.read_csv('./datasets/geo_data_2.csv')
except FileNotFoundError:
    print("Archivos locales no encontrados.")
    #url_base = 'https://practicum-content.s3.us-west-1.amazonaws.com/datasets/'
    #df0 = pd.read_csv(f'{url_base}geo_data_0.csv')
    #df1 = pd.read_csv(f'{url_base}geo_data_1.csv')
    #df2 = pd.read_csv(f'{url_base}geo_data_2.csv')

# Visualización de la distribución de reservas (variable 'product') por región
# Esto nos da una idea inicial de las características de cada zona.
fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=True)
fig.suptitle('Distribución del Volumen de Reservas (product) por Región', fontsize=16)

# Gráfico para Región 0
sns.histplot(df0['product'], ax=axes[0], kde=True, bins=30)
axes[0].set_title('Región 0')
axes[0].axvline(df0['product'].mean(), color='red', linestyle='--', label=f'Media: {df0["product"].mean():.2f}')
axes[0].legend()


# Gráfico para Región 1
sns.histplot(df1['product'], ax=axes[1], kde=True, bins=30)
axes[1].set_title('Región 1')
axes[1].axvline(df1['product'].mean(), color='red', linestyle='--', label=f'Media: {df1["product"].mean():.2f}')
axes[1].legend()

# Gráfico para Región 2
sns.histplot(df2['product'], ax=axes[2], kde=True, bins=30)
axes[2].set_title('Región 2')
axes[2].axvline(df2['product'].mean(), color='red', linestyle='--', label=f'Media: {df2["product"].mean():.2f}')
axes[2].legend()

plt.show()

# Observación: La distribución en la Región 1 es notablemente diferente,
# no sigue una curva normal y parece tener picos discretos.
# Las Regiones 0 y 2 tienen distribuciones más parecidas a una normal.


# ==============================================================================
# SECCIÓN 2: ENTRENAMIENTO Y EVALUACIÓN DE MODELOS
# ==============================================================================
print("\n--- 2. Entrenamiento y Evaluación de Modelos por Región ---")

def train_and_evaluate(df, region_name):
    """
    Función que encapsula el entrenamiento y evaluación para una región.
    Devuelve las predicciones y los valores reales para análisis posteriores.
    """
    # Se excluye 'id' (identificador) y 'product' (objetivo) de las características.
    features = df.drop(['id', 'product'], axis=1)
    target = df['product']

    # División de datos 75/25 como se especifica en las instrucciones.
    features_train, features_valid, target_train, target_valid = train_test_split(
        features, target, test_size=0.25, random_state=12345)

    # Entrenamiento del modelo de Regresión Lineal.
    model = LinearRegression()
    model.fit(features_train, target_train)

    # Predicción sobre el conjunto de validación.
    predictions = pd.Series(model.predict(features_valid), index=target_valid.index)

    # Cálculo de métricas de rendimiento.
    rmse = mean_squared_error(target_valid, predictions)**0.5
    mean_predicted_volume = predictions.mean()

    # Impresión de resultados clave.
    print(f"--- Resultados para la Región: {region_name} ---")
    print(f"Volumen medio de reservas predicho: {mean_predicted_volume:.2f} (miles de barriles)")
    print(f"RMSE del modelo: {rmse:.2f}")

    return predictions, target_valid

# Se aplica la función a cada región y se almacenan los resultados.
predictions_0, target_0 = train_and_evaluate(df0, 'Región 0')
predictions_1, target_1 = train_and_evaluate(df1, 'Región 1')
predictions_2, target_2 = train_and_evaluate(df2, 'Región 2')

# Observación: El RMSE de la Región 1 es casi cero, lo que indica un ajuste
# casi perfecto. Esto puede deberse a que los datos son sintéticos y una
# de las características ('f2') tiene una correlación lineal directa con 'product'.
# Aunque esto mejora la predicción, no garantiza mayores reservas reales.

# Visualización del rendimiento del modelo: Predicciones vs. Valores Reales
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Rendimiento del Modelo: Predicciones vs. Valores Reales', fontsize=16)

# Gráfico para Región 0
sns.scatterplot(x=target_0, y=predictions_0, alpha=0.3, ax=axes[0])
axes[0].plot([0, 190], [0, 190], color='red', linestyle='--') # Línea de predicción perfecta
axes[0].set_title('Región 0')
axes[0].set_xlabel('Volumen Real')
axes[0].set_ylabel('Volumen Predicho')

# Gráfico para Región 1
sns.scatterplot(x=target_1, y=predictions_1, alpha=0.3, ax=axes[1])
axes[1].plot([0, 190], [0, 190], color='red', linestyle='--')
axes[1].set_title('Región 1 (RMSE muy bajo)')
axes[1].set_xlabel('Volumen Real')
axes[1].set_ylabel('Volumen Predicho')

# Gráfico para Región 2
sns.scatterplot(x=target_2, y=predictions_2, alpha=0.3, ax=axes[2])
axes[2].plot([0, 190], [0, 190], color='red', linestyle='--')
axes[2].set_title('Región 2')
axes[2].set_xlabel('Volumen Real')
axes[2].set_ylabel('Volumen Predicho')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Observación: El gráfico confirma el bajo RMSE de la Región 1, donde los puntos
# se alinean casi perfectamente sobre la línea roja. En las Regiones 0 y 2,
# la nube de puntos es mucho más dispersa, mostrando la incertidumbre del modelo.


# ==============================================================================
# SECCIÓN 3: CÁLCULO DE BENEFICIOS Y RIESGOS CON BOOTSTRAPPING
# ==============================================================================
print("\n--- 3. Análisis de Rentabilidad y Riesgo con Bootstrapping ---")

# --- Variables y Constantes del Negocio ---
BUDGET = 100_000_000
WELLS_TO_EXPLORE = 500
WELLS_TO_DEVELOP = 200
PRICE_PER_UNIT = 4500  # Ingreso por 1000 barriles
N_SAMPLES = 1000

def bootstrap_profit_analysis(predictions, target):
    """
    Realiza el análisis de bootstrapping para una región.
    Simula 1000 escenarios de selección de pozos para estimar la distribución de beneficios.
    """
    # Se usa un estado aleatorio para asegurar la reproducibilidad de los resultados.
    state = np.random.RandomState(12345)
    profit_values = []

    # Se combinan predicciones y valores reales para un muestreo correcto.
    data = pd.DataFrame({'predictions': predictions, 'target': target})

    for _ in range(N_SAMPLES):
        # 1. Se toma una sub-muestra de 500 pozos con reemplazo.
        sub_sample = data.sample(n=WELLS_TO_EXPLORE, replace=True, random_state=state)

        # 2. Se eligen los 200 mejores pozos según la PREDICCIÓN del modelo.
        top_200 = sub_sample.sort_values(by='predictions', ascending=False).head(WELLS_TO_DEVELOP)

        # 3. Se calcula el beneficio usando el volumen REAL de esos 200 pozos.
        actual_volume_for_top_200 = top_200['target'].sum()
        profit = (actual_volume_for_top_200 * PRICE_PER_UNIT) - BUDGET
        profit_values.append(profit)

    # Se calculan las métricas finales a partir de la distribución de 1000 beneficios.
    profit_values = pd.Series(profit_values)
    mean_profit = profit_values.mean()
    confidence_interval = (profit_values.quantile(0.025), profit_values.quantile(0.975))
    risk_of_loss = (profit_values < 0).mean() * 100

    return profit_values, mean_profit, confidence_interval, risk_of_loss

# Se almacenan los resultados de cada región.
results_dict = {}
regions_data = {
    'Región 0': (predictions_0, target_0),
    'Región 1': (predictions_1, target_1),
    'Región 2': (predictions_2, target_2)
}

for name, (preds, targs) in regions_data.items():
    profits, mean_p, ci, risk = bootstrap_profit_analysis(preds, targs)
    results_dict[name] = {
        'profits_dist': profits,
        'Beneficio Promedio (M$)': mean_p / 1_000_000,
        'Intervalo de Confianza 95% (M$)': (ci[0] / 1_000_000, ci[1] / 1_000_000),
        'Riesgo de Pérdidas (%)': risk
    }

# Se presenta la tabla de resultados finales.
results_df = pd.DataFrame(results_dict).T.drop(columns=['profits_dist'])
print(results_df.to_string(formatters={
    'Beneficio Promedio (M$)': '{:,.2f}'.format,
    'Intervalo de Confianza 95% (M$)': lambda x: f"({x[0]:.2f}, {x[1]:.2f})",
    'Riesgo de Pérdidas (%)': '{:,.2f}%'.format
}))

# Visualización de la Distribución de Beneficios del Bootstrapping
fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharex=True, sharey=True)
fig.suptitle('Distribución de Beneficios por Región (1000 Simulaciones de Bootstrapping)', fontsize=16)

for i, (name, data) in enumerate(results_dict.items()):
    sns.histplot(data['profits_dist'] / 1_000_000, ax=axes[i], kde=True, bins=30)
    axes[i].axvline(0, color='red', linestyle='--', label='Punto de Equilibrio (0$)')
    axes[i].set_title(f"{name}\nRiesgo de Pérdida: {data['Riesgo de Pérdidas (%)']:.2f}%")
    axes[i].set_xlabel('Beneficio (Millones de $)')
    axes[i].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# ==============================================================================
# SECCIÓN 4: CONCLUSIÓN FINAL
# ==============================================================================
print("\n--- 4. Conclusión Final del Proyecto ---")
print("Basado en el análisis, la recomendación es la siguiente:")
print("1. Criterio de Riesgo: Se deben descartar las regiones con un riesgo de pérdidas > 2.5%.")
print("   - Región 0: Riesgo de 6.00% (DESCARTADA)")
print("   - Región 2: Riesgo de 6.40% (DESCARTADA)")
print("   - Región 1: Riesgo de 1.00% (ACEPTADA)")
print("\n2. Criterio de Rentabilidad: Entre las regiones aceptadas, se elige la de mayor beneficio promedio.")
print("   - La Región 1, además de ser la más segura, ofrece el mayor beneficio promedio estimado de $51.52 millones.")
print("\nRECOMENDACIÓN FINAL: Proceder con la inversión en la REGIÓN 1. 🥇")