"""
Este script realiza un análisis de regresión lineal para predecir las carreras anotadas (runs)
en función del número de bateos en un conjunto de datos de béisbol.
"""

# Importación de librerías necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.model_selection import train_test_split  # Para dividir datos en entrenamiento/prueba
from sklearn.linear_model import LinearRegression  # Modelo de regresión lineal
from sklearn import metrics  # Métricas para evaluación de modelos

# 1. Carga de datos
# ------------------------------
# Leer el archivo CSV que contiene los datos de béisbol
datos = pd.read_csv('beisbol.csv')

# 2. Exploración inicial de datos
# ------------------------------
print("Primeras filas del dataset:")
print(datos.head())  # Muestra las primeras 5 filas para inspección visual

print("\nInformación del dataset:")
print(datos.info())  # Proporciona información sobre tipos de datos y valores no nulos

print("\nEstadísticas descriptivas:")
print(datos.describe())  # Muestra estadísticas básicas (media, desviación estándar, etc.)

# 3. Limpieza y preparación de datos
# ------------------------------
# Eliminar filas con valores nulos en las columnas clave 'bateos' y 'runs'
datos = datos.dropna(subset=['bateos', 'runs'])

# Verificar valores nulos después de la limpieza
print("\nValores nulos por columna después de limpieza:")
print(datos.isnull().sum())  # Cuenta valores nulos por columna

# 4. Preparación de variables
# ------------------------------
# Convertir las columnas a arrays numpy y redimensionarlas para sklearn
X = datos['bateos'].values.reshape(-1,1)  # Variable independiente (predictora)
y = datos['runs'].values.reshape(-1,1)    # Variable dependiente (a predecir)

# 5. División de datos
# ------------------------------
# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
# random_state=42 garantiza reproducibilidad en la división
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Creación y entrenamiento del modelo
# ------------------------------
# Instanciar el modelo de regresión lineal
regressor = LinearRegression()  

# Entrenar el modelo con los datos de entrenamiento
regressor.fit(X_train, y_train)

# 7. Predicciones
# ------------------------------
# Predecir valores de 'runs' para el conjunto de prueba
y_pred = regressor.predict(X_test)

# 8. Evaluación del modelo
# ------------------------------
print('\n--- Métricas de Evaluación ---')
# Error cuadrático medio (MSE) - promedio de los errores al cuadrado
print('Error cuadrático medio (MSE):', metrics.mean_squared_error(y_test, y_pred))

# Raíz del error cuadrático medio (RMSE) - error en las mismas unidades que 'runs'
print('Raíz del error cuadrático medio (RMSE):', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Coeficiente de determinación R² - proporción de varianza explicada
print('Coeficiente de determinación R²:', regressor.score(X_test, y_test))

# 8.1 Reentrenamiento y Optimización del Modelo
# ---------------------------------------------

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Definir rango de valores para alpha (regularización)
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}

# Configurar búsqueda en cuadrícula con validación cruzada
ridge_grid = GridSearchCV(Ridge(), param_grid, scoring='neg_mean_squared_error', cv=5)
ridge_grid.fit(X_train, y_train.ravel())  # ravel() para convertir y_train a 1D

# Extraer mejor modelo y parámetros
best_ridge = ridge_grid.best_estimator_
best_alpha = ridge_grid.best_params_['alpha']
best_mse = -ridge_grid.best_score_

print('\n--- Reentrenamiento con Ridge ---')
print(f"Mejor valor de alpha: {best_alpha}")
print(f"Mejor MSE (validación cruzada): {best_mse:.4f}")

# Evaluar en conjunto de prueba
y_pred_ridge = best_ridge.predict(X_test)
ridge_rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred_ridge))
ridge_r2 = best_ridge.score(X_test, y_test)

print(f"RMSE en prueba: {ridge_rmse:.4f}")
print(f"R² en prueba: {ridge_r2:.4f}")

# Comparar gráficamente con modelo original
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', label='Datos reales')
plt.plot(X_test, y_pred, color='red', label='Regresión Lineal')
plt.plot(X_test, y_pred_ridge, color='green', linestyle='--', label='Ridge Optimizado')
plt.title('Comparación de Modelos: Lineal vs Ridge', fontsize=14)
plt.xlabel('Número de Bateos', fontsize=12)
plt.ylabel('Carreras Anotadas (Runs)', fontsize=12)
plt.legend()
plt.grid(True)
plt.show()

# 9. Parámetros del modelo
# ------------------------------
print('\n--- Parámetros del Modelo ---')
# Intercepto (b) - valor de 'runs' cuando 'bateos' es cero
print('Intercepto (b):', regressor.intercept_)

# Pendiente (m) - cambio esperado en 'runs' por cada unidad de 'bateos'
print('Pendiente (m):', regressor.coef_)

# 10. Interpretación del modelo
# ------------------------------
print('\n--- Interpretación ---')
# Mostrar la ecuación del modelo con formato numérico
print(f"Ecuación del modelo: runs = {regressor.intercept_[0]:.2f} + {regressor.coef_[0][0]:.4f} * bateos")

# Interpretación práctica del coeficiente
print(f"Por cada 100 bateos adicionales, se esperan {100 * regressor.coef_[0][0]:.2f} runs adicionales")

# 11. Visualización de resultados
# ------------------------------
# Crear figura con tamaño específico (10x6 pulgadas)
plt.figure(figsize=(10, 6))

# Graficar datos reales (puntos azules)
plt.scatter(X_test, y_test, color='blue', label='Datos reales')

# Graficar línea de regresión (línea roja)
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Línea de regresión')

# Configuración del gráfico
plt.title('Relación entre Bateos y Carreras (Runs)', fontsize=14)
plt.xlabel('Número de Bateos', fontsize=12)
plt.ylabel('Carreras Anotadas (Runs)', fontsize=12)
plt.legend()  # Mostrar leyenda
plt.grid(True)  # Mostrar cuadrícula
plt.show()  # Mostrar gráfico

# 12. Análisis de residuos
# ------------------------------
# Calcular residuos (diferencia entre valores reales y predichos)
residuos = y_test - y_pred

# Crear gráfico de residuos
plt.figure(figsize=(10, 6))

# Graficar residuos vs valores predichos
plt.scatter(y_pred, residuos, color='green', alpha=0.6)

# Línea horizontal en y=0 como referencia
plt.axhline(y=0, color='red', linestyle='--')

# Configuración del gráfico
plt.title('Análisis de Residuos', fontsize=14)
plt.xlabel('Valores Predichos', fontsize=12)
plt.ylabel('Residuos (Real - Predicho)', fontsize=12)
plt.grid(True)
plt.show()

# 13. Distribución de los residuos
# ------------------------------
plt.figure(figsize=(10, 6))

# Crear histograma de los residuos
plt.hist(residuos, bins=20, color='purple', alpha=0.7)

# Configuración del gráfico
plt.title('Distribución de los Residuos', fontsize=14)
plt.xlabel('Error (Real - Predicho)', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.grid(True)
plt.show()