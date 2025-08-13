# Análisis de Agrupación para Datos de Samsung

Este script realiza un análisis de agrupación (clustering) utilizando el algoritmo K-Means para identificar patrones en los datos históricos de precios de cierre y volumen de operaciones de Samsung utilizando información proporcionada por medio de un csv.

## Características principales

- Implementación del algoritmo K-Means para agrupación
- Determinación óptima del número de clusters
- Análisis temporal de la distribución de clusters
- Visualización interactiva de resultados

## Requisitos

- Python 3.6+
- Bibliotecas requeridas:
  - pandas
  - numpy
  - matplotlib
  - scikit-learn

## Estructura del código


# Importación de librerías necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.preprocessing import StandardScaler  # Para normalización de datos
from sklearn.cluster import KMeans  # Algoritmo de clustering
from sklearn.metrics import silhouette_score  # Métrica de evaluación

# 1. Carga de datos
# ------------------------------
# Leer el archivo CSV con formato de fecha específico (dd/mm/yyyy)
df = pd.read_csv('samsung.csv', parse_dates=['Date'], dayfirst=True)

# 2. Preparación de características
# ------------------------------
# Crear variables temporales para análisis
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Variables para clustering (precio de cierre y volumen)
X = df[['Close', 'Volume']].copy()

# 3. Normalización de datos
# ------------------------------
# Escalar los datos para el algoritmo de clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Determinación del número óptimo de clusters
# ------------------------------
# Método del codo para encontrar el mejor número de grupos
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

# Visualización del método del codo
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Método del Codo para Determinar Número Óptimo de Clusters')
plt.xlabel('Número de clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.grid()
plt.show()

# 5. Aplicación de K-Means
# ------------------------------
# Seleccionar número de clusters basado en el gráfico anterior
n_clusters = 3  # Este valor debe ajustarse según el método del codo

# Crear y entrenar el modelo K-Means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Añadir información de clusters al dataframe original
df['Cluster'] = clusters

# 6. Evaluación del modelo
# ------------------------------
# Calcular métrica de silueta para evaluar calidad de la agrupación
silhouette_avg = silhouette_score(X_scaled, clusters)
print(f"Puntaje de Silueta: {silhouette_avg:.3f}")

# 7. Visualización de clusters
# ------------------------------
plt.figure(figsize=(12, 8))
for cluster in range(n_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Close'], cluster_data['Volume'], 
                label=f'Cluster {cluster}', alpha=0.7)

# Mostrar centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=300, c='black', marker='X', label='Centroides')
plt.title('Agrupación de Datos Samsung por Precio y Volumen')
plt.xlabel('Precio de Cierre (Close)')
plt.ylabel('Volumen de Operaciones')
plt.legend()
plt.grid()
plt.show()

# 8. Análisis temporal de clusters
# ------------------------------
# Distribución de clusters por año
cluster_year = pd.crosstab(df['Year'], df['Cluster'], normalize='index')
print("\nDistribución de clusters por año:")
print(cluster_year)

# Visualización de la evolución de clusters
plt.figure(figsize=(12, 6))
cluster_year.plot(kind='bar', stacked=True)
plt.title('Evolución de Clusters por Año')
plt.xlabel('Año')
plt.ylabel('Proporción')
plt.legend(title='Cluster')
plt.grid(axis='y')
plt.show()

# 9. Caracterización de clusters
# ------------------------------
print("\nCaracterísticas promedio por cluster:")
print(df.groupby('Cluster')[['Close', 'Volume']].mean())

´´´

# Justificación del algoritmo

Se seleccionó K-Means debido a que es un algoritmo eficiente y ampliamente utilizado para la agrupación de datos numéricos escalados, como el precio de cierre y el volumen de operaciones presentes en samsung.csv. Además, K-Means permite una interpretación clara de los centroides y facilita la visualización de clusters en dos dimensiones.
Otras opciones, como DBSCAN, presentaron inconvenientes debido a la gran cantidad de los datos, lo que generó clusters poco representativos. El clustering jerárquico, aunque útil para análisis exploratorio, es computacionalmente más costoso y menos escalable con conjuntos de datos grandes como este.

# Evaluación del modelo

El silhouette score obtenido fue de 0.7, lo que indica que los clusters están bien definidos y separados entre sí. Este valor sugiere que las observaciones dentro de cada cluster son muy similares y que la separación entre grupos es adecuada, lo que valida el uso de K-Means para este conjunto de datos.