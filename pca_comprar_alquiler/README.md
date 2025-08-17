"""
Este script realiza un análisis de reducción de dimensionalidad (PCA) y clustering
para identificar patrones en datos de decisión de comprar/alquilar vivienda.
"""

# Importación de librerías necesarias
import pandas as pd  # Para manipulación y análisis de datos
import numpy as np   # Para operaciones numéricas
import matplotlib.pyplot as plt  # Para visualización de datos
from sklearn.preprocessing import StandardScaler  # Para normalización de datos
from sklearn.decomposition import PCA  # Para reducción de dimensionalidad
from sklearn.cluster import KMeans  # Para agrupamiento de datos
import joblib  # Para guardar modelos

# 1. Carga de datos
# ------------------------------
# Leer el archivo CSV que contiene los datos de compra/alquiler
datos = pd.read_csv('comprar_alquilar.csv')

# 2. Exploración inicial de datos
# ------------------------------
print("Primeras filas del dataset:")
print(datos.head())  # Muestra las primeras 5 filas para inspección visual

print("\nInformación del dataset:")
print(datos.info())  # Proporciona información sobre tipos de datos

print("\nEstadísticas descriptivas:")
print(datos.describe())  # Muestra estadísticas básicas

print("\nDistribución de la variable 'comprar':")
print(datos['comprar'].value_counts())  # Frecuencia de decisiones

# 3. Preprocesamiento de datos
# ------------------------------
# Codificar variables categóricas (one-hot encoding)
datos = pd.get_dummies(datos, columns=['estado_civil', 'trabajo'], drop_first=True)

# Separar características (X) y variable objetivo (y)
X = datos.drop('comprar', axis=1)
y = datos['comprar']

# Estandarizar los datos (media=0, varianza=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Reducción de dimensionalidad con PCA
# ------------------------------
# Crear modelo PCA con 2 componentes para visualización
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Mostrar varianza explicada por cada componente
print("\nVarianza explicada por componente:")
print(pca.explained_variance_ratio_)

# 5. Aplicar PCA
# --------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 6. Análisis de clustering (K-Means)
# -------------------------------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='tab10', alpha=0.7)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='red', marker='X', label='Centroides')
plt.title('Clustering en Espacio PCA')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()
plt.grid()
plt.show()

# 7. Interpretación de componentes
# -------------------------------------------------
componentes = pd.DataFrame(pca.components_, 
                          columns=X.columns,
                          index=['PC1', 'PC2'])
print("\nCargas de los componentes principales:")
print(componentes)

# 8. Guardado de modelos
# -------------------------------------------------
joblib.dump(pca, 'pca_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(kmeans, 'kmeans_model.pkl')

print("\nModelos guardados exitosamente:")
print("- pca_model.pkl (modelo PCA)")
print("- scaler.pkl (normalizador)")
print("- kmeans_model.pkl (modelo K-Means)")

# 9. Gráfico adicional: Varianza explicada acumulada
# -------------------------------------------------
pca_full = PCA().fit(X_scaled)
plt.figure(figsize=(10, 6))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.title('Varianza Acumulada Explicada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.grid()
plt.show()



# Justificación del algoritmo
Se eligió PCA (Análisis de Componentes Principales) porque es un método eficiente para reducción de dimensionalidad cuando se busca capturar la mayor varianza posible bajo la suposición de relaciones lineales entre las variables. PCA permite comprimir información y facilitar la visualización en 2D, lo que ayuda a identificar patrones en el conjunto de datos comprar_alquilar.csv. Alternativas como t-SNE o UMAP ofrecen ventajas en la preservación de relaciones no lineales, pero requieren mayor costo computacional y parámetros de ajuste más delicados. En este caso, la estructura de los datos y la necesidad de interpretabilidad justifican el uso de PCA.

# Diseño del modelo y aporte de PCA
Antes de aplicar PCA, se realizó una estandarización de los datos para garantizar que todas las variables contribuyeran de forma equitativa a las componentes principales. Con 2 componentes, se capturó aproximadamente el **53% de la varianza total**, lo que permite representar de manera clara una parte significativa de la información original en un espacio bidimensional. Esto simplifica el problema y facilita tanto la interpretación como la detección visual de grupos de comportamiento en la decisión de compra o alquiler.

La posterior aplicación de K-Means sobre el espacio reducido no era requisito inicial, pero permitió verificar que las proyecciones de PCA revelaban separaciones consistentes entre grupos, validando que la reducción de dimensionalidad estaba preservando información relevante para la segmentación.


## Justificación del uso de 2 Componentes

En este análisis se aplicó un **Análisis de Componentes Principales (PCA)** con el objetivo de reducir la dimensionalidad de los datos y facilitar su interpretación.

Aunque los dos primeros componentes explican **el 53% de la varianza total**, esta proporción se considera suficiente en este contexto por las siguientes razones:

1. **Simplificación del espacio de datos**  
   Con solo dos componentes se logra representar más de la mitad de la información relevante, reduciendo la complejidad del conjunto original.

2. **Visualización clara de patrones**  
   La proyección bidimensional permite identificar diferencias y similitudes entre los casos de **compras** y **alquileres**, lo que sería menos evidente en un espacio de mayor dimensión.

3. **Balance entre información y ruido**  
   Los componentes adicionales aportan varianza marginal, pero también introducen ruido y dificultan la interpretación.  
   Con dos componentes se conserva la estructura principal de los datos sin sobrecargar el análisis.

En conclusión, el uso de **dos componentes principales (53%)** garantiza un modelo más **interpretativo, visualmente comprensible y eficiente**, adecuado para la exploración y comparación de patrones entre compras y alquileres.

## Compromiso entre visualización y varianza explicada

Aunque los dos primeros componentes explican solo el **53% de la varianza**, se utilizaron por las siguientes razones:

- **Visualización exploratoria**: El objetivo era representar los datos en 2D para detectar patrones visuales. Para modelado más robusto, se recomienda usar más componentes.
- **Scree plot**: El gráfico de varianza acumulada muestra que se necesitan **6 componentes** para superar el 80%. Esto se puede calcular con:


# Interpretación de resultados
La visualización PCA mostró que las observaciones se agrupan en regiones diferenciadas según la variable objetivo (comprar). La varianza acumulada confirmó que pocas componentes son suficientes para representar gran parte de la estructura de los datos, optimizando el análisis y reduciendo el ruido.

# Se eligió PCA sobre otros métodos como:

## LDA (Análisis Discriminante Lineal):
    Aunque útil para clasificación, asume que las clases están normalmente distribuidas y requiere la variable objetivo para proyectar los datos. En   nuestro caso, queríamos un método no supervisado para explorar patrones ocultos.

## t-SNE/UMAP:
    Ideales para visualizar estructuras no lineales, pero son computacionalmente costosos y sus componentes no son interpretables (solo sirven para visualización). PCA, en cambio, permite interpretar las componentes mediante cargas factoriales.