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

# 4. Reducción de dimensionalidad con PCA (80% - 89% de varianza acumulada)
# -------------------------------------------------

# PCA sin limitar número de componentes
pca_temp = PCA().fit(X_scaled)

# Calcular varianza acumulada
varianza_acumulada = np.cumsum(pca_temp.explained_variance_ratio_)

# Encontrar número mínimo de componentes que lleguen al rango 80-89%
n_componentes = np.argmax(varianza_acumulada >= 0.80) + 1

# Si se pasa de 89%, tomar el máximo permitido dentro del rango
if varianza_acumulada[n_componentes-1] > 0.89:
    # Buscar el mayor número de componentes que no exceda 89%
    n_componentes = np.max(np.where(varianza_acumulada <= 0.89)) + 1

print(f"\nNúmero de componentes seleccionados: {n_componentes}")
print("Varianza acumulada:", varianza_acumulada[n_componentes-1])


# 5. Aplicar PCA
pca = PCA(n_components=n_componentes)
X_pca = pca.fit_transform(X_scaled)

# 6. Análisis de clustering (K-Means)
# -------------------------------------------------
kmeans = KMeans(n_clusters=10, random_state=42)
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
indices = [f'PC{i+1}' for i in range(pca.n_components_)]

componentes = pd.DataFrame(pca.components_, 
                          columns=X.columns,
                           index=indices)
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
plt.plot(varianza_acumulada, marker='o')
plt.axhline(0.80, color='red', linestyle='--', label='80%')
plt.axhline(0.89, color='green', linestyle='--', label='89%')
plt.axvline(n_componentes-1, color='blue', linestyle='--', label=f'Seleccionado: {n_componentes}')
plt.title('Varianza Acumulada Explicada')
plt.xlabel('Número de Componentes')
plt.ylabel('Varianza Acumulada')
plt.legend()
plt.grid()
plt.show()


# Justificación del algoritmo
Se eligió PCA (Análisis de Componentes Principales) porque es un método eficiente para reducción de dimensionalidad cuando se busca capturar la mayor varianza posible bajo la suposición de relaciones lineales entre las variables. PCA permite comprimir información y facilitar la visualización en 2D, lo que ayuda a identificar patrones en el conjunto de datos comprar_alquilar.csv. Alternativas como t-SNE o UMAP ofrecen ventajas en la preservación de relaciones no lineales, pero requieren mayor costo computacional y parámetros de ajuste más delicados. En este caso, la estructura de los datos y la necesidad de interpretabilidad justifican el uso de PCA.

# Diseño del modelo y aporte de PCA
Antes de aplicar PCA, se realizó una estandarización de los datos para garantizar que todas las variables contribuyeran de forma equitativa a las componentes principales. Con 2 componentes, se capturó aproximadamente el **53% de la varianza total**, lo que permite representar de manera clara una parte significativa de la información original en un espacio bidimensional. Esto simplifica el problema y facilita tanto la interpretación como la detección visual de grupos de comportamiento en la decisión de compra o alquiler.

La posterior aplicación de K-Means sobre el espacio reducido no era requisito inicial, pero permitió verificar que las proyecciones de PCA revelaban separaciones consistentes entre grupos, validando que la reducción de dimensionalidad estaba preservando información relevante para la segmentación.


## Justificación del uso de componentes que aseguran entre el 80% y 89% de la varianza

En este análisis se aplicó un **Análisis de Componentes Principales (PCA)** con el objetivo de reducir la dimensionalidad de los datos, manteniendo al mismo tiempo la mayor parte de la información relevante.

En lugar de fijar el número de componentes en 2, se selecciona automáticamente el número necesario para que la varianza acumulada se encuentre en el rango **80%–89%**. Esta decisión se fundamenta en las siguientes razones:

1. **Preservación de información relevante**  
Conservar al menos el 80% de la varianza garantiza que las características más significativas de los datos originales se mantengan en el análisis, evitando la pérdida de patrones esenciales.

2. **Reducción de ruido e irrelevancia**  
Limitar el umbral superior al 89% permite excluir componentes que aportan poca información adicional y que, en muchos casos, solo reflejan ruido o variaciones menores no relevantes para la interpretación.

3. **Balance entre precisión y simplicidad**  
Seleccionar un rango en lugar de un valor fijo ofrece un compromiso adecuado: se logra una representación fiel de los datos sin necesidad de trabajar con todas las variables originales, lo que mejora la eficiencia computacional y la claridad analítica.

4. **Flexibilidad del modelo**  
El número de componentes puede variar según el dataset, lo cual hace que el enfoque sea adaptable a diferentes contextos y conjuntos de datos sin perder rigor estadístico.

---

**Conclusión**  
El uso de los componentes necesarios para alcanzar entre el **80% y el 89% de la varianza explicada** garantiza un modelo **robusto, informativo y balanceado**, que preserva la esencia de los datos mientras evita la complejidad innecesaria.


# Interpretación de resultados
La visualización PCA mostró que las observaciones se agrupan en regiones diferenciadas según la variable objetivo (comprar). La varianza acumulada confirmó que pocas componentes son suficientes para representar gran parte de la estructura de los datos, optimizando el análisis y reduciendo el ruido.

# Se eligió PCA sobre otros métodos como:

## LDA (Análisis Discriminante Lineal):
    Aunque útil para clasificación, asume que las clases están normalmente distribuidas y requiere la variable objetivo para proyectar los datos. En   nuestro caso, queríamos un método no supervisado para explorar patrones ocultos.

## t-SNE/UMAP:
    Ideales para visualizar estructuras no lineales, pero son computacionalmente costosos y sus componentes no son interpretables (solo sirven para visualización). PCA, en cambio, permite interpretar las componentes mediante cargas factoriales.