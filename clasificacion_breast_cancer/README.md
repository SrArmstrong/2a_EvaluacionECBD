# Análisis de Clasificación para Diagnóstico de Cáncer de Mama

## Descripción del Proyecto
Este proyecto implementa un modelo de aprendizaje automático para clasificar tumores de mama como benignos o malignos basado en características físicas de las células. El conjunto de datos contiene 30 características computadas a partir de imágenes digitalizadas de masas mamarias.

## Características Técnicas
- **Algoritmo**: Random Forest Classifier
- **Lenguaje**: Python 3.8+
- **Bibliotecas principales**:
  - scikit-learn
  - pandas
  - numpy
  - matplotlib
  - seaborn

## Justificación del Algoritmo (Random Forest)
El algoritmo Random Forest fue seleccionado por las siguientes razones:

1. **Robustez**: Maneja bien datasets con múltiples características (30 en este caso)
2. **Interpretabilidad**: Proporciona importancia de características, crucial para diagnóstico médico
3. **Resistencia al sobreajuste**: Gracias al uso de múltiples árboles y subconjuntos aleatorios
4. **Alto rendimiento**: Demuestra consistentemente buena precisión en problemas de clasificación binaria
5. **Manejo de relaciones no lineales**: Captura automáticamente interacciones complejas entre variables

## Diseño del Modelo

### Preprocesamiento
- Codificación de variable objetivo: M (Maligno) = 1, B (Benigno) = 0
- Estandarización de características (StandardScaler)
- División del dataset: 70% entrenamiento, 30% prueba

### Optimización
Se realizó búsqueda en cuadrícula (GridSearchCV) para optimizar:
- `n_estimators`: [100, 200, 300] (número de árboles)
- `max_depth`: [None, 10, 20, 30] (profundidad máxima)
- `min_samples_split`: [2, 5, 10] (mínimo muestras para dividir nodo)
- `min_samples_leaf`: [1, 2, 4] (mínimo muestras en nodos hoja)

### Evaluación
Métricas principales utilizadas:
- Exactitud (Accuracy)
- AUC-ROC
- Precisión y Recall (especialmente para clase maligna)
- Matriz de confusión

## Resultados y Evaluación

### Métricas Finales
| Métrica         | Valor   |
|-----------------|---------|
| Exactitud       | 0.9824  |
| AUC-ROC         | 0.9952  |
| Precisión (Maligno) | 0.9744 |
| Recall (Maligno)    | 0.9815 |

### Gráficas Clave

#### 1. Matriz de Confusión
![Matriz de Confusión](confusion_matrix.png)
- **Interpretación**: El modelo tiene solo 2 falsos negativos (casos malignos clasificados como benignos), lo cual es crítico en aplicaciones médicas.

#### 2. Curva ROC
![Curva ROC](roc_curve.png)
- **AUC-ROC**: 0.995 indica excelente capacidad discriminativa
- **Interpretación**: El modelo distingue casi perfectamente entre clases

#### 3. Importancia de Características
![Importancia](feature_importance.png)
- **Top 3 características**:
  1. `concave points_worst`
  2. `perimeter_worst`
  3. `concave points_mean`
- **Interpretación**: Las mediciones de "peores" características (valores más extremos) son las más predictivas

## Repositorio del Proyecto
El código completo, modelo serializado y reporte detallado están disponibles en:
[GitHub Repository](https://github.com/tu_usuario/breast_cancer_classifier)

## Cómo Utilizar el Modelo
1. Instalar dependencias: `pip install -r requirements.txt`
2. Cargar el modelo pre-entrenado:
   ```python
   import joblib
   model = joblib.load('breast_cancer_rf_model.pkl')
   scaler = joblib.load('scaler.pkl')