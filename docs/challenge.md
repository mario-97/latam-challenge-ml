# Descripción breve del problema

Se ha proporcionado un cuaderno Jupyter (exploration.ipynb) con el trabajo de un Científico de Datos (en adelante, DS). El DS entrenó un modelo para predecir la probabilidad de retraso para un vuelo que despega o aterriza en el aeropuerto SCL. 

- Un vuelo con mas de 15 min de atraso se considera "Atrasado" (delay = 1), sino, este será etiquetado como no atrasado (delay = 0)

# Correción de errores preliminar

1.- Error en clase Delay: Completar funciones para realizar los test y verificar datos

2.- Error en la función "Barplot": En este caso, se esta entregando de manera incorrecta los parametros a la funcion, si bien, se sabe que el primer elemento corresponde a los datos del eje X, y el segundo al eje Y, estos deben quedar declarados explicitamente con sus nombres de la siguiente manera: Ejemplo, "sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)"

3.- Actualización y librerías faltantes en requeriments.txt

4.- Corrección en test_model.py: Carga de archivo data.csv, y ajuste de Entrenamiento y predicciones 

# Desarrollo de Preprocesamiento, Aplicación de Modelos, Análisis y Conclusiones

1. **Ordenar imports al comienzo del código**
2. **Ordenar funciones al comienzo del código**
3. **Metodología KDD:**
    1. **Selección de datos:** Se utilizaron datos públicos y reales sobre vuelos comerciales de LATAM.
    2. **Preprocesamiento:** Se aplicaron los siguientes preprocesamientos:
        - Codificación de características categóricas mediante one-hot encoding, específicamente a las columnas OPERA, TIPOVUELO y MES.
        - Selección de características importantes con XGBoost: Esto reduce la dimensionalidad de los datos y se enfoca en las características que más influyen en la predicción, mejorando así la eficiencia del modelo y evitando posibles efectos negativos del ruido en datos menos importantes.
        - Balance de clases: Esto implica aumentar o disminuir la cantidad de instancias en cada clase, lo que ayuda a que el modelo no esté sesgado hacia la clase mayoritaria y logre un rendimiento más equilibrado.
        - Normalización de características con StandardScaler: Este proceso ajusta las características para que tengan una media de cero y una desviación estándar de uno, lo que ayuda a que las características estén en la misma escala y evita que algunas características dominen otras en la predicción
        - Nueva codificación de características categóricas, esta vez a todas las columnas presentes, utilizando el método de LabelEncoder: Aunque esto se hizo previamente en un subconjunto de columnas, esta vez se aplicó a todas las columnas categóricas restantes. Esto permite que todas las características categóricas se representen numéricamente de manera coherente en todo el conjunto de datos
    3. **Transformación:** No aplicada.
    4. **Aplicación de modelos:** Se utilizaron los siguientes modelos:
        - XGBoost
        - Logistic Regression
        - Random Forest
    5. **Evaluación, análisis e interpretación:**
        - **Métricas utilizadas:**
            - **ROC AUC (Receiver Operating Characteristic Area Under the Curve):** Mide la capacidad del modelo para distinguir entre clases positivas y negativas utilizando diferentes umbrales de probabilidad.
            - **MCC (Matthews Correlation Coefficient):** Combina la precisión y el recall para evaluar el rendimiento del modelo, especialmente en datos desbalanceados.
            - **Accuracy (Exactitud):** Mide la proporción de predicciones correctas sobre el total de predicciones realizadas por el modelo.
            - **Precision (Precisión):** Mide la proporción de verdaderos positivos sobre el total de predicciones positivas realizadas por el modelo.
            - **Recall (Recuperación o Sensibilidad):** Mide la proporción de verdaderos positivos sobre el total de instancias verdaderamente positivas en el conjunto de datos.
            - **F1-Score:** Es la media armónica entre precisión y recall, útil para conjuntos de datos desbalanceados y cuando se busca un equilibrio entre ambas métricas.
        - **Análisis:** Considerando la naturaleza de los datos y nuestro objetivo de clasificar vuelos atrasados, debemos dar importancia a una alta tasa de predicción de las clases positivas, y con una tasa aceptable de predicciones negativas. Para ello, debemos considerar las siguientes métricas:
            - Precisión y Recall
            - Además:
                - Recall de la clase 0 < 0.60
                - F1-score de la clase 0 < 0.70
                - Recall de la clase 1 > 0.60
                - F1-score de la clase 1 > 0.30
        - **Conclusión:** El modelo que mejor cumple con estos criterios y ha sido más consistente en las distintas pruebas realizadas es el modelo "XGBoost with Feature Importance and with Balance".
        
        |                | precision | recall | f1-score | support |
        |----------------|-----------|--------|----------|---------|
        | 0              | 0.88      | 0.53   | 0.66     | 18294   |
        | 1              | 0.25      | 0.70   | 0.37     | 4214    |
        | accuracy       |           | 0.56   |          | 22508   |
        | macro avg      | 0.57      | 0.61   | 0.52     | 22508   |
        | weighted avg   | 0.77      | 0.56   | 0.61     | 22508   |

        Con el objetivo de seguir potenciando este modelo, se aplica una validación cruzada para mejorar sus entrenamiento:

        **Validacion cruzada: XGBoost with Feature Importance but with Balance**

        |            | precision | recall | f1-score | support |
        |------------|-----------|--------|----------|---------|
        | 0          | 0.88      | 0.52   | 0.66     | 37298   |
        | 1          | 0.24      | 0.67   | 0.36     | 8400    |
        | accuracy   |           | 0.55   |          | 45698   |
        | macro avg  | 0.56      | 0.60   | 0.51     | 45698   |
        | weighted avg | 0.76    | 0.55   | 0.60     | 45698   |

        Por sopresa, el modelo no tuvo ninguna mejora en la predicción de datos, las posibles causas es que el modelo ya esté bien ajustado y que tiene los hiperparámetros adecuados.


# Table of Metrics:
| Model                                            | ROC AUC   | MCC       | Accuracy  | Precision | Recall    | F1-Score  |
|--------------------------------------------------|-----------|-----------|-----------|-----------|-----------|-----------|
| XGBoost                                          | 0.510023  | 0.103196  | 0.815266  | 0.715385  | 0.022069  | 0.042818  |
| LR                                               | 0.513934  | 0.099858  | 0.813755  | 0.540741  | 0.034646  | 0.065120  |
| RF                                               | **0.519775**  | **0.134092**  | **0.816421**  | **0.635762**  | **0.045562**  | **0.085031**  |
| XGBoost with Feature Import. and with Balance    | **0.605388**  | **0.164540**  | 0.553448  | 0.249248  | **0.688420**  | **0.365988**  |
| XGBoost with Feature Import. and without Balance | 0.503232  | 0.058420  | 0.813577  | **0.714286**  | 0.007119  | 0.014098  |
| LR with Feature Import. and with Balance         | **0.603384**  | **0.161447**  | 0.550338  | 0.247715  | **0.688182**  | **0.364299**  |
| LR with Feature Importance and without Balance  | 0.505095  | 0.059186  | 0.813044  | 0.529412  | 0.012814  | 0.025023  |
| RF with Feature Import. and with Balance        | **0.606256**  | **0.165823**  | 0.559312  | 0.250808  | **0.681300**  | **0.366643**  |
| RF with Feature Import. and without Balance     | 0.503232  | 0.058420  | 0.813577  | **0.714286**  | 0.007119  | 0.014098  |

- Según la tabla de métricas:
    - Variabilidad en el Desempeño de Clases: Es notorio cómo el balanceo de clases impacta significativamente en las métricas de Recall y F1-Score, especialmente para la clase "1" (atrasos). Los modelos con balanceo de clases muestran un alto Recall y F1-Score para la clase "1", lo cual es crucial para nuestro objetivo de identificar vuelos atrasados de manera efectiva. Sin embargo, esto se hace a expensas de la Precisión, lo cual indica que el modelo podría estar incurriendo en más Falsos Positivos.
    - Compromiso entre Métricas: Los modelos con balance de clases tienen un rendimiento notablemente mejor en términos de ROC AUC y MCC, que son métricas más robustas para evaluar el rendimiento general del modelo. Esto sugiere que, aunque hay un compromiso entre las diferentes métricas, los modelos con balanceo de clases podrían ser más efectivos para nuestro caso de uso específico, que requiere una alta tasa de detección de la clase minoritaria (atrasos).

## Conclusiones generales

- Al observar los resultados de los 9 modelos entrenados, se puede concluir que:
    - No hay diferencias notables en los resultados entre XGBoost, LogisticRegression y Random Forest.
    - No disminuye el rendimiento del modelo reduciendo las características a las 10 más importantes.
    - Mejora el rendimiento de los modelos al equilibrar clases.
    - No disminuye le rendimiento de los modelos aplicando Normalización de características.
    - El modelo XGBoost no mejoró su rendimiento aplicando un Validación cruzada
    - Tanto "LR with Feature Import. and with Balance", como "RF with Feature Import. and with Balance", tienen una alta tasa en la curva ROC y AUC (0.6 ambos casos), esto quiere decir que estos modelos también son capaces de distinguir de buena manera clases positivas y negativas

**Con esto, el modelo para que sea productivo debe ser aquel que esté entrenado con las 10 características principales y balanceo de clases, pero ¿cuál?** 
- Por lo tanto, el modelo productivo debería ser uno que incorpore las 10 características más importantes, aplique normalización y balanceo de clases. Tanto XGBoost como Random Forest cumplen con estos criterios.

