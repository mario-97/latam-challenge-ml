# Resumen y entendimiento de los datos

resumir aqui

# A considerar:
A partir de este punto, toda mención, ajustes, correción, y/o cualquier implementación es en el archivo model.py. Por lo tanto cuando me refiera a errores en codigo es en alusión a ese archivo.
Si por algun motivo trabajo en otro archivo será explicitamente mencionado.

# Correción de errores preliminar (antes de mi desarrollo) 

1.- Error en clase Delay: Completar funciones para realizar los test y verificar datos

2.- Error en la función "Barplot": En este caso, se esta entregando de manera incorrecta los parametros a la funcion, si bien, se sabe que el primer elemento corresponde a los datos del eje X, y el segundo al eje Y, estos deben quedar declarados explicitamente con sus nombres de la siguiente manera: Ejemplo, "sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)"

3.- Actualización y librerías faltantes en requeriments.txt

4.- Corrección en test_model.py: Carga de archivo csv, y ajuste de Entrenamiento y predicciones 

# Pasos de transcripción del archivo .ipynb a .py., y desarrollo propio de preprocesamiento y modelo RF

1. Ordenar imports al comienzo del código
2. Ordenar funciones al comienzo del código
3. Definir metodología KDD:
    a. Seleccion de datos
    b. Preprocesamiento
    c. Trnasformación
    d. Aplicación de modelos
    e. Evalución, análisis e interpretación

4. Selección de modelo:
    a. Preprocesamiento, razón e importancia de agregar una normalizacion a los datos: 
    a. Se agrega el modelo de Random Forest para aplicar un beachmarking respecto a los otros dos modelos propuesto
    b. Seleccionamos el mejor para aplicar las pruebas, en este caso fue:
    c. Razón de proponer Random Forest: 
    d. Explicación de métricas:
    e. Resultados y conclusiones