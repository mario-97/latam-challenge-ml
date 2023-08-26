# Resumen y entendimiento de los datos

resumir aqui

# A considerar:
A partir de este punto, toda mención, ajustes, correción, y/o cualquier implementación es en el archivo model.py. Por lo tanto cuando me refiera a errores en codigo es en alusión a ese archivo.
Si por algun motivo trabajo en otro archivo será explicitamente mencionado.

# Correción de errores preliminar (antes de mi desarrollo) 

1.- Error en clase Delay: Es un problema en la definición de tipos en la firma de el método 'preprocess'. Y este se encuentra en la forma en que se utiliza 'Union' y 'Tuple' en la anotación de tipos de retorno.

Para corregir esto, se debe ajustar la definición de tipos en el método 'preprocess', de manera que en la linea 16: "-> Union(Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame):", quede de la siguiente manera: "-> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:"

2.- Error en la función "Barplot": En este caso, se esta entregando de manera incorrecta los parametros a la funcion, si bien, se sabe que el primer elemento corresponde a los datos del eje X, y el segundo al eje Y, estos deben quedar declarados explicitamente con sus nombres de la siguiente manera: Ejemplo, "sns.barplot(x=flights_by_airline.index, y=flights_by_airline.values, alpha=0.9)"

3.- Actualización y librerías faltantes en requeriments.txt

4.- Corrección en test_model.py, ajuste de columna inexistente, ajusta de ruta del archivo data.csv

# Pasos de transcripción del archivo .ipynb a .py

1.- Ordenar imports al comienzo del código
2.- Ordenar funciones al comienzo del código