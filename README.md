# Proyecto-Pasantia
Proyecto de ML para detectar y prevenir fraude transaccional.

# README #

En este proyecto se encuentra el codigo completo utilizado en la aplicacion de aprendizaje automatico en transacciones.

### Contiene: ###

* Librerias
* Preprocesamiento
* No supervisado (K-Means)
* Entrenamiento Supervisado (Random Forest)
* Validacion
* Prediccion
* Ejecucion (Este script es el encargado de entrenar el Modelo)
* Exec_predic (Se usa para tomar una data nueva, preprocesarla y hacer las predicciones con el modelo ya entrenado)

### ¿Como se utiliza? ###

* Se debe tener un archivo llamado "dataset.csv" que es la data de entrenamiento original.

* Además debe estar el archivo "NumeroContrato_reportados.csv" con todos los Números de contratos que fueron reportados, con este archivo se asignarán las etiquetas.

* Run Ejecucion.py para el preprocesamiento de la data original y el entrenamiento del modelo pasando por las 2 etapas, ademas ejecuta test de validacion.

* Con Exec_predic.py podremos predecir un nuevo data set realizando el preprocesamiento, añadiendo la columna de Kmeans y haciendo la prediccion con el modelo que sale de Ejecucion.py

### Consideraciones ###

* Entrar a Downloads y descargar acrchivo "all_ordinal_encoders.joblib", este se debe dejar junto a los demás scripts.

### Autor ###

* Alberto Vergara
