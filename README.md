# entrevista-ferraresso
Modelo para clasificar comentarios de peliculas

## Archivos
- EDA.ipynb: Breve analisis del dataset
- training_model.ipynb: Preprocesamiento y entrenamiento de un modelo de clasificacion.
- nlp_transformer.py: Clases para el procesamiento de strings.
- demo.py: Aplicacion con Streamlit para probar un modelo entrenado.

## Aplicacion de prueba
Se puede probar el modelo levantando la imagen docker https://hub.docker.com/repository/docker/ferraresso/demo-review

Se debe ejecutar con el siguiente comando:
docker run -p 8501:8501 ferraresso/demo-review:latest

Y luego ingresar en el navegador a http://localhost:8501
