gender-violence
==============================

Este proyecto busca entender de qué forma se puede utilizar el texto de las 
denuncias que se reciben en un juzgado para identificar la ocurrencia de
 diversos tipos de violencia de género. [Esta presentación](reports/presentacion.pdf) 
 explica la motivación del proyecto, la aproximación metodológica y los
  resultados principales. Los notebooks y scripts principales son:
  
  1. Predicción presencia de violencia de género: [Notebook](notebooks/2.1-YM-gender-violence.ipynb) y [script](src/models/gender_violence_classification.py)
  2. Preddicción tipo de violencia: [Notebook](notebooks/2.2-YM-types-of-violence.ipynb) y [script](src/models/violence_type_classification.py)
  3. Modelo completo:  [Notebook](notebooks/3.1-YM-final-model.ipynb) y [script](src/models/complete_classification.py)
 
 El repositorio contiene todo el código necesario para replicar los
  resultados que se presentan. Estando en el directorio del proyecto, se debe
   correr el siguiente comando en la terminal para generar los datos:
 
 ```
make data
```
Este proceso tarda un tiempo largo debido a que es necesario extraer el texto de los archivos
. Una vez generados los datos, es posible entrenar los modelos. Esto se puede
 hacer simplemente con el comando en la terminal:
 
  ```
make train
```
El repositorio ya contiene los modelos, vectorizadores e imputadores
 utilizados. Al correr este comando se volverán a generar estos archivos
 . Es posible, además, hacer una prueba final a estos modelos estimados. Esto
  se logra con:
  
   ```
make test
```

Finalmente, es posible generar un archivo **.csv** con las probabilidades
 estimadas para cada una de las denuncias en la base de datos de Google Drive
 . Para hacer esto, solo es necesario correr:
 
  ```
make predictions
```
Este comando se puede correr sin necesidad de haber re entrenado el modelo. Solo es necesario haber producido los datos (**make data**).


El proyecto está organizado con el formato que propone [Cookie Cutter Data
 Science](https://drivendata.github.io/cookiecutter-data-science/).

Organización del proyecto 
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    │
    ├── models             <- Models, vectorizers, imputers
    │
    ├── notebooks          <- Jupyter notebooks exploring the project
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Main presentation with results and methodology
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── metadata_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
