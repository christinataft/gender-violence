gender-violence
==============================

## Project Description 

This project aims to understand how the text from judicial complainst can be used to predict the presence of different types of gender violence in the complaint [This presentation (in Spanish)](reports/presentacion.pdf) explains the motivation, methodological approach and main results of the project. The [src](src) folder contains all the scripts for gathering the data, extracting the text, processing the text, crafting features and building models. The notebooks and scripts containing the most relevant models are:
  
  1. Prediction of gender violence: [Notebook](notebooks/2.1-YM-gender-violence.ipynb) y [script](src/models/gender_violence_classification.py)
  2. Preddiction of the different types of gender violence: [Notebook](notebooks/2.2-YM-types-of-violence.ipynb) y [script](src/models/violence_type_classification.py)
  3. Complete model:  [Notebook](notebooks/3.1-YM-final-model.ipynb) y [script](src/models/complete_classification.py)
 
 The repository contains all the code necessary to replicate the models and results. **However, in order to actually run all the code it is necessary for the user to have a Google Service Account with Google's Drive API activated. The key to this account must be stored in the project's directory as "client_secret.json" .**
 
 ## Gather the data

 While on the project's directory, you should run the following command on the terminal in order to extract the data from Google Drive:
 
 ```
make data
```

This process will take some time because it extracts the text from all the PDFs stored in the database.

## Train the model

Once the data has been generated, it is possible to train several models to predict the presence and the type of gender violence. This is done by running the following command on the terminal:

  ```
make train
```

The repository already contains the final models, vectorizers and imputers used. Running this command will replace these files.

## Test the model

After training the model, it is also possible to test the different models. This is done by running the following command on the terminal:
  
   ```
make test
```

## Store predictions

Lastly, it is also possible to generate a **.csv** file with all the estimated probabilities for each judicial complaint in the database. This is done by running:
 
  ```
make predictions
```

This command can by run without training the model (because the final models are all saved in the repository). The only requirement is to run the  ***make data*** command.

## Project structure
------------
This project is organized based on the [Cookie Cutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template


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
  

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
