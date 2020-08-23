import json
import pandas as pd
from pathlib import Path
import os
import pickle
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from text_preprocessing import SpacyTokenizer, TextPreprocessor

SEED = 92
#####################################
# 0. Importing and cleaning the data
#####################################

PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[0]
# import text
text_path = str(PARENT_DIR.joinpath("data/raw/text.json"))
with open(text_path) as json_file:
    raw_data = json.load(json_file)

df_text = pd.DataFrame(raw_data)
# make sure there are no missing values
df_text.dropna(axis=0, how='any', inplace=True)

# import metadata for TRAINING DATA
meta_path = str(PARENT_DIR.joinpath("data/processed/metadata_train.csv"))
df_meta = pd.read_csv(meta_path, index_col=0)
# We will focus exclusively on cases of gender violence
df_meta = df_meta[df_meta['VIOLENCIA_DE_GENERO'] == 1]
# keep only what we need
df_meta = df_meta[["file_id", "V_FISICA", "V_PSIC", "V_ECON",
                   "V_SEX", "V_SOC", "V_AMB", "V_SIMB"]]
# make sure there are no missing values
df_meta.dropna(axis=0, how='any', inplace=True)
# merge both datasets
df = pd.merge(df_meta, df_text,
              on='file_id', how='inner',
              right_index=True)
print(df.shape)

# Imputting missing values
target_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
               "V_SOC", "V_AMB", "V_SIMB"]

median_imputer = SimpleImputer(missing_values=np.nan, strategy='median')
df[target_vars] = median_imputer.fit_transform(df[target_vars])
# save imputer
imp_path = str(PARENT_DIR.joinpath("models/violence_type_imp.sav"))
with open(imp_path, 'wb') as f:
    pickle.dump(median_imputer, f)

#####################################
# 1. Model
#####################################

spacy_tokenizer = SpacyTokenizer()
text_transformer = Pipeline(steps=[
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(max_df=0.85, min_df=0.1,
                                   tokenizer=spacy_tokenizer))])

violence_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
                 "V_SOC", "V_AMB", "V_SIMB"]

X_train, X_test, y_train, y_test = train_test_split(df.text,
                                                    df[violence_vars].astype('int'),
                                                    test_size=0.1,
                                                    random_state=SEED,
                                                    shuffle=True)

# vectorize text
X_train = text_transformer.fit_transform(X_train)
X_test = text_transformer.transform(X_test)
# save vectorizer
vect_path = str(PARENT_DIR.joinpath("models/violence_type_vectorizer.sav"))
with open(vect_path, 'wb') as f:
    pickle.dump(text_transformer, f)


def train_models(model, parameters, violence_vars,
                 X_train, X_test, y_train, y_test):

    """ Function to train a model using GridSearchCV and report on several
    performance metrics

    Args:
        model (class): a model from sklearn
        parameters (dict): set of parameters to explore in GridsearchCV
        violence_vars (list): set of target variables
        X_train, X_test, y_train, y_test: arrays of data
    """

    models = {}
    for var in violence_vars:
        best_model = GridSearchCV(model,
                                  parameters,
                                  cv=5,
                                  scoring="accuracy",
                                  verbose=0)

        best_model.fit(X_train, y_train[var])
        y_train_hat = best_model.predict(X_train)
        y_test_hat = best_model.predict(X_test)

        models[var] = {}
        models[var]["fitted_model"] = best_model

        print("\n**** Results Report for %s  ****\n" % var)
        print(" Training Accuracy: %s" % accuracy_score(y_train[var],
                                                        y_train_hat))
        print(" Test Accuracy: %s \n" % accuracy_score(y_test[var], y_test_hat))
        print(" Training F1: %s" % f1_score(y_train[var], y_train_hat))
        print(" Test F1: %s \n" % f1_score(y_test[var], y_test_hat))
        print(" Training Precision: %s" % precision_score(y_train[var],
                                                          y_train_hat))
        print(
            " Test Precision: %s \n" % precision_score(y_test[var], y_test_hat))
        print(" Training Recall: %s" % recall_score(y_train[var], y_train_hat))
        print(" Test Recall: %s \n" % recall_score(y_test[var], y_test_hat))

    return models


# define candidate models
LR = LogisticRegression(solver='liblinear', max_iter=1000)
KN = KNeighborsClassifier()
NB = MultinomialNB()
RF = RandomForestClassifier()

# define parameters for CV
params_RF = {'n_estimators': (10, 50, 100, 200),
             'max_depth': (None, 10, 20),
             'max_features': ('auto', 'log2')
             }

params_KN = {'n_neighbors': (3, 5, 10),
             'weights': ('distance', 'uniform')
             }

params_LR = {'penalty': ('l1', 'l2'),
             'class_weight': ('balanced', None)}

params_NB = {'alpha': (0.5, 1)}

# combine everything
models = {"LR": {"model": LR, "parameters": params_LR},
          "NB": {"model": NB, "parameters": params_NB},
          "KN": {"model": KN, "parameters": params_KN},
          "RF": {"model": RF, "parameters": params_RF}
          }

#####################################
# 2. Optimize model
#####################################

for name, model in models.items():
    print("\n===== Results Report for %s model ======\n" % str(model["model"]))

    gridcv_models = train_models(model["model"],
                                 model["parameters"],
                                 violence_vars,
                                 X_train, X_test,
                                 y_train, y_test)

    # save the gridcv
    models[name]["gridcv"] = gridcv_models

#####################################
# 3. Selecting the best model
#####################################

model_path = str(PARENT_DIR.joinpath("models/violence_type_model.sav"))
with open(model_path, 'wb') as f:
    pickle.dump(models["RF"]["gridcv"], f)

print(models["RF"]["gridcv"])
