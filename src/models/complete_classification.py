import json
import pandas as pd
from pathlib import Path
import os
import pickle
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, \
    recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

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

features_path = str(PARENT_DIR.joinpath("data/processed/features_train.csv"))
df_features = pd.read_csv(features_path, index_col=0)
df_features.dropna(axis=0, how='any', inplace=True)

# join numerical features and text
df = pd.merge(df_features, df_text,
              on='file_id', how='inner', right_index=True)

# get target variable information
target_path = str(PARENT_DIR.joinpath("data/processed/target_vars_train.csv"))
df_target = pd.read_csv(target_path, index_col=0)
# We will focus exclusively on cases of gender violence
df_target = df_target[df_target['VIOLENCIA_DE_GENERO'] == 1]
df_target.replace('no_corresponde', np.nan, inplace=True)

# Imputting missing values
target_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
               "V_SOC", "V_AMB", "V_SIMB"]

# load imputer previosly used
imp_path = str(PARENT_DIR.joinpath("models/violence_type_imp.sav"))
with open(imp_path, 'rb') as f:
    median_imputer = pickle.load(f)

df_target[target_vars] = median_imputer.transform(df_target[target_vars])
df_target.drop(columns=['file_id'], inplace=True)

# join everything
df = df.join(df_target)
print(df.shape)

####################################################
# 1. Predict presence of gender violence from text
####################################################

print("\n==== Step 1: Predicting the presence of gender violence using text "
      "===\n")
# load vectorizer
vect_path = str(PARENT_DIR.joinpath("models/gender_violence_vectorizer.sav"))
with open(vect_path, 'rb') as f:
    vectorizer_gender_violence = pickle.load(f)
    print(vectorizer_gender_violence)

# vectorize text
text_features = vectorizer_gender_violence.transform(df.text)

# load best model
model_path = str(PARENT_DIR.joinpath("models/gender_violence_model.sav"))
with open(model_path, 'rb') as f:
    gender_violence_model = pickle.load(f)
    print(gender_violence_model)

# get predictions
gender_violence_proba = gender_violence_model.predict_proba(text_features)
gender_violence_proba = gender_violence_proba[:, 1]

#####################################################
# 2. Predict the types of violence from text
#####################################################

print("\n==== Step 2: Predicting the type of gender violence using text "
      "===\n")
# load vectorizer
vect_path = str(PARENT_DIR.joinpath("models/violence_type_vectorizer.sav"))
with open(vect_path, 'rb') as f:
    vectorizer_violence_type = pickle.load(f)
    print(vectorizer_violence_type)

# vectorize text
text_features = vectorizer_violence_type.transform(df.text)

# load best model
model_path = str(PARENT_DIR.joinpath("models/violence_type_model.sav"))
with open(model_path, 'rb') as f:
    violence_type_model = pickle.load(f)
    print(violence_type_model)

target_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
               "V_SOC", "V_AMB", "V_SIMB"]

# get predictions for each type of violence
predicted_probas = {}
for var in target_vars:
    model = violence_type_model[var]['fitted_model']
    col_name = str(var + "_proba")
    predicted_probas[col_name] = model.predict_proba(text_features)[:, 1]

# consolidate results
df_probas = pd.DataFrame(
    dict(file_id=df.file_id, gender_violence_proba=gender_violence_proba))
df_probas.reset_index(drop=True, inplace=True)
df_probas = pd.concat([df_probas, pd.DataFrame(predicted_probas)], axis=1)
df_probas.set_index(df.index, inplace=True)
df_probas.drop(columns=['file_id'], inplace=True)
df = df.join(df_probas)

#####################################################
# 3. Final model
#####################################################
print("\n==== Step 3: Build a complete model"
      "===\n")
violence_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
                 "V_SOC", "V_AMB", "V_SIMB"]
y = df[violence_vars].astype('int')
drop_cols = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX", "file_id", "text",
             "V_SOC", "V_AMB", "V_SIMB", "VIOLENCIA_DE_GENERO"]
X = df.drop(labels=drop_cols, axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=SEED,
                                                    shuffle=True)

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

models = {"LR": {"model": LR, "parameters": params_LR},
          "NB": {"model": NB, "parameters": params_NB},
          "KN": {"model": KN, "parameters": params_KN},
          "RF": {"model": RF, "parameters": params_RF}
          }


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
# 4. Selecting the best model
#####################################

model_path = str(PARENT_DIR.joinpath("models/complete_model.sav"))
with open(model_path, 'wb') as f:
    pickle.dump(models['RF'], f)

print(models['RF']['model'])
