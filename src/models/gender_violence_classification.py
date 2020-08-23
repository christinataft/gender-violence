import json
import pandas as pd
from pathlib import Path
import os
import pickle

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
# 0. Importing the data
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
# keep only what we need
df_meta = df_meta[["file_id", "VIOLENCIA_DE_GENERO"]]
# make sure there are no missing values
df_meta.dropna(axis=0, how='any', inplace=True)
# merge both datasets
df = pd.merge(df_meta, df_text,
              on='file_id', how='inner',
              right_index=True)
print(df.shape)

#####################################
# 1. Model
#####################################

y = df["VIOLENCIA_DE_GENERO"]
X_train, X_test, y_train, y_test = train_test_split(df.text,
                                                    y.astype(int),
                                                    test_size=0.1,
                                                    random_state=SEED,
                                                    shuffle=True)

# define a pipeline for text preprocessing and vectorizing
spacy_tokenizer = SpacyTokenizer()
text_transformer = Pipeline(steps=[
    ('preprocessor', TextPreprocessor()),
    ('vectorizer', TfidfVectorizer(max_df=0.85, min_df=0.1,
                                   tokenizer=spacy_tokenizer))])

# vectorize text
print(text_transformer)
X_train = text_transformer.fit_transform(X_train)
X_test = text_transformer.transform(X_test)
print("Text vectorized")
# save vectorizer
vect_path = str(PARENT_DIR.joinpath("models/gender_violence_vectorizer.sav"))
with open(vect_path, 'wb') as f:
    pickle.dump(text_transformer, f)

# define candidate models and pipe
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

for name, model in models.items():

    best_model = GridSearchCV(model["model"],
                              model["parameters"],
                              cv=5,
                              scoring="accuracy",
                              verbose=0)

    best_model.fit(X_train, y_train)
    # save the best model
    models[name]["best_model"] = best_model
    y_train_hat = best_model.predict(X_train)
    y_test_hat = best_model.predict(X_test)

    print("\n**** Results Report for %s model ****\n" % str(model["model"]))
    print(" Training Accuracy: %s" % accuracy_score(y_train, y_train_hat))
    print(" Test Accuracy: %s \n" % accuracy_score(y_test, y_test_hat))
    print(" Training F1: %s" % f1_score(y_train, y_train_hat))
    print(" Test F1: %s \n" % f1_score(y_test, y_test_hat))
    print(" Training Precision: %s" % precision_score(y_train, y_train_hat))
    print(" Test Precision: %s \n" % precision_score(y_test, y_test_hat))
    print(" Training Recall: %s" % recall_score(y_train, y_train_hat))
    print(" Test Recall: %s \n" % recall_score(y_test, y_test_hat))

#####################################
# 2. Selecting the best model
#####################################

model_path = str(PARENT_DIR.joinpath("models/gender_violence_model.sav"))
with open(model_path, 'wb') as f:
    pickle.dump(models["RF"]["best_model"], f)

print(models["RF"]["best_model"])
