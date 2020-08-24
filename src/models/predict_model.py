import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import pickle


def preprocess(parent_dir):
    """ Function to preprocess the raw spreadsheet
    """
    file_path = str(parent_dir.joinpath("data/raw/spreadsheet_complete.csv"))
    df = pd.read_csv(file_path, index_col=0)
    df['file_id'] = [get_id(link) for link in df['LINK']]
    df.replace('s/d', np.nan, inplace=True)
    df.replace('si', 1, inplace=True)
    df.replace('SI', 1, inplace=True)
    df.replace('no', 0, inplace=True)
    df.replace('NO', 0, inplace=True)

    # save results
    df.to_csv(str(parent_dir.joinpath("data/processed/metadata_complete.csv")))

    return df


def get_id(link):
    """ Function to extract the file ID from its URL
    Args:
        link (str): Google Drive URL for the file

    Returns:
        String with the file's ID
    """

    file_id = np.nan
    try:
        if "/file/" in link:
            file_id = link.split("/file/d/")[1]
        elif "id=" in link:
            file_id = link.split("id=")[1]
        elif "/document/" in link:
            file_id = link.split("/document/d/")[1]

    except TypeError:
        return np.nan

    # remove additional characters from the link
    if "/" in str(file_id):
        file_id = file_id.split("/")[0]

    return file_id

#####################################
# 0. Importing and cleaning the data
#####################################

PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[0]

# get text data
text_path = str(PARENT_DIR.joinpath("data/raw/text.json"))
with open(text_path) as json_file:
    raw_data = json.load(json_file)

df_text = pd.DataFrame(raw_data)
# make sure there are no missing values
df_text.dropna(axis=0, how='any', inplace=True)

# preprocess raw spreadsheet data
df_meta = preprocess(PARENT_DIR)

# join data together
df = pd.merge(df_meta, df_text,
              on='file_id', how='inner',
              right_index=True)
print(df.shape)

####################################################
# 1. Predict presence of gender violence from text
####################################################

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

# get predictions
gender_violence_proba = gender_violence_model.predict_proba(text_features)
gender_violence_proba = gender_violence_proba[:, 1]

####################################################
# 2. Predict type of violence from text
####################################################

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

target_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
               "V_SOC", "V_AMB", "V_SIMB"]

# get predictions for each type of violence
predicted_probas = {}
for var in target_vars:
    model = violence_type_model[var]['fitted_model']
    col_name = str(var + "_proba")
    predicted_probas[col_name] = model.predict_proba(text_features)[:, 1]

# consolidate results
df_probas = pd.DataFrame({'file_id': df.file_id,
                          'gender_violence_proba': gender_violence_proba})
df_probas.reset_index(drop=True, inplace=True)
df_probas = pd.concat([df_probas, pd.DataFrame(predicted_probas)], axis=1)
df_probas.set_index(df.index, inplace=True)

####################################################
# 3. Generate features
####################################################

# get the names of features use for training
file_path = str(PARENT_DIR.joinpath("models/column_names.txt"))
file = open(file_path, "r+")
col_names = file.read().split(",")
col_names.remove("")
# subset the data
df = df[col_names]
original_index = df.index

# make all responses lowercase
for var in list(col_names):
    if var == "file_id":
        continue
    else:
        df[var] = df[var].str.lower()

# load imputer previosly used
imp_path = str(PARENT_DIR.joinpath("models/metadata_imp.sav"))
with open(imp_path, 'rb') as f:
    meta_imputer = pickle.load(f)

df_imp = meta_imputer.transform(df)
# reconvert the imputed data into a pandas DataFrame
df_imp = pd.DataFrame(df_imp, index=original_index, columns=df.columns)

cols = list(df_imp.columns)
cols.remove('file_id')
# generate dummies from categorical columns
df_features = pd.get_dummies(df_imp, columns=cols)

# keep only those that were used for training
file_path = str(PARENT_DIR.joinpath("models/features_names.txt"))
file = open(file_path, "r+")
features_names = file.read().split(",")
features_names.remove("")

features_filter = []
for feature in features_names:
    if feature in list(df_features.columns):
        features_filter.append(feature)
df_features = df_features[features_filter]

# However, now there might be columns that were in the
# training dataset but not in the test new
# We are going to add them with zero values for all rows

not_in_test = [f for f in features_names if f not in list(df_features.columns)]

for f in not_in_test:
    df_features = df_features.join(
        pd.DataFrame({f: [0] * df_features.shape[0]}))

# lastly, we need to give the columns the same order as they had in the
# training data
df_features = df_features[features_names]
df_features.drop(columns=['file_id'], inplace=True)
df_features.replace(np.nan, 0, inplace=True)
df_final = df_features.join(df_probas)
print(len(df_final))

####################################################
# 3. Generate predictions
####################################################

X = df_final.drop(columns=["file_id"])

# load best model
model_path = str(PARENT_DIR.joinpath("models/complete_model.sav"))
with open(model_path, 'rb') as f:
    complete_model = pickle.load(f)

target_vars = ["V_FISICA", "V_PSIC", "V_ECON", "V_SEX",
               "V_SOC", "V_AMB", "V_SIMB"]

# get predictions for each type of violence
predictions = {}
for var in target_vars:
    model = complete_model['gridcv'][var]['fitted_model']
    col_name = str(var + "_pred")
    predictions[col_name] = model.predict_proba(X)

predictions = pd.DataFrame(predictions, index=X.index)
predictions.to_csv(str(PARENT_DIR.joinpath("data/predictions.csv")),
               index=True)

print("Predictions generated!")




