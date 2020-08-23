import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import os
import pickle
from pathlib import Path

PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[0]
df = pd.read_csv(str(PARENT_DIR.joinpath(
    "data/processed/metadata_train.csv")), index_col=0)
# We will focus exclusively on cases of gender violence
df = df[df['VIOLENCIA_DE_GENERO'] == 1]

# drop variables that we cannot see
df.drop(['ART_INFRINGIDO', 'CODIGO_O_LEY', 'CONDUCTA',
         'VIOLENCIA_DE_GENERO', 'FRASES_AGRESION'],
        axis=1, inplace=True)

# select those columns with useful (and clean) information
informative = ['file_id',
               'MATERIA',
               'GENERO_ACUSADO/A',
               'NACIONALIDAD_ACUSADO/A',
               #'EDAD_ACUSADO/A AL MOMENTO DEL HECHO',
               'NIVEL_INSTRUCCION_ACUSADO/A',
               'GENERO_DENUNCIANTE',
               'NACIONALIDAD_DENUNCIANTE',
               #'EDAD_DENUNCIANTE_AL_MOMENTO_DEL_HECHO',
               'NIVEL_INSTRUCCION_DENUNCIANTE',
               'FRECUENCIA_EPISODIOS',
               'RELACION_Y_TIPO_ENTRE_ACUSADO/A_Y_DENUNCIANTE',
               'HIJOS/AS_EN_COMUN',
               'TIPO_DE_RESOLUCION'
              ]
# subset the data
df = df[informative]
# replace with na's
df.replace('no_corresponde', np.nan, inplace=True)
#Exploring the pressence of missing values (percentage)
missing = df.isna().sum()/df.shape[0]
high_missing = df.columns[missing > 0.5]
df = df.drop(axis=1,labels = high_missing)

# make all responses lowercase
for var in list(df.columns):
    if var == "file_id":
        continue
    else:
        df[var] = df[var].str.lower()

# impute missing values
simp_imputer = SimpleImputer(strategy = 'most_frequent')
df_imp = simp_imputer.fit_transform(df)
# save imputer and the columns imputed
names_path = str(PARENT_DIR.joinpath("models/column_names.txt"))
with open(names_path, 'w') as filehandle:
    for c in list(df.columns):
        filehandle.write('%s,' % c)

imp_path = str(PARENT_DIR.joinpath("models/metadata_imp.sav"))
with open(imp_path, 'wb') as f:
    pickle.dump(simp_imputer, f)

# reconvert the imputed data into a pandas DataFrame
df_imp = pd.DataFrame(df_imp, index=df.index, columns=df.columns)
cols = list(df_imp.columns)
cols.remove('file_id')
# generate dummies from categorical columns
df_features = pd.get_dummies(df_imp, columns=cols)
# save results
df_path = str(PARENT_DIR.joinpath("data/processed/features_train.csv"))
df_features.to_csv(df_path)



names_path = str(PARENT_DIR.joinpath("models/features_names.txt"))
with open(names_path, 'w') as filehandle:
    for c in list(df_features.columns):
        filehandle.write('%s,' % c)


print("Generated features from metadata")