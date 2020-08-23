import numpy as np
import pandas as pd
from pathlib import Path
import os


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


def preprocess():
    """ Function to preprocess the raw spreadsheet
    """
    PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[2]
    paths = ['train', 'test']
    for p in paths:
        file_path = str(PARENT_DIR.joinpath("data/raw/spreadsheet_%s.csv" % p))
        df = pd.read_csv(file_path, index_col=0)
        df['file_id'] = [get_id(link) for link in df['LINK']]
        df.replace('s/d', np.nan, inplace=True)
        df.replace('si', 1, inplace=True)
        df.replace('SI', 1, inplace=True)
        df.replace('no', 0, inplace=True)
        df.replace('NO', 0, inplace=True)

        # save results
        df.to_csv(str(PARENT_DIR.joinpath("data/processed/metadata_%s.csv" %
                                          p)))

        # save a dataframe only for target variables
        df_target = df[["file_id", "VIOLENCIA_DE_GENERO", "V_FISICA", "V_PSIC",
                        "V_ECON", "V_SEX", "V_SOC", "V_AMB", "V_SIMB"]]
        df_target.to_csv(str(PARENT_DIR.joinpath(
            "data/processed/target_vars_%s.csv" % p)))

    print("Raw data preprocessed successfully")

    return None
