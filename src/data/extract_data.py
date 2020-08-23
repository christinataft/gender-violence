import pandas as pd
import numpy as np
import gspread
import os
from pathlib import Path
from sklearn.model_selection import train_test_split


def read_worksheet(client_secret, url, worksheet_name):
    """ Function to read a worksheet from a Google Drive spreadsheet
    Args:
        client_secret (str): path of json file containing Google Drive'
        service account secrets
        url (str): URL to Google Drive
        worksheet_name (str): full name of worksheet form the spreadsheet
    Returns:
        List of lists containing the data in the selected worksheet
    """

    gc = gspread.service_account(filename=client_secret)
    sh = gc.open_by_url(url)
    worksheet = sh.worksheet(worksheet_name)
    return worksheet.get_all_values()


def get_spreadsheet():
    """ Function to download spreadsheet and save 3 versions of it:
        - Complete
        - Train
        - Test
    """
    PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[2]
    SECRETS_PATH = str(PARENT_DIR.joinpath("client_secret.json"))
    SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1uAi-Yfq" \
                      "-rJl_cqQaVe9Fv1rLlBJcmEtDpUB0NTOrLAs/edit#gid=625331269"
    WORKSHEET_NAME = "set_de_datos_unificado"

    raw_data = read_worksheet(SECRETS_PATH,
                              SPREADSHEET_URL,
                              WORKSHEET_NAME)

    df = pd.DataFrame(raw_data[1:], columns=raw_data[0])
    df.replace('', np.nan, inplace=True)
    # remove empty rows
    df.dropna(axis=0, how="all", subset=["N", "NRO_REGISTRO"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # we will save a portion of the data so that we never use it for training
    df_train, df_test = train_test_split(df, test_size=0.15)

    df.to_csv(str(PARENT_DIR.joinpath("data/raw/spreadsheet_complete.csv")),
              index=True)

    df_train.to_csv(str(PARENT_DIR.joinpath("data/raw/spreadsheet_train.csv")),
                    index=True)

    df_test.to_csv(str(PARENT_DIR.joinpath("data/raw/spreadsheet_test.csv")),
                   index=True)

    print("Data successfully downloaded from Google Drive and saved locally")

    return None
