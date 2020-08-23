import os
import io
import docx
import pandas as pd
import numpy as np
from pathlib import Path

from googleapiclient.errors import HttpError
from google.oauth2 import service_account
from apiclient import http
from googleapiclient.discovery import build

from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter

from odf import teletype
from odf import text as odf_text
from odf.opendocument import load


###########################
# Functions definitions
###########################


def write_bytesio_to_file(file_name, bytesio):
    """ Function to write the contents of the given BytesIO to a file.
    """
    with open(file_name, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())


def download_file(service, file_id, destination):
    """ Function to download a file stored in Google Drive
    Args:
        service (Google Drive API): object to interact with the content
        file_id (str): Google Drive ID for the file to retrieve
        destination (str): local path to save the file

    Returns:
        Dictionary with the metadata from the downloaded file
    """

    info = service.files().get(fileId=file_id).execute()
    # Google Docs files
    if info["mimeType"] == "application/vnd.google-apps.document":
        request = service.files().export_media(fileId=info['id'],
                                               mimeType="application/vnd"
                                                        ".oasis.opendocument"
                                                        ".text")
    # Files stored in Google Drive
    else:
        request = service.files().get_media(fileId=info['id'])

    fh = io.BytesIO()
    downloader = http.MediaIoBaseDownload(fh, request)
    done = False
    while done is False:
        status, done = downloader.next_chunk()
        # print("Download %d%%." % int(status.progress() * 100))

    write_bytesio_to_file(destination, fh)

    return info


def get_id(link):
    """ Function to extract the Google Drive file ID from its URL
    Args:
        link (str): Google Drive URL for the file

    Returns:
        Tuple with the ID of the file and the document type
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


def get_text_pdf(file_path):
    """ Function to extract the text from a PDF file
    Args:
        file_path (str): local path to file

    Returns:
        String with all the text from the PDF
    """

    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle,
                              laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    with open(file_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)

        text = fake_file_handle.getvalue()

    return text


def get_text_word(file_path):
    """ Function to extract the text from a Word file
    Args:
        file_path (str): local path to file

    Returns:
        String with all the text from the Word file
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return ' '.join(full_text)


def get_text_odt(file_path):
    """ Function to extract the text from an ODT file
    Args:
        file_path (str): local path to file

    Returns:
        String with all the text from the ODT file
    """
    textdoc = load(file_path)
    full_text = []
    for para in textdoc.getElementsByType(odf_text.P):
        full_text.append(teletype.extractText(para))

    return ' '.join(full_text)


def download_extract(file_id, service, parent_dir):
    """ Function to download a file stored in Google Drive and extract
    its text according to its file type
    Args:
        file_id (str): Google Drive ID for the file to retrieve
        service (Google Drive API ): object to interact with the content
        parent_dir (Path): local directory of the project
    Returns:
        A tuple with a dictionary containing the metadata from the retrieved
        file and the extracted text
    """

    info = service.files().get(fileId=file_id).execute()

    if info['name'][-3:] == 'pdf':
        # print('PDF found at position %s' % i)
        file_type = 'pdf'
        destination = str(parent_dir.joinpath("data/interim/file." + file_type))
        download_file(service, file_id, destination)
        text = get_text_pdf(str(parent_dir.joinpath("data/interim/file." +
                                                    file_type)))

        # remove document
        os.remove(str(parent_dir.joinpath("data/interim/file." + file_type)))

    elif info['name'][-4:] == 'docx':
        # print('Word document found at position %s' % i)
        file_type = 'docx'
        destination = str(
            parent_dir.joinpath("data/interim/file." + file_type))
        download_file(service, file_id, destination)
        text = get_text_word(
            str(parent_dir.joinpath("data/interim/file." + file_type)))

        # remove document
        os.remove(
            str(parent_dir.joinpath("data/interim/file." + file_type)))

    else:
        # print('ODT file found at position %s' % i)
        file_type = 'odt'
        destination = str(
            parent_dir.joinpath("data/interim/file." + file_type))
        download_file(service, file_id, destination)
        text = get_text_odt(
            str(parent_dir.joinpath("data/interim/file." + file_type)))

        # remove document
        os.remove(
            str(parent_dir.joinpath("data/interim/file." + file_type)))

    return info, text


###########################
# Text extraction
###########################


def get_text():
    # parameters
    URL = "https://docs.google.com/uc?export=download"
    PARENT_DIR = Path(os.path.basename(__file__)).resolve().parents[0]
    CLIENT_SECRETS = str(PARENT_DIR.joinpath("client_secret.json"))
    SCOPES = ['https://www.googleapis.com/auth/drive']
    CREDENTIALS = service_account.Credentials.from_service_account_file(
        CLIENT_SECRETS, scopes=SCOPES)
    SERVICE = build('drive', 'v3', credentials=CREDENTIALS)

    # get raw data from spreadsheet
    df = pd.read_csv(str(PARENT_DIR.joinpath(
        "data/raw/spreadsheet_complete.csv")))
    ids = [get_id(link) for link in df['LINK'] if link != np.nan]
    ids = pd.DataFrame(ids, columns=['id'])
    ids.dropna(inplace=True)
    unique_ids = set(ids['id'])

    # objects to save results
    text_id = {}
    failed = 0
    print("Starting to process %s files" % len(unique_ids))
    for i, file_id in enumerate(unique_ids):

        if (i % 50) == 0:
            print("Processing document %s" % str(i))

        try:
            info, text = download_extract(file_id, SERVICE, PARENT_DIR)
            text_id[file_id] = text
        except HttpError as e:
            print("\n-------------------- HTTP ERROR -----------")
            print(file_id)
            print(e)
            print("\n------------------------------------------")
            text_id[file_id] = np.nan
            failed += 1
            continue

    print("Extraction failed for %s documents" % failed)

    # save results
    df_final = pd.DataFrame(text_id.items(), columns=['file_id', 'text'])
    df_final.to_json(str(PARENT_DIR.joinpath("data/raw/text.json")),
                     orient="records")
    print("Text successfully extracted from files and saved locally")

    return None

