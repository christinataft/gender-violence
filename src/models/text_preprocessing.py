import re
import string
from sklearn.base import TransformerMixin
import spacy
import unidecode

def remove_symbols(text, symbols):
    """ Function that uses regular expressions to delete all symbols
        given by the user from the text

    Args:
        text (str): raw text
        symbols (list): list containing all the symbols to replace

    Returns:
        clean text (str)
    """

    # generate regular expression
    rx = '[' + re.escape(''.join(symbols)) + ']'

    return re.sub(rx, '', text)


def remove_expressions(text, expressions):
    for exp in expressions:
        text = re.sub(exp, "", text)

    return text


def text_preprocessor(text):
    """ Function to execute a customized text preprocessing pipeline

    Args:
        text (str): raw text

    Return:
        processed text (str)
    """

    # letters left alone
    expressions = ["x{2}","\s[a-z]\s", "\\ufeff1"]
    symbols = list(string.punctuation)
    symbols.extend(["°", "º", "–", "‘", "’", "�", "€", "•",
                    "„", "”", "“", "©", "§", "¡"])

    # lowercase everything
    text = text.lower()
    # remove accents
    text = unidecode.unidecode(text)
    # remove numbers
    text = re.sub(r"\d+", "", text)
    # remove roman numbers
    text = re.sub(r"(?<=^)m{0,4}(cm|cd|d?C{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,"
                  r"3})(?=$)", "", text)
    # remove whitespaces
    text = " ".join(text.split())
    # remove punctuation and other symbols
    text = remove_symbols(text, symbols)
    # remove expressions
    text = remove_expressions(text, expressions)

    return text


def spacy_tokenizer(text, nlp, stop_words):
    """ Function to perform tokenization and lemmatizaion using Spacy

    Args:
        text (str): raw text
        nlp: Spacy language model
        stop_words (list): containing words to remove

    Returns:
        text represented as a list of lemmas
    """

    # Creating our token object, which is used to create documents with
    # linguistic annotations
    tokens = nlp(text)
    # Lemmatizing each token
    lemmas = [word.lemma_ for word in tokens if word.lemma_ != "-PRON-"]
    # Removing stop words
    lemmas = [word for word in lemmas if word not in stop_words]

    return lemmas

class SpacyTokenizer(object):
    """ A class used to tokennize and lemmatize text using Spacy.
    Constructed to be part of an sklearn Pipeline

    Attributes
    ----------
    nlp: Spacy language model
    stop_words (list): containing words to remove from text

    """

    def __init__(self):
        self.nlp = spacy.load("es_core_news_sm")
        self.stop_words = spacy.lang.es.stop_words.STOP_WORDS

    def __call__(self, text):
        return spacy_tokenizer(text, self.nlp, self.stop_words)


class TextPreprocessor(TransformerMixin):
    """ A class to apply a custom preprocessing pipeline.
    Constructed to be part of an sklearn Pipeline

    Methods
    ----------
    transform (X): applies a custom preprocessing function to all documents in
    the list

    fit(X):

    get_params():

    """

    def transform(self, X, **transform_params):
        return [text_preprocessor(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}
