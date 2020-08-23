# -*- coding: utf-8 -*-
import extract_data
import extract_text
import preprocess_raw_data

if __name__ == '__main__':
    extract_data.get_spreadsheet()
    extract_text.get_text()
    preprocess_raw_data.preprocess()
