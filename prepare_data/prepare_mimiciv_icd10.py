import logging
import random
from collections import Counter
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from icdmappings import Mapper
import simple_icd_10 as icd
import simple_icd_10_cm as cm

from prepare_data.utils import (
    TextPreprocessor,
    load_gz_file_into_df,
    preprocess_documents,
    reformat_code_dataframe,
    reformat_icd,
)
from src.settings import (
    DATA_DIRECTORY_MIMICIV_ICD9,
    DATA_DIRECTORY_MIMICIV_ICD10,
    DOWNLOAD_DIRECTORY_MIMICIV,
    DOWNLOAD_DIRECTORY_MIMICIV_NOTE,
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times

    Args:
        df (pd.DataFrame): The codes dataframe
        columns (list[str]): The column names of the codes
        min_count (int): The minimum number of times a code must appear

    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
        print(f"Number of unique codes in {col} before filtering: {len(code_counts)}")
        print(f"Number of unique codes in {col} after filtering: {len(codes_to_keep)}")

    return df

def parse_codes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the codes dataframe by replacing ICD codes with ICD chapters."""
    df = df.rename(columns={"hadm_id": ID_COLUMN, "subject_id": SUBJECT_ID_COLUMN})
    df = df.dropna(subset=["icd_code"])
    df = df.drop_duplicates(subset=[ID_COLUMN, "icd_code"])
    df = (
        df.groupby([SUBJECT_ID_COLUMN, ID_COLUMN, "icd_version"])
        .apply(partial(reformat_code_dataframe, col="icd_code"))
        .reset_index()
    )
    return df


def parse_notes_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the notes dataframe"""
    df = df.rename(
        columns={
            "hadm_id": ID_COLUMN,
            "subject_id": SUBJECT_ID_COLUMN,
            "text": TEXT_COLUMN,
        }
    )
    df = df.dropna(subset=[TEXT_COLUMN])
    df = df.drop_duplicates(subset=[ID_COLUMN, TEXT_COLUMN])
    return df


MIN_TARGET_COUNT = 10  # Minimum number of times a code must appear to be included
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=False,
    remove_digits=True,
    remove_accents=False,
    remove_brackets=False,
    convert_danish_characters=False,
)

random.seed(10)

logging.basicConfig(level=logging.INFO)

download_dir_note = Path(DOWNLOAD_DIRECTORY_MIMICIV_NOTE)
download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIV)
output_dir_icd9 = Path(DATA_DIRECTORY_MIMICIV_ICD9)
output_dir_icd9.mkdir(parents=True, exist_ok=True)

output_dir_icd10 = Path(DATA_DIRECTORY_MIMICIV_ICD10)
output_dir_icd10.mkdir(parents=True, exist_ok=True)

#load the data
mimic_notes = load_gz_file_into_df(download_dir_note / "note/discharge.csv.gz")
mimic_proc = load_gz_file_into_df(
    download_dir / "hosp/procedures_icd.csv.gz", dtype={"icd_code": str}
)
mimic_diag = load_gz_file_into_df(
    download_dir / "hosp/diagnoses_icd.csv.gz", dtype={"icd_code": str}
)

print("mimic-proc: info")
print( mimic_proc.info())
print("mimic-diag: info")
print( mimic_diag.info())
mapper = Mapper()
def icd9_to_icd10(row):
    if row['icd_version'] == 9:
        mapped_code = mapper.map(row['icd_code'], source='icd9', target='icd10')
        if mapped_code is not None:
            return mapped_code, 10
    #return the original icd_code and icd_version if no mapping is found
    return row['icd_code'], row['icd_version']

for index, row in mimic_proc.iterrows():
    mapped_code, icd_version = icd9_to_icd10(row)
    mimic_proc.at[index, 'icd_code'] = mapped_code
    mimic_proc.at[index, 'icd_version'] = icd_version
    
mimic_proc = mimic_proc[mimic_proc['icd_version'] != 9]
mimic_proc = mimic_proc.dropna(subset=['icd_code', 'icd_version'])
mimic_proc = mimic_proc[~mimic_proc['icd_code'].str.contains('NoDx')]
print("mimic_proc icd9 codes converted to icd10")
print(mimic_proc.info())

for index, row in mimic_diag.iterrows():
    mapped_code, icd_version = icd9_to_icd10(row)
    mimic_diag.at[index, 'icd_code'] = mapped_code
    mimic_diag.at[index, 'icd_version'] = icd_version

mimic_diag = mimic_diag[mimic_diag['icd_version'] != 9]
mimic_diag = mimic_diag.dropna(subset=['icd_code', 'icd_version'])
mimic_diag = mimic_diag[~mimic_diag['icd_code'].str.contains('NoDx')]
print("mimic_diag icd9 codes converted to icd10")
print(mimic_diag.info())

mimic_proc = mimic_proc.reset_index(drop=True)
mimic_diag = mimic_diag.reset_index(drop=True)

#format the codes by adding decimal points
mimic_proc["icd_code"] = mimic_proc.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=False
    ),
    axis=1,
)
mimic_diag["icd_code"] = mimic_diag.apply(
    lambda row: reformat_icd(
        code=row["icd_code"], version=row["icd_version"], is_diag=True
    ),
    axis=1,
)

mimic_proc = parse_codes_dataframe(mimic_proc)
mimic_diag = parse_codes_dataframe(mimic_diag)
mimic_notes = parse_notes_dataframe(mimic_notes)

mimic_proc_10 = mimic_proc[mimic_proc["icd_version"] == 10]
mimic_proc_10 = mimic_proc_10.rename(columns={"icd_code": "icd10_proc"})

mimic_diag_10 = mimic_diag[mimic_diag["icd_version"] == 10]
mimic_diag_10 = mimic_diag_10.rename(columns={"icd_code": "icd10_diag"})

mimiciv_10 = mimic_notes.merge(
    mimic_proc_10[[ID_COLUMN, "icd10_proc"]], on=ID_COLUMN, how="left"
)
mimiciv_10 = mimiciv_10.merge(
    mimic_diag_10[[ID_COLUMN, "icd10_diag"]], on=ID_COLUMN, how="left"
)

#remove notes with no codes
mimiciv_10 = mimiciv_10.dropna(subset=["icd10_proc", "icd10_diag"], how="all")

mimiciv_10["icd10_proc"] = mimiciv_10["icd10_proc"].apply(
    lambda x: [] if x is np.nan else x
)
mimiciv_10["icd10_diag"] = mimiciv_10["icd10_diag"].apply(
    lambda x: [] if x is np.nan else x
)

mimiciv_10 = filter_codes(mimiciv_10, ["icd10_proc", "icd10_diag"], MIN_TARGET_COUNT)

mimiciv_10[TARGET_COLUMN] = mimiciv_10["icd10_proc"] + mimiciv_10["icd10_diag"]

mimiciv_10 = mimiciv_10[mimiciv_10[TARGET_COLUMN].apply(lambda x: len(x) > 0)]

mimiciv_10 = mimiciv_10.reset_index(drop=True)

#text preprocess notes
mimiciv_10 = preprocess_documents(df=mimiciv_10, preprocessor=preprocessor)
print(mimiciv_10.info())
print(mimiciv_10[['subject_id', 'icd10_proc', 'icd10_diag', 'target', 'num_targets']].head(10))
    
mimiciv_10.to_feather(output_dir_icd10 / "mimiciv_icd10_mapped.feather")
