from collections import defaultdict
from pathlib import Path
import polars as pl
from omegaconf import OmegaConf

from src.data.datatypes import Data
from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN


def get_code_system2code_counts(
    df: pl.DataFrame, code_systems: list[str]
) -> dict[str, dict[str, int]]:
    """
    Args:
        df (pl.DataFrame): The dataset in Polars dataframe format
        code_systems (list[str]): list of code systems to get counts for
    Returns:
        dict[str, dict[str, int]]: A dictionary with code systems as keys and a dictionary of code counts as values
    """
    code_system2code_counts = defaultdict(dict)
    
    for col in code_systems:
        #extract the column, count occurrences of each code, and convert it to a list of dictionaries
        value_counts = (
            df.select(pl.col(col)) 
            .explode(col)
            .group_by(col)
            .agg(pl.count().alias("count"))
            .to_dicts()
        )

        code_system2code_counts[col] = {code[col]: code["count"] for code in value_counts}

    return code_system2code_counts


def data_pipeline(config: OmegaConf) -> Data:
    """
    The data pipeline.

    Args:
        config (OmegaConf): The config.

    Returns:
        Data: The data.
    """
    dir = Path(config.dir)

    df = pl.read_ipc(
        dir / config.data_filename,
        columns=[
            ID_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            "num_words",
            "num_targets",
        ]
        + config.code_column_names,
        memory_map = False
    )

    splits = pl.read_ipc(dir / config.split_filename, memory_map = False)
    
    gen_data = pl.read_ipc( 
            dir / "gen_data_full.feather",
            columns=[
            ID_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            "num_words",
            "num_targets",
            "split",
        ]
        + config.code_column_names,
        memory_map = False
    )
    print("Shape and Dtypes of gen_data df:")
    print(gen_data.shape)
    print(gen_data.dtypes)
    print(gen_data.columns)
    #inner join on the ID column
    df = df.join(splits, on=ID_COLUMN, how="inner")


    gen_data = gen_data.select(df.columns)
    df = df.vstack(gen_data)
    print("Shape of final df:")
    print(df.shape)
    #print(df.dtypes)
    #print(df.columns)

    '''
    import pandas as pd
    #save to csv for local exp
    # Convert Polars DataFrame clone to Pandas and save as CSV
    
    df_for_csv = df.clone().to_pandas()
    print(df_for_csv.info())
    print(df_for_csv['target'].head())
    output_csv_path = dir / "joined_data.csv"
    df_for_csv.to_csv(output_csv_path, index=False)
    print(f"Data successfully saved to {output_csv_path}")
    '''
    
    code_system2code_counts = get_code_system2code_counts(df, config.code_column_names)
    schema = {
        ID_COLUMN: pl.Int64,
        TEXT_COLUMN: pl.Utf8,
        TARGET_COLUMN: pl.List(pl.Utf8),
        "split": pl.Utf8,
        "num_words": pl.Int64,
        "num_targets": pl.Int64,
    }

    df = df.select(
        [
            ID_COLUMN,
            TEXT_COLUMN,
            TARGET_COLUMN,
            "split",
            "num_words",
            "num_targets",
        ]
    ).cast(schema)

    return Data(df, code_system2code_counts)
