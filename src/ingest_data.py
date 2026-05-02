from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_california_housing


RAW_DATA_PATH = Path("data/raw/housing.csv")


def ingest_data(output_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Download the California Housing dataset and save it as a raw CSV file.

    This simulates a real production data ingestion step where data is pulled
    from a source system and stored before preprocessing.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    df = ingest_data()
    print(f"Saved raw dataset to {RAW_DATA_PATH}")
    print(f"Dataset shape: {df.shape}")