from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("data/raw/housing.csv")
PROCESSED_DATA_DIR = Path("data/processed")
TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"


def preprocess_data(
    raw_data_path: Path = RAW_DATA_PATH,
    train_data_path: Path = TRAIN_DATA_PATH,
    test_data_path: Path = TEST_DATA_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load raw housing data, perform basic validation, and split it into
    train/test datasets.

    In production ML, this step makes model training reproducible.
    """
    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at {raw_data_path}. "
            "Run `python src/ingest_data.py` first."
        )

    df = pd.read_csv(raw_data_path)

    if df.empty:
        raise ValueError("Raw dataset is empty.")

    if "MedHouseVal" not in df.columns:
        raise ValueError("Expected target column 'MedHouseVal' not found.")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
    )

    train_df.to_csv(train_data_path, index=False)
    test_df.to_csv(test_data_path, index=False)

    return train_df, test_df


if __name__ == "__main__":
    train_df, test_df = preprocess_data()
    print(f"Saved training data to {TRAIN_DATA_PATH}")
    print(f"Saved testing data to {TEST_DATA_PATH}")
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")