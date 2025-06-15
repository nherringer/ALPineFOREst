from pathlib import Path
from typing import List, Tuple
import pandas as pd


def make_labeled_dataset(
    candidate_list: List[Tuple],
    ddg_results: List[Tuple[float, float]]
) -> pd.DataFrame:
    """Create a labeled dataset with features + ddG + SEM."""
    rows = [list(features) + [ddg, sem] for features, (ddg, sem) in zip(candidate_list, ddg_results)]
    return pd.DataFrame(rows)


def save_labeled_dataset(df: pd.DataFrame, path: Path):
    """Save labeled dataset to CSV with header."""
    columns = [f"f{i}" for i in range(df.shape[1] - 2)] + ["ddg", "sem"]
    df.columns = columns
    df.to_csv(path, index=False)


def load_labeled_dataset(path: Path) -> pd.DataFrame:
    """Load labeled dataset from CSV."""
    return pd.read_csv(path)


def append_new_data(
    previous: pd.DataFrame,
    new: pd.DataFrame
) -> pd.DataFrame:
    """Append new evaluation data to prior labeled dataset."""
    return pd.concat([previous, new], ignore_index=True)

