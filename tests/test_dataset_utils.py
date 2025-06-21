# tests/test_dataset_utils.py
import pytest
import pandas as pd
import numpy as np
from alpfore.utils import dataset_utils as du
from pathlib import Path
import tempfile
import os


def test_make_labeled_dataset():
    candidate_list = [
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6)
    ]
    ddg_results = [
        (-1.2, 0.1),
        (-0.5, 0.2)
    ]
    df = du.make_labeled_dataset(candidate_list, ddg_results)
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 5)  # 3 features + ddG + SEM
    np.testing.assert_array_equal(df.iloc[0].values, [0.1, 0.2, 0.3, -1.2, 0.1])


def test_save_and_load_labeled_dataset():
    df = pd.DataFrame([
        [0.1, 0.2, 0.3, -1.2, 0.1],
        [0.4, 0.5, 0.6, -0.5, 0.2]
    ])
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        path = Path(tmp.name)
        du.save_labeled_dataset(df, path)
        loaded = du.load_labeled_dataset(path)
        os.unlink(tmp.name)

    assert isinstance(loaded, pd.DataFrame)
    assert loaded.shape == (2, 5)
    np.testing.assert_allclose(loaded.values, df.values)


def test_append_new_data():
    df1 = pd.DataFrame([
        [0.1, 0.2, 0.3, -1.2, 0.1]
    ])
    df2 = pd.DataFrame([
        [0.4, 0.5, 0.6, -0.5, 0.2]
    ])
    result = du.append_new_data(df1, df2)
    assert result.shape == (2, 5)
    np.testing.assert_allclose(result.iloc[1].values, [0.4, 0.5, 0.6, -0.5, 0.2])


