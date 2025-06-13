# src/alpfore/selectors/thompson_sampling.py
from __future__ import annotations

import numpy as np

from alpfore.core.candidate_selector import BaseSelector
from alpfore.core.model import BaseModel


class ThompsonSamplingSelector(BaseSelector):
    """
    Gaussian Thompson sampling for batch candidate selection.

    Parameters
    ----------
    batch_size
        Number of points to return each cycle.
    rng
        Optional NumPy random Generator for reproducibility.
    """

    def __init__(self, batch_size: int = 1, rng: np.random.Generator | None = None):
        self.batch_size = batch_size
        self.rng = rng or np.random.default_rng()

    # --------------------------------------------------------------------- #
    # The pipeline will call this method:                                   #
    # --------------------------------------------------------------------- #
    def select(
        self,
        model: BaseModel,
        search_space: np.ndarray,
    ) -> np.ndarray:
        """
        Draw one posterior sample for every point in `search_space`
        and return the batch_size points with the best (highest) draws.
        Assumes a *single-output* surrogate; extend with axis handling
        if you have multi-output targets.
        """
        mean, var = model.predict(search_space)  # shape (N, 1) each
        std = np.sqrt(var.clip(min=1e-12))  # numerical safety
        draws = self.rng.normal(mean, std)  # Thompson sample

        top_idx = np.argpartition(-draws.squeeze(), self.batch_size - 1)[
            : self.batch_size
        ]
        return search_space[top_idx]
