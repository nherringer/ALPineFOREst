"""
Core abstractions for the *Candidate‑Selection* stage of ALPine FOREst.

A Selector takes a fitted model and proposes the next batch of input vectors
to evaluate (Bayesian optimisation, Thompson sampling, random, …).

Concrete implementations belong in ``alpfore.selectors.*``.
"""

from __future__ import annotations

import abc
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from alpfore.core.model import BaseModel


class BaseSelector(abc.ABC):
    """Abstract contract for proposing new candidates."""

    batch_size: int = 1  # subclasses can override in __init__

    @abc.abstractmethod
    def select(
        self,
        model: "BaseModel",
        search_space: "np.ndarray",
    ) -> "np.ndarray":
        """
        Parameters
        ----------
        model
            A fitted surrogate model.
        search_space
            Array of shape (N_pool, d) representing possible inputs.

        Returns
        -------
        np.ndarray
            Batch of input vectors, shape ``(self.batch_size, d)``.
        """
        ...


__all__ = ["BaseSelector"]
