# src/alpfore/encoding/system_encoder.py
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any


class SystemEncoder:
    """Turn human-readable DNA-NP parameters into a numeric feature vector."""

    def __init__(self, scales: Dict[str, Dict[str, float]], seq_vocab: str):
        """
        scales : {"ssl": {"min": 6, "max": 20}, ...}
        seq_vocab : ordered string of allowed bases, e.g. "ATCG"
        """
        self.scales = scales
        self.seq_vocab = seq_vocab
        self.vocab_map = {ch: i for i, ch in enumerate(seq_vocab)}

    # ---- helpers ----------------------------------------------------- #
    def _scale(self, key: str, val: float) -> float:
        rng = self.scales[key]["max"] - self.scales[key]["min"]
        return np.round((val - self.scales[key]["min"]) / rng, 3)

# inside SystemEncoder
    def _one_hot_seq(self, seq: str, width: int = 12) -> np.ndarray:
        """
        Encode a variable-length DNA sequence into a flattened 12 × 3 array.

        Rules
        -----
        • Allowed bases: 'T' or 'A'  (upper- or lower-case)  
        • If `len(seq) < width`, pad **on the left** with the token Ø → [0,0,1].  
        • If `len(seq) > width`, raise an error.

        Returned shape
        --------------
        (width * 3,)  →  36-element 1-D numpy array.
        """
        seq = seq.upper()
        if len(seq) > width:
            raise ValueError(f"Sequence longer than {width} bp: {seq!r}")

        pad_len = width - len(seq)
        tokens = ["Ø"] * pad_len + list(seq)          # Ø = padding

        # mapping to one-hot rows
        map_vec = {
            "T": np.array([1., 0., 0.]),
            "A": np.array([0., 1., 0.]),
            "Ø": np.array([0., 0., 1.]),
        }

        rows = [map_vec[b] for b in tokens]
        return np.concatenate(rows, axis=0)            # flatten to (36,)

    # ---- public API -------------------------------------------------- #
    def encode(
        self,
        ssl: int,
        lsl: int,
        sgd: int,
        seq: str,
    ) -> np.ndarray:
        meta = np.array(
            [
                self._scale("ssl", ssl),
                self._scale("lsl", lsl),
                self._scale("sgd", sgd),
                self._scale("seqlen", len(seq)),
            ],
            dtype=float,
        )
        one_hot = self._one_hot_seq(seq)
        return np.concatenate([meta, one_hot])

    # Factory for loading scales/vocab from json/yaml
    @classmethod
    def from_json(cls, path: str | Path) -> "SystemEncoder":
        cfg = json.loads(Path(path).read_text())
        return cls(scales=cfg["scales"], seq_vocab=cfg["seq_vocab"])

