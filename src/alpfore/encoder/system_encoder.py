# src/alpfore/encoding/system_encoder.py
from __future__ import annotations
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, List


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
        tokens = ["Ø"] * pad_len + list(seq)  # Ø = padding

        # mapping to one-hot rows
        map_vec = {
            "T": np.array([1.0, 0.0, 0.0]),
            "A": np.array([0.0, 1.0, 0.0]),
            "Ø": np.array([0.0, 0.0, 1.0]),
        }

        rows = [map_vec[b] for b in tokens]
        return np.concatenate(rows, axis=0)  # flatten to (36,)

    # ---- public API -------------------------------------------------- #
    def encode(
        self,
        seq: str,
        ssl: int,
        lsl: int,
        sgd: int,
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

    def decode(self, X: np.ndarray) -> np.ndarray:
        """
        Inverts `encode`.

        Parameters
        ----------
        X : array_like, shape (n, 40) or (40,)
            4 scaled meta-features  +  36 one-hot bits (12 × 3).

        Returns
        -------
        ndarray, shape (n, 4)  (dtype=object)
            Columns: sequence (str), ssl (int), lsl (int), sgd (int)
        """
        # ---------- normalise shape ----------
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != 40:
            raise ValueError("decode expects 40 columns")

        meta_scaled, one_hot = X[:, :4], X[:, 4:]

        # ---------- un-scale meta ----------
        print(self)
        print(type(self))
        rng = self.scales

        def _unscale(key, v):
            lo, hi = rng[key]["min"], rng[key]["max"]
            return v * (hi - lo) + lo

        ssl = _unscale("ssl", meta_scaled[:, 0]).round().astype(int)
        lsl = _unscale("lsl", meta_scaled[:, 1]).round().astype(int)
        sgd = _unscale("sgd", meta_scaled[:, 2]).round().astype(int)
        L_true = _unscale("seqlen", meta_scaled[:, 3]).round().astype(int)  # 4–12

        # ---------- decode sequences ----------
        vocab = np.array(list("TAØ"))  # 0→T, 1→A, 2→Ø
        oh = one_hot.reshape(X.shape[0], 12, 3)

        seqs = []
        for row_oh, L in zip(oh, L_true):
            idx = row_oh.argmax(axis=1)  # (12,)
            chars = vocab[idx]  # array(['T','Ø','A',...])
            # keep only non-padding symbols
            seq = "".join(c for c in chars if c != "Ø")
            if len(seq) != L:  # guard against mismatch
                raise ValueError(
                    f"Decoded length {len(seq)} ≠ expected {L}. "
                    "Ensure encode/decode use identical padding."
                )
            seqs.append(seq)

        # ---------- assemble object array ----------
        out = np.empty((X.shape[0], 4), dtype=object)
        out[:, 0] = seqs
        out[:, 1] = ssl
        out[:, 2] = lsl
        out[:, 3] = sgd
        return out

    # Factory for loading scales/vocab from json/yaml
    @classmethod
    def from_json(cls, path: str | Path) -> "SystemEncoder":
        cfg = json.loads(Path(path).read_text())
        return cls(scales=cfg["scales"], seq_vocab=cfg["seq_vocab"])
