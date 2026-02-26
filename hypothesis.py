"""
Hypothesis — a fitted symbolic expression candidate.
HypothesisBeam — maintains the top-K scored hypotheses.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from primitives import Primitive


@dataclass
class Hypothesis:
    primitive: Primitive
    params: np.ndarray      # fitted coefficients
    train_rmse: float
    val_rmse: float
    aic: float              # Akaike Information Criterion (penalises complexity)

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    @property
    def score(self) -> float:
        """Lower is better. Primary sort key for the beam."""
        return self.val_rmse

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.primitive.name

    @property
    def description(self) -> str:
        return self.primitive.description

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.primitive.fn(x, *self.params)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{p:.4g}" for p in self.params)
        return (
            f"Hypothesis({self.name} | [{params_str}] | "
            f"val_rmse={self.val_rmse:.4g} | aic={self.aic:.4g})"
        )


class HypothesisBeam:
    """Keeps the top-K hypotheses ranked by val_rmse (ascending)."""

    def __init__(self, k: int = 5):
        self.k = k
        self._beam: List[Hypothesis] = []

    def update(self, hypotheses: List[Hypothesis]) -> None:
        valid = [h for h in hypotheses if np.isfinite(h.score)]
        self._beam = sorted(valid, key=lambda h: h.score)[: self.k]

    @property
    def best(self) -> Optional[Hypothesis]:
        return self._beam[0] if self._beam else None

    @property
    def all(self) -> List[Hypothesis]:
        return list(self._beam)

    def __len__(self) -> int:
        return len(self._beam)

    def summary(self, top_k: int = 5) -> str:
        lines = [f"  Beam (showing top {min(top_k, len(self._beam))})"]
        for i, h in enumerate(self._beam[:top_k]):
            marker = "★" if i == 0 else " "
            lines.append(f"  {marker} [{i + 1}] {h}")
        return "\n".join(lines)
