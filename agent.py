"""
SymbolicAgent — the core learning loop.

Protocol:
    agent.observe(x, y)      feed in a window of observed data; triggers refit
    agent.predict(x_future)  predict future values using the current best hypothesis
    agent.best_hypothesis    read-only access to the top-ranked Hypothesis
"""

import numpy as np
from typing import List, Optional

from hypothesis import Hypothesis, HypothesisBeam
from fitter import fit_all
from primitives import PRIMITIVES


class SymbolicAgent:
    """
    Maintains a beam of symbolic hypotheses about the stream's generating
    function and uses the best-scoring one to make predictions.

    Design choices
    --------------
    - train/val split on the observed window (default 70/30)
    - hypothesis ranked by val_rmse  (lower = better)
    - refit triggered on every call to observe()
    - predict() returns None until enough data has been seen
    """

    def __init__(
        self,
        beam_size: int = 5,
        train_ratio: float = 0.7,
        min_points: int = 15,
        n_restarts: int = 5,
        verbose: bool = True,
    ):
        self.beam = HypothesisBeam(k=beam_size)
        self.train_ratio = train_ratio
        self.min_points = min_points
        self.n_restarts = n_restarts
        self.verbose = verbose

        self._x: List[float] = []
        self._y: List[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe(self, x: np.ndarray, y: np.ndarray) -> None:
        """Ingest new (x, y) pairs and refit the hypothesis beam."""
        self._x.extend(x.tolist())
        self._y.extend(y.tolist())

        n = len(self._x)
        if n < self.min_points:
            if self.verbose:
                print(f"  [Agent] warming up — {n}/{self.min_points} points so far")
            return

        x_all = np.array(self._x)
        y_all = np.array(self._y)

        split = max(int(n * self.train_ratio), self.min_points // 2)
        x_train, y_train = x_all[:split], y_all[:split]
        x_val, y_val = x_all[split:], y_all[split:]

        hypotheses = fit_all(
            x_train, y_train, x_val, y_val,
            primitives=PRIMITIVES,
            n_restarts=self.n_restarts,
        )
        self.beam.update(hypotheses)

        if self.verbose:
            print(
                f"\n[Agent] fit on {n} pts  "
                f"(train={split}, val={n - split})"
            )
            print(self.beam.summary())

    def predict(self, x_future: np.ndarray) -> Optional[np.ndarray]:
        """Return predictions for x_future using the best hypothesis."""
        if self.beam.best is None:
            return None
        return self.beam.best.predict(x_future)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def best_hypothesis(self) -> Optional[Hypothesis]:
        return self.beam.best

    @property
    def n_observed(self) -> int:
        return len(self._x)
