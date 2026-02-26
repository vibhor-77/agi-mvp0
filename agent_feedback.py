"""
FeedbackAgent — system identification + control agent.

Protocol:
    agent.observe_triples(states, actions, next_states)  # ingest, no refit
    agent.refit()                                         # fit beam
    agent.predict_next(state, action) -> float            # f_hat(s, a)
    agent.choose_action(current_state, target) -> float   # invert model
    agent.run_control(stream, targets, noise_std)         # closed-loop control
"""

import numpy as np
from typing import List, Optional, Tuple
from scipy.optimize import minimize_scalar

from hypothesis import Hypothesis, HypothesisBeam
from fitter_2d import fit_all_2d
from primitives_2d import PRIMITIVES_2D


class FeedbackAgent:
    """
    Maintains a beam of 2D symbolic hypotheses about the transition function
    g(state, action) -> next_state and inverts the best model to choose actions
    that steer the state toward a target.
    """

    def __init__(
        self,
        beam_size: int = 5,
        train_ratio: float = 0.7,
        min_points: int = 20,
        n_restarts: int = 5,
        action_range: Tuple[float, float] = (-2.0, 2.0),
        verbose: bool = True,
    ):
        self.beam = HypothesisBeam(k=beam_size)
        self.train_ratio = train_ratio
        self.min_points = min_points
        self.n_restarts = n_restarts
        self.action_range = action_range
        self.verbose = verbose

        self._states: List[float] = []
        self._actions: List[float] = []
        self._next_states: List[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def observe_triples(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        next_states: np.ndarray,
    ) -> None:
        """Ingest (state, action, next_state) triples without refitting."""
        self._states.extend(states.tolist())
        self._actions.extend(actions.tolist())
        self._next_states.extend(next_states.tolist())

    def refit(self) -> None:
        """Fit the hypothesis beam on all observed triples."""
        n = len(self._states)
        if n < self.min_points:
            if self.verbose:
                print(f"  [FeedbackAgent] warming up — {n}/{self.min_points} points so far")
            return

        states_all = np.array(self._states)
        actions_all = np.array(self._actions)
        next_all = np.array(self._next_states)

        split = max(int(n * self.train_ratio), self.min_points // 2)
        states_tr = states_all[:split]
        actions_tr = actions_all[:split]
        next_tr = next_all[:split]
        states_v = states_all[split:]
        actions_v = actions_all[split:]
        next_v = next_all[split:]

        hypotheses = fit_all_2d(
            states_tr, actions_tr, next_tr,
            states_v, actions_v, next_v,
            primitives=PRIMITIVES_2D,
            n_restarts=self.n_restarts,
        )
        self.beam.update(hypotheses)

        if self.verbose:
            print(
                f"\n[FeedbackAgent] fit on {n} triples "
                f"(train={split}, val={n - split})"
            )
            print(self.beam.summary())

    def predict_next(self, state: float, action: float) -> float:
        """Predict next state given current state and action using best model."""
        best = self.beam.best
        if best is None:
            return state  # identity fallback
        X = np.array([[state], [action]])
        result = best.primitive.fn(X, *best.params)
        return float(result[0])

    def choose_action(self, current_state: float, target: float) -> float:
        """
        Invert the best model to find the action that drives next_state ≈ target.
        Uses bounded scalar minimisation over action_range.
        Falls back to zero action if no model is available.
        """
        best = self.beam.best
        if best is None:
            return 0.0

        s = current_state
        lo, hi = self.action_range

        result = minimize_scalar(
            lambda a: (
                best.primitive.fn(np.array([[s], [a]]), *best.params)[0] - target
            ) ** 2,
            bounds=(lo, hi),
            method="bounded",
        )
        return float(result.x)

    def run_control(
        self,
        stream,
        targets: np.ndarray,
        noise_std: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run closed-loop control for len(targets) steps.

        At each step i:
          - current state ctrl_states[i]
          - choose action targeting targets[i]
          - observe resulting next state

        Returns (ctrl_states, chosen_actions, targets) each of length n,
        where ctrl_states[i] is the state *after* applying the action at step i.
        """
        n = len(targets)
        rng = np.random.default_rng(seed=123)

        # Buffer for n+1 states; ctrl_buf[0] = initial state
        ctrl_buf = np.zeros(n + 1)
        if self._next_states:
            ctrl_buf[0] = self._next_states[-1]
        else:
            ctrl_buf[0] = stream.state_init

        chosen_actions = np.zeros(n)

        for i in range(n):
            s = ctrl_buf[i]
            a = self.choose_action(s, targets[i])
            chosen_actions[i] = a
            s_next = stream.true_g(s, a)
            if noise_std > 0:
                s_next += float(rng.normal(0, noise_std))
            ctrl_buf[i + 1] = s_next

        # Return resulting states (after applying actions) paired with targets
        ctrl_states = ctrl_buf[1:]   # shape (n,)
        return ctrl_states, chosen_actions, targets

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def best_hypothesis(self) -> Optional[Hypothesis]:
        return self.beam.best

    @property
    def n_observed(self) -> int:
        return len(self._states)
