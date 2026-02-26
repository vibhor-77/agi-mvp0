"""
2D Coefficient fitter — fits each 2D Primitive to observed
(state, action, next_state) triples via scipy curve_fit.

Mirrors fitter.py. Key change: xdata passed to curve_fit is
X = np.vstack([states, actions]) of shape (2, n) instead of 1D.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional

from primitives import Primitive
from primitives_2d import PRIMITIVES_2D
from hypothesis import Hypothesis
from fitter import _rmse, _aic


def fit_primitive_2d(
    primitive: Primitive,
    states_tr: np.ndarray,
    actions_tr: np.ndarray,
    next_tr: np.ndarray,
    states_v: np.ndarray,
    actions_v: np.ndarray,
    next_v: np.ndarray,
    n_restarts: int = 5,
) -> Optional[Hypothesis]:
    """
    Fit one 2D primitive to training data, evaluate on validation data.
    Returns None if all restarts fail.
    """
    X_train = np.vstack([states_tr, actions_tr])  # shape (2, n_train)
    X_val = np.vstack([states_v, actions_v])      # shape (2, n_val)

    best_params: Optional[np.ndarray] = None
    best_train_rmse = np.inf

    rng = np.random.default_rng(seed=42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = list(primitive.p0)
        else:
            noise_scale = 0.5 * (restart / n_restarts)
            p0 = (
                np.array(primitive.p0)
                + rng.normal(0, noise_scale, primitive.n_params)
            ).tolist()

        try:
            params, _ = curve_fit(
                primitive.fn,
                X_train,
                next_tr,
                p0=p0,
                bounds=primitive.bounds,
                maxfev=10_000,
                method="trf",
                ftol=1e-10,
                xtol=1e-10,
            )
        except Exception:
            continue

        y_pred_train = primitive.fn(X_train, *params)
        rmse = _rmse(next_tr, y_pred_train)
        if rmse < best_train_rmse:
            best_train_rmse = rmse
            best_params = params

    if best_params is None:
        return None

    y_pred_val = primitive.fn(X_val, *best_params)
    val_rmse = _rmse(next_v, y_pred_val)
    aic = _aic(len(next_tr), primitive.n_params, best_train_rmse)

    return Hypothesis(
        primitive=primitive,
        params=best_params,
        train_rmse=best_train_rmse,
        val_rmse=val_rmse,
        aic=aic,
    )


def fit_all_2d(
    states_tr: np.ndarray,
    actions_tr: np.ndarray,
    next_tr: np.ndarray,
    states_v: np.ndarray,
    actions_v: np.ndarray,
    next_v: np.ndarray,
    primitives: List[Primitive] = PRIMITIVES_2D,
    n_restarts: int = 5,
) -> List[Hypothesis]:
    """Fit every 2D primitive; return all successful hypotheses."""
    results = []
    for prim in primitives:
        h = fit_primitive_2d(
            prim,
            states_tr, actions_tr, next_tr,
            states_v, actions_v, next_v,
            n_restarts,
        )
        if h is not None:
            results.append(h)
    return results
