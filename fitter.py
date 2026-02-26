"""
Coefficient fitter — tries to fit each Primitive to observed data using
scipy's curve_fit (Levenberg-Marquardt / Trust Region Reflective).

Multiple random restarts increase robustness against local minima.
"""

import numpy as np
from scipy.optimize import curve_fit
from typing import List, Optional

from primitives import Primitive, PRIMITIVES
from hypothesis import Hypothesis


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = np.isfinite(y_pred) & np.isfinite(y_true)
    if mask.sum() < 2:
        return np.inf
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def _aic(n: int, k: int, rmse: float) -> float:
    """
    AIC for Gaussian noise model.
    log-likelihood ∝ -n·log(rmse)  →  AIC = 2k - 2·log_likelihood
    """
    if rmse <= 0 or not np.isfinite(rmse):
        return np.inf
    log_likelihood = -n * np.log(rmse + 1e-12)
    return 2 * k - 2 * log_likelihood


# ---------------------------------------------------------------------------
# Single-primitive fitting
# ---------------------------------------------------------------------------

def fit_primitive(
    primitive: Primitive,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    n_restarts: int = 5,
) -> Optional[Hypothesis]:
    """
    Fit one primitive to training data, evaluate on validation data.
    Returns None if all restarts fail.
    """
    best_params: Optional[np.ndarray] = None
    best_train_rmse = np.inf

    rng = np.random.default_rng(seed=42)

    for restart in range(n_restarts):
        if restart == 0:
            p0 = list(primitive.p0)
        else:
            # Perturb initial guess — helps escape local minima
            noise_scale = 0.5 * (restart / n_restarts)
            p0 = (
                np.array(primitive.p0)
                + rng.normal(0, noise_scale, primitive.n_params)
            ).tolist()

        try:
            params, _ = curve_fit(
                primitive.fn,
                x_train,
                y_train,
                p0=p0,
                bounds=primitive.bounds,
                maxfev=10_000,
                method="trf",
                ftol=1e-10,
                xtol=1e-10,
            )
        except Exception:
            continue

        y_pred_train = primitive.fn(x_train, *params)
        rmse = _rmse(y_train, y_pred_train)
        if rmse < best_train_rmse:
            best_train_rmse = rmse
            best_params = params

    if best_params is None:
        return None

    y_pred_val = primitive.fn(x_val, *best_params)
    val_rmse = _rmse(y_val, y_pred_val)
    aic = _aic(len(y_train), primitive.n_params, best_train_rmse)

    return Hypothesis(
        primitive=primitive,
        params=best_params,
        train_rmse=best_train_rmse,
        val_rmse=val_rmse,
        aic=aic,
    )


# ---------------------------------------------------------------------------
# Fit entire library
# ---------------------------------------------------------------------------

def fit_all(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    primitives: List[Primitive] = PRIMITIVES,
    n_restarts: int = 5,
) -> List[Hypothesis]:
    """Fit every primitive; return all successful hypotheses."""
    results = []
    for prim in primitives:
        h = fit_primitive(prim, x_train, y_train, x_val, y_val, n_restarts)
        if h is not None:
            results.append(h)
    return results
