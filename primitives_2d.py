"""
2D Primitive function library for feedback/control systems.

Each Primitive accepts X: ndarray shape (2, n), where X[0]=states and X[1]=actions.
Returns shape (n,).

Reuses the Primitive dataclass from primitives.py unchanged.
"""

import numpy as np
from typing import List

from primitives import Primitive


INF = 1e9


def _safe_2d(fn):
    """
    Wrap fn so it returns NaN instead of raising on domain errors.
    Unlike _safe, always returns shape (n,) where n = X.shape[1].
    """
    def wrapped(X, *params):
        with np.errstate(all="ignore"):
            try:
                result = fn(X, *params)
                return np.where(np.isfinite(result), result, np.nan)
            except Exception:
                return np.full(X.shape[1], np.nan, dtype=float)
    return wrapped


PRIMITIVES_2D: List[Primitive] = [

    Primitive(
        name="linear_state",
        fn=_safe_2d(lambda X, a, b: a * X[0] + b),
        n_params=2,
        p0=[1.0, 0.0],
        bounds=([-INF, -INF], [INF, INF]),
        description="a·s + b",
    ),

    Primitive(
        name="linear_action",
        fn=_safe_2d(lambda X, a, b, c: a * X[0] + b * X[1] + c),
        n_params=3,
        p0=[1.0, 0.1, 0.0],
        bounds=([-INF, -INF, -INF], [INF, INF, INF]),
        description="a·s + b·act + c",
    ),

    Primitive(
        name="bilinear",
        fn=_safe_2d(lambda X, a, b, c, d: a * X[0] * X[1] + b * X[0] + c * X[1] + d),
        n_params=4,
        p0=[0.1, 1.0, 0.1, 0.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="a·s·act + b·s + c·act + d",
    ),

    Primitive(
        name="quadratic_state",
        fn=_safe_2d(lambda X, a, b, c, d: a * X[0] ** 2 + b * X[0] + c * X[1] + d),
        n_params=4,
        p0=[0.1, 1.0, 0.1, 0.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="a·s² + b·s + c·act + d",
    ),

    Primitive(
        name="sin_state",
        fn=_safe_2d(lambda X, a, b, c, d, e: a * np.sin(b * X[0] + c) + d * X[1] + e),
        n_params=5,
        p0=[1.0, 1.0, 0.0, 0.1, 0.0],
        bounds=([-INF, -INF, -2 * np.pi, -INF, -INF], [INF, INF, 2 * np.pi, INF, INF]),
        description="a·sin(b·s + c) + d·act + e",
    ),

    Primitive(
        name="quadratic_action",
        fn=_safe_2d(lambda X, a, b, c, d: a * X[0] + b * X[1] ** 2 + c * X[1] + d),
        n_params=4,
        p0=[1.0, 0.1, 0.1, 0.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="a·s + b·act² + c·act + d",
    ),

    Primitive(
        name="linear_gain",
        fn=_safe_2d(lambda X, a, b, c, d: (a * X[0] + b) * X[1] + c * X[0] + d),
        n_params=4,
        p0=[0.1, 0.0, 1.0, 0.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="(a·s + b)·act + c·s + d",
    ),
]

PRIMITIVES_2D_MAP = {p.name: p for p in PRIMITIVES_2D}
