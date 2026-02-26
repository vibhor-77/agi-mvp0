"""
Primitive function library.

Each Primitive is a named symbolic template:  f(x, *params)
The agent searches over this library to find which template best explains the stream.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, List, Tuple


INF = 1e9


@dataclass
class Primitive:
    name: str
    fn: Callable            # fn(x: ndarray, *params) -> ndarray
    n_params: int
    p0: List[float]         # initial parameter guess for curve_fit
    bounds: Tuple           # (lower_bounds, upper_bounds)
    description: str        # human-readable e.g. "a·x² + b·x + c"


def _safe(fn: Callable) -> Callable:
    """Wrap fn so it returns NaN instead of raising on domain errors."""
    def wrapped(x, *params):
        with np.errstate(all="ignore"):
            try:
                result = fn(x, *params)
                return np.where(np.isfinite(result), result, np.nan)
            except Exception:
                return np.full_like(x, np.nan, dtype=float)
    return wrapped


PRIMITIVES: List[Primitive] = [

    Primitive(
        name="constant",
        fn=_safe(lambda x, a: np.full_like(x, a, dtype=float)),
        n_params=1,
        p0=[1.0],
        bounds=([-INF], [INF]),
        description="a",
    ),

    Primitive(
        name="linear",
        fn=_safe(lambda x, a, b: a * x + b),
        n_params=2,
        p0=[1.0, 0.0],
        bounds=([-INF, -INF], [INF, INF]),
        description="a·x + b",
    ),

    Primitive(
        name="quadratic",
        fn=_safe(lambda x, a, b, c: a * x**2 + b * x + c),
        n_params=3,
        p0=[1.0, 0.0, 0.0],
        bounds=([-INF, -INF, -INF], [INF, INF, INF]),
        description="a·x² + b·x + c",
    ),

    Primitive(
        name="cubic",
        fn=_safe(lambda x, a, b, c, d: a * x**3 + b * x**2 + c * x + d),
        n_params=4,
        p0=[0.1, 0.0, 1.0, 0.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="a·x³ + b·x² + c·x + d",
    ),

    Primitive(
        name="sin",
        fn=_safe(lambda x, a, b, c, d: a * np.sin(b * x + c) + d),
        n_params=4,
        p0=[1.0, 1.0, 0.0, 0.0],
        bounds=([-INF, -INF, -2 * np.pi, -INF], [INF, INF, 2 * np.pi, INF]),
        description="a·sin(b·x + c) + d",
    ),

    Primitive(
        name="cos",
        fn=_safe(lambda x, a, b, c, d: a * np.cos(b * x + c) + d),
        n_params=4,
        p0=[1.0, 1.0, 0.0, 0.0],
        bounds=([-INF, -INF, -2 * np.pi, -INF], [INF, INF, 2 * np.pi, INF]),
        description="a·cos(b·x + c) + d",
    ),

    Primitive(
        name="exp",
        fn=_safe(lambda x, a, b, c: a * np.exp(np.clip(b * x, -50, 50)) + c),
        n_params=3,
        p0=[1.0, 0.1, 0.0],
        bounds=([-INF, -10.0, -INF], [INF, 10.0, INF]),
        description="a·exp(b·x) + c",
    ),

    Primitive(
        name="log",
        fn=_safe(lambda x, a, b, c: a * np.log(np.abs(x) + np.abs(b) + 1e-6) + c),
        n_params=3,
        p0=[1.0, 1.0, 0.0],
        bounds=([-INF, -INF, -INF], [INF, INF, INF]),
        description="a·log(|x| + |b|) + c",
    ),

    Primitive(
        name="power",
        fn=_safe(lambda x, a, b, c: a * (np.abs(x) ** np.clip(b, 0.01, 10)) + c),
        n_params=3,
        p0=[1.0, 2.0, 0.0],
        bounds=([-INF, 0.01, -INF], [INF, 10.0, INF]),
        description="a·|x|^b + c",
    ),

    # --- Composite primitives ---

    Primitive(
        name="sin+linear",
        fn=_safe(lambda x, a, b, c, d, e: a * np.sin(b * x + c) + d * x + e),
        n_params=5,
        p0=[1.0, 1.0, 0.0, 0.1, 0.0],
        bounds=([-INF] * 5, [INF] * 5),
        description="a·sin(b·x + c) + d·x + e",
    ),

    Primitive(
        name="cos+linear",
        fn=_safe(lambda x, a, b, c, d, e: a * np.cos(b * x + c) + d * x + e),
        n_params=5,
        p0=[1.0, 1.0, 0.0, 0.1, 0.0],
        bounds=([-INF] * 5, [INF] * 5),
        description="a·cos(b·x + c) + d·x + e",
    ),

    Primitive(
        name="sin+quadratic",
        fn=_safe(
            lambda x, a, b, c, d, e, f: a * np.sin(b * x + c) + d * x**2 + e * x + f
        ),
        n_params=6,
        p0=[1.0, 1.0, 0.0, 0.1, 0.0, 0.0],
        bounds=([-INF] * 6, [INF] * 6),
        description="a·sin(b·x + c) + d·x² + e·x + f",
    ),

    # Damped oscillation
    Primitive(
        name="sin·exp",
        fn=_safe(
            lambda x, a, b, c, d, e: a
            * np.sin(b * x + c)
            * np.exp(np.clip(d * x, -50, 50))
            + e
        ),
        n_params=5,
        p0=[1.0, 1.0, 0.0, -0.05, 0.0],
        bounds=([-INF, -INF, -2 * np.pi, -5.0, -INF], [INF, INF, 2 * np.pi, 5.0, INF]),
        description="a·sin(b·x + c)·exp(d·x) + e",
    ),

    # Linear rational
    Primitive(
        name="rational",
        fn=_safe(lambda x, a, b, c, d: (a * x + b) / (c * x + d + 1e-9)),
        n_params=4,
        p0=[1.0, 0.0, 0.0, 1.0],
        bounds=([-INF] * 4, [INF] * 4),
        description="(a·x + b) / (c·x + d)",
    ),
]

PRIMITIVE_MAP = {p.name: p for p in PRIMITIVES}
