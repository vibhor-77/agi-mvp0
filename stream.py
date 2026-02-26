"""
Stream generators — produce deterministic (optionally noisy) time series
from known mathematical functions.

Each stream exposes:
    stream.name        — short identifier
    stream.true_fn     — human-readable description of the true generating function
    stream.generator   — callable(n_points, noise_std) -> (x, y)  both np.ndarray
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class Stream:
    name: str
    true_fn: str
    generator: Callable  # (n: int, noise_std: float) -> (x: ndarray, y: ndarray)


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def make_linear(a: float = 3.0, b: float = 7.0) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return x, a * x + b + np.random.normal(0, noise_std, n)
    return Stream("linear", f"{a}·n + {b}", gen)


def make_quadratic(a: float = 0.5, b: float = 2.0, c: float = 1.0) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return x, a * x**2 + b * x + c + np.random.normal(0, noise_std, n)
    return Stream("quadratic", f"{a}·n² + {b}·n + {c}", gen)


def make_cubic(a: float = 0.05, b: float = -0.5, c: float = 2.0, d: float = 0.0) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return x, a * x**3 + b * x**2 + c * x + d + np.random.normal(0, noise_std, n)
    return Stream("cubic", f"{a}·n³ + {b}·n² + {c}·n + {d}", gen)


def make_sinusoidal(
    a: float = 3.0, b: float = 0.5, c: float = 0.0, d: float = 1.0
) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return x, a * np.sin(b * x + c) + d + np.random.normal(0, noise_std, n)
    return Stream("sinusoidal", f"{a}·sin({b}·n + {c}) + {d}", gen)


def make_exp(a: float = 1.0, b: float = 0.05, c: float = 0.0) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return x, a * np.exp(b * x) + c + np.random.normal(0, noise_std, n)
    return Stream("exponential", f"{a}·exp({b}·n) + {c}", gen)


def make_sin_plus_linear(
    a: float = 2.0, b: float = 0.3, c: float = 0.0, d: float = 0.5, e: float = 0.0
) -> Stream:
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return (
            x,
            a * np.sin(b * x + c) + d * x + e + np.random.normal(0, noise_std, n),
        )
    return Stream("sin+linear", f"{a}·sin({b}·n + {c}) + {d}·n + {e}", gen)


def make_damped_sin(
    a: float = 5.0, b: float = 0.8, c: float = 0.0, d: float = -0.05, e: float = 0.0
) -> Stream:
    """Damped sinusoid: a·sin(b·n + c)·exp(d·n) + e"""
    def gen(n: int, noise_std: float = 0.0):
        x = np.arange(n, dtype=float)
        return (
            x,
            a * np.sin(b * x + c) * np.exp(d * x) + e
            + np.random.normal(0, noise_std, n),
        )
    return Stream("damped_sin", f"{a}·sin({b}·n + {c})·exp({d}·n) + {e}", gen)


def make_mandelbrot_escape(
    c_real: float = -0.7, c_imag: float = 0.27, escape_radius: float = 10.0
) -> Stream:
    """
    Iterate z_{n+1} = z_n² + c from z_0 = 0.
    Stream = |z_n|, resetting when |z| exceeds escape_radius.

    This is a deterministic but chaotic sequence — no single primitive will
    fit it well.  It demonstrates the limits of Phase 1 and motivates
    Phase 2 compositional search.
    """
    def gen(n: int, noise_std: float = 0.0):
        c = complex(c_real, c_imag)
        z = 0 + 0j
        trajectory = []
        for _ in range(n):
            z = z * z + c
            trajectory.append(abs(z))
            if abs(z) > escape_radius:
                z = 0 + 0j  # reset — continuing stream
        x = np.arange(n, dtype=float)
        y = np.array(trajectory) + np.random.normal(0, noise_std, n)
        return x, y
    return Stream(
        "mandelbrot_escape",
        f"|z_n| for z→z²+({c_real}+{c_imag}i)",
        gen,
    )


# ---------------------------------------------------------------------------
# Default suites
# ---------------------------------------------------------------------------

DEFAULT_STREAMS = [
    make_linear(),
    make_quadratic(),
    make_cubic(),
    make_sinusoidal(),
    make_exp(),
    make_sin_plus_linear(),
    make_damped_sin(),
    make_mandelbrot_escape(),
]

NOISY_STREAMS = [
    make_linear(),       # used with noise_std > 0 in eval
    make_quadratic(),
    make_sinusoidal(),
]
