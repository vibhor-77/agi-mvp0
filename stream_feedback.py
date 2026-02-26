"""
Feedback stream generators — dynamical systems with controllable inputs.

Each FeedbackStream has a step interface:
    true_g(s: float, a: float) -> float

Factory functions return streams suitable for system-identification + control.
"""

import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class FeedbackStream:
    name: str
    true_fn: str
    true_g: Callable        # true_g(s: float, a: float) -> float
    state_init: float
    action_range: Tuple[float, float]


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def generate_triples(stream, n_steps, actions, noise_std, rng):
    """
    Simulate the stream for n_steps using the given actions sequence.
    Returns (states, actions, next_states) arrays of length n_steps.
    """
    states = np.zeros(n_steps)
    next_states = np.zeros(n_steps)

    s = stream.state_init
    for i in range(n_steps):
        a = float(actions[i])
        states[i] = s
        try:
            s_next = stream.true_g(s, a)
        except (OverflowError, ValueError, ZeroDivisionError):
            s_next = 0.0  # reset on divergence
        if not np.isfinite(float(s_next)):
            s_next = 0.0
        if noise_std > 0:
            s_next += float(rng.normal(0, noise_std))
        next_states[i] = s_next
        s = s_next

    return states, actions.copy(), next_states


def random_exploration(stream, n_steps, noise_std=0.0, seed=42):
    """
    Explore the stream with uniformly random actions from action_range.
    Returns (states, actions, next_states).
    """
    rng = np.random.default_rng(seed)
    lo, hi = stream.action_range
    actions = rng.uniform(lo, hi, n_steps)
    return generate_triples(stream, n_steps, actions, noise_std, rng)


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def make_linear_control(a=0.8, b=0.5, c=0.0):
    """x_{n+1} = a·x_n + b·action + c  (canonical LTI control)"""
    return FeedbackStream(
        name="linear_control",
        true_fn=f"{a}·s + {b}·act + {c}",
        true_g=lambda s, act: a * s + b * act + c,
        state_init=0.0,
        action_range=(-2.0, 2.0),
    )


def make_bilinear_control(a=0.3, b=0.7, c=0.2, d=0.0):
    """x_{n+1} = a·x_n·action + b·x_n + c·action + d"""
    return FeedbackStream(
        name="bilinear_control",
        true_fn=f"{a}·s·act + {b}·s + {c}·act + {d}",
        true_g=lambda s, act: a * s * act + b * s + c * act + d,
        state_init=1.0,
        action_range=(-1.5, 1.5),
    )


def make_sin_state_control(a=1.0, b=1.0, c=0.0, d=0.5, e=0.0):
    """x_{n+1} = a·sin(b·x_n + c) + d·action + e"""
    return FeedbackStream(
        name="sin_state_control",
        true_fn=f"{a}·sin({b}·s + {c}) + {d}·act + {e}",
        true_g=lambda s, act: a * np.sin(b * s + c) + d * act + e,
        state_init=0.5,
        action_range=(-2.0, 2.0),
    )


def make_quadratic_state_control(a=-0.5, b=2.0, c=1.0, d=0.5):
    """
    x_{n+1} = a·x_n² + b·x_n + c·action + d

    Downward-opening parabola: stable fixed point near x*=2.4.
    Operating range [1, 3] with action in [-1, 1].
    """
    return FeedbackStream(
        name="quadratic_state_control",
        true_fn=f"{a}·s² + {b}·s + {c}·act + {d}",
        true_g=lambda s, act: a * s ** 2 + b * s + c * act + d,
        state_init=2.0,
        action_range=(-1.0, 1.0),
    )
