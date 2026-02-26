"""
Entry point.

Usage:
    python main.py                  # run default suite (clean)
    python main.py --noise 0.5      # add Gaussian noise (std=0.5)
    python main.py --observe 80 --predict 20
    python main.py --stream quadratic sinusoidal
"""

import argparse
import numpy as np

from stream import (
    DEFAULT_STREAMS,
    make_linear,
    make_quadratic,
    make_cubic,
    make_sinusoidal,
    make_exp,
    make_sin_plus_linear,
    make_damped_sin,
    make_mandelbrot_escape,
)
from eval import run_evaluation


STREAM_REGISTRY = {
    "linear": make_linear,
    "quadratic": make_quadratic,
    "cubic": make_cubic,
    "sinusoidal": make_sinusoidal,
    "exp": make_exp,
    "sin+linear": make_sin_plus_linear,
    "damped_sin": make_damped_sin,
    "mandelbrot": make_mandelbrot_escape,
}


def parse_args():
    p = argparse.ArgumentParser(description="Symbolic Regression Agent — Phase 1")
    p.add_argument("--observe", type=int, default=60, help="# points to observe")
    p.add_argument("--predict", type=int, default=15, help="# future points to predict")
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std")
    p.add_argument(
        "--stream",
        nargs="+",
        choices=list(STREAM_REGISTRY.keys()),
        default=None,
        help="Which stream(s) to run (default: all)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    return p.parse_args()


def main():
    args = parse_args()

    if args.stream:
        streams = [STREAM_REGISTRY[s]() for s in args.stream]
    else:
        streams = DEFAULT_STREAMS

    np.random.seed(0)  # reproducibility

    run_evaluation(
        streams=streams,
        n_observe=args.observe,
        n_predict=args.predict,
        noise_std=args.noise,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
