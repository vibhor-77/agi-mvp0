"""
Phase 2 demo — Feedback Token Loop.

Usage:
    python demo_feedback.py [--system linear|bilinear|sin_state|quadratic]
                            [--explore 80] [--control 50] [--noise 0.0]
                            [--target-pattern step|sine|ramp] [--quiet]

Saves a 4-panel figure to results_feedback.png.
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stream_feedback import (
    make_linear_control,
    make_bilinear_control,
    make_sin_state_control,
    make_quadratic_state_control,
    random_exploration,
)
from agent_feedback import FeedbackAgent


SYSTEM_REGISTRY = {
    "linear": make_linear_control,
    "bilinear": make_bilinear_control,
    "sin_state": make_sin_state_control,
    "quadratic": make_quadratic_state_control,
}


def make_targets(pattern: str, n: int) -> np.ndarray:
    t = np.arange(n, dtype=float)
    if pattern == "sine":
        return 2.0 * np.sin(2 * np.pi * t / n) + 2.0
    elif pattern == "ramp":
        return np.linspace(0.5, 4.0, n)
    else:  # step (default)
        return np.where(t < n // 2, 1.0, 3.0)


def parse_args():
    p = argparse.ArgumentParser(description="Symbolic Regression Agent — Phase 2 Feedback")
    p.add_argument(
        "--system",
        choices=list(SYSTEM_REGISTRY.keys()),
        default="linear",
        help="Which dynamical system to use",
    )
    p.add_argument("--explore", type=int, default=80, help="# exploration steps")
    p.add_argument("--control", type=int, default=50, help="# control steps")
    p.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std")
    p.add_argument(
        "--target-pattern",
        choices=["step", "sine", "ramp"],
        default="step",
        help="Target reference pattern for control phase",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress per-step output")
    return p.parse_args()


def main():
    args = parse_args()
    verbose = not args.quiet

    stream = SYSTEM_REGISTRY[args.system]()

    if verbose:
        print(f"\n=== Phase 2: Feedback Token Loop ===")
        print(f"System    : {stream.name}")
        print(f"True fn   : {stream.true_fn}")
        print(f"Exploration: {args.explore} steps  |  Control: {args.control} steps")
        print(f"Noise     : {args.noise}  |  Target pattern: {args.target_pattern}\n")

    # ------------------------------------------------------------------
    # Phase 1: Random exploration
    # ------------------------------------------------------------------
    if verbose:
        print("--- Exploration phase ---")

    states_exp, actions_exp, next_exp = random_exploration(
        stream, args.explore, noise_std=args.noise, seed=0
    )

    agent = FeedbackAgent(
        action_range=stream.action_range,
        verbose=verbose,
    )
    agent.observe_triples(states_exp, actions_exp, next_exp)
    agent.refit()

    best = agent.best_hypothesis
    if verbose:
        if best is not None:
            print(f"\nBest identified model: {best}")
        else:
            print("\nNo model identified during exploration.")

    # ------------------------------------------------------------------
    # Model-ID validation set predictions (top 30% of exploration data)
    # ------------------------------------------------------------------
    n_exp = len(states_exp)
    split = int(n_exp * 0.7)
    states_v = states_exp[split:]
    actions_v = actions_exp[split:]
    next_v = next_exp[split:]

    if best is not None:
        X_val = np.vstack([states_v, actions_v])
        pred_v = best.primitive.fn(X_val, *best.params)
    else:
        pred_v = np.full_like(next_v, np.nan)

    # ------------------------------------------------------------------
    # Phase 2: Closed-loop control
    # ------------------------------------------------------------------
    if verbose:
        print("\n--- Control phase ---")

    targets = make_targets(args.target_pattern, args.control)
    ctrl_states, chosen_actions, targets_out = agent.run_control(
        stream, targets, noise_std=args.noise
    )

    tracking_error = np.abs(ctrl_states - targets_out)
    mean_error = float(np.mean(tracking_error))

    if verbose:
        print(f"Mean tracking error : {mean_error:.4f}")

    # ------------------------------------------------------------------
    # 4-panel plot
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle(
        f"Phase 2: {stream.name}  |  noise={args.noise}  |  target={args.target_pattern}",
        fontsize=13,
        fontweight="bold",
    )

    steps_exp = np.arange(len(states_exp))
    steps_ctrl = np.arange(len(ctrl_states))

    # Panel 1 — Exploration trajectory (state vs step, action magnitude as colour)
    ax = axes[0, 0]
    sc = ax.scatter(
        steps_exp, states_exp,
        c=np.abs(actions_exp), cmap="viridis", s=20, zorder=3,
    )
    plt.colorbar(sc, ax=ax, label="|action|")
    ax.set_xlabel("Step")
    ax.set_ylabel("State")
    ax.set_title("Exploration Trajectory")
    ax.grid(True, alpha=0.3)

    # Panel 2 — Model-ID scatter: predicted vs true on validation set
    ax = axes[0, 1]
    valid = np.isfinite(pred_v) & np.isfinite(next_v)
    if valid.any():
        ax.scatter(next_v[valid], pred_v[valid], s=20, alpha=0.6, color="steelblue", label="val pts")
        lo_v = min(next_v[valid].min(), pred_v[valid].min())
        hi_v = max(next_v[valid].max(), pred_v[valid].max())
        ax.plot([lo_v, hi_v], [lo_v, hi_v], "r--", lw=1.5, label="y=x")
    model_name = best.name if best is not None else "none"
    val_rmse = best.val_rmse if best is not None else float("nan")
    ax.set_xlabel("True $x_{n+1}$")
    ax.set_ylabel("Predicted $x_{n+1}$")
    ax.set_title(f"Model ID: {model_name}  (val_rmse={val_rmse:.4g})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3 — Control phase: state trajectory vs target reference
    ax = axes[1, 0]
    ax.plot(steps_ctrl, ctrl_states, "b-", lw=1.5, label="state", zorder=3)
    ax.plot(steps_ctrl, targets_out, "r--", lw=1.5, label="target", zorder=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("State")
    ax.set_title("Control Phase: State vs Target")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4 — Tracking error over time
    ax = axes[1, 1]
    ax.plot(steps_ctrl, tracking_error, "m-", lw=1.5, label="error")
    ax.axhline(
        mean_error, color="k", linestyle=":", lw=1.2,
        label=f"mean={mean_error:.4g}",
    )
    ax.set_xlabel("Step")
    ax.set_ylabel("|state − target|")
    ax.set_title("Tracking Error")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = "results_feedback.png"
    plt.savefig(out_path, dpi=120)

    if verbose:
        print(f"\nPlot saved to {out_path}")


if __name__ == "__main__":
    main()
