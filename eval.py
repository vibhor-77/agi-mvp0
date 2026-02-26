"""
Evaluation harness — runs the SymbolicAgent on a suite of streams,
measures prediction accuracy, and plots results.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
from typing import List, Optional

from stream import Stream
from agent import SymbolicAgent


# ---------------------------------------------------------------------------
# Single-stream evaluation
# ---------------------------------------------------------------------------

def evaluate_stream(
    stream: Stream,
    n_observe: int = 60,
    n_predict: int = 15,
    noise_std: float = 0.0,
    verbose: bool = True,
) -> dict:
    """
    1. Generate n_observe + n_predict points from stream.
    2. Feed the first n_observe points to the agent.
    3. Ask the agent to predict the next n_predict points.
    4. Return a result dict with metrics and raw arrays.
    """
    n_total = n_observe + n_predict
    x_all, y_all = stream.generator(n_total, noise_std)

    x_obs, y_obs = x_all[:n_observe], y_all[:n_observe]
    x_fut, y_fut = x_all[n_observe:], y_all[n_observe:]

    agent = SymbolicAgent(verbose=verbose)
    agent.observe(x_obs, y_obs)
    y_pred = agent.predict(x_fut)

    best = agent.best_hypothesis
    pred_rmse = (
        float(np.sqrt(np.mean((y_fut - y_pred) ** 2)))
        if y_pred is not None
        else np.inf
    )

    return {
        "stream": stream.name,
        "true_fn": stream.true_fn,
        "best_hypothesis": best.name if best else None,
        "best_description": best.description if best else None,
        "best_params": best.params if best else None,
        "val_rmse": best.val_rmse if best else np.inf,
        "pred_rmse": pred_rmse,
        "x_obs": x_obs,
        "y_obs": y_obs,
        "x_fut": x_fut,
        "y_fut": y_fut,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: List[dict], title: str = "Symbolic Agent — Phase 1") -> None:
    n = len(results)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, results):
        x_obs, y_obs = r["x_obs"], r["y_obs"]
        x_fut, y_fut = r["x_fut"], r["y_fut"]
        y_pred = r.get("y_pred")

        ax.plot(x_obs, y_obs, "b.", ms=3, alpha=0.55, label="observed")
        ax.plot(x_fut, y_fut, "g.", ms=5, label="true future")

        if y_pred is not None:
            ax.plot(x_fut, y_pred, "r--", lw=2, label="predicted")
            # shade error between true and predicted
            ax.fill_between(
                x_fut, y_fut, y_pred, alpha=0.15, color="red"
            )

        ax.axvline(x_obs[-1], color="k", linestyle=":", alpha=0.4, lw=1)

        rmse_str = (
            f"{r['pred_rmse']:.3g}"
            if np.isfinite(r.get("pred_rmse", np.inf))
            else "∞"
        )
        ax.set_title(
            f"{r['stream']}\n"
            f"True:  {r.get('true_fn', '?')}\n"
            f"Fit:   {r.get('best_description', '?')}\n"
            f"RMSE:  {rmse_str}",
            fontsize=8,
            loc="left",
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.set_xlabel("n", fontsize=8)

    # Hide unused axes
    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    out = "results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n[Eval] plot saved → {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Full evaluation run
# ---------------------------------------------------------------------------

def run_evaluation(
    streams=None,
    n_observe: int = 60,
    n_predict: int = 15,
    noise_std: float = 0.0,
    verbose: bool = True,
) -> List[dict]:
    from stream import DEFAULT_STREAMS

    if streams is None:
        streams = DEFAULT_STREAMS

    sep = "=" * 65
    print(sep)
    print("  SYMBOLIC REGRESSION AGENT  —  Phase 1")
    print(sep)

    results = []
    for stream in streams:
        print(f"\n{'─' * 65}")
        print(f"  Stream : {stream.name}")
        print(f"  True   : {stream.true_fn}")
        print(f"{'─' * 65}")
        r = evaluate_stream(
            stream,
            n_observe=n_observe,
            n_predict=n_predict,
            noise_std=noise_std,
            verbose=verbose,
        )
        results.append(r)
        rmse_str = (
            f"{r['pred_rmse']:.6g}"
            if np.isfinite(r["pred_rmse"])
            else "∞"
        )
        print(f"\n  → best hypothesis : {r['best_hypothesis']}")
        print(f"  → description     : {r['best_description']}")
        print(f"  → pred RMSE       : {rmse_str}")

    # Summary table
    print(f"\n{sep}")
    print("  SUMMARY")
    print(sep)
    col = "{:<20} {:<38} {:<18} {:>10}"
    print(col.format("Stream", "True function", "Best hypothesis", "Pred RMSE"))
    print("-" * 90)
    for r in results:
        rmse_str = (
            f"{r['pred_rmse']:.4g}"
            if np.isfinite(r["pred_rmse"])
            else "∞"
        )
        print(
            col.format(
                r["stream"][:19],
                r.get("true_fn", "?")[:37],
                r.get("best_hypothesis", "?")[:17],
                rmse_str,
            )
        )

    plot_results(results)
    return results
