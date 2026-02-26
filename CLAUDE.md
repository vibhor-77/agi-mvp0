# CLAUDE.md — AGI MVP0 Codebase Guide

This file provides context for AI assistants working in this repository.

## Project Overview

**AGI MVP0** is a two-phase symbolic regression research project in pure Python.

- **Phase 1**: Passive stream observation — learns symbolic expressions from 1D time-series data using a hypothesis beam search over a library of primitive functions.
- **Phase 2**: Feedback/control loop — extends Phase 1 with 2D system identification (state, action → next_state) and closed-loop control via model inversion.

The goal is a minimal, readable demonstration of symbolic AI: observe data, hypothesize a functional form, fit coefficients, and use the model to act.

---

## Repository Structure

```
agi-mvp0/
├── requirements.txt          # numpy, scipy, matplotlib
│
├── Phase 1 — Passive Stream Learning
│   ├── main.py               # CLI entry point
│   ├── agent.py              # SymbolicAgent (observe + predict)
│   ├── stream.py             # Stream dataclass + 8 factory functions
│   ├── primitives.py         # 15 1D primitive function templates
│   ├── hypothesis.py         # Hypothesis dataclass + HypothesisBeam
│   ├── fitter.py             # scipy curve_fit wrapper (1D)
│   └── eval.py               # Evaluation harness + matplotlib output
│
├── Phase 2 — Feedback/Control
│   ├── demo_feedback.py      # CLI entry point
│   ├── agent_feedback.py     # FeedbackAgent (system ID + control)
│   ├── stream_feedback.py    # FeedbackStream + 4 control factories
│   ├── primitives_2d.py      # 7 2D primitive templates
│   └── fitter_2d.py          # scipy curve_fit wrapper (2D)
│
└── Output (generated)
    ├── results.png            # Phase 1 visualization
    └── results_feedback.png   # Phase 2 visualization
```

---

## Technology Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Numerics | NumPy ≥ 1.24 |
| Optimization | SciPy ≥ 1.10 (`curve_fit`, `minimize_scalar`) |
| Visualization | Matplotlib ≥ 3.7 |
| Testing | None (demo scripts + visual output) |
| CI/CD | None |
| Packaging | None (flat module layout) |

No web frameworks, databases, or external services are used.

---

## Running the Code

### Phase 1 — Passive Learning

```bash
# Install dependencies
pip install -r requirements.txt

# Run default suite of 8 streams
python main.py

# Options
python main.py --noise 0.5 --observe 60 --predict 15
python main.py --stream quadratic sinusoidal
python main.py --quiet   # suppress per-stream output
```

Key CLI flags for `main.py`:
- `--observe N` — number of data points to feed the agent (default: 50)
- `--predict N` — number of future steps to forecast (default: 10)
- `--noise FLOAT` — Gaussian noise std added to streams (default: 0.0)
- `--stream NAME [NAME...]` — run only specified stream(s)
- `--quiet` — suppress verbose output

Output: summary table printed to stdout + `results.png` saved to disk.

### Phase 2 — Feedback Control

```bash
python demo_feedback.py

# Options
python demo_feedback.py --system linear_control --explore 80 --control 40
python demo_feedback.py --noise 0.1 --target-pattern sine
```

Key CLI flags for `demo_feedback.py`:
- `--system NAME` — feedback system to use (default: `linear_control`)
- `--explore N` — number of random-action exploration steps (default: 60)
- `--control N` — number of closed-loop control steps (default: 40)
- `--noise FLOAT` — observation noise std (default: 0.05)
- `--target-pattern NAME` — reference trajectory shape (`sine`, `step`, etc.)

Output: `results_feedback.png` with 4-panel visualization.

---

## Architecture & Key Conventions

### Phase 1 Data Flow

```
Stream → SymbolicAgent.observe(x, y)
           → fit_all(primitives, x_train, y_train)
           → HypothesisBeam.update(hypotheses)
         SymbolicAgent.predict(x_future)
           → beam.best_hypothesis.predict(x_future)
```

### Phase 2 Data Flow

```
FeedbackStream.random_exploration()
  → (state, action, next_state) triples
FeedbackAgent.observe_triples(triples)
  → fit_all_2d(primitives_2d, states, actions, next_states)
FeedbackAgent.run_control(target)
  → choose_action(state) via minimize_scalar on best 2D model
  → step system → observe → refit
```

### Hypothesis Ranking

Hypotheses are ranked by **validation RMSE** on a held-out 30% split of observed data. AIC is computed but not used for ranking. The beam keeps the top-K best hypotheses (default K=5).

### Primitive Library Design

- Each `Primitive` is a dataclass with: `name`, `fn` (callable), `n_params`, `p0` (initial guess), `bounds`, `description`.
- 1D primitives (Phase 1) take `(x, *params)` signature.
- 2D primitives (Phase 2) take `((state, action), *params)` signature.
- All primitives are wrapped in `_safe()` / `_safe_2d()` to catch NaN/overflow, returning `np.inf` on failure.

### Fitting Strategy

- `scipy.optimize.curve_fit` with `method='trf'` (Trust Region Reflective) is used throughout.
- Multiple random restarts (default 5) help escape local minima.
- Bounds are set per-primitive to enforce physical plausibility.
- Failures (singular covariance, etc.) are caught and the primitive is skipped.

### Reproducibility

Fixed random seeds are used: `seed=0`, `seed=42`, `seed=123`. Do not remove them — they ensure deterministic output for comparison.

---

## Module Responsibilities

| File | Class/Key API | Responsibility |
|---|---|---|
| `agent.py` | `SymbolicAgent` | Maintains hypothesis beam; `observe()`, `predict()` |
| `stream.py` | `Stream`, `DEFAULT_STREAMS` | Time-series data generators |
| `primitives.py` | `Primitive`, `PRIMITIVES` | 1D function template library |
| `hypothesis.py` | `Hypothesis`, `HypothesisBeam` | Beam search over hypotheses |
| `fitter.py` | `fit_all()`, `fit_primitive()` | 1D curve fitting via scipy |
| `eval.py` | `evaluate_stream()`, `run_evaluation()` | Harness + matplotlib plotting |
| `agent_feedback.py` | `FeedbackAgent` | System ID + control via model inversion |
| `stream_feedback.py` | `FeedbackStream` | Controlled-system simulators |
| `primitives_2d.py` | `PRIMITIVES_2D` | 2D (state, action) function library |
| `fitter_2d.py` | `fit_all_2d()` | 2D curve fitting via scipy |

---

## Development Conventions

### Code Style

- Pure Python with no linting configuration; aim for PEP 8 compliance.
- Module-level docstrings are present in every file — keep them current.
- Functions have brief inline comments at decision points (not line-by-line narration).
- No type annotations are used; keep this consistent unless refactoring deliberately.

### Adding a New Primitive (Phase 1)

1. Define a function `def my_fn(x, a, b, ...)` with safe fallback behavior.
2. Create a `Primitive(name=..., fn=my_fn, n_params=N, p0=[...], bounds=(...), description=...)`.
3. Append it to the `PRIMITIVES` list in `primitives.py`.
4. It will automatically be tried on every `observe()` call.

### Adding a New Stream (Phase 1)

1. Define `true_fn` (deterministic ground truth) and a generator function.
2. Create a `Stream(name=..., true_fn=..., generator=...)`.
3. Append to `DEFAULT_STREAMS` in `stream.py`.

### Adding a New 2D Primitive (Phase 2)

Follow the same pattern in `primitives_2d.py` but the function signature must be `fn((state, action), *params)` and it must be added to `PRIMITIVES_2D`.

### Adding a New Feedback System (Phase 2)

Define `true_fn`, `true_g` (state transition), initial state, and action range, then add a factory function and register it in `stream_feedback.py`.

---

## Testing & Verification

There is no automated test suite. Correctness is verified visually:

- Run `python main.py` and inspect `results.png` — each panel should show the agent's prediction tracking the ground-truth curve.
- Run `python demo_feedback.py` and inspect `results_feedback.png` — the control phase panel should show the state converging to the target.

When making changes, regenerate both output images and confirm they look correct before committing.

---

## Git Workflow

The project uses a `master` main branch. Feature development is done on `claude/` prefixed branches.

```bash
# Standard workflow
git checkout -b claude/<feature-name>
# ... make changes ...
git add <specific files>
git commit -m "Clear, descriptive message"
git push -u origin claude/<feature-name>
```

Do not push directly to `master`.

---

## What This Project Is NOT

- Not a production system — no error handling for malformed user input beyond argparse.
- Not a library — no `__init__.py`, no installable package.
- Not tested with CI — all validation is manual and visual.
- Not designed for large-scale data — streams are small (50–200 points), fitting is iterative.
