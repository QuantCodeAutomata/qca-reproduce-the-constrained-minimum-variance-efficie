# Repository Knowledge: qca-reproduce-the-constrained-minimum-variance-efficie

## Overview
This repository reproduces four portfolio optimization experiments from a paper
using the skfolio library with the built-in S&P 500 dataset.

## Environment
- Python 3.10+
- skfolio==0.6.0 (key library)
- copulae==0.7.9 (for copula modeling)
- No VineCopula in skfolio v0.6.0 — use copulae.StudentCopula instead

## Key Findings
- `max_weights` in MeanRisk accepts `dict[str, float]` for per-asset caps
  e.g., `max_weights={"AAPL": 0.2}`
- `prices_to_returns(asset_prices, factor_prices)` returns `(X, y)` tuple
  with automatic date alignment on common intersection
- `WalkForward(train_size=252, test_size=60)` creates strictly forward folds
- `cross_val_predict` returns `MultiPeriodPortfolio` objects
- `Population([p1, p2]).summary()` generates comparison table

## File Structure
- `exp1_efficient_frontier.py` — Exp 1: constrained min-variance frontier
- `exp2_factor_model.py` — Exp 2: factor prior with ridge loadings
- `exp3_walk_forward_stacking.py` — Exp 3: walk-forward stacking vs EW
- `exp4_copula.py` — Exp 4: copula synthetic prior + visualization
- `run_all_experiments.py` — Main runner, saves to results/
- `tests/test_experiments.py` — Comprehensive pytest suite
- `results/RESULTS.md` — Generated metrics and findings

## Running
```bash
python run_all_experiments.py   # run all experiments
pytest tests/ -v                # run all tests
```

## Dataset
- Built-in S&P 500: 8313 rows × 20 columns, 1990-01-02 to 2022-12-28
- All 20 tickers confirmed including AAPL, JPM, LLY
- Factor dataset: load_factors_dataset() from skfolio.datasets

## Copula Workflow
1. `pseudo_obs(returns)` → uniform marginals
2. `StudentCopula(dim=n).fit(u)` → fitted copula
3. `cop.random(n, seed=42)` → uniform samples
4. `np.quantile(hist, u_samples)` → back-transform to return space
