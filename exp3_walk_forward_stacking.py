"""
Experiment 3: Walk-Forward Stacking Optimization vs Equal-Weight Benchmark
===========================================================================
Reproduces the paper's walk-forward portfolio evaluation example using:
  - A stacking ensemble of three base estimators:
      (a) Inverse Volatility
      (b) MeanRisk with CVaR
      (c) Hierarchical Risk Parity
  - A CVaR-based MeanRisk final estimator
  - WalkForward cross-validator: train_size=252, test_size=60
  - Equal-weight benchmark evaluated on identical folds

Validates that time-series-aware validation is appropriate for portfolio
strategies and that stacking can be integrated into the unified framework.

# Using skfolio StackingOptimization, WalkForward, cross_val_predict — Context7 confirmed
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfolio
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.model_selection import WalkForward, cross_val_predict
from skfolio.optimization import (
    EqualWeighted,
    HierarchicalRiskParity,
    InverseVolatility,
    MeanRisk,
    ObjectiveFunction,
    StackingOptimization,
)
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
TRAIN_SIZE: int = 252   # observations per training window
TEST_SIZE_WF: int = 60  # observations per test window
OUTER_TEST_SIZE: float = 0.2  # outer chronological split ratio
MIN_REQUIRED_OBS: int = TRAIN_SIZE + TEST_SIZE_WF  # minimum for one fold


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_prepare_data(
    use_outer_split: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the built-in S&P 500 dataset and prepare the return matrix.

    Parameters
    ----------
    use_outer_split : bool
        If True, apply an outer 80/20 chronological split and use the test
        portion for walk-forward (matching the paper's literal workflow).
        If False, apply walk-forward to the full return matrix.

    Returns
    -------
    X : pd.DataFrame
        Return matrix to use for walk-forward evaluation.
    metadata : dict
        Dataset metadata.
    """
    prices = load_sp500_dataset()
    X_full = prices_to_returns(prices)

    metadata: dict[str, Any] = {
        "full_start": str(X_full.index[0].date()),
        "full_end": str(X_full.index[-1].date()),
        "n_full_obs": len(X_full),
        "n_assets": X_full.shape[1],
        "asset_tickers": list(X_full.columns),
        "skfolio_version": skfolio.__version__,
        "use_outer_split": use_outer_split,
    }

    logger.info("Full dataset: %s to %s, %d obs, %d assets",
                metadata["full_start"], metadata["full_end"],
                metadata["n_full_obs"], metadata["n_assets"])

    if use_outer_split:
        # Paper's literal workflow: outer split, then walk-forward on X_test
        _, X = train_test_split(X_full, test_size=OUTER_TEST_SIZE, shuffle=False)
        metadata["wf_sample"] = "outer_test_split"
        logger.info("Using outer test split for walk-forward (paper literal workflow)")
    else:
        X = X_full
        metadata["wf_sample"] = "full_sample"
        logger.info("Using full sample for walk-forward")

    if len(X) < MIN_REQUIRED_OBS:
        raise ValueError(
            f"Insufficient observations ({len(X)}) for walk-forward with "
            f"train_size={TRAIN_SIZE}, test_size={TEST_SIZE_WF}. "
            f"Need at least {MIN_REQUIRED_OBS}."
        )

    metadata["wf_start"] = str(X.index[0].date())
    metadata["wf_end"] = str(X.index[-1].date())
    metadata["n_wf_obs"] = len(X)

    logger.info("Walk-forward sample: %s to %s (%d obs)",
                metadata["wf_start"], metadata["wf_end"], metadata["n_wf_obs"])

    return X, metadata


# ── Estimator construction ────────────────────────────────────────────────────

def build_stacking_model() -> StackingOptimization:
    """Build the stacking optimization model.

    Base estimators (paper-specified):
      (a) Inverse Volatility
      (b) MeanRisk with CVaR
      (c) Hierarchical Risk Parity

    Final estimator (paper-specified):
      MeanRisk with CVaR

    Returns
    -------
    StackingOptimization
        Configured stacking model.
    """
    # Base estimators — Using skfolio InverseVolatility, MeanRisk, HRP — Context7 confirmed
    base_estimators = [
        ("inverse_vol", InverseVolatility()),
        ("mean_cvar", MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
            portfolio_params=dict(name="MeanRisk-CVaR", tag="base"),
        )),
        ("hrp", HierarchicalRiskParity(
            portfolio_params=dict(name="HRP", tag="base"),
        )),
    ]

    # Final estimator — Using skfolio MeanRisk with CVaR — Context7 confirmed
    final_estimator = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.CVAR,
        portfolio_params=dict(name="Stacking-Final", tag="final"),
    )

    # Using skfolio StackingOptimization — Context7 confirmed
    model = StackingOptimization(
        estimators=base_estimators,
        final_estimator=final_estimator,
        portfolio_params=dict(name="Stacking", tag="stacking"),
    )

    logger.info("Stacking model: 3 base estimators + MeanRisk(CVaR) final")
    logger.info("Base estimators: %s", [name for name, _ in base_estimators])

    return model


def build_benchmark() -> EqualWeighted:
    """Build the equal-weight benchmark estimator.

    Returns
    -------
    EqualWeighted
        Equal-weight portfolio estimator.
    """
    # Using skfolio EqualWeighted — Context7 confirmed
    benchmark = EqualWeighted(
        portfolio_params=dict(name="Equal-Weight", tag="benchmark"),
    )
    logger.info("Benchmark: EqualWeighted")
    return benchmark


# ── Walk-forward evaluation ───────────────────────────────────────────────────

def run_walk_forward(
    model: Any,
    X: pd.DataFrame,
    cv: WalkForward,
    label: str,
) -> Any:
    """Run walk-forward cross-validated prediction.

    Parameters
    ----------
    model : estimator
        Portfolio optimizer.
    X : pd.DataFrame
        Return matrix.
    cv : WalkForward
        Walk-forward cross-validator.
    label : str
        Label for logging.

    Returns
    -------
    MultiPeriodPortfolio or Population
        Cross-validated portfolio predictions.
    """
    logger.info("Running walk-forward for %s ...", label)
    # Using skfolio cross_val_predict — Context7 confirmed
    pred = cross_val_predict(model, X, cv=cv)
    logger.info("%s walk-forward complete. Type: %s", label, type(pred).__name__)
    return pred


# ── Fold validation ───────────────────────────────────────────────────────────

def validate_folds(X: pd.DataFrame, cv: WalkForward) -> dict[str, Any]:
    """Validate that walk-forward folds are strictly chronological.

    Parameters
    ----------
    X : pd.DataFrame
        Return matrix.
    cv : WalkForward
        Walk-forward cross-validator.

    Returns
    -------
    dict
        Fold validation results.
    """
    folds = list(cv.split(X))
    n_folds = len(folds)

    fold_info = []
    for i, (train_idx, test_idx) in enumerate(folds):
        train_end = X.index[train_idx[-1]]
        test_start = X.index[test_idx[0]]
        test_end = X.index[test_idx[-1]]
        no_leakage = train_end < test_start
        fold_info.append({
            "fold": i,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "train_end": str(train_end.date()),
            "test_start": str(test_start.date()),
            "test_end": str(test_end.date()),
            "no_leakage": no_leakage,
        })

    all_no_leakage = all(f["no_leakage"] for f in fold_info)

    results = {
        "n_folds": n_folds,
        "all_folds_no_leakage": all_no_leakage,
        "fold_details": fold_info,
    }

    logger.info("Walk-forward: %d folds, all no-leakage: %s", n_folds, all_no_leakage)
    assert all_no_leakage, "Look-ahead leakage detected in walk-forward folds!"

    return results


# ── Population summary ────────────────────────────────────────────────────────

def get_population_summary(
    stacking_pred: Any,
    benchmark_pred: Any,
) -> pd.DataFrame:
    """Aggregate predictions into a population and generate summary.

    Parameters
    ----------
    stacking_pred : MultiPeriodPortfolio
        Stacking model walk-forward predictions.
    benchmark_pred : MultiPeriodPortfolio
        Benchmark walk-forward predictions.

    Returns
    -------
    pd.DataFrame
        Population summary.
    """
    # Using skfolio Population — Context7 confirmed
    population = Population([stacking_pred, benchmark_pred])
    summary = population.summary()
    logger.info("Population summary generated.")
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_cumulative_returns(
    stacking_pred: Any,
    benchmark_pred: Any,
    output_path: str,
) -> None:
    """Plot cumulative returns for stacking vs benchmark.

    Parameters
    ----------
    stacking_pred : MultiPeriodPortfolio
        Stacking model predictions.
    benchmark_pred : MultiPeriodPortfolio
        Benchmark predictions.
    output_path : str
        File path to save the figure.
    """
    stacking_returns = np.asarray(stacking_pred)
    benchmark_returns = np.asarray(benchmark_pred)

    stacking_cum = np.cumprod(1 + stacking_returns)
    benchmark_cum = np.cumprod(1 + benchmark_returns)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(stacking_cum, label="Stacking (InvVol + MeanRisk-CVaR + HRP)",
            color="steelblue", linewidth=2)
    ax.plot(benchmark_cum, label="Equal-Weight Benchmark",
            color="tomato", linestyle="--", linewidth=2)
    ax.set_xlabel("Walk-Forward Test Observations", fontsize=12)
    ax.set_ylabel("Cumulative Return (1 = start)", fontsize=12)
    ax.set_title(
        f"Walk-Forward Stacking vs Equal-Weight Benchmark\n"
        f"(train_size={TRAIN_SIZE}, test_size={TEST_SIZE_WF})",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Cumulative returns plot saved to %s", output_path)


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(results_dir: str = "results") -> dict[str, Any]:
    """Run the full Experiment 3 pipeline.

    Parameters
    ----------
    results_dir : str
        Directory to save output artifacts.

    Returns
    -------
    dict
        All experiment results and metadata.
    """
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    X, metadata = load_and_prepare_data(use_outer_split=True)

    # 2. Build estimators
    stacking_model = build_stacking_model()
    benchmark_model = build_benchmark()

    # 3. Define walk-forward cross-validator
    # Using skfolio WalkForward — Context7 confirmed
    cv = WalkForward(train_size=TRAIN_SIZE, test_size=TEST_SIZE_WF)

    # 4. Validate folds
    fold_validation = validate_folds(X, cv)

    # 5. Run walk-forward for stacking model
    stacking_pred = run_walk_forward(stacking_model, X, cv, "Stacking")

    # 6. Run walk-forward for benchmark (same folds)
    benchmark_pred = run_walk_forward(benchmark_model, X, cv, "EqualWeight")

    # 7. Generate population summary
    summary = get_population_summary(stacking_pred, benchmark_pred)

    # 8. Plot cumulative returns
    plot_path = os.path.join(results_dir, "exp3_walk_forward_cumulative.png")
    plot_cumulative_returns(stacking_pred, benchmark_pred, plot_path)

    # 9. Extract key metrics
    stacking_returns = np.asarray(stacking_pred)
    benchmark_returns = np.asarray(benchmark_pred)

    results = {
        "metadata": metadata,
        "fold_validation": {
            "n_folds": fold_validation["n_folds"],
            "all_no_leakage": fold_validation["all_folds_no_leakage"],
        },
        "stacking_metrics": {
            "annualized_sharpe": float(stacking_pred.annualized_sharpe_ratio),
            "annualized_mean": float(stacking_pred.annualized_mean),
            "annualized_std": float(stacking_pred.annualized_standard_deviation),
            "n_observations": len(stacking_returns),
        },
        "benchmark_metrics": {
            "annualized_sharpe": float(benchmark_pred.annualized_sharpe_ratio),
            "annualized_mean": float(benchmark_pred.annualized_mean),
            "annualized_std": float(benchmark_pred.annualized_standard_deviation),
            "n_observations": len(benchmark_returns),
        },
        "population_summary": summary.to_dict() if hasattr(summary, "to_dict") else str(summary),
        "plot_path": plot_path,
    }

    # Save summary
    if hasattr(summary, "to_csv"):
        summary.to_csv(os.path.join(results_dir, "exp3_population_summary.csv"))
    else:
        pd.DataFrame({"summary": [str(summary)]}).to_csv(
            os.path.join(results_dir, "exp3_population_summary.csv"), index=False
        )

    logger.info("Experiment 3 complete.")
    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Experiment 3 Results ===")
    print(f"Walk-forward sample: {results['metadata']['wf_start']} to "
          f"{results['metadata']['wf_end']}")
    print(f"Number of folds: {results['fold_validation']['n_folds']}")
    print(f"All folds no-leakage: {results['fold_validation']['all_no_leakage']}")
    print(f"\nStacking model:")
    for k, v in results["stacking_metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nEqual-weight benchmark:")
    for k, v in results["benchmark_metrics"].items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
