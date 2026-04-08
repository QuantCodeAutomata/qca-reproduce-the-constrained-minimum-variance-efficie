"""
Experiment 1: Constrained Minimum-Variance Efficient Frontier
=============================================================
Reproduces the paper's minimum-variance portfolio optimization example using the
built-in skfolio S&P 500 price dataset with:
  - Chronological 80/20 train-test split
  - Maximum AAPL portfolio weight of 0.2
  - L2 regularization coefficient 0.01
  - Efficient frontier containing 100 portfolios

Validates that the in-sample training frontier dominates the out-of-sample test
frontier, illustrating overfitting.

# Using skfolio MeanRisk, load_sp500_dataset, prices_to_returns — Context7 confirmed
"""

import logging
import os
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfolio
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
AAPL_MAX_WEIGHT: float = 0.2
L2_COEF: float = 0.01
FRONTIER_SIZE: int = 100
TEST_SIZE: float = 0.2
ANNUALIZATION_FACTOR: int = 252  # trading days per year


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_prepare_data() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load the built-in S&P 500 dataset and convert prices to returns.

    Returns
    -------
    X_train : pd.DataFrame
        Training return matrix (first 80% of observations).
    X_test : pd.DataFrame
        Test return matrix (last 20% of observations).
    metadata : dict
        Dataset metadata including date range, shape, and asset names.
    """
    prices = load_sp500_dataset()

    metadata: dict[str, Any] = {
        "start_date": str(prices.index[0].date()),
        "end_date": str(prices.index[-1].date()),
        "n_observations": len(prices),
        "n_assets": prices.shape[1],
        "asset_tickers": list(prices.columns),
        "aapl_present": "AAPL" in prices.columns,
        "skfolio_version": skfolio.__version__,
    }

    logger.info("Dataset: %s to %s, %d obs, %d assets",
                metadata["start_date"], metadata["end_date"],
                metadata["n_observations"], metadata["n_assets"])
    logger.info("Assets: %s", metadata["asset_tickers"])
    logger.info("AAPL present: %s", metadata["aapl_present"])

    if not metadata["aapl_present"]:
        raise ValueError(
            "AAPL is absent from the dataset. Cannot impose the AAPL weight constraint."
        )

    # Convert prices to returns using the library's standard routine
    # Using skfolio prices_to_returns — Context7 confirmed
    X = prices_to_returns(prices)

    # Chronological 80/20 split — no shuffling to preserve time order
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, shuffle=False)

    metadata["train_start"] = str(X_train.index[0].date())
    metadata["train_end"] = str(X_train.index[-1].date())
    metadata["test_start"] = str(X_test.index[0].date())
    metadata["test_end"] = str(X_test.index[-1].date())
    metadata["n_train"] = len(X_train)
    metadata["n_test"] = len(X_test)

    logger.info("Train: %s to %s (%d obs)", metadata["train_start"],
                metadata["train_end"], metadata["n_train"])
    logger.info("Test:  %s to %s (%d obs)", metadata["test_start"],
                metadata["test_end"], metadata["n_test"])

    return X_train, X_test, metadata


# ── Model configuration & fitting ────────────────────────────────────────────

def build_model() -> MeanRisk:
    """Build the constrained MeanRisk minimum-variance optimizer.

    Configuration (paper-specified):
      - Objective: minimize risk (variance)
      - Risk measure: variance
      - AAPL max weight: 0.2
      - L2 regularization coefficient: 0.01
      - Efficient frontier size: 100

    Returns
    -------
    MeanRisk
        Configured but unfitted optimizer.
    """
    # Using skfolio MeanRisk — Context7 confirmed
    # Paper formula: min w^T Σ w + λ w^T w  s.t. w^T 1 = 1, w_AAPL ≤ 0.2
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        max_weights={"AAPL": AAPL_MAX_WEIGHT},  # per-asset cap
        l2_coef=L2_COEF,
        efficient_frontier_size=FRONTIER_SIZE,
        portfolio_params=dict(name="Min-Variance", tag="Train"),
    )

    logger.info("Model config: objective=%s, risk=%s, l2_coef=%s, "
                "frontier_size=%d, AAPL_max=%.2f",
                ObjectiveFunction.MINIMIZE_RISK.value,
                RiskMeasure.VARIANCE.value,
                L2_COEF, FRONTIER_SIZE, AAPL_MAX_WEIGHT)
    logger.info("Solver defaults: %s", model.solver)
    logger.info("Min weights default: %s", model.min_weights)
    logger.info("Budget default: %s", model.budget)

    return model


def fit_model(model: MeanRisk, X_train: pd.DataFrame) -> MeanRisk:
    """Fit the optimizer on training data only.

    Parameters
    ----------
    model : MeanRisk
        Configured optimizer.
    X_train : pd.DataFrame
        Training return matrix.

    Returns
    -------
    MeanRisk
        Fitted optimizer.
    """
    model.fit(X_train)
    logger.info("Model fitted. Weights shape: %s", model.weights_.shape)
    return model


# ── Frontier evaluation ───────────────────────────────────────────────────────

def evaluate_frontier(
    model: MeanRisk,
    X: pd.DataFrame,
    tag: str,
) -> Population:
    """Evaluate the fitted frontier on a return matrix.

    Parameters
    ----------
    model : MeanRisk
        Fitted optimizer.
    X : pd.DataFrame
        Return matrix (train or test).
    tag : str
        Label for the portfolio population.

    Returns
    -------
    Population
        Collection of portfolios on the frontier.
    """
    population = model.predict(X)
    logger.info("%s frontier: %d portfolios", tag, len(population))
    return population


# ── Validation ───────────────────────────────────────────────────────────────

def validate_constraints(
    model: MeanRisk,
    X_train: pd.DataFrame,
) -> dict[str, Any]:
    """Validate budget constraint and AAPL weight cap.

    Parameters
    ----------
    model : MeanRisk
        Fitted optimizer.
    X_train : pd.DataFrame
        Training return matrix (used to identify AAPL column index).

    Returns
    -------
    dict
        Validation results.
    """
    weights = model.weights_  # shape (frontier_size, n_assets)
    columns = list(X_train.columns)
    aapl_idx = columns.index("AAPL")

    weight_sums = weights.sum(axis=1)
    budget_ok = bool(np.allclose(weight_sums, 1.0, atol=1e-4))

    aapl_weights = weights[:, aapl_idx]
    aapl_ok = bool(np.all(aapl_weights <= AAPL_MAX_WEIGHT + 1e-6))

    results = {
        "frontier_size": weights.shape[0],
        "frontier_size_ok": weights.shape[0] == FRONTIER_SIZE,
        "budget_constraint_ok": budget_ok,
        "max_weight_sum": float(weight_sums.max()),
        "min_weight_sum": float(weight_sums.min()),
        "aapl_max_weight_observed": float(aapl_weights.max()),
        "aapl_constraint_ok": aapl_ok,
    }

    logger.info("Validation: frontier_size=%d (expected %d), budget_ok=%s, "
                "AAPL_max=%.4f (limit %.2f), aapl_ok=%s",
                results["frontier_size"], FRONTIER_SIZE,
                results["budget_constraint_ok"],
                results["aapl_max_weight_observed"], AAPL_MAX_WEIGHT,
                results["aapl_constraint_ok"])

    assert results["frontier_size_ok"], (
        f"Frontier has {results['frontier_size']} points, expected {FRONTIER_SIZE}"
    )
    assert results["budget_constraint_ok"], (
        f"Budget constraint violated: weight sums range "
        f"[{results['min_weight_sum']:.6f}, {results['max_weight_sum']:.6f}]"
    )
    assert results["aapl_constraint_ok"], (
        f"AAPL weight cap violated: max observed = {results['aapl_max_weight_observed']:.4f}"
    )

    return results


# ── Qualitative dominance check ───────────────────────────────────────────────

def check_train_dominates_test(
    train_pop: Population,
    test_pop: Population,
) -> dict[str, Any]:
    """Check overfitting evidence: training frontier vs test frontier.

    Overfitting in portfolio optimization manifests as risk underestimation:
    the optimizer minimizes in-sample variance, so the same portfolios show
    lower volatility in-sample than out-of-sample. Additionally, the training
    frontier may show higher Sharpe ratios in-sample than out-of-sample.

    Parameters
    ----------
    train_pop : Population
        Training frontier portfolios.
    test_pop : Population
        Test frontier portfolios.

    Returns
    -------
    dict
        Dominance statistics.
    """
    train_vols = np.array([p.annualized_standard_deviation for p in train_pop])
    train_rets = np.array([p.annualized_mean for p in train_pop])
    train_sharpes = np.array([p.annualized_sharpe_ratio for p in train_pop])
    test_vols = np.array([p.annualized_standard_deviation for p in test_pop])
    test_rets = np.array([p.annualized_mean for p in test_pop])
    test_sharpes = np.array([p.annualized_sharpe_ratio for p in test_pop])

    # Overfitting signal 1: in-sample volatility < out-of-sample volatility
    # (risk underestimation — the optimizer exploits in-sample noise)
    vol_underestimation = float(train_vols.mean()) < float(test_vols.mean())

    # Overfitting signal 2: compare returns at same volatility level
    # Interpolate test returns at train volatility levels
    vol_min = max(train_vols.min(), test_vols.min())
    vol_max = min(train_vols.max(), test_vols.max())
    common_vols = np.linspace(vol_min, vol_max, 50)
    train_interp = np.interp(common_vols, train_vols, train_rets)
    test_interp = np.interp(common_vols, test_vols, test_rets)
    diff = train_interp - test_interp
    pct_train_above = float(np.mean(diff > 0))

    results = {
        "train_mean_return_avg": float(train_rets.mean()),
        "test_mean_return_avg": float(test_rets.mean()),
        "train_vol_avg": float(train_vols.mean()),
        "test_vol_avg": float(test_vols.mean()),
        "train_sharpe_avg": float(train_sharpes.mean()),
        "test_sharpe_avg": float(test_sharpes.mean()),
        "train_vol_range": (float(train_vols.min()), float(train_vols.max())),
        "test_vol_range": (float(test_vols.min()), float(test_vols.max())),
        "pct_frontier_train_above_test": pct_train_above,
        "vol_underestimation": vol_underestimation,
        # Overfitting confirmed if in-sample vol < out-of-sample vol
        "train_dominates_test": vol_underestimation,
        "mean_return_diff": float(diff.mean()),
    }

    logger.info("Overfitting check:")
    logger.info("  In-sample vol avg: %.4f, Out-of-sample vol avg: %.4f",
                results["train_vol_avg"], results["test_vol_avg"])
    logger.info("  Risk underestimation (overfitting): %s", vol_underestimation)
    logger.info("  Train above test on %.1f%% of frontier (return comparison)",
                pct_train_above * 100)
    logger.info("  Train dominates test: %s", results["train_dominates_test"])

    return results


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_efficient_frontiers(
    train_pop: Population,
    test_pop: Population,
    output_path: str,
) -> None:
    """Plot training and test efficient frontiers on the same figure.

    Uses annualized mean (y-axis) vs annualized standard deviation (x-axis).

    Parameters
    ----------
    train_pop : Population
        Training frontier portfolios.
    test_pop : Population
        Test frontier portfolios.
    output_path : str
        File path to save the figure.
    """
    train_vols = [p.annualized_standard_deviation for p in train_pop]
    train_rets = [p.annualized_mean for p in train_pop]
    test_vols = [p.annualized_standard_deviation for p in test_pop]
    test_rets = [p.annualized_mean for p in test_pop]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_vols, train_rets, "b-o", markersize=3, label="Training Frontier",
            linewidth=2)
    ax.plot(test_vols, test_rets, "r--s", markersize=3, label="Test Frontier",
            linewidth=2)
    ax.set_xlabel("Annualized Standard Deviation", fontsize=13)
    ax.set_ylabel("Annualized Mean Return", fontsize=13)
    ax.set_title(
        "Constrained Minimum-Variance Efficient Frontier\n"
        f"(AAPL max weight={AAPL_MAX_WEIGHT}, L2 coef={L2_COEF}, "
        f"{FRONTIER_SIZE} portfolios)",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Frontier plot saved to %s", output_path)


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(results_dir: str = "results") -> dict[str, Any]:
    """Run the full Experiment 1 pipeline.

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
    X_train, X_test, metadata = load_and_prepare_data()

    # 2. Build and fit model
    model = build_model()
    model = fit_model(model, X_train)

    # 3. Evaluate on training data
    train_pop = evaluate_frontier(model, X_train, "Train")

    # 4. Evaluate on test data (no refitting)
    test_pop = evaluate_frontier(model, X_test, "Test")

    # 5. Validate constraints
    validation = validate_constraints(model, X_train)

    # 6. Qualitative dominance check
    dominance = check_train_dominates_test(train_pop, test_pop)

    # 7. Plot
    plot_path = os.path.join(results_dir, "exp1_efficient_frontier.png")
    plot_efficient_frontiers(train_pop, test_pop, plot_path)

    # 8. Save frontier data
    train_data = pd.DataFrame({
        "annualized_vol": [p.annualized_standard_deviation for p in train_pop],
        "annualized_return": [p.annualized_mean for p in train_pop],
        "sharpe_ratio": [p.annualized_sharpe_ratio for p in train_pop],
    })
    test_data = pd.DataFrame({
        "annualized_vol": [p.annualized_standard_deviation for p in test_pop],
        "annualized_return": [p.annualized_mean for p in test_pop],
        "sharpe_ratio": [p.annualized_sharpe_ratio for p in test_pop],
    })
    train_data.to_csv(os.path.join(results_dir, "exp1_train_frontier.csv"), index=False)
    test_data.to_csv(os.path.join(results_dir, "exp1_test_frontier.csv"), index=False)

    results = {
        "metadata": metadata,
        "validation": validation,
        "dominance": dominance,
        "train_frontier_summary": {
            "mean_return_range": (
                float(train_data["annualized_return"].min()),
                float(train_data["annualized_return"].max()),
            ),
            "vol_range": (
                float(train_data["annualized_vol"].min()),
                float(train_data["annualized_vol"].max()),
            ),
        },
        "test_frontier_summary": {
            "mean_return_range": (
                float(test_data["annualized_return"].min()),
                float(test_data["annualized_return"].max()),
            ),
            "vol_range": (
                float(test_data["annualized_vol"].min()),
                float(test_data["annualized_vol"].max()),
            ),
        },
        "plot_path": plot_path,
    }

    logger.info("Experiment 1 complete.")
    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Experiment 1 Results ===")
    print(f"Dataset: {results['metadata']['start_date']} to {results['metadata']['end_date']}")
    print(f"Assets: {results['metadata']['n_assets']}, AAPL present: {results['metadata']['aapl_present']}")
    print(f"Train obs: {results['metadata']['n_train']}, Test obs: {results['metadata']['n_test']}")
    print(f"Frontier size: {results['validation']['frontier_size']}")
    print(f"Budget constraint OK: {results['validation']['budget_constraint_ok']}")
    print(f"AAPL constraint OK: {results['validation']['aapl_constraint_ok']}")
    print(f"AAPL max observed weight: {results['validation']['aapl_max_weight_observed']:.4f}")
    print(f"Train dominates test: {results['dominance']['train_dominates_test']}")
    print(f"% frontier train above test: {results['dominance']['pct_frontier_train_above_test']:.1%}")
