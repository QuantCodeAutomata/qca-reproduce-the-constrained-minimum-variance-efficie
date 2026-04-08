"""
Experiment 2: Factor-Model Prior Integration with Ridge-Estimated Loadings
==========================================================================
Reproduces the paper's factor-prior portfolio optimization example by:
  - Loading the built-in asset and factor price datasets
  - Aligning them on their common date intersection
  - Converting both to returns
  - Fitting a MeanRisk minimum-variance optimizer with a FactorModel prior
    using LoadingMatrixRegression with ridge regression alpha = 0.1
  - Preserving chronological train-test separation

# Using skfolio FactorModel, LoadingMatrixRegression, MeanRisk — Context7 confirmed
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfolio
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
from skfolio.optimization import MeanRisk, ObjectiveFunction
from skfolio.preprocessing import prices_to_returns
from skfolio.prior import FactorModel, LoadingMatrixRegression

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
AAPL_MAX_WEIGHT: float = 0.2
L2_COEF: float = 0.01
RIDGE_ALPHA: float = 0.1
TEST_SIZE: float = 0.2


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_prepare_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]
]:
    """Load and align asset and factor price datasets, then convert to returns.

    Returns
    -------
    X_train : pd.DataFrame
        Training asset returns.
    X_test : pd.DataFrame
        Test asset returns.
    y_train : pd.DataFrame
        Training factor returns.
    y_test : pd.DataFrame
        Test factor returns.
    metadata : dict
        Dataset metadata.
    """
    asset_prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()

    asset_meta = {
        "asset_start": str(asset_prices.index[0].date()),
        "asset_end": str(asset_prices.index[-1].date()),
        "n_assets": asset_prices.shape[1],
        "asset_tickers": list(asset_prices.columns),
        "n_asset_obs": len(asset_prices),
    }
    factor_meta = {
        "factor_start": str(factor_prices.index[0].date()),
        "factor_end": str(factor_prices.index[-1].date()),
        "n_factors": factor_prices.shape[1],
        "factor_names": list(factor_prices.columns),
        "n_factor_obs": len(factor_prices),
    }

    logger.info("Asset dataset: %s to %s, %d obs, %d assets",
                asset_meta["asset_start"], asset_meta["asset_end"],
                asset_meta["n_asset_obs"], asset_meta["n_assets"])
    logger.info("Factor dataset: %s to %s, %d obs, %d factors",
                factor_meta["factor_start"], factor_meta["factor_end"],
                factor_meta["n_factor_obs"], factor_meta["n_factors"])

    if "AAPL" not in asset_prices.columns:
        raise ValueError("AAPL is absent from the asset dataset.")

    # Convert aligned prices to returns using the joint skfolio routine.
    # prices_to_returns accepts (prices, factor_prices) and returns (X, y)
    # aligned on their common date intersection.
    # Using skfolio prices_to_returns joint conversion — Context7 confirmed
    X, y = prices_to_returns(asset_prices, factor_prices)

    overlap_start = str(X.index[0].date())
    overlap_end = str(X.index[-1].date())
    logger.info("Aligned overlap: %s to %s, %d obs", overlap_start, overlap_end, len(X))

    # Chronological 80/20 split — same dates for both X and y
    X_train, X_test = train_test_split(X, test_size=TEST_SIZE, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=TEST_SIZE, shuffle=False)

    # Verify date alignment
    assert list(X_train.index) == list(y_train.index), "Train date mismatch"
    assert list(X_test.index) == list(y_test.index), "Test date mismatch"

    metadata = {
        **asset_meta,
        **factor_meta,
        "overlap_start": overlap_start,
        "overlap_end": overlap_end,
        "n_overlap_obs": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_start": str(X_train.index[0].date()),
        "train_end": str(X_train.index[-1].date()),
        "test_start": str(X_test.index[0].date()),
        "test_end": str(X_test.index[-1].date()),
        "skfolio_version": skfolio.__version__,
        "aapl_present": "AAPL" in X.columns,
    }

    logger.info("Train: %s to %s (%d obs)", metadata["train_start"],
                metadata["train_end"], metadata["n_train"])
    logger.info("Test:  %s to %s (%d obs)", metadata["test_start"],
                metadata["test_end"], metadata["n_test"])

    return X_train, X_test, y_train, y_test, metadata


# ── Model configuration & fitting ────────────────────────────────────────────

def build_factor_model() -> MeanRisk:
    """Build the MeanRisk optimizer with a FactorModel prior.

    Configuration (paper-specified):
      - LoadingMatrixRegression with Ridge(alpha=0.1)
      - FactorModel prior using the ridge loading estimator
      - Objective: minimize risk (variance)
      - Risk measure: variance
      - AAPL max weight: 0.2
      - L2 regularization coefficient: 0.01

    Returns
    -------
    MeanRisk
        Configured but unfitted optimizer.
    """
    # Ridge regression for loading matrix estimation
    # Using sklearn Ridge — Context7 confirmed
    ridge_regressor = Ridge(alpha=RIDGE_ALPHA)

    # Loading matrix estimator using ridge regression
    # Using skfolio LoadingMatrixRegression — Context7 confirmed
    loading_estimator = LoadingMatrixRegression(linear_regressor=ridge_regressor)

    # Factor model prior
    # Using skfolio FactorModel — Context7 confirmed
    factor_prior = FactorModel(loading_matrix_estimator=loading_estimator)

    # MeanRisk optimizer with factor prior
    # Using skfolio MeanRisk — Context7 confirmed
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        prior_estimator=factor_prior,
        max_weights={"AAPL": AAPL_MAX_WEIGHT},
        l2_coef=L2_COEF,
        portfolio_params=dict(name="Factor-Prior Min-Variance", tag="Train"),
    )

    logger.info("Factor model config: ridge_alpha=%s, l2_coef=%s, AAPL_max=%.2f",
                RIDGE_ALPHA, L2_COEF, AAPL_MAX_WEIGHT)
    logger.info("Solver: %s", model.solver)
    logger.info("Min weights: %s", model.min_weights)

    return model


def fit_model(
    model: MeanRisk,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> MeanRisk:
    """Fit the factor-prior optimizer on training data only.

    Parameters
    ----------
    model : MeanRisk
        Configured optimizer.
    X_train : pd.DataFrame
        Training asset returns.
    y_train : pd.DataFrame
        Training factor returns.

    Returns
    -------
    MeanRisk
        Fitted optimizer.
    """
    model.fit(X_train, y_train)
    logger.info("Factor model fitted. Weights shape: %s", model.weights_.shape)
    return model


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
        Training return matrix.

    Returns
    -------
    dict
        Validation results.
    """
    weights = model.weights_
    columns = list(X_train.columns)
    aapl_idx = columns.index("AAPL")

    # Handle both single portfolio (1D) and frontier (2D) weights
    if weights.ndim == 1:
        weights_2d = weights.reshape(1, -1)
    else:
        weights_2d = weights

    weight_sums = weights_2d.sum(axis=1)
    budget_ok = bool(np.allclose(weight_sums, 1.0, atol=1e-4))
    aapl_weights = weights_2d[:, aapl_idx]
    aapl_ok = bool(np.all(aapl_weights <= AAPL_MAX_WEIGHT + 1e-6))

    results = {
        "n_portfolios": weights_2d.shape[0],
        "budget_constraint_ok": budget_ok,
        "max_weight_sum": float(weight_sums.max()),
        "min_weight_sum": float(weight_sums.min()),
        "aapl_max_weight_observed": float(aapl_weights.max()),
        "aapl_constraint_ok": aapl_ok,
    }

    logger.info("Validation: n_portfolios=%d, budget_ok=%s, "
                "AAPL_max=%.4f (limit %.2f), aapl_ok=%s",
                results["n_portfolios"],
                results["budget_constraint_ok"],
                results["aapl_max_weight_observed"], AAPL_MAX_WEIGHT,
                results["aapl_constraint_ok"])

    assert results["budget_constraint_ok"], (
        f"Budget constraint violated: weight sums range "
        f"[{results['min_weight_sum']:.6f}, {results['max_weight_sum']:.6f}]"
    )
    assert results["aapl_constraint_ok"], (
        f"AAPL weight cap violated: max observed = {results['aapl_max_weight_observed']:.4f}"
    )

    return results


# ── Portfolio summary ─────────────────────────────────────────────────────────

def get_portfolio_summary(
    model: MeanRisk,
    X: pd.DataFrame,
    label: str,
) -> pd.DataFrame:
    """Generate portfolio summary statistics.

    Parameters
    ----------
    model : MeanRisk
        Fitted optimizer.
    X : pd.DataFrame
        Return matrix to evaluate on.
    label : str
        Label for logging.

    Returns
    -------
    pd.DataFrame
        Summary statistics.
    """
    portfolio = model.predict(X)
    logger.info("%s portfolio: annualized_mean=%.4f, annualized_std=%.4f",
                label,
                portfolio.annualized_mean,
                portfolio.annualized_standard_deviation)

    summary = pd.DataFrame({
        "metric": [
            "annualized_mean",
            "annualized_std",
            "annualized_sharpe_ratio",
        ],
        "value": [
            portfolio.annualized_mean,
            portfolio.annualized_standard_deviation,
            portfolio.annualized_sharpe_ratio,
        ],
    })
    return summary


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_factor_weights(
    model: MeanRisk,
    X_train: pd.DataFrame,
    output_path: str,
) -> None:
    """Plot the portfolio weights from the factor-prior model.

    Parameters
    ----------
    model : MeanRisk
        Fitted optimizer.
    X_train : pd.DataFrame
        Training return matrix (for column names).
    output_path : str
        File path to save the figure.
    """
    weights = model.weights_
    if weights.ndim == 2:
        weights = weights[0]  # use first portfolio if frontier

    assets = list(X_train.columns)
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["steelblue" if a != "AAPL" else "tomato" for a in assets]
    ax.bar(assets, weights, color=colors)
    ax.axhline(AAPL_MAX_WEIGHT, color="red", linestyle="--", linewidth=1.5,
               label=f"AAPL max weight ({AAPL_MAX_WEIGHT})")
    ax.set_xlabel("Asset", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.set_title(
        f"Factor-Prior Min-Variance Portfolio Weights\n"
        f"(Ridge α={RIDGE_ALPHA}, L2 coef={L2_COEF}, AAPL max={AAPL_MAX_WEIGHT})",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Factor weights plot saved to %s", output_path)


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(results_dir: str = "results") -> dict[str, Any]:
    """Run the full Experiment 2 pipeline.

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

    # 1. Load and align data
    X_train, X_test, y_train, y_test, metadata = load_and_prepare_data()

    # 2. Build and fit factor-prior model
    model = build_factor_model()
    model = fit_model(model, X_train, y_train)

    # 3. Validate constraints
    validation = validate_constraints(model, X_train)

    # 4. Generate portfolio summaries
    train_summary = get_portfolio_summary(model, X_train, "Train")
    test_summary = get_portfolio_summary(model, X_test, "Test")

    # 5. Plot weights
    plot_path = os.path.join(results_dir, "exp2_factor_weights.png")
    plot_factor_weights(model, X_train, plot_path)

    # 6. Save summaries
    train_summary.to_csv(
        os.path.join(results_dir, "exp2_train_summary.csv"), index=False
    )
    test_summary.to_csv(
        os.path.join(results_dir, "exp2_test_summary.csv"), index=False
    )

    results = {
        "metadata": metadata,
        "validation": validation,
        "train_summary": train_summary.to_dict("records"),
        "test_summary": test_summary.to_dict("records"),
        "plot_path": plot_path,
        "fit_successful": True,
    }

    logger.info("Experiment 2 complete.")
    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Experiment 2 Results ===")
    print(f"Overlap period: {results['metadata']['overlap_start']} to "
          f"{results['metadata']['overlap_end']}")
    print(f"Train obs: {results['metadata']['n_train']}, "
          f"Test obs: {results['metadata']['n_test']}")
    print(f"Factors: {results['metadata']['factor_names']}")
    print(f"Budget constraint OK: {results['validation']['budget_constraint_ok']}")
    print(f"AAPL constraint OK: {results['validation']['aapl_constraint_ok']}")
    print(f"AAPL max observed weight: {results['validation']['aapl_max_weight_observed']:.4f}")
    print(f"Fit successful: {results['fit_successful']}")
    print("\nTrain summary:")
    for row in results["train_summary"]:
        print(f"  {row['metric']}: {row['value']:.4f}")
