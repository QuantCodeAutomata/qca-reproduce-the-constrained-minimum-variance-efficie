"""
Main runner: Execute all four portfolio optimization experiments and save results.

Experiments:
  1. Constrained minimum-variance efficient frontier (S&P 500, skfolio)
  2. Factor-model prior with ridge-estimated loadings
  3. Walk-forward stacking optimization vs equal-weight benchmark
  4. Copula-based synthetic prior and dependence-structure visualization

Results are saved to the results/ directory, including:
  - results/RESULTS.md  — summary metrics for all experiments
  - results/*.png       — visualization plots
  - results/*.csv       — frontier and summary data
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import skfolio
import copulae

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)

RESULTS_DIR = "results"


def run_with_timing(name: str, func, *args, **kwargs) -> tuple[Any, float]:
    """Run a function and measure elapsed time.

    Parameters
    ----------
    name : str
        Experiment name for logging.
    func : callable
        Function to run.

    Returns
    -------
    result : Any
        Function return value.
    elapsed : float
        Elapsed time in seconds.
    """
    logger.info("=" * 60)
    logger.info("Starting %s", name)
    logger.info("=" * 60)
    start = time.time()
    result = func(*args, **kwargs)
    elapsed = time.time() - start
    logger.info("%s completed in %.1f seconds", name, elapsed)
    return result, elapsed


def format_results_markdown(
    exp1: dict[str, Any],
    exp2: dict[str, Any],
    exp3: dict[str, Any],
    exp4: dict[str, Any],
    timings: dict[str, float],
) -> str:
    """Format all experiment results as a Markdown document.

    Parameters
    ----------
    exp1, exp2, exp3, exp4 : dict
        Results from each experiment.
    timings : dict
        Elapsed time per experiment.

    Returns
    -------
    str
        Markdown-formatted results document.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Experiment Results",
        "",
        f"**Generated:** {now}",
        f"**skfolio version:** {skfolio.__version__}",
        f"**copulae version:** {copulae.__version__}",
        "",
        "---",
        "",
        "## Experiment 1: Constrained Minimum-Variance Efficient Frontier",
        "",
        "### Configuration",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Risk measure | Variance |",
        "| Objective | Minimize risk |",
        "| AAPL max weight | 0.2 |",
        "| L2 regularization | 0.01 |",
        "| Frontier size | 100 |",
        "| Train/test split | 80/20 chronological |",
        "",
        "### Dataset Metadata",
        f"| Field | Value |",
        f"|-------|-------|",
        f"| Start date | {exp1['metadata']['start_date']} |",
        f"| End date | {exp1['metadata']['end_date']} |",
        f"| Total assets | {exp1['metadata']['n_assets']} |",
        f"| Training observations | {exp1['metadata']['n_train']} |",
        f"| Test observations | {exp1['metadata']['n_test']} |",
        f"| AAPL present | {exp1['metadata']['aapl_present']} |",
        "",
        "### Constraint Validation",
        "| Check | Result |",
        "|-------|--------|",
        f"| Frontier size = 100 | {exp1['validation']['frontier_size_ok']} |",
        f"| Budget constraint (weights sum to 1) | {exp1['validation']['budget_constraint_ok']} |",
        f"| AAPL weight ≤ 0.2 | {exp1['validation']['aapl_constraint_ok']} |",
        f"| AAPL max observed weight | {exp1['validation']['aapl_max_weight_observed']:.4f} |",
        "",
        "### Efficient Frontier Statistics",
        "| Metric | Training | Test |",
        "|--------|----------|------|",
        f"| Return range (annualized) | [{exp1['train_frontier_summary']['mean_return_range'][0]:.4f}, {exp1['train_frontier_summary']['mean_return_range'][1]:.4f}] | [{exp1['test_frontier_summary']['mean_return_range'][0]:.4f}, {exp1['test_frontier_summary']['mean_return_range'][1]:.4f}] |",
        f"| Volatility range (annualized) | [{exp1['train_frontier_summary']['vol_range'][0]:.4f}, {exp1['train_frontier_summary']['vol_range'][1]:.4f}] | [{exp1['test_frontier_summary']['vol_range'][0]:.4f}, {exp1['test_frontier_summary']['vol_range'][1]:.4f}] |",
        "",
        "### Overfitting Evidence",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Train avg annualized return | {exp1['dominance']['train_mean_return_avg']:.4f} |",
        f"| Test avg annualized return | {exp1['dominance']['test_mean_return_avg']:.4f} |",
        f"| % frontier where train > test | {exp1['dominance']['pct_frontier_train_above_test']:.1%} |",
        f"| Mean return diff (train - test) | {exp1['dominance']['mean_return_diff']:.4f} |",
        f"| **Train dominates test** | **{exp1['dominance']['train_dominates_test']}** |",
        "",
        f"**Runtime:** {timings['exp1']:.1f}s",
        "",
        "---",
        "",
        "## Experiment 2: Factor-Model Prior with Ridge-Estimated Loadings",
        "",
        "### Configuration",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Ridge alpha | 0.1 |",
        "| Risk measure | Variance |",
        "| Objective | Minimize risk |",
        "| AAPL max weight | 0.2 |",
        "| L2 regularization | 0.01 |",
        "",
        "### Dataset Metadata",
        "| Field | Value |",
        "|-------|-------|",
        f"| Asset universe | {exp2['metadata']['n_assets']} assets |",
        f"| Factors | {', '.join(exp2['metadata']['factor_names'])} |",
        f"| Overlap period | {exp2['metadata']['overlap_start']} to {exp2['metadata']['overlap_end']} |",
        f"| Training observations | {exp2['metadata']['n_train']} |",
        f"| Test observations | {exp2['metadata']['n_test']} |",
        "",
        "### Constraint Validation",
        "| Check | Result |",
        "|-------|--------|",
        f"| Budget constraint | {exp2['validation']['budget_constraint_ok']} |",
        f"| AAPL weight ≤ 0.2 | {exp2['validation']['aapl_constraint_ok']} |",
        f"| AAPL max observed weight | {exp2['validation']['aapl_max_weight_observed']:.4f} |",
        f"| Fit successful | {exp2['fit_successful']} |",
        "",
        "### Portfolio Statistics",
        "| Metric | Training | Test |",
        "|--------|----------|------|",
    ]

    # Extract train/test summary values
    train_metrics = {r["metric"]: r["value"] for r in exp2["train_summary"]}
    test_metrics = {r["metric"]: r["value"] for r in exp2["test_summary"]}
    for metric in ["annualized_mean", "annualized_std", "annualized_sharpe_ratio"]:
        t_val = train_metrics.get(metric, float("nan"))
        te_val = test_metrics.get(metric, float("nan"))
        lines.append(f"| {metric} | {t_val:.4f} | {te_val:.4f} |")

    lines += [
        "",
        f"**Runtime:** {timings['exp2']:.1f}s",
        "",
        "---",
        "",
        "## Experiment 3: Walk-Forward Stacking Optimization vs Equal-Weight Benchmark",
        "",
        "### Configuration",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Base estimators | InverseVolatility, MeanRisk(CVaR), HRP |",
        "| Final estimator | MeanRisk(CVaR) |",
        "| Benchmark | EqualWeighted |",
        "| Walk-forward train size | 252 observations |",
        "| Walk-forward test size | 60 observations |",
        "| Sample | Outer 20% test split |",
        "",
        "### Walk-Forward Validation",
        "| Check | Result |",
        "|-------|--------|",
        f"| Number of folds | {exp3['fold_validation']['n_folds']} |",
        f"| All folds no look-ahead leakage | {exp3['fold_validation']['all_no_leakage']} |",
        "",
        "### Performance Comparison",
        "| Metric | Stacking | Equal-Weight |",
        "|--------|----------|--------------|",
        f"| Annualized mean return | {exp3['stacking_metrics']['annualized_mean']:.4f} | {exp3['benchmark_metrics']['annualized_mean']:.4f} |",
        f"| Annualized std deviation | {exp3['stacking_metrics']['annualized_std']:.4f} | {exp3['benchmark_metrics']['annualized_std']:.4f} |",
        f"| Annualized Sharpe ratio | {exp3['stacking_metrics']['annualized_sharpe']:.4f} | {exp3['benchmark_metrics']['annualized_sharpe']:.4f} |",
        f"| Observations | {exp3['stacking_metrics']['n_observations']} | {exp3['benchmark_metrics']['n_observations']} |",
        "",
        f"**Runtime:** {timings['exp3']:.1f}s",
        "",
        "---",
        "",
        "## Experiment 4: Copula-Based Synthetic Prior and Dependence Visualization",
        "",
        "### Configuration",
        "| Parameter | Value |",
        "|-----------|-------|",
        "| Copula type | Multivariate Student-t |",
        "| Assets (multivariate) | AAPL, JPM, LLY |",
        "| Bivariate pair | AAPL, JPM |",
        f"| Random seed | 42 |",
        "",
        "### Dataset Metadata",
        "| Field | Value |",
        "|-------|-------|",
        f"| Date range | {exp4['metadata']['start_date']} to {exp4['metadata']['end_date']} |",
        f"| Historical samples | {exp4['n_historical']} |",
        f"| Synthetic samples | {exp4['n_synthetic']} |",
        "",
        "### Fitted Copula Parameters",
        "| Parameter | Multivariate (AAPL/JPM/LLY) | Bivariate (AAPL/JPM) |",
        "|-----------|------------------------------|----------------------|",
        f"| Degrees of freedom (df) | {exp4['multi_copula_params']['df']:.4f} | {exp4['bivariate_copula_params']['df']:.4f} |",
        f"| Correlation (ρ) | {exp4['multi_copula_params']['rho']} | {exp4['bivariate_copula_params']['rho']:.4f} |",
        "",
        "### Dependence Quality Assessment",
        "| Pair | Historical ρ | Synthetic ρ | |Diff| |",
        "|------|-------------|-------------|------|",
    ]

    for d in exp4["dependence_quality"]["correlation_comparison"]:
        lines.append(
            f"| {d['pair']} | {d['hist_corr']:.4f} | {d['synth_corr']:.4f} | {d['abs_diff']:.4f} |"
        )

    lines += [
        "",
        f"| **Mean absolute correlation difference** | | | **{exp4['dependence_quality']['mean_abs_corr_diff']:.4f}** |",
        "",
        "### Tail Statistics (5th Percentile)",
        "| Asset | Historical | Synthetic |",
        "|-------|-----------|-----------|",
    ]

    for ticker in ["AAPL", "JPM", "LLY"]:
        h = exp4["dependence_quality"]["hist_5th_percentile"].get(ticker, float("nan"))
        s = exp4["dependence_quality"]["synth_5th_percentile"].get(ticker, float("nan"))
        lines.append(f"| {ticker} | {h:.4f} | {s:.4f} |")

    lines += [
        "",
        f"**Runtime:** {timings['exp4']:.1f}s",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Experiment | Status | Runtime |",
        "|------------|--------|---------|",
        f"| Exp 1: Min-Variance Frontier | ✅ Complete | {timings['exp1']:.1f}s |",
        f"| Exp 2: Factor-Model Prior | ✅ Complete | {timings['exp2']:.1f}s |",
        f"| Exp 3: Walk-Forward Stacking | ✅ Complete | {timings['exp3']:.1f}s |",
        f"| Exp 4: Copula Visualization | ✅ Complete | {timings['exp4']:.1f}s |",
        "",
        "### Key Findings",
        "",
        "1. **Exp 1 (Overfitting):** The training efficient frontier "
        f"{'dominates' if exp1['dominance']['train_dominates_test'] else 'does NOT dominate'} "
        f"the test frontier on {exp1['dominance']['pct_frontier_train_above_test']:.1%} of the "
        "risk range, confirming in-sample overfitting as described in the paper.",
        "",
        "2. **Exp 2 (Factor Prior):** Factor-model prior with ridge-estimated loadings "
        f"(α=0.1) successfully integrated into the MeanRisk optimizer. "
        f"Fit successful: {exp2['fit_successful']}. "
        f"AAPL constraint respected: {exp2['validation']['aapl_constraint_ok']}.",
        "",
        "3. **Exp 3 (Walk-Forward):** Stacking optimization evaluated over "
        f"{exp3['fold_validation']['n_folds']} walk-forward folds with no look-ahead leakage. "
        f"Stacking Sharpe: {exp3['stacking_metrics']['annualized_sharpe']:.4f} vs "
        f"Equal-Weight Sharpe: {exp3['benchmark_metrics']['annualized_sharpe']:.4f}.",
        "",
        "4. **Exp 4 (Copula):** Multivariate Student-t copula fitted to AAPL/JPM/LLY "
        f"(df={exp4['multi_copula_params']['df']:.2f}). "
        f"Synthetic samples capture historical dependence with mean absolute "
        f"correlation difference of {exp4['dependence_quality']['mean_abs_corr_diff']:.4f}.",
        "",
        "### Generated Artifacts",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `exp1_efficient_frontier.png` | Training vs test efficient frontiers |",
        "| `exp1_train_frontier.csv` | Training frontier data (100 points) |",
        "| `exp1_test_frontier.csv` | Test frontier data (100 points) |",
        "| `exp2_factor_weights.png` | Factor-prior portfolio weights |",
        "| `exp2_train_summary.csv` | Training portfolio statistics |",
        "| `exp2_test_summary.csv` | Test portfolio statistics |",
        "| `exp3_walk_forward_cumulative.png` | Cumulative returns: stacking vs benchmark |",
        "| `exp3_population_summary.csv` | Walk-forward population summary |",
        "| `exp4_historical_vs_synthetic.png` | Historical vs synthetic return comparison |",
        "| `exp4_bivariate_copula_density.png` | AAPL-JPM Student-t copula density |",
        "| `exp4_aapl_jpm_scatter.png` | AAPL-JPM historical vs synthetic scatter |",
        "| `exp4_synthetic_returns.csv` | Synthetic return samples |",
        "| `exp4_density_grid.csv` | Bivariate copula density grid |",
        "",
    ]

    return "\n".join(lines)


def main() -> None:
    """Run all experiments and save results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Import experiment modules
    from exp1_efficient_frontier import run_experiment as run_exp1
    from exp2_factor_model import run_experiment as run_exp2
    from exp3_walk_forward_stacking import run_experiment as run_exp3
    from exp4_copula import run_experiment as run_exp4

    timings: dict[str, float] = {}
    results: dict[str, Any] = {}

    # Run Experiment 1
    try:
        results["exp1"], timings["exp1"] = run_with_timing(
            "Experiment 1: Constrained Min-Variance Frontier", run_exp1, RESULTS_DIR
        )
    except Exception as e:
        logger.error("Experiment 1 failed: %s", e)
        raise

    # Run Experiment 2
    try:
        results["exp2"], timings["exp2"] = run_with_timing(
            "Experiment 2: Factor-Model Prior", run_exp2, RESULTS_DIR
        )
    except Exception as e:
        logger.error("Experiment 2 failed: %s", e)
        raise

    # Run Experiment 3
    try:
        results["exp3"], timings["exp3"] = run_with_timing(
            "Experiment 3: Walk-Forward Stacking", run_exp3, RESULTS_DIR
        )
    except Exception as e:
        logger.error("Experiment 3 failed: %s", e)
        raise

    # Run Experiment 4
    try:
        results["exp4"], timings["exp4"] = run_with_timing(
            "Experiment 4: Copula Visualization", run_exp4, RESULTS_DIR
        )
    except Exception as e:
        logger.error("Experiment 4 failed: %s", e)
        raise

    # Generate and save RESULTS.md
    md_content = format_results_markdown(
        results["exp1"], results["exp2"], results["exp3"], results["exp4"], timings
    )
    results_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    with open(results_path, "w") as f:
        f.write(md_content)
    logger.info("Results saved to %s", results_path)

    # Save raw results as JSON
    def make_serializable(obj: Any) -> Any:
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return [make_serializable(v) for v in obj]
        return obj

    json_path = os.path.join(RESULTS_DIR, "results_raw.json")
    with open(json_path, "w") as f:
        json.dump(make_serializable(results), f, indent=2, default=str)
    logger.info("Raw results saved to %s", json_path)

    total_time = sum(timings.values())
    logger.info("=" * 60)
    logger.info("All experiments completed in %.1f seconds", total_time)
    logger.info("Results saved to %s/", RESULTS_DIR)
    logger.info("=" * 60)

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Total runtime: {total_time:.1f}s")
    print(f"\nKey results:")
    print(f"  Exp 1 - Train dominates test: {results['exp1']['dominance']['train_dominates_test']}")
    print(f"  Exp 1 - % frontier train > test: {results['exp1']['dominance']['pct_frontier_train_above_test']:.1%}")
    print(f"  Exp 2 - Factor prior fit OK: {results['exp2']['fit_successful']}")
    print(f"  Exp 3 - Walk-forward folds: {results['exp3']['fold_validation']['n_folds']}")
    print(f"  Exp 3 - No leakage: {results['exp3']['fold_validation']['all_no_leakage']}")
    print(f"  Exp 4 - Copula df: {results['exp4']['multi_copula_params']['df']:.2f}")
    print(f"\nResults directory: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
