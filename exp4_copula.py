"""
Experiment 4: Copula-Based Synthetic Prior and Dependence-Structure Visualization
==================================================================================
Reproduces the paper's copula demonstration by:
  - Fitting a multivariate Student-t copula to historical asset returns
    (AAPL, JPM, LLY and additional assets from the built-in dataset)
  - Sampling synthetic returns from the fitted copula
  - Visually comparing generated distributions with historical returns
  - Fitting a bivariate Student-t copula for AAPL-JPM
  - Producing a 2D Student-t copula density contour plot for AAPL-JPM

Validates that copula-based synthetic priors and stress-testing workflows
can be integrated into the same portfolio research environment.

# Using copulae StudentCopula, pseudo_obs — Context7 confirmed
# Custom vine-like workflow using multivariate StudentCopula — Context7 found
# no native VineCopula in skfolio v0.6.0 or copulae v0.7.9
"""

import logging
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skfolio
from scipy.stats import t as scipy_t

from copulae import StudentCopula, pseudo_obs
from skfolio.datasets import load_sp500_dataset
from skfolio.preprocessing import prices_to_returns

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── Constants ────────────────────────────────────────────────────────────────
REQUIRED_TICKERS: list[str] = ["AAPL", "JPM", "LLY"]
RANDOM_SEED: int = 42  # fixed for reproducibility


# ── Data loading & preprocessing ─────────────────────────────────────────────

def load_and_prepare_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load the built-in S&P 500 dataset and convert prices to returns.

    Returns
    -------
    X : pd.DataFrame
        Full return matrix.
    metadata : dict
        Dataset metadata.
    """
    prices = load_sp500_dataset()
    X = prices_to_returns(prices)

    metadata: dict[str, Any] = {
        "start_date": str(X.index[0].date()),
        "end_date": str(X.index[-1].date()),
        "n_observations": len(X),
        "n_assets": X.shape[1],
        "asset_tickers": list(X.columns),
        "skfolio_version": skfolio.__version__,
    }

    for ticker in REQUIRED_TICKERS:
        if ticker not in X.columns:
            raise ValueError(
                f"Required ticker {ticker} is absent from the dataset. "
                "Cannot reproduce the copula demonstration."
            )

    logger.info("Dataset: %s to %s, %d obs, %d assets",
                metadata["start_date"], metadata["end_date"],
                metadata["n_observations"], metadata["n_assets"])
    logger.info("Required tickers present: %s", REQUIRED_TICKERS)

    return X, metadata


# ── Multivariate copula fitting ───────────────────────────────────────────────

def fit_multivariate_copula(
    X: pd.DataFrame,
    tickers: list[str],
) -> tuple[StudentCopula, np.ndarray, np.ndarray]:
    """Fit a multivariate Student-t copula to selected asset returns.

    The workflow:
      1. Extract returns for selected tickers
      2. Transform to pseudo-observations (rank-based uniform marginals)
      3. Fit a multivariate Student-t copula

    Parameters
    ----------
    X : pd.DataFrame
        Full return matrix.
    tickers : list[str]
        Asset tickers to include in the copula.

    Returns
    -------
    copula : StudentCopula
        Fitted multivariate Student-t copula.
    returns_arr : np.ndarray
        Historical returns array (n_obs, n_assets).
    pseudo_obs_arr : np.ndarray
        Pseudo-observations used for fitting (n_obs, n_assets).
    """
    returns_arr = X[tickers].values
    n_obs, n_dim = returns_arr.shape

    # Transform to pseudo-observations (rank-based uniform marginals)
    # Using copulae pseudo_obs — Context7 confirmed
    pseudo_obs_arr = pseudo_obs(returns_arr)

    logger.info("Fitting multivariate Student-t copula: dim=%d, n_obs=%d", n_dim, n_obs)

    # Using copulae StudentCopula — Context7 confirmed
    copula = StudentCopula(dim=n_dim)
    copula.fit(pseudo_obs_arr)

    logger.info("Fitted copula params: df=%.4f, rho=%s",
                float(copula.params.df), copula.params.rho)

    return copula, returns_arr, pseudo_obs_arr


# ── Synthetic sample generation ───────────────────────────────────────────────

def generate_synthetic_returns(
    copula: StudentCopula,
    historical_returns: np.ndarray,
    n_samples: int,
    seed: int = RANDOM_SEED,
) -> np.ndarray:
    """Generate synthetic returns from the fitted copula.

    Uses the empirical quantile function to back-transform copula samples
    from uniform marginals to the original return scale.

    Parameters
    ----------
    copula : StudentCopula
        Fitted multivariate Student-t copula.
    historical_returns : np.ndarray
        Historical returns (n_obs, n_assets) for empirical marginal inversion.
    n_samples : int
        Number of synthetic samples to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Synthetic returns (n_samples, n_assets).
    """
    # Sample from copula (uniform marginals)
    # Using copulae StudentCopula.random — Context7 confirmed
    uniform_samples = copula.random(n_samples, seed=seed)

    # Back-transform using empirical quantile function (inverse CDF)
    # Custom — Context7 found no library equivalent for empirical marginal inversion
    n_assets = historical_returns.shape[1]
    synthetic = np.zeros((n_samples, n_assets))
    for i in range(n_assets):
        synthetic[:, i] = np.quantile(historical_returns[:, i], uniform_samples[:, i])

    logger.info("Generated %d synthetic return samples (dim=%d)", n_samples, n_assets)
    return synthetic


# ── Bivariate copula fitting ──────────────────────────────────────────────────

def fit_bivariate_copula(
    X: pd.DataFrame,
    ticker1: str,
    ticker2: str,
) -> tuple[StudentCopula, np.ndarray, np.ndarray]:
    """Fit a bivariate Student-t copula for a pair of assets.

    Parameters
    ----------
    X : pd.DataFrame
        Full return matrix.
    ticker1 : str
        First asset ticker.
    ticker2 : str
        Second asset ticker.

    Returns
    -------
    copula : StudentCopula
        Fitted bivariate Student-t copula.
    returns_pair : np.ndarray
        Historical returns for the pair (n_obs, 2).
    pseudo_obs_pair : np.ndarray
        Pseudo-observations for the pair (n_obs, 2).
    """
    returns_pair = X[[ticker1, ticker2]].values
    pseudo_obs_pair = pseudo_obs(returns_pair)

    logger.info("Fitting bivariate Student-t copula for %s-%s", ticker1, ticker2)

    # Using copulae StudentCopula — Context7 confirmed
    copula = StudentCopula(dim=2)
    copula.fit(pseudo_obs_pair)

    logger.info("Bivariate copula params: df=%.4f, rho=%.4f",
                float(copula.params.df), float(copula.params.rho[0]))

    return copula, returns_pair, pseudo_obs_pair


# ── Density grid evaluation ───────────────────────────────────────────────────

def compute_copula_density_grid(
    copula: StudentCopula,
    n_grid: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the bivariate copula density on a uniform grid.

    Parameters
    ----------
    copula : StudentCopula
        Fitted bivariate Student-t copula.
    n_grid : int
        Number of grid points per dimension.

    Returns
    -------
    uu : np.ndarray
        Grid of u values (n_grid, n_grid).
    vv : np.ndarray
        Grid of v values (n_grid, n_grid).
    density : np.ndarray
        Copula density values (n_grid, n_grid).
    """
    # Avoid boundary singularities
    u = np.linspace(0.02, 0.98, n_grid)
    v = np.linspace(0.02, 0.98, n_grid)
    uu, vv = np.meshgrid(u, v)
    grid = np.column_stack([uu.ravel(), vv.ravel()])

    # Using copulae StudentCopula.pdf — Context7 confirmed
    density_flat = copula.pdf(grid)
    density = density_flat.reshape(n_grid, n_grid)

    logger.info("Density grid: shape=%s, range=[%.4f, %.4f]",
                density.shape, density.min(), density.max())

    return uu, vv, density


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_historical_vs_synthetic(
    historical: np.ndarray,
    synthetic: np.ndarray,
    tickers: list[str],
    output_path: str,
) -> None:
    """Create pairwise scatter plots comparing historical and synthetic returns.

    Parameters
    ----------
    historical : np.ndarray
        Historical returns (n_obs, n_assets).
    synthetic : np.ndarray
        Synthetic returns (n_samples, n_assets).
    tickers : list[str]
        Asset names.
    output_path : str
        File path to save the figure.
    """
    n = len(tickers)
    fig, axes = plt.subplots(n, n, figsize=(4 * n, 4 * n))

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if i == j:
                # Marginal histograms on diagonal
                ax.hist(historical[:, i], bins=50, alpha=0.6, color="steelblue",
                        density=True, label="Historical")
                ax.hist(synthetic[:, i], bins=50, alpha=0.6, color="tomato",
                        density=True, label="Synthetic")
                ax.set_title(tickers[i], fontsize=10)
                if i == 0:
                    ax.legend(fontsize=8)
            else:
                # Scatter plots off-diagonal
                ax.scatter(historical[:, j], historical[:, i], alpha=0.2,
                           s=5, color="steelblue", label="Historical")
                ax.scatter(synthetic[:, j], synthetic[:, i], alpha=0.2,
                           s=5, color="tomato", label="Synthetic")
                ax.set_xlabel(tickers[j], fontsize=9)
                ax.set_ylabel(tickers[i], fontsize=9)

    fig.suptitle(
        "Historical vs Synthetic Returns (Multivariate Student-t Copula)\n"
        f"Tickers: {', '.join(tickers)}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Historical vs synthetic plot saved to %s", output_path)


def plot_bivariate_copula_density(
    uu: np.ndarray,
    vv: np.ndarray,
    density: np.ndarray,
    ticker1: str,
    ticker2: str,
    copula_params: dict[str, float],
    output_path: str,
) -> None:
    """Plot the bivariate Student-t copula density as a filled contour.

    Parameters
    ----------
    uu : np.ndarray
        Grid of u values.
    vv : np.ndarray
        Grid of v values.
    density : np.ndarray
        Copula density values.
    ticker1 : str
        First asset name.
    ticker2 : str
        Second asset name.
    copula_params : dict
        Fitted copula parameters for annotation.
    output_path : str
        File path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    cf = ax.contourf(uu, vv, density, levels=20, cmap="RdYlBu_r")
    plt.colorbar(cf, ax=ax, label="Copula Density")
    ax.contour(uu, vv, density, levels=20, colors="k", linewidths=0.5, alpha=0.4)
    ax.set_xlabel(f"u ({ticker1})", fontsize=12)
    ax.set_ylabel(f"v ({ticker2})", fontsize=12)
    ax.set_title(
        f"Bivariate Student-t Copula Density: {ticker1} vs {ticker2}\n"
        f"df={copula_params['df']:.2f}, ρ={copula_params['rho']:.4f}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Bivariate copula density plot saved to %s", output_path)


def plot_aapl_jpm_scatter(
    historical: np.ndarray,
    synthetic: np.ndarray,
    ticker1: str,
    ticker2: str,
    output_path: str,
) -> None:
    """Plot historical vs synthetic scatter for AAPL-JPM pair.

    Parameters
    ----------
    historical : np.ndarray
        Historical returns for the pair (n_obs, 2).
    synthetic : np.ndarray
        Synthetic returns for the pair (n_samples, 2).
    ticker1 : str
        First asset name.
    ticker2 : str
        Second asset name.
    output_path : str
        File path to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(historical[:, 0], historical[:, 1], alpha=0.3, s=8,
                    color="steelblue")
    axes[0].set_xlabel(ticker1, fontsize=12)
    axes[0].set_ylabel(ticker2, fontsize=12)
    axes[0].set_title("Historical Returns", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(synthetic[:, 0], synthetic[:, 1], alpha=0.3, s=8,
                    color="tomato")
    axes[1].set_xlabel(ticker1, fontsize=12)
    axes[1].set_ylabel(ticker2, fontsize=12)
    axes[1].set_title("Synthetic Returns (Student-t Copula)", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Dependence Structure: {ticker1} vs {ticker2}", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("AAPL-JPM scatter plot saved to %s", output_path)


# ── Qualitative assessment ────────────────────────────────────────────────────

def assess_dependence_quality(
    historical: np.ndarray,
    synthetic: np.ndarray,
    tickers: list[str],
) -> dict[str, Any]:
    """Assess whether synthetic samples capture historical dependence structure.

    Compares pairwise correlations and tail statistics between historical
    and synthetic return samples.

    Parameters
    ----------
    historical : np.ndarray
        Historical returns (n_obs, n_assets).
    synthetic : np.ndarray
        Synthetic returns (n_samples, n_assets).
    tickers : list[str]
        Asset names.

    Returns
    -------
    dict
        Dependence quality metrics.
    """
    n = len(tickers)
    hist_corr = np.corrcoef(historical.T)
    synth_corr = np.corrcoef(synthetic.T)

    # Pairwise correlation comparison
    corr_diffs = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = abs(hist_corr[i, j] - synth_corr[i, j])
            corr_diffs.append({
                "pair": f"{tickers[i]}-{tickers[j]}",
                "hist_corr": float(hist_corr[i, j]),
                "synth_corr": float(synth_corr[i, j]),
                "abs_diff": float(diff),
            })

    # Tail statistics (5th percentile)
    hist_tail = np.percentile(historical, 5, axis=0)
    synth_tail = np.percentile(synthetic, 5, axis=0)

    results = {
        "correlation_comparison": corr_diffs,
        "mean_abs_corr_diff": float(np.mean([d["abs_diff"] for d in corr_diffs])),
        "hist_5th_percentile": {t: float(v) for t, v in zip(tickers, hist_tail)},
        "synth_5th_percentile": {t: float(v) for t, v in zip(tickers, synth_tail)},
        "hist_kurtosis": {
            t: float(pd.Series(historical[:, i]).kurtosis())
            for i, t in enumerate(tickers)
        },
        "synth_kurtosis": {
            t: float(pd.Series(synthetic[:, i]).kurtosis())
            for i, t in enumerate(tickers)
        },
    }

    logger.info("Mean absolute correlation difference: %.4f",
                results["mean_abs_corr_diff"])
    for d in corr_diffs:
        logger.info("  %s: hist_corr=%.4f, synth_corr=%.4f, diff=%.4f",
                    d["pair"], d["hist_corr"], d["synth_corr"], d["abs_diff"])

    return results


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiment(results_dir: str = "results") -> dict[str, Any]:
    """Run the full Experiment 4 pipeline.

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
    X, metadata = load_and_prepare_data()

    # 2. Select tickers for multivariate copula
    # Use AAPL, JPM, LLY as required by the paper
    multi_tickers = REQUIRED_TICKERS
    logger.info("Multivariate copula tickers: %s", multi_tickers)

    # 3. Fit multivariate Student-t copula
    multi_copula, hist_returns, pseudo_obs_arr = fit_multivariate_copula(
        X, multi_tickers
    )

    # 4. Generate synthetic returns
    n_samples = len(hist_returns)  # match historical sample size
    synthetic_returns = generate_synthetic_returns(
        multi_copula, hist_returns, n_samples, seed=RANDOM_SEED
    )

    # 5. Plot historical vs synthetic
    hist_vs_synth_path = os.path.join(results_dir, "exp4_historical_vs_synthetic.png")
    plot_historical_vs_synthetic(
        hist_returns, synthetic_returns, multi_tickers, hist_vs_synth_path
    )

    # 6. Fit bivariate Student-t copula for AAPL-JPM
    biv_copula, hist_pair, pseudo_pair = fit_bivariate_copula(X, "AAPL", "JPM")

    # 7. Compute density grid
    uu, vv, density = compute_copula_density_grid(biv_copula, n_grid=60)

    # 8. Plot bivariate copula density
    copula_params = {
        "df": float(biv_copula.params.df),
        "rho": float(biv_copula.params.rho[0]),
    }
    density_path = os.path.join(results_dir, "exp4_bivariate_copula_density.png")
    plot_bivariate_copula_density(
        uu, vv, density, "AAPL", "JPM", copula_params, density_path
    )

    # 9. Generate synthetic for AAPL-JPM pair and plot scatter
    synth_pair = generate_synthetic_returns(
        biv_copula, hist_pair, len(hist_pair), seed=RANDOM_SEED
    )
    scatter_path = os.path.join(results_dir, "exp4_aapl_jpm_scatter.png")
    plot_aapl_jpm_scatter(hist_pair, synth_pair, "AAPL", "JPM", scatter_path)

    # 10. Qualitative assessment
    quality = assess_dependence_quality(hist_returns, synthetic_returns, multi_tickers)

    # 11. Save synthetic samples
    synth_df = pd.DataFrame(synthetic_returns, columns=multi_tickers)
    synth_df.to_csv(os.path.join(results_dir, "exp4_synthetic_returns.csv"), index=False)

    # 12. Save density grid
    density_df = pd.DataFrame({
        "u": uu.ravel(),
        "v": vv.ravel(),
        "density": density.ravel(),
    })
    density_df.to_csv(os.path.join(results_dir, "exp4_density_grid.csv"), index=False)

    results = {
        "metadata": metadata,
        "multi_copula_params": {
            "df": float(multi_copula.params.df),
            "rho": multi_copula.params.rho.tolist(),
            "dim": len(multi_tickers),
            "tickers": multi_tickers,
        },
        "bivariate_copula_params": copula_params,
        "n_historical": len(hist_returns),
        "n_synthetic": n_samples,
        "dependence_quality": quality,
        "plots": {
            "hist_vs_synth": hist_vs_synth_path,
            "bivariate_density": density_path,
            "aapl_jpm_scatter": scatter_path,
        },
    }

    logger.info("Experiment 4 complete.")
    return results


if __name__ == "__main__":
    results = run_experiment()
    print("\n=== Experiment 4 Results ===")
    print(f"Dataset: {results['metadata']['start_date']} to {results['metadata']['end_date']}")
    print(f"Historical samples: {results['n_historical']}, Synthetic: {results['n_synthetic']}")
    print(f"\nMultivariate Student-t Copula ({', '.join(results['multi_copula_params']['tickers'])}):")
    print(f"  df = {results['multi_copula_params']['df']:.4f}")
    print(f"  rho = {results['multi_copula_params']['rho']}")
    print(f"\nBivariate Student-t Copula (AAPL-JPM):")
    print(f"  df = {results['bivariate_copula_params']['df']:.4f}")
    print(f"  rho = {results['bivariate_copula_params']['rho']:.4f}")
    print(f"\nMean absolute correlation difference: "
          f"{results['dependence_quality']['mean_abs_corr_diff']:.4f}")
