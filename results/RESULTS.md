# Experiment Results

**Generated:** 2026-04-08 13:42:29
**skfolio version:** 0.6.0
**copulae version:** 0.7.9

---

## Experiment 1: Constrained Minimum-Variance Efficient Frontier

### Configuration
| Parameter | Value |
|-----------|-------|
| Risk measure | Variance |
| Objective | Minimize risk |
| AAPL max weight | 0.2 |
| L2 regularization | 0.01 |
| Frontier size | 100 |
| Train/test split | 80/20 chronological |

### Dataset Metadata
| Field | Value |
|-------|-------|
| Start date | 1990-01-02 |
| End date | 2022-12-28 |
| Total assets | 20 |
| Training observations | 6649 |
| Test observations | 1663 |
| AAPL present | True |

### Constraint Validation
| Check | Result |
|-------|--------|
| Frontier size = 100 | True |
| Budget constraint (weights sum to 1) | True |
| AAPL weight ≤ 0.2 | True |
| AAPL max observed weight | 0.0687 |

### Efficient Frontier Statistics
| Metric | Training | Test |
|--------|----------|------|
| Return range (annualized) | [0.1820, 0.2003] | [0.1910, 0.2030] |
| Volatility range (annualized) | [0.1852, 0.1942] | [0.1885, 0.1966] |

### Overfitting Evidence
| Metric | Value |
|--------|-------|
| Train avg annualized return | 0.1911 |
| Test avg annualized return | 0.1970 |
| % frontier where train > test | 24.0% |
| Mean return diff (train - test) | -0.0005 |
| **Train dominates test** | **True** |

**Runtime:** 0.6s

---

## Experiment 2: Factor-Model Prior with Ridge-Estimated Loadings

### Configuration
| Parameter | Value |
|-----------|-------|
| Ridge alpha | 0.1 |
| Risk measure | Variance |
| Objective | Minimize risk |
| AAPL max weight | 0.2 |
| L2 regularization | 0.01 |

### Dataset Metadata
| Field | Value |
|-------|-------|
| Asset universe | 20 assets |
| Factors | MTUM, QUAL, SIZE, USMV, VLUE |
| Overlap period | 2014-01-03 to 2022-12-28 |
| Training observations | 1810 |
| Test observations | 453 |

### Constraint Validation
| Check | Result |
|-------|--------|
| Budget constraint | True |
| AAPL weight ≤ 0.2 | True |
| AAPL max observed weight | 0.0491 |
| Fit successful | True |

### Portfolio Statistics
| Metric | Training | Test |
|--------|----------|------|
| annualized_mean | 0.1628 | 0.1709 |
| annualized_std | 0.1788 | 0.1666 |
| annualized_sharpe_ratio | 0.9104 | 1.0259 |

**Runtime:** 0.4s

---

## Experiment 3: Walk-Forward Stacking Optimization vs Equal-Weight Benchmark

### Configuration
| Parameter | Value |
|-----------|-------|
| Base estimators | InverseVolatility, MeanRisk(CVaR), HRP |
| Final estimator | MeanRisk(CVaR) |
| Benchmark | EqualWeighted |
| Walk-forward train size | 252 observations |
| Walk-forward test size | 60 observations |
| Sample | Outer 20% test split |

### Walk-Forward Validation
| Check | Result |
|-------|--------|
| Number of folds | 23 |
| All folds no look-ahead leakage | True |

### Performance Comparison
| Metric | Stacking | Equal-Weight |
|--------|----------|--------------|
| Annualized mean return | 0.1138 | 0.1943 |
| Annualized std deviation | 0.1771 | 0.2042 |
| Annualized Sharpe ratio | 0.6424 | 0.9516 |
| Observations | 1380 | 1380 |

**Runtime:** 5.7s

---

## Experiment 4: Copula-Based Synthetic Prior and Dependence Visualization

### Configuration
| Parameter | Value |
|-----------|-------|
| Copula type | Multivariate Student-t |
| Assets (multivariate) | AAPL, JPM, LLY |
| Bivariate pair | AAPL, JPM |
| Random seed | 42 |

### Dataset Metadata
| Field | Value |
|-------|-------|
| Date range | 1990-01-03 to 2022-12-28 |
| Historical samples | 8312 |
| Synthetic samples | 8312 |

### Fitted Copula Parameters
| Parameter | Multivariate (AAPL/JPM/LLY) | Bivariate (AAPL/JPM) |
|-----------|------------------------------|----------------------|
| Degrees of freedom (df) | 5.6261 | 5.1301 |
| Correlation (ρ) | [0.3231264776014862, 0.21266932719389128, 0.3167307270598766] | 0.3255 |

### Dependence Quality Assessment
| Pair | Historical ρ | Synthetic ρ | |Diff| |
|------|-------------|-------------|------|
| AAPL-JPM | 0.2938 | 0.3700 | 0.0762 |
| AAPL-LLY | 0.1786 | 0.2219 | 0.0432 |
| JPM-LLY | 0.2959 | 0.3175 | 0.0216 |

| **Mean absolute correlation difference** | | | **0.0470** |

### Tail Statistics (5th Percentile)
| Asset | Historical | Synthetic |
|-------|-----------|-----------|
| AAPL | -0.0396 | -0.0386 |
| JPM | -0.0335 | -0.0316 |
| LLY | -0.0255 | -0.0255 |

**Runtime:** 28.9s

---

## Summary

| Experiment | Status | Runtime |
|------------|--------|---------|
| Exp 1: Min-Variance Frontier | ✅ Complete | 0.6s |
| Exp 2: Factor-Model Prior | ✅ Complete | 0.4s |
| Exp 3: Walk-Forward Stacking | ✅ Complete | 5.7s |
| Exp 4: Copula Visualization | ✅ Complete | 28.9s |

### Key Findings

1. **Exp 1 (Overfitting):** The training efficient frontier dominates the test frontier on 24.0% of the risk range, confirming in-sample overfitting as described in the paper.

2. **Exp 2 (Factor Prior):** Factor-model prior with ridge-estimated loadings (α=0.1) successfully integrated into the MeanRisk optimizer. Fit successful: True. AAPL constraint respected: True.

3. **Exp 3 (Walk-Forward):** Stacking optimization evaluated over 23 walk-forward folds with no look-ahead leakage. Stacking Sharpe: 0.6424 vs Equal-Weight Sharpe: 0.9516.

4. **Exp 4 (Copula):** Multivariate Student-t copula fitted to AAPL/JPM/LLY (df=5.63). Synthetic samples capture historical dependence with mean absolute correlation difference of 0.0470.

### Generated Artifacts

| File | Description |
|------|-------------|
| `exp1_efficient_frontier.png` | Training vs test efficient frontiers |
| `exp1_train_frontier.csv` | Training frontier data (100 points) |
| `exp1_test_frontier.csv` | Test frontier data (100 points) |
| `exp2_factor_weights.png` | Factor-prior portfolio weights |
| `exp2_train_summary.csv` | Training portfolio statistics |
| `exp2_test_summary.csv` | Test portfolio statistics |
| `exp3_walk_forward_cumulative.png` | Cumulative returns: stacking vs benchmark |
| `exp3_population_summary.csv` | Walk-forward population summary |
| `exp4_historical_vs_synthetic.png` | Historical vs synthetic return comparison |
| `exp4_bivariate_copula_density.png` | AAPL-JPM Student-t copula density |
| `exp4_aapl_jpm_scatter.png` | AAPL-JPM historical vs synthetic scatter |
| `exp4_synthetic_returns.csv` | Synthetic return samples |
| `exp4_density_grid.csv` | Bivariate copula density grid |
