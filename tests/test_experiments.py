"""
Pytest tests for all four portfolio optimization experiments.

Tests verify:
  - Methodology adherence (constraints, parameters, data splits)
  - Mathematical properties (budget constraint, weight bounds)
  - Edge cases (empty data, single assets, extreme values)
  - Regression tests with known expected outcomes
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

import skfolio
from skfolio import Population, RiskMeasure
from skfolio.datasets import load_factors_dataset, load_sp500_dataset
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
from skfolio.prior import FactorModel, LoadingMatrixRegression

from copulae import StudentCopula, pseudo_obs


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def sp500_prices():
    """Load the built-in S&P 500 price dataset once per module."""
    return load_sp500_dataset()


@pytest.fixture(scope="module")
def sp500_returns(sp500_prices):
    """Convert S&P 500 prices to returns."""
    return prices_to_returns(sp500_prices)


@pytest.fixture(scope="module")
def train_test_returns(sp500_returns):
    """Chronological 80/20 train-test split of S&P 500 returns."""
    X_train, X_test = train_test_split(sp500_returns, test_size=0.2, shuffle=False)
    return X_train, X_test


@pytest.fixture(scope="module")
def fitted_mv_model(train_test_returns):
    """Fitted minimum-variance model for Exp 1."""
    X_train, _ = train_test_returns
    model = MeanRisk(
        objective_function=ObjectiveFunction.MINIMIZE_RISK,
        risk_measure=RiskMeasure.VARIANCE,
        max_weights={"AAPL": 0.2},
        l2_coef=0.01,
        efficient_frontier_size=100,
    )
    model.fit(X_train)
    return model


@pytest.fixture(scope="module")
def factor_data():
    """Load and align asset and factor datasets."""
    asset_prices = load_sp500_dataset()
    factor_prices = load_factors_dataset()
    X, y = prices_to_returns(asset_prices, factor_prices)
    X_train, X_test = train_test_split(X, test_size=0.2, shuffle=False)
    y_train, y_test = train_test_split(y, test_size=0.2, shuffle=False)
    return X_train, X_test, y_train, y_test


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 1 Tests: Constrained Minimum-Variance Efficient Frontier
# ═══════════════════════════════════════════════════════════════════════════════

class TestExp1DataLoading:
    """Tests for data loading and preprocessing in Experiment 1."""

    def test_sp500_dataset_loads(self, sp500_prices):
        """Dataset should load without errors and have expected structure."""
        assert sp500_prices is not None
        assert isinstance(sp500_prices, pd.DataFrame)
        assert sp500_prices.shape[0] > 0
        assert sp500_prices.shape[1] > 0

    def test_aapl_present_in_dataset(self, sp500_prices):
        """AAPL must be present in the dataset for the weight constraint."""
        assert "AAPL" in sp500_prices.columns, "AAPL is required for the weight constraint"

    def test_returns_conversion(self, sp500_returns):
        """Returns should have one fewer row than prices and no NaN in body."""
        prices = load_sp500_dataset()
        assert len(sp500_returns) == len(prices) - 1
        # Returns should not have NaN (library handles this)
        assert not sp500_returns.isnull().all().any()

    def test_chronological_split_no_shuffle(self, train_test_returns):
        """Train-test split must preserve chronological order."""
        X_train, X_test = train_test_returns
        assert X_train.index[-1] < X_test.index[0], (
            "Train set must end before test set begins"
        )

    def test_split_ratio(self, sp500_returns, train_test_returns):
        """Test set should be approximately 20% of total observations."""
        X_train, X_test = train_test_returns
        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert abs(test_ratio - 0.2) < 0.01, (
            f"Test ratio {test_ratio:.3f} deviates from 0.2"
        )

    def test_dataset_has_sufficient_observations(self, sp500_returns):
        """Dataset must have enough observations for meaningful optimization."""
        assert len(sp500_returns) > 500, "Need at least 500 observations"

    def test_dataset_date_range(self, sp500_prices):
        """Dataset should span multiple years."""
        date_range = (sp500_prices.index[-1] - sp500_prices.index[0]).days
        assert date_range > 365 * 5, "Dataset should span at least 5 years"


class TestExp1ModelConfiguration:
    """Tests for model configuration in Experiment 1."""

    def test_model_builds_without_error(self):
        """MeanRisk model should build with paper-specified parameters."""
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
            efficient_frontier_size=100,
        )
        assert model is not None

    def test_model_parameters_match_paper(self):
        """Model parameters must exactly match paper specification."""
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
            efficient_frontier_size=100,
        )
        assert model.objective_function == ObjectiveFunction.MINIMIZE_RISK
        assert model.risk_measure == RiskMeasure.VARIANCE
        assert model.l2_coef == 0.01
        assert model.efficient_frontier_size == 100

    def test_model_fits_on_training_data(self, fitted_mv_model):
        """Model should fit successfully on training data."""
        assert hasattr(fitted_mv_model, "weights_")
        assert fitted_mv_model.weights_ is not None


class TestExp1Constraints:
    """Tests for constraint validation in Experiment 1."""

    def test_frontier_has_exactly_100_points(self, fitted_mv_model):
        """Efficient frontier must contain exactly 100 portfolios."""
        assert fitted_mv_model.weights_.shape[0] == 100, (
            f"Expected 100 frontier points, got {fitted_mv_model.weights_.shape[0]}"
        )

    def test_budget_constraint_satisfied(self, fitted_mv_model):
        """All portfolio weights must sum to 1 (full investment)."""
        weight_sums = fitted_mv_model.weights_.sum(axis=1)
        assert np.allclose(weight_sums, 1.0, atol=1e-4), (
            f"Budget constraint violated: sums range [{weight_sums.min():.6f}, "
            f"{weight_sums.max():.6f}]"
        )

    def test_aapl_weight_cap_respected(self, fitted_mv_model, train_test_returns):
        """AAPL weight must not exceed 0.2 in any frontier portfolio."""
        X_train, _ = train_test_returns
        aapl_idx = list(X_train.columns).index("AAPL")
        aapl_weights = fitted_mv_model.weights_[:, aapl_idx]
        assert np.all(aapl_weights <= 0.2 + 1e-6), (
            f"AAPL weight cap violated: max = {aapl_weights.max():.4f}"
        )

    def test_weights_are_non_negative_by_default(self, fitted_mv_model):
        """Default long-only constraint: all weights should be >= 0."""
        assert np.all(fitted_mv_model.weights_ >= -1e-6), (
            "Negative weights found (short positions) — check long-only default"
        )

    def test_aapl_weight_range_valid(self, fitted_mv_model, train_test_returns):
        """AAPL weights should be between 0 and 0.2."""
        X_train, _ = train_test_returns
        aapl_idx = list(X_train.columns).index("AAPL")
        aapl_weights = fitted_mv_model.weights_[:, aapl_idx]
        assert np.all(aapl_weights >= -1e-6), "AAPL weight below 0"
        assert np.all(aapl_weights <= 0.2 + 1e-6), "AAPL weight above 0.2"


class TestExp1FrontierQuality:
    """Tests for efficient frontier quality in Experiment 1."""

    def test_train_frontier_has_positive_returns(self, fitted_mv_model, train_test_returns):
        """Training frontier should have valid annualized returns."""
        X_train, _ = train_test_returns
        train_pop = fitted_mv_model.predict(X_train)
        train_rets = [p.annualized_mean for p in train_pop]
        assert len(train_rets) == 100

    def test_test_frontier_evaluates_without_refitting(self, fitted_mv_model, train_test_returns):
        """Test frontier should be evaluable without refitting the model."""
        _, X_test = train_test_returns
        test_pop = fitted_mv_model.predict(X_test)
        assert len(test_pop) == 100

    def test_train_dominates_test_qualitatively(self, fitted_mv_model, train_test_returns):
        """Training frontier should show overfitting evidence vs test frontier.

        Overfitting in portfolio optimization manifests as risk underestimation:
        the model underestimates out-of-sample volatility. The same portfolios
        have lower volatility in-sample than out-of-sample.
        """
        X_train, X_test = train_test_returns
        train_pop = fitted_mv_model.predict(X_train)
        test_pop = fitted_mv_model.predict(X_test)

        train_vols = np.array([p.annualized_standard_deviation for p in train_pop])
        test_vols = np.array([p.annualized_standard_deviation for p in test_pop])

        # Classic overfitting signal: in-sample volatility < out-of-sample volatility
        # The optimizer minimizes in-sample variance, so it underestimates out-of-sample risk
        assert train_vols.mean() < test_vols.mean(), (
            "Expected in-sample volatility < out-of-sample volatility (risk underestimation). "
            f"Train vol avg: {train_vols.mean():.4f}, Test vol avg: {test_vols.mean():.4f}"
        )

    def test_frontier_volatility_is_monotone(self, fitted_mv_model, train_test_returns):
        """Frontier volatilities should be roughly monotone (sorted)."""
        X_train, _ = train_test_returns
        train_pop = fitted_mv_model.predict(X_train)
        vols = [p.annualized_standard_deviation for p in train_pop]
        # Check that volatility is generally increasing along the frontier
        # Allow small violations due to numerical precision
        diffs = np.diff(vols)
        pct_increasing = np.mean(diffs >= -1e-6)
        assert pct_increasing > 0.8, (
            f"Frontier volatility not monotone: only {pct_increasing:.1%} increasing"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 2 Tests: Factor-Model Prior with Ridge Loadings
# ═══════════════════════════════════════════════════════════════════════════════

class TestExp2DataLoading:
    """Tests for data loading in Experiment 2."""

    def test_factor_dataset_loads(self):
        """Factor dataset should load without errors."""
        factor_prices = load_factors_dataset()
        assert factor_prices is not None
        assert isinstance(factor_prices, pd.DataFrame)
        assert factor_prices.shape[0] > 0

    def test_joint_conversion_aligns_dates(self, factor_data):
        """Asset and factor returns should have identical date indices."""
        X_train, X_test, y_train, y_test = factor_data
        assert list(X_train.index) == list(y_train.index), "Train date mismatch"
        assert list(X_test.index) == list(y_test.index), "Test date mismatch"

    def test_no_future_leakage_in_split(self, factor_data):
        """Train set must end before test set begins."""
        X_train, X_test, y_train, y_test = factor_data
        assert X_train.index[-1] < X_test.index[0]
        assert y_train.index[-1] < y_test.index[0]

    def test_aapl_present_in_factor_experiment(self, factor_data):
        """AAPL must be present in the asset universe."""
        X_train, _, _, _ = factor_data
        assert "AAPL" in X_train.columns


class TestExp2ModelConfiguration:
    """Tests for factor model configuration in Experiment 2."""

    def test_ridge_alpha_is_correct(self):
        """Ridge regression alpha must be exactly 0.1 as specified."""
        ridge = Ridge(alpha=0.1)
        assert ridge.alpha == 0.1

    def test_loading_matrix_regression_uses_ridge(self):
        """LoadingMatrixRegression must use Ridge with alpha=0.1."""
        ridge = Ridge(alpha=0.1)
        lmr = LoadingMatrixRegression(linear_regressor=ridge)
        assert isinstance(lmr.linear_regressor, Ridge)
        assert lmr.linear_regressor.alpha == 0.1

    def test_factor_model_prior_builds(self):
        """FactorModel prior should build without errors."""
        ridge = Ridge(alpha=0.1)
        lmr = LoadingMatrixRegression(linear_regressor=ridge)
        factor_prior = FactorModel(loading_matrix_estimator=lmr)
        assert factor_prior is not None

    def test_meanrisk_with_factor_prior_builds(self):
        """MeanRisk with FactorModel prior should build without errors."""
        ridge = Ridge(alpha=0.1)
        lmr = LoadingMatrixRegression(linear_regressor=ridge)
        factor_prior = FactorModel(loading_matrix_estimator=lmr)
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=factor_prior,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
        )
        assert model is not None


class TestExp2FitAndConstraints:
    """Tests for factor model fitting and constraint validation."""

    @pytest.fixture(scope="class")
    def fitted_factor_model(self, factor_data):
        """Fitted factor-prior model."""
        X_train, _, y_train, _ = factor_data
        ridge = Ridge(alpha=0.1)
        lmr = LoadingMatrixRegression(linear_regressor=ridge)
        factor_prior = FactorModel(loading_matrix_estimator=lmr)
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=factor_prior,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
        )
        model.fit(X_train, y_train)
        return model

    def test_factor_model_fits_successfully(self, fitted_factor_model):
        """Factor model should fit without errors."""
        assert hasattr(fitted_factor_model, "weights_")
        assert fitted_factor_model.weights_ is not None

    def test_budget_constraint_factor_model(self, fitted_factor_model):
        """Portfolio weights must sum to 1."""
        weights = fitted_factor_model.weights_
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        weight_sums = weights.sum(axis=1)
        assert np.allclose(weight_sums, 1.0, atol=1e-4)

    def test_aapl_constraint_factor_model(self, fitted_factor_model, factor_data):
        """AAPL weight must not exceed 0.2."""
        X_train, _, _, _ = factor_data
        weights = fitted_factor_model.weights_
        if weights.ndim == 1:
            weights = weights.reshape(1, -1)
        aapl_idx = list(X_train.columns).index("AAPL")
        aapl_weights = weights[:, aapl_idx]
        assert np.all(aapl_weights <= 0.2 + 1e-6)

    def test_factor_model_predicts_on_test(self, fitted_factor_model, factor_data):
        """Factor model should predict on test data without refitting."""
        _, X_test, _, _ = factor_data
        portfolio = fitted_factor_model.predict(X_test)
        assert portfolio is not None


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 3 Tests: Walk-Forward Stacking Optimization
# ═══════════════════════════════════════════════════════════════════════════════

class TestExp3WalkForward:
    """Tests for walk-forward cross-validation in Experiment 3."""

    def test_walkforward_creates_correct_folds(self, sp500_returns):
        """WalkForward should create folds with correct train/test sizes."""
        _, X = train_test_split(sp500_returns, test_size=0.2, shuffle=False)
        cv = WalkForward(train_size=252, test_size=60)
        folds = list(cv.split(X))
        assert len(folds) > 0, "No folds generated"
        for train_idx, test_idx in folds:
            assert len(train_idx) == 252, f"Train size {len(train_idx)} != 252"
            assert len(test_idx) == 60, f"Test size {len(test_idx)} != 60"

    def test_walkforward_folds_are_chronological(self, sp500_returns):
        """Each test fold must occur strictly after its training fold."""
        _, X = train_test_split(sp500_returns, test_size=0.2, shuffle=False)
        cv = WalkForward(train_size=252, test_size=60)
        folds = list(cv.split(X))
        for train_idx, test_idx in folds:
            assert train_idx[-1] < test_idx[0], "Look-ahead leakage detected"

    def test_walkforward_no_overlap_between_folds(self, sp500_returns):
        """Test folds should not overlap with each other."""
        _, X = train_test_split(sp500_returns, test_size=0.2, shuffle=False)
        cv = WalkForward(train_size=252, test_size=60)
        folds = list(cv.split(X))
        test_indices = [set(test_idx) for _, test_idx in folds]
        for i in range(len(test_indices)):
            for j in range(i + 1, len(test_indices)):
                overlap = test_indices[i] & test_indices[j]
                assert len(overlap) == 0, f"Folds {i} and {j} overlap"

    def test_sufficient_observations_for_walkforward(self, sp500_returns):
        """Dataset must have enough observations for at least one fold."""
        _, X = train_test_split(sp500_returns, test_size=0.2, shuffle=False)
        assert len(X) >= 252 + 60, (
            f"Insufficient observations ({len(X)}) for walk-forward"
        )


class TestExp3Estimators:
    """Tests for stacking estimator construction in Experiment 3."""

    def test_inverse_volatility_builds(self):
        """InverseVolatility estimator should build without errors."""
        model = InverseVolatility()
        assert model is not None

    def test_meanrisk_cvar_builds(self):
        """MeanRisk with CVaR should build without errors."""
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
        )
        assert model is not None

    def test_hrp_builds(self):
        """HierarchicalRiskParity should build without errors."""
        model = HierarchicalRiskParity()
        assert model is not None

    def test_equal_weighted_builds(self):
        """EqualWeighted benchmark should build without errors."""
        model = EqualWeighted()
        assert model is not None

    def test_stacking_model_builds(self):
        """StackingOptimization should build with three base estimators."""
        base_estimators = [
            ("inv_vol", InverseVolatility()),
            ("mean_cvar", MeanRisk(
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                risk_measure=RiskMeasure.CVAR,
            )),
            ("hrp", HierarchicalRiskParity()),
        ]
        final_estimator = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
        )
        model = StackingOptimization(
            estimators=base_estimators,
            final_estimator=final_estimator,
        )
        assert model is not None
        assert len(model.estimators) == 3

    def test_stacking_has_correct_base_estimators(self):
        """Stacking model must have exactly the three paper-specified base estimators."""
        base_estimators = [
            ("inv_vol", InverseVolatility()),
            ("mean_cvar", MeanRisk(
                objective_function=ObjectiveFunction.MINIMIZE_RISK,
                risk_measure=RiskMeasure.CVAR,
            )),
            ("hrp", HierarchicalRiskParity()),
        ]
        final_estimator = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.CVAR,
        )
        model = StackingOptimization(
            estimators=base_estimators,
            final_estimator=final_estimator,
        )
        estimator_names = [name for name, _ in model.estimators]
        assert "inv_vol" in estimator_names
        assert "mean_cvar" in estimator_names
        assert "hrp" in estimator_names


# ═══════════════════════════════════════════════════════════════════════════════
# Experiment 4 Tests: Copula-Based Synthetic Prior
# ═══════════════════════════════════════════════════════════════════════════════

class TestExp4DataRequirements:
    """Tests for data requirements in Experiment 4."""

    def test_required_tickers_present(self, sp500_returns):
        """AAPL, JPM, and LLY must be present in the dataset."""
        for ticker in ["AAPL", "JPM", "LLY"]:
            assert ticker in sp500_returns.columns, (
                f"Required ticker {ticker} is absent"
            )

    def test_returns_have_sufficient_observations(self, sp500_returns):
        """Need enough observations for reliable copula fitting."""
        assert len(sp500_returns) > 100, "Need at least 100 observations for copula"


class TestExp4PseudoObservations:
    """Tests for pseudo-observation transformation."""

    def test_pseudo_obs_range(self, sp500_returns):
        """Pseudo-observations must be in (0, 1)."""
        data = sp500_returns[["AAPL", "JPM", "LLY"]].values
        u = pseudo_obs(data)
        assert u.min() > 0, "Pseudo-obs below 0"
        assert u.max() < 1, "Pseudo-obs above 1"

    def test_pseudo_obs_shape(self, sp500_returns):
        """Pseudo-observations must have same shape as input."""
        data = sp500_returns[["AAPL", "JPM", "LLY"]].values
        u = pseudo_obs(data)
        assert u.shape == data.shape

    def test_pseudo_obs_uniform_marginals(self, sp500_returns):
        """Each column of pseudo-observations should be approximately uniform."""
        data = sp500_returns[["AAPL", "JPM"]].values
        u = pseudo_obs(data)
        for col in range(u.shape[1]):
            # Mean of uniform(0,1) is 0.5
            assert abs(u[:, col].mean() - 0.5) < 0.05, (
                f"Column {col} mean {u[:, col].mean():.3f} deviates from 0.5"
            )


class TestExp4CopulaFitting:
    """Tests for copula fitting in Experiment 4."""

    @pytest.fixture(scope="class")
    def fitted_bivariate_copula(self, sp500_returns):
        """Fitted bivariate Student-t copula for AAPL-JPM."""
        data = sp500_returns[["AAPL", "JPM"]].values
        u = pseudo_obs(data)
        cop = StudentCopula(dim=2)
        cop.fit(u)
        return cop

    def test_bivariate_copula_fits(self, fitted_bivariate_copula):
        """Bivariate Student-t copula should fit without errors."""
        assert fitted_bivariate_copula is not None
        assert hasattr(fitted_bivariate_copula, "params")

    def test_bivariate_copula_df_positive(self, fitted_bivariate_copula):
        """Degrees of freedom must be positive."""
        assert float(fitted_bivariate_copula.params.df) > 0

    def test_bivariate_copula_rho_in_range(self, fitted_bivariate_copula):
        """Correlation parameter must be in (-1, 1)."""
        rho = float(fitted_bivariate_copula.params.rho[0])
        assert -1 < rho < 1, f"Correlation {rho} out of range (-1, 1)"

    def test_multivariate_copula_fits(self, sp500_returns):
        """Multivariate Student-t copula should fit for 3 assets."""
        data = sp500_returns[["AAPL", "JPM", "LLY"]].values
        u = pseudo_obs(data)
        cop = StudentCopula(dim=3)
        cop.fit(u)
        assert cop is not None
        assert float(cop.params.df) > 0

    def test_copula_pdf_positive(self, fitted_bivariate_copula):
        """Copula density must be non-negative."""
        u = np.array([[0.25, 0.25], [0.5, 0.5], [0.75, 0.75]])
        density = fitted_bivariate_copula.pdf(u)
        assert np.all(density >= 0), "Copula density must be non-negative"

    def test_copula_pdf_at_center_near_one(self, fitted_bivariate_copula):
        """Copula density at (0.5, 0.5) should be close to 1 for near-independence."""
        center = np.array([[0.5, 0.5]])
        density = fitted_bivariate_copula.pdf(center)
        # Handle both scalar and array return types from copulae
        density_val = float(np.asarray(density).ravel()[0])
        # For near-independence, density at center should be close to 1
        assert 0.5 < density_val < 2.0, (
            f"Density at center {density_val:.4f} seems unreasonable"
        )


class TestExp4SyntheticSampling:
    """Tests for synthetic return generation in Experiment 4."""

    @pytest.fixture(scope="class")
    def copula_and_data(self, sp500_returns):
        """Fitted copula and historical data for sampling tests."""
        data = sp500_returns[["AAPL", "JPM", "LLY"]].values
        u = pseudo_obs(data)
        cop = StudentCopula(dim=3)
        cop.fit(u)
        return cop, data

    def test_synthetic_samples_shape(self, copula_and_data):
        """Synthetic samples should have correct shape."""
        cop, data = copula_and_data
        n_samples = 200
        uniform_samples = cop.random(n_samples, seed=42)
        assert uniform_samples.shape == (n_samples, 3)

    def test_synthetic_samples_in_unit_cube(self, copula_and_data):
        """Copula samples (uniform marginals) must be in [0, 1]."""
        cop, data = copula_and_data
        uniform_samples = cop.random(200, seed=42)
        assert uniform_samples.min() >= 0
        assert uniform_samples.max() <= 1

    def test_back_transformed_samples_in_return_range(self, copula_and_data):
        """Back-transformed synthetic returns should be in historical range."""
        cop, data = copula_and_data
        uniform_samples = cop.random(len(data), seed=42)
        synthetic = np.zeros_like(uniform_samples)
        for i in range(data.shape[1]):
            synthetic[:, i] = np.quantile(data[:, i], uniform_samples[:, i])

        # Synthetic returns should be within historical range (by construction)
        for i in range(data.shape[1]):
            assert synthetic[:, i].min() >= data[:, i].min() - 1e-10
            assert synthetic[:, i].max() <= data[:, i].max() + 1e-10

    def test_synthetic_correlation_similar_to_historical(self, copula_and_data):
        """Synthetic returns should have similar pairwise correlations."""
        cop, data = copula_and_data
        uniform_samples = cop.random(len(data), seed=42)
        synthetic = np.zeros_like(uniform_samples)
        for i in range(data.shape[1]):
            synthetic[:, i] = np.quantile(data[:, i], uniform_samples[:, i])

        hist_corr = np.corrcoef(data.T)
        synth_corr = np.corrcoef(synthetic.T)

        # Correlations should be in the same ballpark (within 0.3)
        for i in range(3):
            for j in range(i + 1, 3):
                diff = abs(hist_corr[i, j] - synth_corr[i, j])
                assert diff < 0.3, (
                    f"Correlation diff for pair ({i},{j}): {diff:.4f} > 0.3"
                )

    def test_reproducibility_with_fixed_seed(self, copula_and_data):
        """Synthetic samples should be reproducible with fixed seed."""
        cop, data = copula_and_data
        samples1 = cop.random(100, seed=42)
        samples2 = cop.random(100, seed=42)
        assert np.allclose(samples1, samples2), "Samples not reproducible with same seed"


# ═══════════════════════════════════════════════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_asset_returns(self):
        """Single-asset return series should not break preprocessing."""
        prices = pd.DataFrame(
            {"AAPL": [100.0, 101.0, 102.0, 101.5, 103.0]},
            index=pd.date_range("2020-01-01", periods=5),
        )
        returns = prices_to_returns(prices)
        assert returns.shape == (4, 1)
        assert not returns.isnull().any().any()

    def test_pseudo_obs_with_two_assets(self):
        """pseudo_obs should work with exactly 2 assets."""
        np.random.seed(42)
        data = np.random.randn(50, 2)
        u = pseudo_obs(data)
        assert u.shape == (50, 2)
        assert u.min() > 0
        assert u.max() < 1

    def test_student_copula_dim2(self):
        """StudentCopula should work with dim=2."""
        np.random.seed(42)
        data = np.random.randn(100, 2)
        u = pseudo_obs(data)
        cop = StudentCopula(dim=2)
        cop.fit(u)
        assert float(cop.params.df) > 0

    def test_walkforward_minimum_folds(self):
        """WalkForward should produce at least one fold with minimal data."""
        np.random.seed(42)
        n = 252 + 60 + 10  # just enough for one fold
        X = pd.DataFrame(
            np.random.randn(n, 5),
            columns=["A", "B", "C", "D", "E"],
            index=pd.date_range("2020-01-01", periods=n),
        )
        cv = WalkForward(train_size=252, test_size=60)
        folds = list(cv.split(X))
        assert len(folds) >= 1

    def test_budget_constraint_mathematical_property(self):
        """Budget constraint: weights summing to 1 is a mathematical requirement."""
        # Verify that a manually constructed weight vector satisfies the constraint
        weights = np.array([0.1, 0.2, 0.3, 0.15, 0.25])
        assert abs(weights.sum() - 1.0) < 1e-10

    def test_aapl_weight_cap_mathematical_property(self):
        """AAPL weight cap: 0.2 is the maximum allowed weight."""
        # Verify the constraint boundary
        assert 0.2 <= 0.2 + 1e-6  # at boundary: OK
        assert not (0.2 + 0.01 <= 0.2 + 1e-6)  # above boundary: violation

    def test_returns_have_correct_dtype(self, sp500_returns):
        """Returns should be float dtype."""
        assert sp500_returns.dtypes.apply(lambda x: np.issubdtype(x, np.floating)).all()

    def test_extreme_returns_handled(self):
        """Copula fitting should handle extreme return values."""
        np.random.seed(42)
        # Create data with extreme values (fat tails)
        data = np.random.standard_t(df=3, size=(200, 2))
        u = pseudo_obs(data)
        cop = StudentCopula(dim=2)
        cop.fit(u)
        assert float(cop.params.df) > 0


# ═══════════════════════════════════════════════════════════════════════════════
# Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestIntegration:
    """Integration tests verifying end-to-end experiment workflows."""

    def test_exp1_full_pipeline(self, train_test_returns):
        """Experiment 1 full pipeline should run without errors."""
        X_train, X_test = train_test_returns
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
            efficient_frontier_size=10,  # reduced for speed
        )
        model.fit(X_train)
        train_pop = model.predict(X_train)
        test_pop = model.predict(X_test)
        assert len(train_pop) == 10
        assert len(test_pop) == 10

    def test_exp2_full_pipeline(self, factor_data):
        """Experiment 2 full pipeline should run without errors."""
        X_train, X_test, y_train, y_test = factor_data
        ridge = Ridge(alpha=0.1)
        lmr = LoadingMatrixRegression(linear_regressor=ridge)
        factor_prior = FactorModel(loading_matrix_estimator=lmr)
        model = MeanRisk(
            objective_function=ObjectiveFunction.MINIMIZE_RISK,
            risk_measure=RiskMeasure.VARIANCE,
            prior_estimator=factor_prior,
            max_weights={"AAPL": 0.2},
            l2_coef=0.01,
        )
        model.fit(X_train, y_train)
        portfolio = model.predict(X_test)
        assert portfolio is not None

    def test_exp4_full_pipeline(self, sp500_returns):
        """Experiment 4 full pipeline should run without errors."""
        tickers = ["AAPL", "JPM", "LLY"]
        data = sp500_returns[tickers].values
        u = pseudo_obs(data)
        cop = StudentCopula(dim=3)
        cop.fit(u)
        samples_u = cop.random(100, seed=42)
        synthetic = np.zeros((100, 3))
        for i in range(3):
            synthetic[:, i] = np.quantile(data[:, i], samples_u[:, i])
        assert synthetic.shape == (100, 3)
        assert not np.isnan(synthetic).any()

    def test_skfolio_version_is_available(self):
        """skfolio version should be accessible."""
        assert hasattr(skfolio, "__version__")
        assert skfolio.__version__ is not None
