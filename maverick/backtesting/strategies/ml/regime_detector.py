"""Market regime detection (KMeans / GaussianMixture / threshold).

Split out of `regime_aware.py` (see the Task 6 report): the ported legacy
module put `MarketRegimeDetector` and `RegimeAwareStrategy` (now
`regime_aware.py`, imports `MarketRegimeDetector` from here) in one 899-line
file, over this repo's 500-line-per-module cap. `extract_regime_features`
moved to the standalone `regime_features.py` (it never referenced `self`);
this class now delegates to it. No behavior changed by any of these moves.

One new seam, disclosed per the Task 6 determinism rule:
`MarketRegimeDetector` gained a trailing `random_state: int | None = None`
constructor parameter. The `GaussianMixture`/`KMeans` estimators already
hardcode `random_state=42` with no way to override it, so their default
behavior is unaffected either way. The one genuinely unseeded spot is the
zero-variance-feature noise injection in `fit_regimes`, which called the
*global* `np.random.normal` with no seam at all. When `random_state` is
`None` (the default), behavior is unchanged: `_initialize_model` still
hardcodes 42 for the estimator, and noise injection still draws from
`np.random.normal`. When a seed is supplied, both the estimator's
`random_state` and the noise-injection generator use it, so tests can pin a
different seed without mutating global numpy state.
"""

import logging

import numpy as np
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .regime_features import extract_regime_features

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detect market regimes using various statistical methods."""

    def __init__(
        self,
        method: str = "hmm",
        n_regimes: int = 3,
        lookback_period: int = 50,
        random_state: int | None = None,
    ):
        """Initialize regime detector.

        Args:
            method: Detection method ('hmm', 'kmeans', 'threshold')
            n_regimes: Number of market regimes to detect
            lookback_period: Period for regime detection
            random_state: Optional seed. `None` (default) preserves legacy
                behavior: the estimator is seeded to 42 and zero-variance
                noise injection draws from the global `np.random` state.
        """
        self.method = method
        self.n_regimes = n_regimes
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self.random_state = random_state
        self._rng = (
            np.random.default_rng(random_state) if random_state is not None else None
        )

        # Initialize detection model
        self.model = None
        self.is_fitted = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize regime detection model with better configurations."""
        seed = self.random_state if self.random_state is not None else 42
        if self.method == "hmm":
            # Use GaussianMixture with more stable configuration
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type="diag",  # Use diagonal covariance for stability
                random_state=seed,
                max_iter=200,
                tol=1e-6,
                reg_covar=1e-6,  # Regularization for numerical stability
                init_params="kmeans",  # Better initialization
                warm_start=False,
            )
        elif self.method == "kmeans":
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=seed,
                n_init=10,
                max_iter=500,
                tol=1e-6,
                algorithm="lloyd",  # More stable algorithm
            )
        elif self.method == "threshold":
            # Threshold-based regime detection
            self.model = None
        else:
            raise ValueError(f"Unsupported regime detection method: {self.method}")

    def extract_regime_features(self, data: DataFrame) -> np.ndarray:
        """Extract robust features for regime detection.

        Delegates to the standalone `regime_features.extract_regime_features`
        (moved there because it never referenced `self`).

        Args:
            data: Price data

        Returns:
            Feature array with consistent dimensionality and stability
        """
        return extract_regime_features(data)

    def detect_regime_threshold(self, data: DataFrame) -> int:
        """Detect regime using threshold-based method.

        Args:
            data: Price data

        Returns:
            Regime label (0: bear/declining, 1: sideways, 2: bull/trending)
        """
        if len(data) < 20:
            return 1  # Default to sideways

        # Calculate trend and volatility measures
        returns = data["close"].pct_change()

        # Trend measure (20-day slope)
        x = np.arange(20)
        y = data["close"].iloc[-20:].values
        trend_slope = np.polyfit(x, y, 1)[0] / y[-1]  # Normalized slope

        # Volatility measure
        vol_20 = returns.rolling(20).std().iloc[-1] * np.sqrt(252)

        # Define regime thresholds
        trend_threshold = 0.001  # 0.1% daily trend threshold
        vol_threshold = 0.25  # 25% annual volatility threshold

        # Classify regime
        if trend_slope > trend_threshold and vol_20 < vol_threshold:
            return 2  # Bull/trending market
        elif trend_slope < -trend_threshold and vol_20 > vol_threshold:
            return 0  # Bear/declining market
        else:
            return 1  # Sideways/uncertain market

    def fit_regimes(self, data: DataFrame) -> None:
        """Fit regime detection model to historical data with enhanced robustness.

        Args:
            data: Historical price data
        """
        if self.method == "threshold":
            self.is_fitted = True
            return

        try:
            # Need sufficient data for stable regime detection
            min_required_samples = max(50, self.n_regimes * 20)
            if len(data) < min_required_samples + self.lookback_period:
                logger.warning(
                    f"Insufficient data for regime fitting: {len(data)} < {min_required_samples + self.lookback_period}"
                )
                self.is_fitted = True
                return

            # Extract features for regime detection with temporal consistency
            feature_list = []
            feature_consistency_count = None

            # Use overlapping windows for more stable regime detection
            step_size = max(1, self.lookback_period // 10)

            for i in range(self.lookback_period, len(data), step_size):
                window_data = data.iloc[max(0, i - self.lookback_period) : i + 1]
                features = self.extract_regime_features(window_data)

                if len(features) > 0 and np.all(np.isfinite(features)):
                    # Check feature consistency
                    if feature_consistency_count is None:
                        feature_consistency_count = len(features)
                    elif len(features) != feature_consistency_count:
                        logger.warning(
                            f"Feature dimension mismatch: expected {feature_consistency_count}, got {len(features)}"
                        )
                        continue

                    feature_list.append(features)

            if len(feature_list) < min_required_samples:
                logger.warning(
                    f"Insufficient valid samples for regime fitting: {len(feature_list)} < {min_required_samples}"
                )
                self.is_fitted = True
                return

            # Ensure we have valid feature_list before creating array
            if len(feature_list) == 0:
                logger.warning(
                    "Empty feature list after filtering, cannot create feature matrix"
                )
                self.is_fitted = True
                return

            X = np.array(feature_list)

            # Additional data quality checks
            if X.size == 0:
                logger.warning("Empty feature matrix, cannot fit regime detector")
                self.is_fitted = True
                return
            elif np.any(np.isnan(X)) or np.any(np.isinf(X)):
                logger.warning("Found NaN or inf values in feature matrix, cleaning...")
                X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)

            # Check for zero variance features
            feature_std = np.std(X, axis=0)
            zero_variance_features = np.where(feature_std < 1e-8)[0]
            if len(zero_variance_features) > 0:
                logger.debug(
                    f"Found {len(zero_variance_features)} zero-variance features"
                )
                # Add small noise to zero-variance features
                normal = self._rng.normal if self._rng is not None else np.random.normal
                for idx in zero_variance_features:
                    X[:, idx] += normal(0, 1e-6, X.shape[0])

            # Scale features with robust scaler
            X_scaled = self.scaler.fit_transform(X)

            # Fit model with better error handling
            try:
                assert self.model is not None  # set by _initialize_model
                if self.method == "hmm":
                    # For GaussianMixture, ensure numerical stability
                    self.model.fit(X_scaled)

                    # Validate fitted model
                    # (`weights_`/`converged_` are fit-time-only GaussianMixture
                    # attributes absent from the stub -- `getattr` sidesteps
                    # attribute resolution here without changing behavior)
                    weights = getattr(self.model, "weights_", None)  # noqa: B009
                    if weights is None or len(weights) != self.n_regimes:
                        raise ValueError("Model fitting failed - invalid weights")

                    # Check convergence
                    if not getattr(self.model, "converged_"):  # noqa: B009
                        logger.warning(
                            "GaussianMixture did not converge, but will proceed"
                        )

                elif self.method == "kmeans":
                    self.model.fit(X_scaled)

                    # Validate fitted model (`cluster_centers_` is likewise
                    # fit-time-only and absent from the KMeans stub)
                    centers = getattr(self.model, "cluster_centers_", None)  # noqa: B009
                    if centers is None or len(centers) != self.n_regimes:
                        raise ValueError(
                            "KMeans fitting failed - invalid cluster centers"
                        )

                self.is_fitted = True

                # Log fitting success with model diagnostics
                if self.method == "hmm":
                    avg_log_likelihood = self.model.score(X_scaled) / len(X_scaled)
                    logger.info(
                        f"Fitted {self.method} regime detector with {len(X)} samples, avg log-likelihood: {avg_log_likelihood:.4f}"
                    )
                else:
                    inertia = getattr(self.model, "inertia_", "N/A")  # noqa: B009
                    logger.info(
                        f"Fitted {self.method} regime detector with {len(X)} samples, inertia: {inertia}"
                    )

            except Exception as model_error:
                logger.error(f"Model fitting failed: {model_error}")
                logger.info("Falling back to threshold method")
                self.method = "threshold"  # Fallback to threshold method
                self.is_fitted = True

        except Exception as e:
            logger.error(f"Error fitting regime detector: {e}")
            self.is_fitted = True  # Allow fallback to threshold method

    def detect_current_regime(self, data: DataFrame) -> int:
        """Detect current market regime with enhanced error handling.

        Args:
            data: Recent price data

        Returns:
            Regime label (0: bear, 1: sideways, 2: bull)
        """
        if not self.is_fitted:
            logger.debug("Regime detector not fitted, using threshold method")
            return self.detect_regime_threshold(data)

        try:
            if self.method == "threshold":
                return self.detect_regime_threshold(data)

            # Extract features for current regime
            features = self.extract_regime_features(data)

            if len(features) == 0:
                logger.debug("No features extracted, falling back to threshold method")
                return self.detect_regime_threshold(data)

            # Check for non-finite features only if features array is not empty
            if features.size > 0 and np.any(~np.isfinite(features)):
                logger.debug("Non-finite features detected, cleaning and proceeding")
                features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)

            # Validate feature consistency with training
            expected_features = (
                self.scaler.n_features_in_
                if hasattr(self.scaler, "n_features_in_")
                else None
            )
            if expected_features is not None and len(features) != expected_features:
                logger.warning(
                    f"Feature count mismatch in prediction: expected {expected_features}, got {len(features)}"
                )
                return self.detect_regime_threshold(data)

            # Scale features and predict regime
            try:
                assert self.model is not None  # method != "threshold" implies set
                X = self.scaler.transform([features])
                regime = self.model.predict(X)[0]

                # Validate regime prediction
                if regime < 0 or regime >= self.n_regimes:
                    logger.warning(
                        f"Invalid regime prediction: {regime}, using threshold method"
                    )
                    return self.detect_regime_threshold(data)

                return int(regime)

            except Exception as pred_error:
                logger.debug(
                    f"Prediction error: {pred_error}, falling back to threshold method"
                )
                return self.detect_regime_threshold(data)

        except Exception as e:
            logger.error(f"Error detecting current regime: {e}")
            return self.detect_regime_threshold(data)  # Always fallback to threshold

    def get_regime_probabilities(self, data: DataFrame) -> np.ndarray:
        """Get probabilities for each regime.

        Args:
            data: Recent price data

        Returns:
            Array of regime probabilities
        """
        if not self.is_fitted or self.method == "threshold":
            # For threshold method, return deterministic probabilities
            regime = self.detect_current_regime(data)
            probs = np.zeros(self.n_regimes)
            probs[regime] = 1.0
            return probs

        try:
            features = self.extract_regime_features(data)

            if len(features) == 0:
                return np.ones(self.n_regimes) / self.n_regimes
            elif features.size > 0 and np.any(np.isnan(features)):
                return np.ones(self.n_regimes) / self.n_regimes

            assert self.model is not None  # method != "threshold" implies set
            X = self.scaler.transform([features])

            if hasattr(self.model, "predict_proba"):
                # `predict_proba` only exists on GaussianMixture, not KMeans;
                # `getattr` sidesteps the union-member attribute check that
                # `hasattr` alone doesn't narrow, without changing behavior.
                predict_proba = getattr(self.model, "predict_proba")  # noqa: B009
                return predict_proba(X)[0]
            else:
                # For methods without probabilities, return one-hot encoding
                regime = self.model.predict(X)[0]
                probs = np.zeros(self.n_regimes)
                probs[regime] = 1.0
                return probs

        except Exception as e:
            logger.error(f"Error getting regime probabilities: {e}")
            return np.ones(self.n_regimes) / self.n_regimes
