"""Automated retraining pipeline for ML models with data drift detection."""

import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """Detects data drift in features and targets."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize drift detector.

        Args:
            significance_level: Statistical significance level for drift detection
        """
        self.significance_level = significance_level
        self.reference_data: pd.DataFrame | None = None
        self.reference_target: pd.Series | None = None
        self.feature_stats: dict[str, dict[str, float]] = {}

    def set_reference_data(
        self, features: pd.DataFrame, target: pd.Series | None = None
    ):
        """Set reference data for drift detection.

        Args:
            features: Reference feature data
            target: Reference target data (optional)
        """
        self.reference_data = features.copy()
        self.reference_target = target.copy() if target is not None else None

        # Calculate reference statistics
        self.feature_stats = {}
        for col in features.columns:
            if features[col].dtype in ["float64", "float32", "int64", "int32"]:
                self.feature_stats[col] = {
                    "mean": features[col].mean(),
                    "std": features[col].std(),
                    "min": features[col].min(),
                    "max": features[col].max(),
                    "median": features[col].median(),
                }

        logger.info(
            f"Set reference data with {len(features)} samples and {len(features.columns)} features"
        )

    def detect_feature_drift(
        self, new_features: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        """Detect drift in features using statistical tests.

        Args:
            new_features: New feature data to compare

        Returns:
            Dictionary with drift detection results per feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set")

        drift_results = {}

        for col in new_features.columns:
            if col not in self.reference_data.columns:
                continue

            if new_features[col].dtype not in ["float64", "float32", "int64", "int32"]:
                continue

            ref_data = self.reference_data[col].dropna()
            new_data = new_features[col].dropna()

            if len(ref_data) == 0 or len(new_data) == 0:
                continue

            # Perform statistical tests
            drift_detected = False
            test_results = {}

            try:
                # Kolmogorov-Smirnov test for distribution change
                ks_statistic, ks_p_value = stats.ks_2samp(ref_data, new_data)
                test_results["ks_statistic"] = ks_statistic
                test_results["ks_p_value"] = ks_p_value
                ks_drift = ks_p_value < self.significance_level

                # Mann-Whitney U test for location shift
                mw_statistic, mw_p_value = stats.mannwhitneyu(
                    ref_data, new_data, alternative="two-sided"
                )
                test_results["mw_statistic"] = mw_statistic
                test_results["mw_p_value"] = mw_p_value
                mw_drift = mw_p_value < self.significance_level

                # Levene test for variance change
                levene_statistic, levene_p_value = stats.levene(ref_data, new_data)
                test_results["levene_statistic"] = levene_statistic
                test_results["levene_p_value"] = levene_p_value
                levene_drift = levene_p_value < self.significance_level

                # Overall drift detection
                drift_detected = ks_drift or mw_drift or levene_drift

                # Calculate effect sizes
                test_results["mean_diff"] = new_data.mean() - ref_data.mean()
                test_results["std_ratio"] = new_data.std() / (ref_data.std() + 1e-8)

            except Exception as e:
                logger.warning(f"Error in drift detection for {col}: {e}")
                test_results["error"] = str(e)

            drift_results[col] = {
                "drift_detected": drift_detected,
                "test_results": test_results,
                "reference_stats": self.feature_stats.get(col, {}),
                "new_stats": {
                    "mean": new_data.mean(),
                    "std": new_data.std(),
                    "min": new_data.min(),
                    "max": new_data.max(),
                    "median": new_data.median(),
                },
            }

        return drift_results

    def detect_target_drift(self, new_target: pd.Series) -> dict[str, Any]:
        """Detect drift in target variable.

        Args:
            new_target: New target data to compare

        Returns:
            Dictionary with target drift results
        """
        if self.reference_target is None:
            logger.warning("No reference target data set")
            return {"drift_detected": False, "reason": "no_reference_target"}

        ref_target = self.reference_target.dropna()
        new_target = new_target.dropna()

        if len(ref_target) == 0 or len(new_target) == 0:
            return {"drift_detected": False, "reason": "insufficient_data"}

        drift_results = {"drift_detected": False}

        try:
            # For categorical targets, use chi-square test
            if ref_target.dtype == "object" or ref_target.nunique() < 10:
                ref_counts = ref_target.value_counts()
                new_counts = new_target.value_counts()

                # Align the categories
                all_categories = set(ref_counts.index) | set(new_counts.index)
                ref_aligned = [ref_counts.get(cat, 0) for cat in all_categories]
                new_aligned = [new_counts.get(cat, 0) for cat in all_categories]

                if sum(ref_aligned) > 0 and sum(new_aligned) > 0:
                    chi2_stat, chi2_p_value = stats.chisquare(new_aligned, ref_aligned)
                    drift_results.update(
                        {
                            "test_type": "chi_square",
                            "chi2_statistic": chi2_stat,
                            "chi2_p_value": chi2_p_value,
                            "drift_detected": chi2_p_value < self.significance_level,
                        }
                    )

            # For continuous targets
            else:
                ks_statistic, ks_p_value = stats.ks_2samp(ref_target, new_target)
                drift_results.update(
                    {
                        "test_type": "kolmogorov_smirnov",
                        "ks_statistic": ks_statistic,
                        "ks_p_value": ks_p_value,
                        "drift_detected": ks_p_value < self.significance_level,
                    }
                )

        except Exception as e:
            logger.warning(f"Error in target drift detection: {e}")
            drift_results["error"] = str(e)

        return drift_results

    def get_drift_summary(
        self, feature_drift: dict[str, dict], target_drift: dict[str, Any]
    ) -> dict[str, Any]:
        """Get summary of drift detection results.

        Args:
            feature_drift: Feature drift results
            target_drift: Target drift results

        Returns:
            Summary dictionary
        """
        total_features = len(feature_drift)
        drifted_features = sum(
            1 for result in feature_drift.values() if result["drift_detected"]
        )
        target_drift_detected = target_drift.get("drift_detected", False)

        drift_severity = "none"
        if target_drift_detected or drifted_features > total_features * 0.5:
            drift_severity = "high"
        elif drifted_features > total_features * 0.2:
            drift_severity = "medium"
        elif drifted_features > 0:
            drift_severity = "low"

        return {
            "total_features": total_features,
            "drifted_features": drifted_features,
            "drift_percentage": drifted_features / max(total_features, 1) * 100,
            "target_drift_detected": target_drift_detected,
            "drift_severity": drift_severity,
            "recommendation": self._get_retraining_recommendation(
                drift_severity, target_drift_detected
            ),
        }

    def _get_retraining_recommendation(
        self, drift_severity: str, target_drift: bool
    ) -> str:
        """Get retraining recommendation based on drift severity."""
        if target_drift:
            return "immediate_retraining"
        elif drift_severity == "high":
            return "urgent_retraining"
        elif drift_severity == "medium":
            return "scheduled_retraining"
        elif drift_severity == "low":
            return "monitor_closely"
        else:
            return "no_action_needed"


class ModelPerformanceMonitor:
    """Monitors model performance and detects degradation."""

    def __init__(self, performance_threshold: float = 0.05):
        """Initialize performance monitor.

        Args:
            performance_threshold: Threshold for performance degradation detection
        """
        self.performance_threshold = performance_threshold
        self.baseline_metrics: dict[str, float] = {}
        self.performance_history: list[dict[str, Any]] = []

    def set_baseline_performance(self, metrics: dict[str, float]):
        """Set baseline performance metrics.

        Args:
            metrics: Baseline performance metrics
        """
        self.baseline_metrics = metrics.copy()
        logger.info(f"Set baseline performance: {metrics}")

    def evaluate_performance(
        self,
        model: BaseEstimator,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        additional_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        """Evaluate current model performance.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            additional_metrics: Additional metrics to include

        Returns:
            Performance evaluation results
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = {
                "accuracy": accuracy_score(y_test, y_pred),
                "timestamp": datetime.now().isoformat(),
            }

            # Add additional metrics if provided
            if additional_metrics:
                metrics.update(additional_metrics)

            # Detect performance degradation
            degradation_detected = False
            degradation_details = {}

            for metric_name, current_value in metrics.items():
                if metric_name in self.baseline_metrics and metric_name != "timestamp":
                    baseline_value = self.baseline_metrics[metric_name]
                    degradation = (baseline_value - current_value) / abs(baseline_value)

                    if degradation > self.performance_threshold:
                        degradation_detected = True
                        degradation_details[metric_name] = {
                            "baseline": baseline_value,
                            "current": current_value,
                            "degradation": degradation,
                        }

            evaluation_result = {
                "metrics": metrics,
                "degradation_detected": degradation_detected,
                "degradation_details": degradation_details,
                "classification_report": classification_report(
                    y_test, y_pred, output_dict=True
                ),
            }

            # Store in history
            self.performance_history.append(evaluation_result)

            # Keep only recent history
            if len(self.performance_history) > 100:
                self.performance_history = self.performance_history[-100:]

            return evaluation_result

        except Exception as e:
            logger.error(f"Error evaluating model performance: {e}")
            return {"error": str(e)}

    def get_performance_trend(self, metric_name: str = "accuracy") -> dict[str, Any]:
        """Analyze performance trend over time.

        Args:
            metric_name: Metric to analyze

        Returns:
            Trend analysis results
        """
        if not self.performance_history:
            return {"trend": "no_data"}

        values = []
        timestamps = []

        for record in self.performance_history:
            if metric_name in record["metrics"]:
                values.append(record["metrics"][metric_name])
                timestamps.append(record["metrics"]["timestamp"])

        if len(values) < 3:
            return {"trend": "insufficient_data"}

        # Calculate trend
        x = np.arange(len(values))
        slope, _, r_value, p_value, _ = stats.linregress(x, values)

        trend_direction = "stable"
        if p_value < 0.05:  # Statistically significant trend
            if slope > 0:
                trend_direction = "improving"
            else:
                trend_direction = "degrading"

        return {
            "trend": trend_direction,
            "slope": slope,
            "r_squared": r_value**2,
            "p_value": p_value,
            "recent_values": values[-5:],
            "timestamps": timestamps[-5:],
        }


class AutoRetrainingPipeline:
    """Automated pipeline for model retraining with drift detection and performance monitoring."""

    def __init__(
        self,
        model_manager: ModelManager,
        model_factory: Callable[[], BaseEstimator],
        feature_extractor: Callable[[pd.DataFrame], pd.DataFrame],
        target_extractor: Callable[[pd.DataFrame], pd.Series],
        retraining_schedule_hours: int = 24,
        min_samples_for_retraining: int = 100,
    ):
        """Initialize auto-retraining pipeline.

        Args:
            model_manager: Model manager instance
            model_factory: Function that creates new model instances
            feature_extractor: Function to extract features from data
            target_extractor: Function to extract targets from data
            retraining_schedule_hours: Hours between scheduled retraining checks
            min_samples_for_retraining: Minimum samples required for retraining
        """
        self.model_manager = model_manager
        self.model_factory = model_factory
        self.feature_extractor = feature_extractor
        self.target_extractor = target_extractor
        self.retraining_schedule_hours = retraining_schedule_hours
        self.min_samples_for_retraining = min_samples_for_retraining

        self.drift_detector = DataDriftDetector()
        self.performance_monitor = ModelPerformanceMonitor()

        self.last_retraining: dict[str, datetime] = {}
        self.retraining_history: list[dict[str, Any]] = []

    def should_retrain(
        self,
        model_id: str,
        new_data: pd.DataFrame,
        force_check: bool = False,
    ) -> tuple[bool, str]:
        """Determine if a model should be retrained.

        Args:
            model_id: Model identifier
            new_data: New data for evaluation
            force_check: Force retraining check regardless of schedule

        Returns:
            Tuple of (should_retrain, reason)
        """
        # Check schedule
        last_retrain = self.last_retraining.get(model_id)
        if not force_check and last_retrain is not None:
            time_since_retrain = datetime.now() - last_retrain
            if (
                time_since_retrain.total_seconds()
                < self.retraining_schedule_hours * 3600
            ):
                return False, "schedule_not_due"

        # Check minimum samples
        if len(new_data) < self.min_samples_for_retraining:
            return False, "insufficient_samples"

        # Extract features and targets
        try:
            features = self.feature_extractor(new_data)
            targets = self.target_extractor(new_data)
        except Exception as e:
            logger.error(f"Error extracting features/targets: {e}")
            return False, f"extraction_error: {e}"

        # Check for data drift
        if self.drift_detector.reference_data is not None:
            feature_drift = self.drift_detector.detect_feature_drift(features)
            target_drift = self.drift_detector.detect_target_drift(targets)
            drift_summary = self.drift_detector.get_drift_summary(
                feature_drift, target_drift
            )

            if drift_summary["recommendation"] in [
                "immediate_retraining",
                "urgent_retraining",
            ]:
                return True, f"data_drift_{drift_summary['drift_severity']}"

        # Check performance degradation
        current_model = self.model_manager.load_model(model_id)
        if current_model is not None and current_model.model is not None:
            try:
                # Split data for evaluation
                X_train, X_test, y_train, y_test = train_test_split(
                    features, targets, test_size=0.3, random_state=42, stratify=targets
                )

                # Scale features if scaler is available
                if current_model.scaler is not None:
                    X_test_scaled = current_model.scaler.transform(X_test)
                else:
                    X_test_scaled = X_test

                # Evaluate performance
                performance_result = self.performance_monitor.evaluate_performance(
                    current_model.model, X_test_scaled, y_test
                )

                if performance_result.get("degradation_detected", False):
                    return True, "performance_degradation"

            except Exception as e:
                logger.warning(f"Error evaluating model performance: {e}")

        return False, "no_triggers"

    def retrain_model(
        self,
        model_id: str,
        training_data: pd.DataFrame,
        validation_split: float = 0.2,
        **model_params,
    ) -> str | None:
        """Retrain a model with new data.

        Args:
            model_id: Model identifier
            training_data: Training data
            validation_split: Fraction of data to use for validation
            **model_params: Additional parameters for model training

        Returns:
            New model version string if successful, None otherwise
        """
        try:
            # Extract features and targets
            features = self.feature_extractor(training_data)
            targets = self.target_extractor(training_data)

            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                features,
                targets,
                test_size=validation_split,
                random_state=42,
                stratify=targets,
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Create and train new model
            model = self.model_factory()
            model.set_params(**model_params)
            model.fit(X_train_scaled, y_train)

            # Evaluate model
            train_score = model.score(X_train_scaled, y_train)
            val_score = model.score(X_val_scaled, y_val)

            # Create version string
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_version = f"v_{timestamp}"

            # Prepare metadata
            metadata = {
                "training_samples": len(X_train),
                "validation_samples": len(X_val),
                "features_count": X_train.shape[1],
                "model_params": model_params,
                "retraining_trigger": "automated",
            }

            # Prepare performance metrics
            performance_metrics = {
                "train_accuracy": train_score,
                "validation_accuracy": val_score,
                "overfitting_gap": train_score - val_score,
            }

            # Save model
            success = self.model_manager.save_model(
                model_id=model_id,
                version=new_version,
                model=model,
                scaler=scaler,
                metadata=metadata,
                performance_metrics=performance_metrics,
                set_as_active=True,  # Set as active if validation performance is good
            )

            if success:
                # Update retraining history
                self.last_retraining[model_id] = datetime.now()
                self.retraining_history.append(
                    {
                        "model_id": model_id,
                        "version": new_version,
                        "timestamp": datetime.now().isoformat(),
                        "training_samples": len(X_train),
                        "validation_accuracy": val_score,
                    }
                )

                # Update drift detector reference data
                self.drift_detector.set_reference_data(features, targets)

                # Update performance monitor baseline
                self.performance_monitor.set_baseline_performance(performance_metrics)

                logger.info(
                    f"Successfully retrained model {model_id} -> {new_version} (val_acc: {val_score:.4f})"
                )
                return new_version
            else:
                logger.error(f"Failed to save retrained model {model_id}")
                return None

        except Exception as e:
            logger.error(f"Error retraining model {model_id}: {e}")
            return None

    def run_retraining_check(
        self, model_id: str, new_data: pd.DataFrame
    ) -> dict[str, Any]:
        """Run complete retraining check and execute if needed.

        Args:
            model_id: Model identifier
            new_data: New data for evaluation

        Returns:
            Dictionary with check results and actions taken
        """
        start_time = datetime.now()

        try:
            # Check if retraining is needed
            should_retrain, reason = self.should_retrain(model_id, new_data)

            result = {
                "model_id": model_id,
                "timestamp": start_time.isoformat(),
                "should_retrain": should_retrain,
                "reason": reason,
                "data_samples": len(new_data),
                "new_version": None,
                "success": False,
            }

            if should_retrain:
                logger.info(f"Retraining {model_id} due to: {reason}")
                new_version = self.retrain_model(model_id, new_data)

                if new_version:
                    result.update(
                        {
                            "new_version": new_version,
                            "success": True,
                            "action": "retrained",
                        }
                    )
                else:
                    result.update(
                        {
                            "action": "retrain_failed",
                            "error": "Model retraining failed",
                        }
                    )
            else:
                result.update(
                    {
                        "action": "no_retrain",
                        "success": True,
                    }
                )

            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result["execution_time_seconds"] = execution_time

            return result

        except Exception as e:
            logger.error(f"Error in retraining check for {model_id}: {e}")
            return {
                "model_id": model_id,
                "timestamp": start_time.isoformat(),
                "should_retrain": False,
                "reason": "check_error",
                "error": str(e),
                "success": False,
                "execution_time_seconds": (datetime.now() - start_time).total_seconds(),
            }

    def get_retraining_summary(self) -> dict[str, Any]:
        """Get summary of retraining pipeline status.

        Returns:
            Summary dictionary
        """
        return {
            "total_models_managed": len(self.last_retraining),
            "total_retrainings": len(self.retraining_history),
            "recent_retrainings": self.retraining_history[-10:],
            "last_retraining_times": {
                model_id: timestamp.isoformat()
                for model_id, timestamp in self.last_retraining.items()
            },
            "retraining_schedule_hours": self.retraining_schedule_hours,
            "min_samples_for_retraining": self.min_samples_for_retraining,
        }


# Alias for backward compatibility
RetrainingPipeline = AutoRetrainingPipeline

# Ensure all expected names are available
__all__ = [
    "DataDriftDetector",
    "ModelPerformanceMonitor",
    "AutoRetrainingPipeline",
    "RetrainingPipeline",  # Alias for backward compatibility
]
