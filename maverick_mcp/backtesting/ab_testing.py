"""A/B testing framework for comparing ML model performance."""

import logging
import random
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from .model_manager import ModelManager

logger = logging.getLogger(__name__)


class ABTestGroup:
    """Represents a group in an A/B test."""

    def __init__(
        self,
        group_id: str,
        model_id: str,
        model_version: str,
        traffic_allocation: float,
        description: str = "",
    ):
        """Initialize A/B test group.

        Args:
            group_id: Unique identifier for the group
            model_id: Model identifier
            model_version: Model version
            traffic_allocation: Fraction of traffic allocated to this group (0-1)
            description: Description of the group
        """
        self.group_id = group_id
        self.model_id = model_id
        self.model_version = model_version
        self.traffic_allocation = traffic_allocation
        self.description = description
        self.created_at = datetime.now()

        # Performance tracking
        self.predictions: list[Any] = []
        self.actual_values: list[Any] = []
        self.prediction_timestamps: list[datetime] = []
        self.prediction_confidence: list[float] = []

    def add_prediction(
        self,
        prediction: Any,
        actual: Any,
        confidence: float = 1.0,
        timestamp: datetime | None = None,
    ):
        """Add a prediction result to the group.

        Args:
            prediction: Model prediction
            actual: Actual value
            confidence: Prediction confidence score
            timestamp: Prediction timestamp
        """
        self.predictions.append(prediction)
        self.actual_values.append(actual)
        self.prediction_confidence.append(confidence)
        self.prediction_timestamps.append(timestamp or datetime.now())

    def get_metrics(self) -> dict[str, float]:
        """Calculate performance metrics for the group.

        Returns:
            Dictionary of performance metrics
        """
        if not self.predictions or not self.actual_values:
            return {}

        try:
            predictions = np.array(self.predictions)
            actuals = np.array(self.actual_values)

            metrics = {
                "sample_count": len(predictions),
                "accuracy": accuracy_score(actuals, predictions),
                "precision": precision_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
                "recall": recall_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
                "f1_score": f1_score(
                    actuals, predictions, average="weighted", zero_division=0
                ),
                "avg_confidence": np.mean(self.prediction_confidence),
            }

            # Add confusion matrix for binary/multiclass
            unique_labels = np.unique(np.concatenate([predictions, actuals]))
            if len(unique_labels) <= 10:  # Reasonable number of classes
                from sklearn.metrics import confusion_matrix

                cm = confusion_matrix(actuals, predictions, labels=unique_labels)
                metrics["confusion_matrix"] = cm.tolist()
                metrics["unique_labels"] = unique_labels.tolist()

            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics for group {self.group_id}: {e}")
            return {"error": str(e)}

    def to_dict(self) -> dict[str, Any]:
        """Convert group to dictionary representation."""
        return {
            "group_id": self.group_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "traffic_allocation": self.traffic_allocation,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "metrics": self.get_metrics(),
        }


class ABTest:
    """Manages an A/B test between different model versions."""

    def __init__(
        self,
        test_id: str,
        name: str,
        description: str = "",
        random_seed: int | None = None,
    ):
        """Initialize A/B test.

        Args:
            test_id: Unique identifier for the test
            name: Human-readable name for the test
            description: Description of the test
            random_seed: Random seed for reproducible traffic splitting
        """
        self.test_id = test_id
        self.name = name
        self.description = description
        self.created_at = datetime.now()
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self.status = "created"  # created, running, completed, cancelled

        # Groups in the test
        self.groups: dict[str, ABTestGroup] = {}

        # Traffic allocation
        self.traffic_splitter = TrafficSplitter(random_seed)

        # Test configuration
        self.min_samples_per_group = 100
        self.confidence_level = 0.95
        self.minimum_detectable_effect = 0.05

    def add_group(
        self,
        group_id: str,
        model_id: str,
        model_version: str,
        traffic_allocation: float,
        description: str = "",
    ) -> bool:
        """Add a group to the A/B test.

        Args:
            group_id: Unique identifier for the group
            model_id: Model identifier
            model_version: Model version
            traffic_allocation: Fraction of traffic (0-1)
            description: Description of the group

        Returns:
            True if successful
        """
        if self.status != "created":
            logger.error(
                f"Cannot add group to test {self.test_id} - test already started"
            )
            return False

        if group_id in self.groups:
            logger.error(f"Group {group_id} already exists in test {self.test_id}")
            return False

        # Validate traffic allocation
        current_total = sum(g.traffic_allocation for g in self.groups.values())
        if (
            current_total + traffic_allocation > 1.0001
        ):  # Small tolerance for floating point
            logger.error(
                f"Traffic allocation would exceed 100%: {current_total + traffic_allocation}"
            )
            return False

        group = ABTestGroup(
            group_id=group_id,
            model_id=model_id,
            model_version=model_version,
            traffic_allocation=traffic_allocation,
            description=description,
        )

        self.groups[group_id] = group
        self.traffic_splitter.update_allocation(
            {gid: g.traffic_allocation for gid, g in self.groups.items()}
        )

        logger.info(f"Added group {group_id} to test {self.test_id}")
        return True

    def start_test(self) -> bool:
        """Start the A/B test.

        Returns:
            True if successful
        """
        if self.status != "created":
            logger.error(
                f"Cannot start test {self.test_id} - invalid status: {self.status}"
            )
            return False

        if len(self.groups) < 2:
            logger.error(f"Cannot start test {self.test_id} - need at least 2 groups")
            return False

        # Validate traffic allocation sums to approximately 1.0
        total_allocation = sum(g.traffic_allocation for g in self.groups.values())
        if abs(total_allocation - 1.0) > 0.01:
            logger.error(f"Traffic allocation does not sum to 1.0: {total_allocation}")
            return False

        self.status = "running"
        self.started_at = datetime.now()
        logger.info(f"Started A/B test {self.test_id} with {len(self.groups)} groups")
        return True

    def assign_traffic(self, user_id: str | None = None) -> str | None:
        """Assign traffic to a group.

        Args:
            user_id: User identifier for consistent assignment

        Returns:
            Group ID or None if test not running
        """
        if self.status != "running":
            return None

        return self.traffic_splitter.assign_group(user_id)

    def record_prediction(
        self,
        group_id: str,
        prediction: Any,
        actual: Any,
        confidence: float = 1.0,
        timestamp: datetime | None = None,
    ) -> bool:
        """Record a prediction result for a group.

        Args:
            group_id: Group identifier
            prediction: Model prediction
            actual: Actual value
            confidence: Prediction confidence
            timestamp: Prediction timestamp

        Returns:
            True if successful
        """
        if group_id not in self.groups:
            logger.error(f"Group {group_id} not found in test {self.test_id}")
            return False

        self.groups[group_id].add_prediction(prediction, actual, confidence, timestamp)
        return True

    def get_results(self) -> dict[str, Any]:
        """Get current A/B test results.

        Returns:
            Dictionary with test results
        """
        results = {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "groups": {},
            "statistical_analysis": {},
        }

        # Group results
        for group_id, group in self.groups.items():
            results["groups"][group_id] = group.to_dict()

        # Statistical analysis
        if len(self.groups) >= 2:
            results["statistical_analysis"] = self._perform_statistical_analysis()

        return results

    def _perform_statistical_analysis(self) -> dict[str, Any]:
        """Perform statistical analysis of A/B test results.

        Returns:
            Statistical analysis results
        """
        analysis = {
            "ready_for_analysis": True,
            "sample_size_adequate": True,
            "statistical_significance": {},
            "effect_sizes": {},
            "recommendations": [],
        }

        # Check sample sizes
        sample_sizes = {
            group_id: len(group.predictions) for group_id, group in self.groups.items()
        }

        min_samples = min(sample_sizes.values()) if sample_sizes else 0
        if min_samples < self.min_samples_per_group:
            analysis["ready_for_analysis"] = False
            analysis["sample_size_adequate"] = False
            analysis["recommendations"].append(
                f"Need at least {self.min_samples_per_group} samples per group (current min: {min_samples})"
            )

        if not analysis["ready_for_analysis"]:
            return analysis

        # Pairwise comparisons
        group_ids = list(self.groups.keys())
        for i, group_a_id in enumerate(group_ids):
            for group_b_id in group_ids[i + 1 :]:
                comparison_key = f"{group_a_id}_vs_{group_b_id}"

                try:
                    group_a = self.groups[group_a_id]
                    group_b = self.groups[group_b_id]

                    # Compare accuracy scores
                    accuracy_a = accuracy_score(
                        group_a.actual_values, group_a.predictions
                    )
                    accuracy_b = accuracy_score(
                        group_b.actual_values, group_b.predictions
                    )

                    # Perform statistical test
                    # For classification accuracy, we can use a proportion test
                    n_correct_a = sum(
                        np.array(group_a.predictions) == np.array(group_a.actual_values)
                    )
                    n_correct_b = sum(
                        np.array(group_b.predictions) == np.array(group_b.actual_values)
                    )
                    n_total_a = len(group_a.predictions)
                    n_total_b = len(group_b.predictions)

                    # Two-proportion z-test
                    p_combined = (n_correct_a + n_correct_b) / (n_total_a + n_total_b)
                    se = np.sqrt(
                        p_combined * (1 - p_combined) * (1 / n_total_a + 1 / n_total_b)
                    )

                    if se > 0:
                        z_score = (accuracy_a - accuracy_b) / se
                        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

                        # Effect size (Cohen's h for proportions)
                        h = 2 * (
                            np.arcsin(np.sqrt(accuracy_a))
                            - np.arcsin(np.sqrt(accuracy_b))
                        )

                        analysis["statistical_significance"][comparison_key] = {
                            "accuracy_a": accuracy_a,
                            "accuracy_b": accuracy_b,
                            "difference": accuracy_a - accuracy_b,
                            "z_score": z_score,
                            "p_value": p_value,
                            "significant": p_value < (1 - self.confidence_level),
                            "effect_size_h": h,
                        }

                        # Recommendations based on results
                        if p_value < (1 - self.confidence_level):
                            if accuracy_a > accuracy_b:
                                analysis["recommendations"].append(
                                    f"Group {group_a_id} significantly outperforms {group_b_id} "
                                    f"(p={p_value:.4f}, effect_size={h:.4f})"
                                )
                            else:
                                analysis["recommendations"].append(
                                    f"Group {group_b_id} significantly outperforms {group_a_id} "
                                    f"(p={p_value:.4f}, effect_size={h:.4f})"
                                )
                        else:
                            analysis["recommendations"].append(
                                f"No significant difference between {group_a_id} and {group_b_id} "
                                f"(p={p_value:.4f})"
                            )

                except Exception as e:
                    logger.error(
                        f"Error in statistical analysis for {comparison_key}: {e}"
                    )
                    analysis["statistical_significance"][comparison_key] = {
                        "error": str(e)
                    }

        return analysis

    def stop_test(self, reason: str = "completed") -> bool:
        """Stop the A/B test.

        Args:
            reason: Reason for stopping

        Returns:
            True if successful
        """
        if self.status != "running":
            logger.error(f"Cannot stop test {self.test_id} - not running")
            return False

        self.status = "completed" if reason == "completed" else "cancelled"
        self.ended_at = datetime.now()
        logger.info(f"Stopped A/B test {self.test_id}: {reason}")
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert test to dictionary representation."""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "groups": {gid: g.to_dict() for gid, g in self.groups.items()},
            "configuration": {
                "min_samples_per_group": self.min_samples_per_group,
                "confidence_level": self.confidence_level,
                "minimum_detectable_effect": self.minimum_detectable_effect,
            },
        }


class TrafficSplitter:
    """Handles traffic splitting for A/B tests."""

    def __init__(self, random_seed: int | None = None):
        """Initialize traffic splitter.

        Args:
            random_seed: Random seed for reproducible splitting
        """
        self.random_seed = random_seed
        self.group_allocation: dict[str, float] = {}
        self.cumulative_allocation: list[tuple[str, float]] = []

    def update_allocation(self, allocation: dict[str, float]):
        """Update group traffic allocation.

        Args:
            allocation: Dictionary mapping group_id to allocation fraction
        """
        self.group_allocation = allocation.copy()

        # Create cumulative distribution for sampling
        self.cumulative_allocation = []
        cumulative = 0.0

        for group_id, fraction in allocation.items():
            cumulative += fraction
            self.cumulative_allocation.append((group_id, cumulative))

    def assign_group(self, user_id: str | None = None) -> str | None:
        """Assign a user to a group.

        Args:
            user_id: User identifier for consistent assignment

        Returns:
            Group ID or None if no groups configured
        """
        if not self.cumulative_allocation:
            return None

        # Generate random value
        if user_id is not None:
            # Hash user_id for consistent assignment
            import hashlib

            hash_object = hashlib.md5(user_id.encode())
            hash_int = int(hash_object.hexdigest(), 16)
            rand_value = (hash_int % 10000) / 10000.0  # Normalize to [0, 1)
        else:
            if self.random_seed is not None:
                random.seed(self.random_seed)
            rand_value = random.random()

        # Find group based on cumulative allocation
        for group_id, cumulative_threshold in self.cumulative_allocation:
            if rand_value <= cumulative_threshold:
                return group_id

        # Fallback to last group
        return self.cumulative_allocation[-1][0] if self.cumulative_allocation else None


class ABTestManager:
    """Manages multiple A/B tests."""

    def __init__(self, model_manager: ModelManager):
        """Initialize A/B test manager.

        Args:
            model_manager: Model manager instance
        """
        self.model_manager = model_manager
        self.tests: dict[str, ABTest] = {}

    def create_test(
        self,
        test_id: str,
        name: str,
        description: str = "",
        random_seed: int | None = None,
    ) -> ABTest:
        """Create a new A/B test.

        Args:
            test_id: Unique identifier for the test
            name: Human-readable name
            description: Description
            random_seed: Random seed for reproducible splitting

        Returns:
            ABTest instance
        """
        if test_id in self.tests:
            raise ValueError(f"Test {test_id} already exists")

        test = ABTest(test_id, name, description, random_seed)
        self.tests[test_id] = test
        logger.info(f"Created A/B test {test_id}: {name}")
        return test

    def get_test(self, test_id: str) -> ABTest | None:
        """Get an A/B test by ID.

        Args:
            test_id: Test identifier

        Returns:
            ABTest instance or None
        """
        return self.tests.get(test_id)

    def list_tests(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """List all A/B tests.

        Args:
            status_filter: Filter by status (created, running, completed, cancelled)

        Returns:
            List of test summaries
        """
        tests = []
        for test in self.tests.values():
            if status_filter is None or test.status == status_filter:
                tests.append(
                    {
                        "test_id": test.test_id,
                        "name": test.name,
                        "status": test.status,
                        "groups_count": len(test.groups),
                        "created_at": test.created_at.isoformat(),
                        "started_at": test.started_at.isoformat()
                        if test.started_at
                        else None,
                    }
                )

        # Sort by creation time (newest first)
        tests.sort(key=lambda x: x["created_at"], reverse=True)
        return tests

    def run_model_comparison(
        self,
        test_name: str,
        model_versions: list[tuple[str, str]],  # (model_id, version)
        test_data: pd.DataFrame,
        feature_extractor: Any,
        target_extractor: Any,
        traffic_allocation: list[float] | None = None,
        test_duration_hours: int = 24,
    ) -> str:
        """Run a model comparison A/B test.

        Args:
            test_name: Name for the test
            model_versions: List of (model_id, version) tuples to compare
            test_data: Test data for evaluation
            feature_extractor: Function to extract features
            target_extractor: Function to extract targets
            traffic_allocation: Custom traffic allocation (defaults to equal split)
            test_duration_hours: Duration to run the test

        Returns:
            Test ID
        """
        # Generate unique test ID
        test_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create test
        test = self.create_test(
            test_id=test_id,
            name=test_name,
            description=f"Comparing {len(model_versions)} model versions",
        )

        # Default equal traffic allocation
        if traffic_allocation is None:
            allocation_per_group = 1.0 / len(model_versions)
            traffic_allocation = [allocation_per_group] * len(model_versions)

        # Add groups
        for i, (model_id, version) in enumerate(model_versions):
            group_id = f"group_{i}_{model_id}_{version}"
            test.add_group(
                group_id=group_id,
                model_id=model_id,
                model_version=version,
                traffic_allocation=traffic_allocation[i],
                description=f"Model {model_id} version {version}",
            )

        # Start test
        test.start_test()

        # Extract features and targets
        features = feature_extractor(test_data)
        targets = target_extractor(test_data)

        # Simulate predictions for each group
        for _, row in features.iterrows():
            # Assign traffic
            group_id = test.assign_traffic(str(row.name))  # Use row index as user_id
            if group_id is None:
                continue

            # Get corresponding group's model
            group = test.groups[group_id]
            model_version = self.model_manager.load_model(
                group.model_id, group.model_version
            )

            if model_version is None or model_version.model is None:
                logger.warning(f"Could not load model for group {group_id}")
                continue

            try:
                # Make prediction
                X = row.values.reshape(1, -1)
                if model_version.scaler is not None:
                    X = model_version.scaler.transform(X)

                prediction = model_version.model.predict(X)[0]

                # Get confidence if available
                confidence = 1.0
                if hasattr(model_version.model, "predict_proba"):
                    proba = model_version.model.predict_proba(X)[0]
                    confidence = max(proba)

                # Get actual value
                actual = targets.loc[row.name]

                # Record prediction
                test.record_prediction(group_id, prediction, actual, confidence)

            except Exception as e:
                logger.warning(f"Error making prediction for group {group_id}: {e}")

        logger.info(f"Completed model comparison test {test_id}")
        return test_id

    def get_test_summary(self) -> dict[str, Any]:
        """Get summary of all A/B tests.

        Returns:
            Summary dictionary
        """
        total_tests = len(self.tests)
        status_counts = {}

        for test in self.tests.values():
            status_counts[test.status] = status_counts.get(test.status, 0) + 1

        recent_tests = sorted(
            [
                {
                    "test_id": test.test_id,
                    "name": test.name,
                    "status": test.status,
                    "created_at": test.created_at.isoformat(),
                }
                for test in self.tests.values()
            ],
            key=lambda x: x["created_at"],
            reverse=True,
        )[:10]

        return {
            "total_tests": total_tests,
            "status_distribution": status_counts,
            "recent_tests": recent_tests,
        }
