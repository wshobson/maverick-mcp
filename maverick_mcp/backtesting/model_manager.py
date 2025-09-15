"""ML Model Manager for backtesting strategies with versioning and persistence."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ModelVersion:
    """Represents a specific version of an ML model with metadata."""

    def __init__(
        self,
        model_id: str,
        version: str,
        model: BaseEstimator,
        scaler: StandardScaler | None = None,
        metadata: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
    ):
        """Initialize model version.

        Args:
            model_id: Unique identifier for the model
            version: Version string (e.g., "1.0.0")
            model: The trained ML model
            scaler: Feature scaler (if used)
            metadata: Additional metadata about the model
            performance_metrics: Performance metrics from training/validation
        """
        self.model_id = model_id
        self.version = version
        self.model = model
        self.scaler = scaler
        self.metadata = metadata or {}
        self.performance_metrics = performance_metrics or {}
        self.created_at = datetime.now()
        self.last_used = None
        self.usage_count = 0

        # Add default metadata
        self.metadata.update(
            {
                "model_type": type(model).__name__,
                "created_at": self.created_at.isoformat(),
                "sklearn_version": getattr(model, "_sklearn_version", "unknown"),
            }
        )

    def increment_usage(self):
        """Increment usage counter and update last used timestamp."""
        self.usage_count += 1
        self.last_used = datetime.now()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "metadata": self.metadata,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at.isoformat(),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "usage_count": self.usage_count,
        }


class ModelManager:
    """Manages ML models with versioning, persistence, and performance tracking."""

    def __init__(self, base_path: str | Path = "./models"):
        """Initialize model manager.

        Args:
            base_path: Base directory for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.models: dict[str, dict[str, ModelVersion]] = {}
        self.active_models: dict[str, str] = {}  # model_id -> active_version

        # Performance tracking
        self.performance_history: dict[str, list[dict[str, Any]]] = {}

        # Load existing models
        self._load_registry()

    def _get_model_path(self, model_id: str, version: str) -> Path:
        """Get file path for a specific model version."""
        return self.base_path / model_id / f"{version}.pkl"

    def _get_metadata_path(self, model_id: str, version: str) -> Path:
        """Get metadata file path for a specific model version."""
        return self.base_path / model_id / f"{version}_metadata.json"

    def _get_registry_path(self) -> Path:
        """Get registry file path."""
        return self.base_path / "registry.json"

    def _load_registry(self):
        """Load model registry from disk."""
        registry_path = self._get_registry_path()
        if registry_path.exists():
            try:
                with open(registry_path) as f:
                    registry_data = json.load(f)

                self.active_models = registry_data.get("active_models", {})
                models_info = registry_data.get("models", {})

                # Lazy load model metadata (don't load actual models until needed)
                for model_id, versions in models_info.items():
                    self.models[model_id] = {}
                    for version, version_info in versions.items():
                        # Create placeholder ModelVersion (model will be loaded on demand)
                        model_version = ModelVersion(
                            model_id=model_id,
                            version=version,
                            model=None,  # Will be loaded on demand
                            metadata=version_info.get("metadata", {}),
                            performance_metrics=version_info.get(
                                "performance_metrics", {}
                            ),
                        )
                        model_version.created_at = datetime.fromisoformat(
                            version_info.get("created_at", datetime.now().isoformat())
                        )
                        model_version.last_used = (
                            datetime.fromisoformat(version_info["last_used"])
                            if version_info.get("last_used")
                            else None
                        )
                        model_version.usage_count = version_info.get("usage_count", 0)
                        self.models[model_id][version] = model_version

                logger.info(
                    f"Loaded model registry with {len(self.models)} model types"
                )

            except Exception as e:
                logger.error(f"Error loading model registry: {e}")

    def _save_registry(self):
        """Save model registry to disk."""
        try:
            registry_data = {"active_models": self.active_models, "models": {}}

            for model_id, versions in self.models.items():
                registry_data["models"][model_id] = {}
                for version, model_version in versions.items():
                    registry_data["models"][model_id][version] = model_version.to_dict()

            registry_path = self._get_registry_path()
            with open(registry_path, "w") as f:
                json.dump(registry_data, f, indent=2)

            logger.debug("Saved model registry")

        except Exception as e:
            logger.error(f"Error saving model registry: {e}")

    def save_model(
        self,
        model_id: str,
        version: str,
        model: BaseEstimator,
        scaler: StandardScaler | None = None,
        metadata: dict[str, Any] | None = None,
        performance_metrics: dict[str, float] | None = None,
        set_as_active: bool = True,
    ) -> bool:
        """Save a model version to disk.

        Args:
            model_id: Unique identifier for the model
            version: Version string
            model: Trained ML model
            scaler: Feature scaler (if used)
            metadata: Additional metadata
            performance_metrics: Performance metrics
            set_as_active: Whether to set this as the active version

        Returns:
            True if successful
        """
        try:
            # Create model directory
            model_dir = self.base_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save model and scaler using joblib (better for sklearn models)
            model_path = self._get_model_path(model_id, version)
            model_data = {
                "model": model,
                "scaler": scaler,
            }
            joblib.dump(model_data, model_path)

            # Create ModelVersion instance
            model_version = ModelVersion(
                model_id=model_id,
                version=version,
                model=model,
                scaler=scaler,
                metadata=metadata,
                performance_metrics=performance_metrics,
            )

            # Save metadata separately
            metadata_path = self._get_metadata_path(model_id, version)
            with open(metadata_path, "w") as f:
                json.dump(model_version.to_dict(), f, indent=2)

            # Update registry
            if model_id not in self.models:
                self.models[model_id] = {}
            self.models[model_id][version] = model_version

            # Set as active if requested
            if set_as_active:
                self.active_models[model_id] = version

            # Save registry
            self._save_registry()

            logger.info(
                f"Saved model {model_id} v{version} ({'active' if set_as_active else 'inactive'})"
            )
            return True

        except Exception as e:
            logger.error(f"Error saving model {model_id} v{version}: {e}")
            return False

    def load_model(
        self, model_id: str, version: str | None = None
    ) -> ModelVersion | None:
        """Load a specific model version.

        Args:
            model_id: Model identifier
            version: Version to load (defaults to active version)

        Returns:
            ModelVersion instance or None if not found
        """
        try:
            if version is None:
                version = self.active_models.get(model_id)
                if version is None:
                    logger.warning(f"No active version found for model {model_id}")
                    return None

            if model_id not in self.models or version not in self.models[model_id]:
                logger.warning(f"Model {model_id} v{version} not found in registry")
                return None

            model_version = self.models[model_id][version]

            # Load actual model if not already loaded
            if model_version.model is None:
                model_path = self._get_model_path(model_id, version)
                if not model_path.exists():
                    logger.error(f"Model file not found: {model_path}")
                    return None

                model_data = joblib.load(model_path)
                model_version.model = model_data["model"]
                model_version.scaler = model_data.get("scaler")

            # Update usage statistics
            model_version.increment_usage()
            self._save_registry()

            logger.debug(f"Loaded model {model_id} v{version}")
            return model_version

        except Exception as e:
            logger.error(f"Error loading model {model_id} v{version}: {e}")
            return None

    def list_models(self) -> dict[str, list[str]]:
        """List all available models and their versions.

        Returns:
            Dictionary mapping model_id to list of versions
        """
        return {
            model_id: list(versions.keys())
            for model_id, versions in self.models.items()
        }

    def list_model_versions(self, model_id: str) -> list[dict[str, Any]]:
        """List all versions of a specific model with metadata.

        Args:
            model_id: Model identifier

        Returns:
            List of version information dictionaries
        """
        if model_id not in self.models:
            return []

        versions_info = []
        for version, model_version in self.models[model_id].items():
            info = model_version.to_dict()
            info["is_active"] = self.active_models.get(model_id) == version
            versions_info.append(info)

        # Sort by creation date (newest first)
        versions_info.sort(key=lambda x: x["created_at"], reverse=True)

        return versions_info

    def set_active_version(self, model_id: str, version: str) -> bool:
        """Set the active version for a model.

        Args:
            model_id: Model identifier
            version: Version to set as active

        Returns:
            True if successful
        """
        if model_id not in self.models or version not in self.models[model_id]:
            logger.error(f"Model {model_id} v{version} not found")
            return False

        self.active_models[model_id] = version
        self._save_registry()
        logger.info(f"Set {model_id} v{version} as active")
        return True

    def delete_model_version(self, model_id: str, version: str) -> bool:
        """Delete a specific model version.

        Args:
            model_id: Model identifier
            version: Version to delete

        Returns:
            True if successful
        """
        try:
            if model_id not in self.models or version not in self.models[model_id]:
                logger.warning(f"Model {model_id} v{version} not found")
                return False

            # Don't delete active version
            if self.active_models.get(model_id) == version:
                logger.error(f"Cannot delete active version {model_id} v{version}")
                return False

            # Delete files
            model_path = self._get_model_path(model_id, version)
            metadata_path = self._get_metadata_path(model_id, version)

            if model_path.exists():
                model_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()

            # Remove from registry
            del self.models[model_id][version]

            # Clean up empty model entry
            if not self.models[model_id]:
                del self.models[model_id]
                if model_id in self.active_models:
                    del self.active_models[model_id]

            self._save_registry()
            logger.info(f"Deleted model {model_id} v{version}")
            return True

        except Exception as e:
            logger.error(f"Error deleting model {model_id} v{version}: {e}")
            return False

    def cleanup_old_versions(
        self, keep_versions: int = 5, min_age_days: int = 30
    ) -> int:
        """Clean up old model versions.

        Args:
            keep_versions: Number of versions to keep per model
            min_age_days: Minimum age in days before deletion

        Returns:
            Number of versions deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now() - timedelta(days=min_age_days)

        for model_id, versions in list(self.models.items()):
            # Sort versions by creation date (newest first)
            sorted_versions = sorted(
                versions.items(), key=lambda x: x[1].created_at, reverse=True
            )

            # Keep active version and recent versions
            active_version = self.active_models.get(model_id)
            versions_to_delete = []

            for i, (version, model_version) in enumerate(sorted_versions):
                # Skip if it's the active version
                if version == active_version:
                    continue

                # Skip if we haven't kept enough versions yet
                if i < keep_versions:
                    continue

                # Skip if it's too new
                if model_version.created_at > cutoff_date:
                    continue

                versions_to_delete.append(version)

            # Delete old versions
            for version in versions_to_delete:
                if self.delete_model_version(model_id, version):
                    deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old model versions")

        return deleted_count

    def get_model_performance_history(self, model_id: str) -> list[dict[str, Any]]:
        """Get performance history for a model.

        Args:
            model_id: Model identifier

        Returns:
            List of performance records
        """
        return self.performance_history.get(model_id, [])

    def log_model_performance(
        self,
        model_id: str,
        version: str,
        metrics: dict[str, float],
        additional_data: dict[str, Any] | None = None,
    ):
        """Log performance metrics for a model.

        Args:
            model_id: Model identifier
            version: Model version
            metrics: Performance metrics
            additional_data: Additional data to log
        """
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []

        performance_record = {
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "metrics": metrics,
            "additional_data": additional_data or {},
        }

        self.performance_history[model_id].append(performance_record)

        # Keep only recent performance records (last 1000)
        if len(self.performance_history[model_id]) > 1000:
            self.performance_history[model_id] = self.performance_history[model_id][
                -1000:
            ]

        logger.debug(f"Logged performance for {model_id} v{version}")

    def compare_model_versions(
        self, model_id: str, versions: list[str] | None = None
    ) -> pd.DataFrame:
        """Compare performance metrics across model versions.

        Args:
            model_id: Model identifier
            versions: Versions to compare (defaults to all versions)

        Returns:
            DataFrame with comparison results
        """
        if model_id not in self.models:
            return pd.DataFrame()

        if versions is None:
            versions = list(self.models[model_id].keys())

        comparison_data = []
        for version in versions:
            if version in self.models[model_id]:
                model_version = self.models[model_id][version]
                row_data = {
                    "version": version,
                    "created_at": model_version.created_at,
                    "usage_count": model_version.usage_count,
                    "is_active": self.active_models.get(model_id) == version,
                }
                row_data.update(model_version.performance_metrics)
                comparison_data.append(row_data)

        return pd.DataFrame(comparison_data)

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics for the model manager.

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        total_models = 0
        total_versions = 0

        for model_id, versions in self.models.items():
            total_models += 1
            for version in versions:
                total_versions += 1
                model_path = self._get_model_path(model_id, version)
                if model_path.exists():
                    total_size += model_path.stat().st_size

        return {
            "total_models": total_models,
            "total_versions": total_versions,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "base_path": str(self.base_path),
        }
