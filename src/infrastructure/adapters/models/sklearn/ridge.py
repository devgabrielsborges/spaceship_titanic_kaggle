"""Ridge Regression adapter."""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeClassifier

from src.domain.entities.experiment_config import TaskType
from src.domain.ports.model_port import ModelPort


class RidgeAdapter(ModelPort):
    """Adapter for sklearn Ridge Regression."""

    def __init__(self, task_type: TaskType = TaskType.REGRESSION):
        self._model: Ridge | RidgeClassifier | None = None
        self._task_type = task_type
        self._is_classifier = task_type in (
            TaskType.BINARY_CLASSIFICATION,
            TaskType.MULTICLASS_CLASSIFICATION,
        )

    @property
    def name(self) -> str:
        return "RidgeClassifier" if self._is_classifier else "RidgeRegression"

    def build(self, hyperparameters: dict) -> None:
        if self._is_classifier:
            self._model = RidgeClassifier(**hyperparameters)
        else:
            self._model = Ridge(**hyperparameters)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        self._model.fit(X, y)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._model.predict(X)

    def get_search_space(self, trial: Any) -> dict:
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 100.0, log=True),
            "solver": trial.suggest_categorical(
                "solver", ["auto", "svd", "cholesky", "lsqr", "saga"]
            ),
            "max_iter": trial.suggest_int("max_iter", 100, 5000),
        }

    def get_model(self) -> Ridge | RidgeClassifier:
        return self._model

    def get_default_trials(self) -> int:
        """Ridge has moderate search space: alpha (continuous) × 5 solvers × max_iter."""
        return 50
