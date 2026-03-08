"""
Blending ensemble for model combination.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ...core.base import BaseEnsemble, BaseModel


class BlendingEnsemble(BaseEnsemble):
    """
    Blending ensemble with optimized weights.
    
    Combines predictions from multiple models using weighted averaging.
    Weights are optimized to maximize performance on validation data.
    """

    def __init__(
        self,
        name: str = "BlendingEnsemble",
        models: Optional[List[BaseModel]] = None,
        optimization_metric: str = "sharpe",
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, models, config)
        self.optimization_metric = optimization_metric
        self._model_weights: Dict[str, float] = {}

    def add_model(self, model: BaseModel, weight: float = 1.0) -> "BlendingEnsemble":
        """Add a model to the ensemble."""
        self.models.append(model)
        self._model_weights[model.name] = weight
        return self

    def fit_weights(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "BlendingEnsemble":
        """
        Optimize ensemble weights using validation data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = self._get_all_predictions(X)
        
        # Optimize weights
        n_models = len(self.models)
        initial_weights = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1, each weight >= 0
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_models)]
        
        # Optimize
        result = minimize(
            lambda w: self._objective(w, predictions, y),
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )
        
        # Store optimized weights
        self._weights = result.x
        for i, model in enumerate(self.models):
            self._model_weights[model.name] = result.x[i]
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        if self._weights is None:
            # Use equal weights if not optimized
            self._weights = np.ones(len(self.models)) / len(self.models)
        
        # Get predictions from all models
        predictions = self._get_all_predictions(X)
        
        # Weighted average
        ensemble_pred = np.zeros(len(X))
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * self._weights[i]
        
        return ensemble_pred

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        if self._weights is None:
            self._weights = np.ones(len(self.models)) / len(self.models)
        
        # Get probability predictions from all models
        probas = []
        for model in self.models:
            proba = model.predict_proba(X)
            probas.append(proba)
        
        # Weighted average of probabilities
        ensemble_proba = np.zeros_like(probas[0])
        for i, proba in enumerate(probas):
            ensemble_proba += proba * self._weights[i]
        
        # Normalize to ensure probabilities sum to 1
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba

    def _get_all_predictions(self, X: pd.DataFrame) -> List[np.ndarray]:
        """Get predictions from all models."""
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        return predictions

    def _objective(
        self,
        weights: np.ndarray,
        predictions: List[np.ndarray],
        y: Union[pd.Series, np.ndarray],
    ) -> float:
        """
        Objective function for weight optimization.
        
        Returns negative metric to minimize.
        """
        # Calculate weighted predictions
        ensemble_pred = np.zeros(len(y))
        for i, pred in enumerate(predictions):
            ensemble_pred += pred * weights[i]
        
        if self.optimization_metric == "accuracy":
            # Classification accuracy
            y_class = np.sign(y)
            pred_class = np.sign(ensemble_pred)
            metric = np.mean(y_class == pred_class)
        
        elif self.optimization_metric == "mse":
            # Mean squared error
            metric = -np.mean((y - ensemble_pred) ** 2)
        
        elif self.optimization_metric == "sharpe":
            # Sharpe ratio of strategy returns
            returns = ensemble_pred * y  # Simplified
            if np.std(returns) > 0:
                metric = np.mean(returns) / np.std(returns)
            else:
                metric = 0
        
        else:
            metric = -np.mean((y - ensemble_pred) ** 2)
        
        return -metric  # Minimize negative metric

    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights."""
        return self._model_weights.copy()
