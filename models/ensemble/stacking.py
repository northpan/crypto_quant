"""
Stacking ensemble for model combination.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from ...core.base import BaseEnsemble, BaseModel


class StackingEnsemble(BaseEnsemble):
    """
    Stacking ensemble with meta-learner.
    
    Uses predictions from base models as features for a meta-learner.
    """

    def __init__(
        self,
        name: str = "StackingEnsemble",
        models: Optional[List[BaseModel]] = None,
        meta_learner: Optional[BaseModel] = None,
        n_folds: int = 5,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, models, config)
        self.meta_learner = meta_learner
        self.n_folds = n_folds
        self._meta_features: Optional[pd.DataFrame] = None

    def add_model(self, model: BaseModel, weight: float = 1.0) -> "StackingEnsemble":
        """Add a base model to the ensemble."""
        self.models.append(model)
        return self

    def set_meta_learner(self, meta_learner: BaseModel) -> "StackingEnsemble":
        """Set the meta-learner model."""
        self.meta_learner = meta_learner
        return self

    def fit_weights(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> "StackingEnsemble":
        """
        Train the stacking ensemble.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Self for method chaining
        """
        if not self.models:
            raise ValueError("No base models in ensemble")
        
        if self.meta_learner is None:
            raise ValueError("No meta-learner set")
        
        # Generate meta-features using cross-validation
        meta_features = self._generate_meta_features(X, y)
        
        # Train meta-learner on meta-features
        self.meta_learner.fit(meta_features, y)
        
        # Retrain base models on full dataset
        for model in self.models:
            model.fit(X, y)
        
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions."""
        # Generate meta-features from base models
        meta_features = self._predict_meta_features(X)
        
        # Use meta-learner for final prediction
        return self.meta_learner.predict(meta_features)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate ensemble probability predictions."""
        # Generate meta-features from base models
        meta_features = self._predict_meta_features(X)
        
        # Use meta-learner for final prediction
        return self.meta_learner.predict_proba(meta_features)

    def _generate_meta_features(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Generate meta-features using cross-validation.
        
        This prevents information leakage by using out-of-fold predictions.
        """
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        meta_features = np.zeros((len(X), len(self.models)))
        
        for train_idx, val_idx in kfold.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx] if isinstance(y, pd.Series) else y[train_idx]
            
            for i, model in enumerate(self.models):
                # Clone and train model on training fold
                model_copy = self._clone_model(model)
                model_copy.fit(X_train, y_train)
                
                # Predict on validation fold
                meta_features[val_idx, i] = model_copy.predict(X_val)
        
        return pd.DataFrame(
            meta_features,
            columns=[f"model_{i}" for i in range(len(self.models))],
            index=X.index,
        )

    def _predict_meta_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features from trained base models."""
        meta_features = np.zeros((len(X), len(self.models)))
        
        for i, model in enumerate(self.models):
            meta_features[:, i] = model.predict(X)
        
        return pd.DataFrame(
            meta_features,
            columns=[f"model_{i}" for i in range(len(self.models))],
            index=X.index,
        )

    def _clone_model(self, model: BaseModel) -> BaseModel:
        """Clone a model with the same parameters."""
        import copy
        return copy.deepcopy(model)

    def get_model_weights(self) -> Dict[str, float]:
        """Get model weights (not applicable for stacking)."""
        # For stacking, weights are learned by meta-learner
        if hasattr(self.meta_learner, '_feature_importance'):
            importance = self.meta_learner._feature_importance
            if importance:
                return {
                    self.models[i].name: importance.get(f"model_{i}", 1.0 / len(self.models))
                    for i in range(len(self.models))
                }
        
        return {model.name: 1.0 / len(self.models) for model in self.models}
