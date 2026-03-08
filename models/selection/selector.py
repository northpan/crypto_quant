"""
Feature and model selection utilities.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif


class FeatureSelector:
    """
    Feature selection for model training.
    """

    def __init__(
        self,
        method: str = "importance",
        k: int = 50,
        threshold: float = 0.01,
    ):
        self.method = method
        self.k = k
        self.threshold = threshold
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        feature_importance: Optional[Dict[str, float]] = None,
    ) -> "FeatureSelector":
        """Fit feature selector."""
        if self.method == "importance" and feature_importance:
            # Select based on provided feature importance
            sorted_features = sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            self._selected_features = [f for f, _ in sorted_features[:self.k]]
            self._feature_scores = feature_importance
        
        elif self.method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=self.k)
            selector.fit(X, y)
            
            self._selected_features = [
                X.columns[i] for i in selector.get_support(indices=True)
            ]
            self._feature_scores = {
                X.columns[i]: selector.scores_[i]
                for i in range(len(X.columns))
            }
        
        elif self.method == "f_classif":
            selector = SelectKBest(f_classif, k=self.k)
            selector.fit(X, y)
            
            self._selected_features = [
                X.columns[i] for i in selector.get_support(indices=True)
            ]
            self._feature_scores = {
                X.columns[i]: selector.scores_[i]
                for i in range(len(X.columns))
            }
        
        elif self.method == "correlation":
            # Remove highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            to_drop = [
                column for column in upper.columns
                if any(upper[column] > 0.95)
            ]
            
            self._selected_features = [
                c for c in X.columns if c not in to_drop
            ][:self.k]
        
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data by selecting features."""
        return X[self._selected_features]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)

    def get_selected_features(self) -> List[str]:
        """Get list of selected features."""
        return self._selected_features.copy()

    def get_feature_scores(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return self._feature_scores.copy()


class ModelSelector:
    """
    Model selection and hyperparameter tuning.
    """

    def __init__(
        self,
        models: List[Any],
        cv_folds: int = 5,
        scoring: str = "accuracy",
    ):
        self.models = models
        self.cv_folds = cv_folds
        self.scoring = scoring
        self._best_model: Optional[Any] = None
        self._cv_results: Dict[str, List[float]] = {}

    def select(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Any:
        """Select best model using cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        best_score = -np.inf
        
        for model in self.models:
            scores = cross_val_score(
                model, X, y,
                cv=self.cv_folds,
                scoring=self.scoring,
            )
            
            self._cv_results[model.name] = scores.tolist()
            
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                self._best_model = model
        
        return self._best_model

    def get_cv_results(self) -> Dict[str, List[float]]:
        """Get cross-validation results."""
        return self._cv_results.copy()

    def get_best_model(self) -> Optional[Any]:
        """Get best model."""
        return self._best_model
