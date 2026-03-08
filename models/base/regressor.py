"""
Regression models for return prediction.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ...core.base import BaseModel


class LGBMRegressorModel(BaseModel):
    """
    LightGBM regressor for return prediction.
    """

    def __init__(
        self,
        name: str = "LGBMRegressor",
        prediction_horizon: int = 1,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.params = params or {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 100,
            'random_state': 42,
        }
        
        self._model = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "LGBMRegressorModel":
        """Train the LightGBM regressor."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self._feature_names = list(X.columns)
        
        # Create datasets
        train_data = lgb.Dataset(X, label=y)
        
        valid_sets = [train_data]
        if validation_data:
            X_val, y_val = validation_data
            valid_data = lgb.Dataset(X_val, label=y_val)
            valid_sets.append(valid_data)
        
        # Train model
        self._model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params.get('n_estimators', 100),
            valid_sets=valid_sets,
            callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)],
        )
        
        # Get feature importance
        importance = self._model.feature_importance(importance_type='gain')
        self._feature_importance = {
            name: float(imp) 
            for name, imp in zip(self._feature_names, importance)
        }
        
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate probability-like predictions.
        
        For regression, we convert predictions to probabilities
        based on the sign and magnitude.
        """
        predictions = self.predict(X)
        
        # Convert to 3-class probabilities
        proba = np.zeros((len(predictions), 3))
        
        for i, pred in enumerate(predictions):
            if pred > 0.01:  # Strong positive
                proba[i] = [0.1, 0.2, 0.7]
            elif pred > 0:  # Weak positive
                proba[i] = [0.2, 0.4, 0.4]
            elif pred > -0.01:  # Weak negative
                proba[i] = [0.4, 0.4, 0.2]
            else:  # Strong negative
                proba[i] = [0.7, 0.2, 0.1]
        
        return proba


class XGBRegressorModel(BaseModel):
    """
    XGBoost regressor for return prediction.
    """

    def __init__(
        self,
        name: str = "XGBRegressor",
        prediction_horizon: int = 1,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.params = params or {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
        }
        
        self._model = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "XGBRegressorModel":
        """Train the XGBoost regressor."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self._feature_names = list(X.columns)
        
        # Create model
        self._model = xgb.XGBRegressor(**self.params)
        
        # Prepare validation data
        eval_set = None
        if validation_data:
            X_val, y_val = validation_data
            eval_set = [(X_val, y_val)]
        
        # Train model
        self._model.fit(
            X, y,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False,
        )
        
        # Get feature importance
        importance = self._model.feature_importances_
        self._feature_importance = {
            name: float(imp) 
            for name, imp in zip(self._feature_names, importance)
        }
        
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self._model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability-like predictions."""
        predictions = self.predict(X)
        
        proba = np.zeros((len(predictions), 3))
        
        for i, pred in enumerate(predictions):
            if pred > 0.01:
                proba[i] = [0.1, 0.2, 0.7]
            elif pred > 0:
                proba[i] = [0.2, 0.4, 0.4]
            elif pred > -0.01:
                proba[i] = [0.4, 0.4, 0.2]
            else:
                proba[i] = [0.7, 0.2, 0.1]
        
        return proba
