"""
Classification models for direction prediction.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ...core.base import BaseModel


class LightGBMModel(BaseModel):
    """
    LightGBM classifier for direction prediction.
    """

    def __init__(
        self,
        name: str = "LightGBM",
        prediction_horizon: int = 1,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.params = params or {
            'objective': 'multiclass',
            'num_class': 3,  # -1, 0, 1
            'metric': 'multi_logloss',
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
    ) -> "LightGBMModel":
        """Train the LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        self._feature_names = list(X.columns)
        
        # Convert y to categorical (0, 1, 2 for -1, 0, 1)
        y_mapped = self._map_labels(y)
        
        # Create datasets
        train_data = lgb.Dataset(X, label=y_mapped)
        
        valid_sets = [train_data]
        if validation_data:
            X_val, y_val = validation_data
            y_val_mapped = self._map_labels(y_val)
            valid_data = lgb.Dataset(X_val, label=y_val_mapped)
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
        
        # Get probabilities
        proba = self._model.predict(X)
        
        # Convert to class labels (-1, 0, 1)
        predictions = np.argmax(proba, axis=1) - 1
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self._model.predict(X)

    def _map_labels(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Map labels from (-1, 0, 1) to (0, 1, 2)."""
        y_array = np.array(y)
        return y_array + 1  # -1->0, 0->1, 1->2


class XGBoostModel(BaseModel):
    """
    XGBoost classifier for direction prediction.
    """

    def __init__(
        self,
        name: str = "XGBoost",
        prediction_horizon: int = 1,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.params = params or {
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': 'mlogloss',
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
    ) -> "XGBoostModel":
        """Train the XGBoost model."""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        
        self._feature_names = list(X.columns)
        
        # Convert y to categorical (0, 1, 2 for -1, 0, 1)
        y_mapped = self._map_labels(y)
        
        # Create model
        self._model = xgb.XGBClassifier(**self.params)
        
        # Prepare validation data
        eval_set = None
        if validation_data:
            X_val, y_val = validation_data
            y_val_mapped = self._map_labels(y_val)
            eval_set = [(X_val, y_val_mapped)]
        
        # Train model
        self._model.fit(
            X, y_mapped,
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
        
        proba = self._model.predict_proba(X)
        predictions = np.argmax(proba, axis=1) - 1
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self._model.predict_proba(X)

    def _map_labels(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Map labels from (-1, 0, 1) to (0, 1, 2)."""
        y_array = np.array(y)
        return y_array + 1


class CatBoostModel(BaseModel):
    """
    CatBoost classifier for direction prediction.
    """

    def __init__(
        self,
        name: str = "CatBoost",
        prediction_horizon: int = 1,
        params: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.params = params or {
            'loss_function': 'MultiClass',
            'classes_count': 3,
            'depth': 6,
            'learning_rate': 0.05,
            'iterations': 100,
            'random_seed': 42,
            'verbose': False,
        }
        
        self._model = None
        self._feature_names: List[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "CatBoostModel":
        """Train the CatBoost model."""
        try:
            from catboost import CatBoostClassifier, Pool
        except ImportError:
            raise ImportError("CatBoost not installed. Run: pip install catboost")
        
        self._feature_names = list(X.columns)
        
        # Convert y to categorical (0, 1, 2 for -1, 0, 1)
        y_mapped = self._map_labels(y)
        
        # Create pools
        train_pool = Pool(X, label=y_mapped)
        
        eval_set = None
        if validation_data:
            X_val, y_val = validation_data
            y_val_mapped = self._map_labels(y_val)
            eval_set = Pool(X_val, label=y_val_mapped)
        
        # Create and train model
        self._model = CatBoostClassifier(**self.params)
        
        self._model.fit(
            train_pool,
            eval_set=eval_set,
            early_stopping_rounds=10,
            verbose=False,
        )
        
        # Get feature importance
        importance = self._model.get_feature_importance()
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
        
        proba = self._model.predict_proba(X)
        predictions = np.argmax(proba, axis=1) - 1
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        return self._model.predict_proba(X)

    def _map_labels(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Map labels from (-1, 0, 1) to (0, 1, 2)."""
        y_array = np.array(y)
        return y_array + 1


class NeuralNetworkModel(BaseModel):
    """
    Neural Network classifier using PyTorch.
    """

    def __init__(
        self,
        name: str = "NeuralNetwork",
        prediction_horizon: int = 1,
        hidden_dims: List[int] = [128, 64, 32],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 256,
        epochs: int = 100,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name, prediction_horizon, config)
        
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        
        self._model = None
        self._feature_names: List[str] = []
        self._input_dim: int = 0

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[pd.Series, np.ndarray],
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
    ) -> "NeuralNetworkModel":
        """Train the neural network model."""
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch not installed. Run: pip install torch")
        
        self._feature_names = list(X.columns)
        self._input_dim = X.shape[1]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X.values)
        y_mapped = self._map_labels(y)
        y_tensor = torch.LongTensor(y_mapped)
        
        # Create data loader
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )
        
        # Create model
        self._model = self._build_network()
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self._model.parameters(),
            lr=self.learning_rate,
        )
        
        # Training loop
        self._model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self._model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {total_loss / len(train_loader):.4f}")
        
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        import torch
        
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self._model(X_tensor)
            predictions = torch.argmax(outputs, dim=1).numpy() - 1
        
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Generate probability predictions."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call fit() first.")
        
        import torch
        import torch.nn.functional as F
        
        self._model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.values)
            outputs = self._model(X_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy()
        
        return probabilities

    def _build_network(self):
        """Build neural network architecture."""
        import torch
        import torch.nn as nn
        
        layers = []
        prev_dim = self._input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer (3 classes: -1, 0, 1)
        layers.append(nn.Linear(prev_dim, 3))
        
        return nn.Sequential(*layers)

    def _map_labels(self, y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Map labels from (-1, 0, 1) to (0, 1, 2)."""
        y_array = np.array(y)
        return y_array + 1
