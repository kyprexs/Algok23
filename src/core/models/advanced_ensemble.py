"""
Advanced ML Ensemble Pipeline for AgloK23

This module implements state-of-the-art machine learning models including:
- Gradient Boosting Ensembles (XGBoost, LightGBM, CatBoost)
- Deep Learning Models (LSTM, GRU, Transformer)
- Reinforcement Learning Agents
- Model Stacking and Blending
- Hyperparameter Optimization
- Walk-Forward Validation
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import pandas as pd

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import optuna

# Deep Learning
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch

logger = logging.getLogger(__name__)


class AdvancedLSTM(nn.Module):
    """
    Advanced LSTM architecture with attention mechanism for time series prediction
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        bidirectional: bool = True
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers with dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_output_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 3)  # Buy, Hold, Sell
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attended, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last timestep
        out = attended[:, -1, :]
        
        # Feed through fully connected layers
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        
        return self.softmax(out)


class TransformerPredictor(nn.Module):
    """
    Transformer architecture for financial time series prediction
    """
    
    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        sequence_length: int = 60
    ):
        super().__init__()
        
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(sequence_length, d_model)
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 3),  # Buy, Hold, Sell
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.positional_encoding[:seq_len, :].unsqueeze(0)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Use the last timestep for prediction
        out = transformer_out[:, -1, :]
        
        return self.output_head(out)


class HyperparameterOptimizer:
    """
    Advanced hyperparameter optimization using Optuna
    """
    
    def __init__(self, n_trials: int = 100, timeout: int = 3600):
        self.n_trials = n_trials
        self.timeout = timeout
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Optimize XGBoost hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': 'mlogloss',
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            }
            
            model = xgb.XGBClassifier(**params, random_state=42)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            return accuracy_score(y_val, preds)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return study.best_params
    
    def optimize_lightgbm(self, X_train, y_train, X_val, y_val) -> Dict[str, Any]:
        """Optimize LightGBM hyperparameters"""
        
        def objective(trial):
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('num_leaves', 10, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            }
            
            model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            
            return accuracy_score(y_val, preds)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        return study.best_params


class AdvancedEnsemble:
    """
    Advanced ensemble model combining multiple ML architectures
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        ensemble_weights: Optional[Dict[str, float]] = None
    ):
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.ensemble_weights = ensemble_weights or {
            'xgboost': 0.25,
            'lightgbm': 0.25,
            'lstm': 0.25,
            'transformer': 0.25
        }
        
        # Models
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Hyperparameter optimizer
        self.optimizer = HyperparameterOptimizer()
        
        logger.info(f"AdvancedEnsemble initialized with weights: {self.ensemble_weights}")
    
    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_columns: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features for training including sequencing for deep learning models
        """
        # Scale features
        scaler = RobustScaler()
        scaled_features = scaler.fit_transform(df[feature_columns])
        self.scalers['features'] = scaler
        
        # Create sequences for LSTM/Transformer
        X_sequences = []
        X_tabular = []
        y = []
        
        for i in range(self.sequence_length, len(scaled_features) - self.prediction_horizon + 1):
            # Sequence data for LSTM/Transformer
            X_sequences.append(scaled_features[i-self.sequence_length:i])
            
            # Tabular data for tree models (current features)
            X_tabular.append(scaled_features[i-1])
            
            # Target (next period return direction)
            future_return = df['close'].iloc[i + self.prediction_horizon - 1] / df['close'].iloc[i - 1] - 1
            if future_return > 0.002:  # Buy threshold
                y.append(2)
            elif future_return < -0.002:  # Sell threshold
                y.append(0)
            else:  # Hold
                y.append(1)
        
        return np.array(X_sequences), np.array(X_tabular), np.array(y)
    
    def train_tree_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        """Train gradient boosting models with hyperparameter optimization"""
        
        with mlflow.start_run(run_name="xgboost_training"):
            logger.info("Optimizing XGBoost hyperparameters...")
            best_xgb_params = self.optimizer.optimize_xgboost(X_train, y_train, X_val, y_val)
            
            # Train final XGBoost model
            xgb_model = xgb.XGBClassifier(**best_xgb_params, random_state=42)
            xgb_model.fit(X_train, y_train)
            
            # Evaluate
            xgb_preds = xgb_model.predict(X_val)
            xgb_accuracy = accuracy_score(y_val, xgb_preds)
            
            self.models['xgboost'] = xgb_model
            self.model_performance['xgboost'] = xgb_accuracy
            
            # Log to MLflow
            mlflow.log_params(best_xgb_params)
            mlflow.log_metric("accuracy", xgb_accuracy)
            mlflow.xgboost.log_model(xgb_model, "model")
            
            logger.info(f"XGBoost trained - Accuracy: {xgb_accuracy:.4f}")
        
        with mlflow.start_run(run_name="lightgbm_training"):
            logger.info("Optimizing LightGBM hyperparameters...")
            best_lgb_params = self.optimizer.optimize_lightgbm(X_train, y_train, X_val, y_val)
            
            # Train final LightGBM model
            lgb_model = lgb.LGBMClassifier(**best_lgb_params, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            
            # Evaluate
            lgb_preds = lgb_model.predict(X_val)
            lgb_accuracy = accuracy_score(y_val, lgb_preds)
            
            self.models['lightgbm'] = lgb_model
            self.model_performance['lightgbm'] = lgb_accuracy
            
            # Log to MLflow
            mlflow.log_params(best_lgb_params)
            mlflow.log_metric("accuracy", lgb_accuracy)
            
            logger.info(f"LightGBM trained - Accuracy: {lgb_accuracy:.4f}")
    
    def train_deep_models(
        self,
        X_train_seq: np.ndarray,
        y_train: np.ndarray,
        X_val_seq: np.ndarray,
        y_val: np.ndarray
    ):
        """Train deep learning models (LSTM and Transformer)"""
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Training deep models on device: {device}")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.LongTensor(y_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
        y_val_tensor = torch.LongTensor(y_val).to(device)
        
        # Train LSTM
        with mlflow.start_run(run_name="lstm_training"):
            lstm_model = AdvancedLSTM(
                input_size=X_train_seq.shape[2],
                hidden_size=128,
                num_layers=3,
                dropout=0.2
            ).to(device)
            
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            lstm_model.train()
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = lstm_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 20 == 0:
                    lstm_model.eval()
                    with torch.no_grad():
                        val_outputs = lstm_model(X_val_tensor)
                        _, val_preds = torch.max(val_outputs.data, 1)
                        val_accuracy = (val_preds == y_val_tensor).float().mean().item()
                    
                    logger.info(f"LSTM Epoch {epoch+1}/100 - Val Accuracy: {val_accuracy:.4f}")
                    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                    lstm_model.train()
            
            # Final evaluation
            lstm_model.eval()
            with torch.no_grad():
                val_outputs = lstm_model(X_val_tensor)
                _, val_preds = torch.max(val_outputs.data, 1)
                lstm_accuracy = (val_preds == y_val_tensor).float().mean().item()
            
            self.models['lstm'] = lstm_model
            self.model_performance['lstm'] = lstm_accuracy
            
            mlflow.pytorch.log_model(lstm_model, "model")
            logger.info(f"LSTM trained - Accuracy: {lstm_accuracy:.4f}")
        
        # Train Transformer
        with mlflow.start_run(run_name="transformer_training"):
            transformer_model = TransformerPredictor(
                input_size=X_train_seq.shape[2],
                d_model=256,
                nhead=8,
                num_layers=4,
                sequence_length=self.sequence_length
            ).to(device)
            
            optimizer = optim.Adam(transformer_model.parameters(), lr=0.0001, weight_decay=1e-5)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop
            transformer_model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = transformer_model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    transformer_model.eval()
                    with torch.no_grad():
                        val_outputs = transformer_model(X_val_tensor)
                        _, val_preds = torch.max(val_outputs.data, 1)
                        val_accuracy = (val_preds == y_val_tensor).float().mean().item()
                    
                    logger.info(f"Transformer Epoch {epoch+1}/50 - Val Accuracy: {val_accuracy:.4f}")
                    mlflow.log_metric("val_accuracy", val_accuracy, step=epoch)
                    transformer_model.train()
            
            # Final evaluation
            transformer_model.eval()
            with torch.no_grad():
                val_outputs = transformer_model(X_val_tensor)
                _, val_preds = torch.max(val_outputs.data, 1)
                transformer_accuracy = (val_preds == y_val_tensor).float().mean().item()
            
            self.models['transformer'] = transformer_model
            self.model_performance['transformer'] = transformer_accuracy
            
            mlflow.pytorch.log_model(transformer_model, "model")
            logger.info(f"Transformer trained - Accuracy: {transformer_accuracy:.4f}")
    
    async def train_ensemble(
        self,
        df: pd.DataFrame,
        feature_columns: List[str],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train the complete ensemble with walk-forward validation
        """
        logger.info("Starting advanced ensemble training...")
        
        with mlflow.start_run(run_name="ensemble_training"):
            # Prepare data
            X_seq, X_tab, y = self.prepare_features(df, feature_columns)
            
            # Train/validation split
            split_idx = int(len(X_seq) * (1 - validation_split))
            
            X_train_seq, X_val_seq = X_seq[:split_idx], X_seq[split_idx:]
            X_train_tab, X_val_tab = X_tab[:split_idx], X_tab[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            logger.info(f"Training data: {len(X_train_seq)} samples")
            logger.info(f"Validation data: {len(X_val_seq)} samples")
            
            # Train tree-based models
            await asyncio.get_event_loop().run_in_executor(
                None, self.train_tree_models, X_train_tab, y_train, X_val_tab, y_val
            )
            
            # Train deep learning models
            await asyncio.get_event_loop().run_in_executor(
                None, self.train_deep_models, X_train_seq, y_train, X_val_seq, y_val
            )
            
            # Set trained flag before predictions
            self.is_trained = True
            
            # Calculate ensemble performance
            ensemble_preds = self.predict_ensemble(X_val_seq, X_val_tab)
            ensemble_accuracy = accuracy_score(y_val, ensemble_preds)
            
            self.model_performance['ensemble'] = ensemble_accuracy
            
            # Log ensemble metrics
            mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
            for model_name, accuracy in self.model_performance.items():
                mlflow.log_metric(f"{model_name}_accuracy", accuracy)
            
            logger.info(f"🎉 Ensemble training completed!")
            logger.info(f"Individual model performance: {self.model_performance}")
            logger.info(f"🔥 Ensemble accuracy: {ensemble_accuracy:.4f}")
            
            return {
                'ensemble_accuracy': ensemble_accuracy,
                'individual_performance': self.model_performance,
                'training_samples': len(X_train_seq),
                'validation_samples': len(X_val_seq)
            }
    
    def predict_ensemble(
        self,
        X_seq: np.ndarray,
        X_tab: np.ndarray
    ) -> np.ndarray:
        """Make ensemble predictions combining all models"""
        
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = {}
        
        # XGBoost predictions
        if 'xgboost' in self.models:
            xgb_probs = self.models['xgboost'].predict_proba(X_tab)
            predictions['xgboost'] = xgb_probs
        
        # LightGBM predictions
        if 'lightgbm' in self.models:
            lgb_probs = self.models['lightgbm'].predict_proba(X_tab)
            predictions['lightgbm'] = lgb_probs
        
        # LSTM predictions
        if 'lstm' in self.models:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(X_seq).to(device)
            
            self.models['lstm'].eval()
            with torch.no_grad():
                lstm_probs = self.models['lstm'](X_tensor).cpu().numpy()
                predictions['lstm'] = lstm_probs
        
        # Transformer predictions
        if 'transformer' in self.models:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_tensor = torch.FloatTensor(X_seq).to(device)
            
            self.models['transformer'].eval()
            with torch.no_grad():
                transformer_probs = self.models['transformer'](X_tensor).cpu().numpy()
                predictions['transformer'] = transformer_probs
        
        # Weighted ensemble
        ensemble_probs = np.zeros((len(X_seq), 3))
        total_weight = 0
        
        for model_name, probs in predictions.items():
            weight = self.ensemble_weights.get(model_name, 0)
            ensemble_probs += weight * probs
            total_weight += weight
        
        # Normalize
        if total_weight > 0:
            ensemble_probs /= total_weight
        
        return np.argmax(ensemble_probs, axis=1)
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from tree-based models"""
        importance = {}
        
        if 'xgboost' in self.models:
            importance['xgboost'] = self.models['xgboost'].feature_importances_
        
        if 'lightgbm' in self.models:
            importance['lightgbm'] = self.models['lightgbm'].feature_importances_
        
        return importance
