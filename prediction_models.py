#!/usr/bin/env python3
"""
WNBA Prediction Models
Implements ensemble ML models for predicting player statistics.
Based on research from DARKO, XGBoost Synergy, and Neural Network approaches.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
import joblib
import os
from dataclasses import asdict
import warnings

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from data_models import (
    PlayerPrediction, ModelMetrics, PredictionConfig, ModelType,
    WNBAModelError, WNBAPredictionError
)
from feature_engineer import WNBAFeatureEngineer
from team_mapping import TeamNameMapper
from utils import mmss_to_float

warnings.filterwarnings('ignore')


class StatsDataset(Dataset):
    """PyTorch Dataset for player statistics."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Initialize dataset.
        
        Args:
            X: Feature array
            y: Target array
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        return self.X[idx], self.y[idx]


class StatsNeuralNetwork(nn.Module):
    """
    Neural network for predicting player statistics with uncertainty quantification.
    
    Uses dropout for Monte Carlo uncertainty estimation.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [128, 64, 32, 16],
        dropout_rate: float = 0.3
    ):
        """
        Initialize neural network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate for uncertainty
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            # Add dropout (more in earlier layers)
            dropout = dropout_rate * (1 - i * 0.1)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)
    
    def predict_with_uncertainty(
        self, 
        x: torch.Tensor, 
        n_samples: int = 100
    ) -> Tuple[float, float]:
        """
        Predict with uncertainty using Monte Carlo dropout.
        
        Args:
            x: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean_prediction, uncertainty)
        """
        self.train()  # Enable dropout
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x).item()
                predictions.append(pred)
        
        mean_pred = np.mean(predictions)
        uncertainty = np.std(predictions)
        return float(mean_pred), float(uncertainty)


class WNBAPredictionModel:
    """
    Ensemble prediction model for WNBA player statistics.
    
    Implements multiple ML approaches and combines them for robust predictions.
    
    Attributes:
        config: Configuration for predictions
        feature_engineer: Feature engineering component
        models: Dictionary of trained models by statistic
        scalers: Dictionary of feature scalers by statistic
        feature_columns: List of feature column names
        is_trained: Whether models are trained
        model_version: Version string for tracking
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        model_save_dir: str = "wnba_models"
    ):
        """
        Initialize prediction model.
        
        Args:
            config: Configuration object
            model_save_dir: Directory to save trained models
        """
        self.config = config or PredictionConfig()
        self.model_save_dir = model_save_dir
        self.feature_engineer = WNBAFeatureEngineer(config=self.config)
        
        # Model storage
        self.models: Dict[str, Dict[str, Any]] = {stat: {} for stat in self.config.target_stats}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_columns: List[str] = []
        self.is_trained = False
        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.uncertainty_calibrators: Dict[str, Any] = {}  # stat -> calibrator
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Create model directory
        os.makedirs(self.model_save_dir, exist_ok=True)

    def prepare_data(self, game_logs_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for training using feature engineering.
        
        Args:
            game_logs_df: Raw game logs DataFrame
            
        Returns:
            DataFrame with engineered features
            
        Raises:
            WNBAModelError: If data preparation fails
        """
        try:
            self.logger.info("Preparing data for model training...")
            
            # Engineer features
            features_df = self.feature_engineer.create_all_features(game_logs_df)
            
            # Store feature columns
            self.feature_columns = self.feature_engineer.feature_columns
            
            # Ensure 'float_minutes' is always present
            if 'float_minutes' not in features_df.columns and 'MP' in features_df.columns:
                features_df['float_minutes'] = features_df['MP'].apply(mmss_to_float)
            elif 'float_minutes' not in features_df.columns:
                features_df['float_minutes'] = 0.0
            # Use 'float_minutes' everywhere minutes as a float are needed
            
            # Validate we have enough data
            min_games = self.config.min_games_for_prediction
            player_counts = features_df.groupby('player').size()
            valid_players = player_counts[player_counts >= min_games].index
            if not isinstance(valid_players, pd.Index):
                valid_players = pd.Index(valid_players)
            if len(valid_players) == 0:
                raise WNBAModelError(f"No players with at least {min_games} games")
            # Filter to valid players and reset index for safe splitting
            features_df = features_df[features_df['player'].isin(valid_players)]
            if not isinstance(features_df, pd.DataFrame):
                features_df = pd.DataFrame(features_df)
            features_df = features_df.reset_index(drop=True)
            
            self.logger.info(f"Data prepared: {len(features_df)} games for {len(valid_players)} players")
            self.logger.info(f"Features: {len(self.feature_columns)} columns")
            
            return features_df
            
        except Exception as e:
            raise WNBAModelError(f"Data preparation failed: {e}")

    def _player_out_split(self, df, holdout_games=3):
        # Ensure df is a DataFrame with a default RangeIndex if needed
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input to _player_out_split must be a DataFrame")
        df = df.sort_values(['player', 'date'])
        if not df.index.is_monotonic_increasing or not isinstance(df.index, pd.RangeIndex):
            df = df.reset_index(drop=True)
        train_idx, test_idx = [], []
        for p, grp in df.groupby('player'):
            idx = grp.index.to_list()
            if len(idx) <= holdout_games:
                continue
            train_idx.extend(idx[:-holdout_games])
            test_idx.extend(idx[-holdout_games:])
        return np.array(train_idx), np.array(test_idx)

    def _create_base_models(self, stat_name: str = None) -> Dict[str, Any]:
        """
        Create base models for ensemble using hyperparameters from config.models or Optuna best params.
        If stat_name is provided, use stat-specific Optuna parameters if available.
        Returns:
            Dictionary of model instances
        """
        from config_loader import ConfigLoader, ModelConfig
        import json
        import copy
        optuna_params = None
        try:
            with open("optuna_best_params.json", "r") as f:
                optuna_params = json.load(f)
        except Exception:
            optuna_params = None

        # Use config.models if available, else fallback to defaults
        model_config = None
        if hasattr(self.config, 'models'):
            model_config = self.config.models
        else:
            try:
                wnba_config = ConfigLoader.load_config()
                model_config = wnba_config.models
            except Exception:
                model_config = ModelConfig()

        # Helper to filter params for each model
        def filter_params(params, allowed_keys):
            return {k: v for k, v in params.items() if k in allowed_keys}

        # Helper to get params: prefer stat-specific optuna, then global optuna, then config, then fallback
        def get_params(model_name, config_dict, optuna_key=None, allowed_keys=None):
            params = None
            # Try stat-specific Optuna params
            if optuna_params and stat_name and stat_name in optuna_params and optuna_key and optuna_key in optuna_params[stat_name]:
                params = copy.deepcopy(optuna_params[stat_name][optuna_key]["best_params"])
            # Fallback to global Optuna params (legacy)
            elif optuna_params and optuna_key and optuna_key in optuna_params:
                params = copy.deepcopy(optuna_params[optuna_key]["best_params"])
            # Fallback to config
            elif config_dict:
                params = config_dict
            else:
                params = {}
            # Add required defaults if missing
            if model_name == "xgboost":
                params.setdefault("random_state", 42)
                params.setdefault("verbosity", 0)
            elif model_name == "lightgbm":
                params.setdefault("random_state", 42)
                params.setdefault("verbosity", -1)
            elif model_name == "random_forest":
                params.setdefault("random_state", 42)
                params.setdefault("n_jobs", -1)
            if allowed_keys:
                params = filter_params(params, allowed_keys)
            return params

        xgb_allowed = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state", "verbosity"}
        lgb_allowed = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "random_state", "verbosity"}
        rf_allowed = {"n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "random_state", "n_jobs"}
        bayes_allowed = {"alpha_1", "alpha_2", "lambda_1", "lambda_2"}

        xgb_params = get_params("xgboost", getattr(model_config, "xgboost", {}), "XGBoost", xgb_allowed)
        lgb_params = get_params("lightgbm", getattr(model_config, "lightgbm", {}), "LightGBM", lgb_allowed)
        rf_params = get_params("random_forest", getattr(model_config, "random_forest", {}), "RandomForest", rf_allowed)
        # For BayesianRidge, ensure only the four float hyperparameters are passed
        def get_bayes_params():
            params = get_params("bayesian_ridge", {}, "BayesianRidge", bayes_allowed)
            # Only keep the four float params and ensure no extra keys
            filtered = {k: float(params[k]) for k in bayes_allowed if k in params and params[k] is not None}
            # Fill missing with defaults
            for k in bayes_allowed:
                if k not in filtered:
                    filtered[k] = 1e-6
            # Remove any extra keys
            filtered = {k: float(filtered[k]) for k in bayes_allowed}
            return filtered

        bayes_params = get_bayes_params()

        return {
            ModelType.XGBOOST.value: xgb.XGBRegressor(**xgb_params),
            ModelType.LIGHTGBM.value: lgb.LGBMRegressor(**lgb_params),
            ModelType.RANDOM_FOREST.value: RandomForestRegressor(**rf_params),
            ModelType.BAYESIAN_RIDGE.value: BayesianRidge(**bayes_params)
        }

    def train_models_for_stat(
        self, 
        features_df: pd.DataFrame, 
        stat_name: str
    ) -> Dict[str, ModelMetrics]:
        """
        Train all models for a specific statistic.
        
        Args:
            features_df: DataFrame with features and targets
            stat_name: Name of statistic to predict
            
        Returns:
            Dictionary of model metrics
            
        Raises:
            WNBAModelError: If training fails
        """
        if stat_name not in self.config.target_stats:
            raise WNBAModelError(f"Invalid stat name: {stat_name}")
        
        self.logger.info(f"Training models for {stat_name}...")
        
        # Prepare data
        X = features_df[self.feature_columns].fillna(0)
        y = features_df[stat_name]
        
        # Player-out split: hold out last n games per player
        train_idx, test_idx = self._player_out_split(features_df, holdout_games=3)
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise WNBAModelError("Insufficient data for player-out split")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[stat_name] = scaler
        
        # Train models
        base_models = self._create_base_models(stat_name=stat_name)
        trained_models = {}
        metrics = {}
        
        for model_name, model in base_models.items():
            try:
                # Train model
                if model_name == ModelType.BAYESIAN_RIDGE.value:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                # Simple Brier score (convert to binary classification)
                median_val = y_test.median()
                y_binary = (y_test > median_val).astype(int)
                y_pred_prob = (y_pred > median_val).astype(float)
                brier_score = np.mean((y_pred_prob - y_binary) ** 2)
                
                # Coverage (simplified)
                residuals = np.abs(y_test - y_pred)
                coverage = np.mean(residuals <= 1.96 * np.std(residuals))
                
                trained_models[model_name] = model
                metrics[model_name] = ModelMetrics(
                    model_name=model_name,
                    stat_type=stat_name,
                    mae=mae,
                    mse=mse,
                    r2_score=r2,
                    brier_score=brier_score,
                    coverage_95=coverage,
                    n_predictions=len(y_test)
                )
                
                self.logger.info(f"  {model_name}: R² = {r2:.3f}, MAE = {mae:.3f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to train {model_name} for {stat_name}: {e}")
                continue
        
        # Train neural network
        try:
            nn_model, nn_metrics = self._train_neural_network(
                X_train_scaled, y_train, X_test_scaled, y_test, stat_name
            )
            if nn_model:
                trained_models[ModelType.NEURAL_NETWORK.value] = nn_model
                metrics[ModelType.NEURAL_NETWORK.value] = nn_metrics
                self.logger.info(f"  Neural Network: R² = {nn_metrics.r2_score:.3f}, MAE = {nn_metrics.mae:.3f}")
        except Exception as e:
            self.logger.warning(f"Neural network training failed for {stat_name}: {e}")
        
        self.models[stat_name] = trained_models
        return metrics

    def _train_neural_network(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_test: np.ndarray, 
        y_test: np.ndarray,
        stat_name: str
    ) -> Tuple[Optional[StatsNeuralNetwork], Optional[ModelMetrics]]:
        """
        Train neural network with uncertainty quantification - FIXED VERSION.
        
        CRITICAL FIX: Properly convert pandas Series/DataFrame to numpy arrays
        """
        try:
            # FIXED: Ensure we have numpy arrays, not pandas Series/DataFrame
            if hasattr(X_train, 'values'):
                X_train = X_train.values
            if hasattr(y_train, 'values'):
                y_train = y_train.values
            if hasattr(X_test, 'values'):
                X_test = X_test.values
            if hasattr(y_test, 'values'):
                y_test = y_test.values
            
            # Ensure arrays are the right shape and type
            X_train = np.asarray(X_train, dtype=np.float32)
            y_train = np.asarray(y_train, dtype=np.float32)
            X_test = np.asarray(X_test, dtype=np.float32)
            y_test = np.asarray(y_test, dtype=np.float32)
            
            # Ensure y arrays are 1D
            if y_train.ndim > 1:
                y_train = y_train.ravel()
            if y_test.ndim > 1:
                y_test = y_test.ravel()
            
            # Create datasets
            train_dataset = StatsDataset(X_train, y_train)
            test_dataset = StatsDataset(X_test, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create model
            model = StatsNeuralNetwork(X_train.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            best_model_state = None
            
            for epoch in range(100):
                # Training
                model.train()
                train_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs.squeeze(), batch_y).item()
                
                val_loss /= len(test_loader)
                
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Load best model
            if best_model_state:
                model.load_state_dict(best_model_state)
            
            # Calculate final metrics
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test)
                y_pred = model(X_test_tensor).squeeze().numpy()
            
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Brier score
            median_val = np.median(y_test)
            y_binary = (y_test > median_val).astype(int)
            y_pred_prob = (y_pred > median_val).astype(float)
            brier_score = np.mean((y_pred_prob - y_binary) ** 2)
            
            # Coverage
            residuals = np.abs(y_test - y_pred)
            coverage = np.mean(residuals <= 1.96 * np.std(residuals))
            
            metrics = ModelMetrics(
                model_name=ModelType.NEURAL_NETWORK.value,
                stat_type=stat_name,
                mae=mae,
                mse=mse,
                r2_score=r2,
                brier_score=brier_score,
                coverage_95=coverage,
                n_predictions=len(y_test)
            )
            
            return model, metrics
            
        except Exception as e:
            self.logger.warning(f"Neural network training failed: {e}")
            # Print more detailed error for debugging
            import traceback
            self.logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None, None

    def train_all_models(self, game_logs_df: pd.DataFrame) -> Dict[str, Dict[str, ModelMetrics]]:
        """
        Train models for all target statistics.
        
        Args:
            game_logs_df: Raw game logs DataFrame
            
        Returns:
            Dictionary of metrics by statistic and model
            
        Raises:
            WNBAModelError: If training fails
        """
        try:
            self.logger.info("Starting model training for all statistics...")
            
            # Prepare data
            features_df = self.prepare_data(game_logs_df)
            
            # Train models for each statistic
            all_metrics = {}
            for stat in self.config.target_stats:
                stat_metrics = self.train_models_for_stat(features_df, stat)
                all_metrics[stat] = stat_metrics
            
            self.is_trained = True
            self.logger.info("✅ All models trained successfully!")
            
            return all_metrics
            
        except Exception as e:
            raise WNBAModelError(f"Model training failed: {e}")

    def set_uncertainty_calibrators(self, calibrators: Dict[str, Any]):
        """
        Set uncertainty calibrators (e.g., isotonic regression models) for each stat.
        Args:
            calibrators: dict mapping stat name to fitted calibrator (must have .predict method)
        """
        self.uncertainty_calibrators = calibrators or {}

    def predict_player_stats(
        self, 
        player_features: Union[Dict[str, float], pd.Series, np.ndarray]
    ) -> PlayerPrediction:
        """
        Predict statistics for a single player.
        If self.uncertainty_calibrators is set, use them to calibrate uncertainties for each stat.
        
        Args:
            player_features: Features for prediction (dict, Series, or array)
            
        Returns:
            PlayerPrediction object with predictions and uncertainties
            
        Raises:
            WNBAPredictionError: If prediction fails
        """
        if not self.is_trained:
            raise WNBAPredictionError("Models not trained. Call train_all_models first.")
        
        try:
            # Convert features to array format
            if isinstance(player_features, dict):
                feature_array = np.array([player_features.get(col, 0.0) for col in self.feature_columns])
            elif isinstance(player_features, pd.Series):
                feature_array = player_features[self.feature_columns].fillna(0.0).values
            else:
                feature_array = np.array(player_features)
            
            feature_array = feature_array.reshape(1, -1)
            
            predictions = {}
            uncertainties = {}
            
            # Get predictions from each statistic
            for stat in self.config.target_stats:
                if stat not in self.models or not self.models[stat]:
                    predictions[stat] = 0.0
                    uncertainties[stat] = 1.0
                    continue
                
                stat_predictions = []
                
                # Get prediction from each model
                for model_name, model in self.models[stat].items():
                    try:
                        if model_name == ModelType.BAYESIAN_RIDGE.value:
                            # Use scaled features
                            scaled_features = self.scalers[stat].transform(feature_array)
                            pred = model.predict(scaled_features)[0]
                        elif model_name == ModelType.NEURAL_NETWORK.value:
                            scaled_features = self.scalers[stat].transform(feature_array)
                            X_tensor = torch.FloatTensor(scaled_features)
                            pred, _ = model.predict_with_uncertainty(X_tensor)
                        else:
                            pred = model.predict(feature_array)[0]
                        
                        stat_predictions.append(max(0, pred))  # Non-negative
                        
                    except Exception as e:
                        self.logger.warning(f"Prediction failed for {model_name} ({stat}): {e}")
                        continue
                
                if stat_predictions:
                    # Ensemble prediction using configured weights
                    # Use weights from ModelConfig
                    model_weights = getattr(self.config, 'models', None)
                    if model_weights and hasattr(model_weights, 'weights'):
                        weights_dict = model_weights.weights
                    else:
                        # fallback: equal weights
                        weights_dict = {name: 1.0 for name in self.models[stat].keys()}
                    weights = [weights_dict.get(name, 0.25) for name in self.models[stat].keys()]
                    weights = np.array(weights[:len(stat_predictions)])
                    if weights.sum() > 0:
                        weights = weights / weights.sum()  # Normalize
                    else:
                        weights = np.ones_like(weights) / len(weights)
                    predictions[stat] = np.average(stat_predictions, weights=weights)
                    raw_uncertainty = np.std(stat_predictions) if len(stat_predictions) > 1 else 1.0
                    # Apply calibrator if available
                    calibrator = self.uncertainty_calibrators.get(stat)
                    if calibrator is not None:
                        try:
                            calibrated = calibrator.predict([raw_uncertainty])[0]
                            uncertainties[stat] = float(np.clip(calibrated, 0.0, None))
                        except Exception:
                            uncertainties[stat] = raw_uncertainty
                    else:
                        uncertainties[stat] = raw_uncertainty
                else:
                    predictions[stat] = 0.0
                    uncertainties[stat] = 1.0
            
            # Calculate overall confidence
            confidence_score = 1.0 - np.mean(list(uncertainties.values())) / 10.0  # Normalize
            confidence_score = np.clip(confidence_score, 0.0, 1.0)
            
            # Create prediction object (would need game context for full object)
            # This is a simplified version - full implementation would need game details
            if not (isinstance(player_features, dict) or hasattr(player_features, 'get')):
                raise WNBAPredictionError("player_features must be a dict or have a .get method")
            team = player_features.get('team')
            opponent = player_features.get('opponent')
            if not team or not opponent:
                raise WNBAPredictionError("Missing team or opponent in player_features. Only real teams from team_mapping.py are allowed.")
            team = TeamNameMapper.to_abbreviation(team)
            opponent = TeamNameMapper.to_abbreviation(opponent)
            if not team or not opponent:
                raise WNBAPredictionError(f"Unknown team or opponent: {team}, {opponent}. Only real teams from team_mapping.py are allowed.")
            return PlayerPrediction(
                game_id="unknown",
                player="unknown",
                team=team,
                opponent=opponent,
                home_away="H",  # Would come from game context
                predicted_points=predictions.get('points', 0.0),
                predicted_rebounds=predictions.get('total_rebounds', 0.0),
                predicted_assists=predictions.get('assists', 0.0),
                points_uncertainty=uncertainties.get('points', 1.0),
                rebounds_uncertainty=uncertainties.get('total_rebounds', 1.0),
                assists_uncertainty=uncertainties.get('assists', 1.0),
                confidence_score=confidence_score,
                model_version=self.model_version
            )
            
        except Exception as e:
            raise WNBAPredictionError(f"Prediction failed: {e}")

    def save_models(self) -> str:
        """
        Save trained models to disk.
        
        Returns:
            Path to saved models directory
            
        Raises:
            WNBAModelError: If saving fails
        """
        if not self.is_trained:
            raise WNBAModelError("No trained models to save")
        
        try:
            save_path = os.path.join(self.model_save_dir, f"models_{self.model_version}")
            os.makedirs(save_path, exist_ok=True)
            
            # Save models and scalers
            for stat, stat_models in self.models.items():
                for model_name, model in stat_models.items():
                    model_file = os.path.join(save_path, f"{stat}_{model_name}.joblib")
                    joblib.dump(model, model_file)
                
                # Save scaler
                if stat in self.scalers:
                    scaler_file = os.path.join(save_path, f"{stat}_scaler.joblib")
                    joblib.dump(self.scalers[stat], scaler_file)
            
            # Save metadata
            metadata = {
                'model_version': self.model_version,
                'feature_columns': self.feature_columns,
                'target_stats': self.config.target_stats,
                'config': asdict(self.config)
            }
            
            metadata_file = os.path.join(save_path, "metadata.joblib")
            joblib.dump(metadata, metadata_file)
            
            self.logger.info(f"Models saved to: {save_path}")
            return save_path
            
        except Exception as e:
            raise WNBAModelError(f"Failed to save models: {e}")

    def load_models(self, model_path: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            model_path: Path to saved models directory
            
        Raises:
            WNBAModelError: If loading fails
        """
        try:
            # Load metadata
            metadata_file = os.path.join(model_path, "metadata.joblib")
            if not os.path.exists(metadata_file):
                raise WNBAModelError(f"Metadata file not found: {metadata_file}")
            
            metadata = joblib.load(metadata_file)
            self.model_version = metadata['model_version']
            self.feature_columns = metadata['feature_columns']
            
            # Load models
            for stat in metadata['target_stats']:
                self.models[stat] = {}
                
                # Load stat models
                for model_type in ModelType:
                    model_file = os.path.join(model_path, f"{stat}_{model_type.value}.joblib")
                    if os.path.exists(model_file):
                        self.models[stat][model_type.value] = joblib.load(model_file)
                
                # Load scaler
                scaler_file = os.path.join(model_path, f"{stat}_scaler.joblib")
                if os.path.exists(scaler_file):
                    self.scalers[stat] = joblib.load(scaler_file)
            
            self.is_trained = True
            self.logger.info(f"Models loaded from: {model_path}")
            
        except Exception as e:
            raise WNBAModelError(f"Failed to load models: {e}")


def main():
    """
    Example usage of the prediction model.
    """
    print("🏀 WNBA Prediction Models - Example")
    print("=" * 40)
    
    print("❌ No sample data provided - requires real game logs")
    print("💡 Example usage:")
    print("   model = WNBAPredictionModel()")
    print("   metrics = model.train_all_models(game_logs_df)")
    print("   prediction = model.predict_player_stats(player_features)")


if __name__ == "__main__":
    main()