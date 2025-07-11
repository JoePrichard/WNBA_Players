import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
import logging
import os
from glob import glob
import json
from typing import Tuple, List, Dict, Callable, Any

from feature_engineer import WNBAFeatureEngineer
from prediction_models import WNBAPredictionModel, StatsNeuralNetwork, StatsDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# --- CONFIGURABLE ---
STATS_TO_PREDICT = ['points', 'total_rebounds', 'assists']  # Default stats to optimize
CSV_FILENAME = None  # Set to a specific CSV filename in wnba_game_data/ or leave as None to auto-detect latest
N_TRIALS = 50
RANDOM_STATE = 42

import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("optuna_sweeps")

# --- DATA LOADING ---
def find_latest_csv(data_dir: str = "wnba_game_data") -> str:
    csv_files = sorted(glob(os.path.join(data_dir, "*.csv")), key=os.path.getmtime, reverse=True)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}/")
    return csv_files[0]

def load_and_engineer_data() -> Tuple[pd.DataFrame, List[str]]:
    # Load from CSV instead of scraping
    if CSV_FILENAME:
        csv_path = os.path.join("wnba_game_data", CSV_FILENAME)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Specified CSV not found: {csv_path}")
    else:
        csv_path = find_latest_csv()
    logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} rows.")

    # --- DIAGNOSTIC PRINTS ---
    logger.info("Engineering features...")
    fe = WNBAFeatureEngineer()
    df = fe.create_all_features(df)
    feature_columns = fe.feature_columns
    logger.info(f"Feature columns: {len(feature_columns)}")
    return df, feature_columns

# --- TRAIN/TEST SPLIT ---
def get_train_test(
    df: pd.DataFrame,
    feature_columns: List[str],
    stat: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = df[feature_columns].fillna(0)
    y = df[stat]
    tscv = TimeSeriesSplit(n_splits=3)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    return X_train, X_test, y_train, y_test

# --- OPTUNA OBJECTIVES ---
def objective_xgb(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE,
        'verbosity': 0
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def objective_lgb(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE,
        'verbosity': -1
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def objective_rf(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

def objective_bayes(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    params = {
        'alpha_1': trial.suggest_float('alpha_1', 1e-7, 1e-5, log=True),
        'alpha_2': trial.suggest_float('alpha_2', 1e-7, 1e-5, log=True),
        'lambda_1': trial.suggest_float('lambda_1', 1e-7, 1e-5, log=True),
        'lambda_2': trial.suggest_float('lambda_2', 1e-7, 1e-5, log=True),
    }
    model = BayesianRidge(**params)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    return mean_absolute_error(y_test, y_pred)

def objective_nn(
    trial: optuna.trial.Trial,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    # Hyperparameters for NN
    hidden_dim = trial.suggest_categorical('hidden_dim', [32, 64, 128])
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
    epochs = 50

    # Prepare data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    y_train_np = np.asarray(y_train)
    y_test_np = np.asarray(y_test)
    X_train_scaled = np.asarray(X_train_scaled)
    X_test_scaled = np.asarray(X_test_scaled)

    train_dataset = StatsDataset(X_train_scaled, y_train_np)
    test_dataset = StatsDataset(X_test_scaled, y_test_np)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Build model
    input_dim = X_train_scaled.shape[1]
    layers = []
    prev_dim = input_dim
    for i in range(n_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, 1))
    model = nn.Sequential(*layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.FloatTensor(X_test_scaled)).squeeze().numpy()
    return mean_absolute_error(y_test_np, y_pred)

# --- MAIN SWEEP FUNCTION ---
def run_optuna_sweep(
    model_name: str,
    objective_fn: Callable[[optuna.trial.Trial, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series], float],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> Tuple[float, Dict[str, Any]]:
    logger.info(f"Running Optuna sweep for {model_name}...")
    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(lambda trial: objective_fn(trial, X_train, X_test, y_train, y_test), n_trials=N_TRIALS, show_progress_bar=True)
    logger.info(f"Best MAE for {model_name}: {study.best_value:.4f}")
    logger.info(f"Best params for {model_name}: {study.best_params}")
    return study.best_value, study.best_params

# --- BASELINE (VANILLA) MAE ---
def get_vanilla_mae(
    model_cls: Any,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> float:
    if model_cls == xgb.XGBRegressor:
        model = model_cls(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=0)
    elif model_cls == lgb.LGBMRegressor:
        model = model_cls(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=RANDOM_STATE, verbosity=-1)
    elif model_cls == BayesianRidge:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = model_cls(alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return mean_absolute_error(y_test, y_pred)
    elif model_cls == RandomForestRegressor:
        model = model_cls(n_estimators=100, max_depth=8, random_state=RANDOM_STATE, n_jobs=-1)
    else:
        model = model_cls()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

# --- MAIN SCRIPT ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sweeps for WNBA models.")
    parser.add_argument('--stat', type=str, default=None, help='Stat to optimize (points, total_rebounds, assists). If not set, runs all.')
    args = parser.parse_args()

    stats_to_run = [args.stat] if args.stat else STATS_TO_PREDICT
    best_params_dict = {}

    for stat in stats_to_run:
        logger.info(f"\n=== Optimizing for stat: {stat} ===")
        df, feature_columns = load_and_engineer_data()
        X_train, X_test, y_train, y_test = get_train_test(df, feature_columns, stat)
        best_params_dict[stat] = {}

        # XGBoost
        vanilla_mae_xgb = get_vanilla_mae(xgb.XGBRegressor, X_train, X_test, y_train, y_test)
        best_mae_xgb, best_params_xgb = run_optuna_sweep('XGBoost', objective_xgb, X_train, X_test, y_train, y_test)
        logger.info(f"\nXGBoost: Vanilla MAE = {vanilla_mae_xgb:.4f}, Optuna Best MAE = {best_mae_xgb:.4f}, Improvement = {100*(vanilla_mae_xgb-best_mae_xgb)/vanilla_mae_xgb:.1f}%")
        logger.info(f"Best XGBoost params: {best_params_xgb}")
        best_params_dict[stat]['XGBoost'] = {'vanilla_mae': vanilla_mae_xgb, 'best_mae': best_mae_xgb, 'best_params': best_params_xgb}

        # LightGBM
        vanilla_mae_lgb = get_vanilla_mae(lgb.LGBMRegressor, X_train, X_test, y_train, y_test)
        best_mae_lgb, best_params_lgb = run_optuna_sweep('LightGBM', objective_lgb, X_train, X_test, y_train, y_test)
        logger.info(f"\nLightGBM: Vanilla MAE = {vanilla_mae_lgb:.4f}, Optuna Best MAE = {best_mae_lgb:.4f}, Improvement = {100*(vanilla_mae_lgb-best_mae_lgb)/vanilla_mae_lgb:.1f}%")
        logger.info(f"Best LightGBM params: {best_params_lgb}")
        best_params_dict[stat]['LightGBM'] = {'vanilla_mae': vanilla_mae_lgb, 'best_mae': best_mae_lgb, 'best_params': best_params_lgb}

        # Random Forest
        vanilla_mae_rf = get_vanilla_mae(RandomForestRegressor, X_train, X_test, y_train, y_test)
        best_mae_rf, best_params_rf = run_optuna_sweep('Random Forest', objective_rf, X_train, X_test, y_train, y_test)
        logger.info(f"\nRandom Forest: Vanilla MAE = {vanilla_mae_rf:.4f}, Optuna Best MAE = {best_mae_rf:.4f}, Improvement = {100*(vanilla_mae_rf-best_mae_rf)/vanilla_mae_rf:.1f}%")
        logger.info(f"Best Random Forest params: {best_params_rf}")
        best_params_dict[stat]['RandomForest'] = {'vanilla_mae': vanilla_mae_rf, 'best_mae': best_mae_rf, 'best_params': best_params_rf}

        # Bayesian Ridge
        vanilla_mae_bayes = get_vanilla_mae(BayesianRidge, X_train, X_test, y_train, y_test)
        best_mae_bayes, best_params_bayes = run_optuna_sweep('Bayesian Ridge', objective_bayes, X_train, X_test, y_train, y_test)
        logger.info(f"\nBayesian Ridge: Vanilla MAE = {vanilla_mae_bayes:.4f}, Optuna Best MAE = {best_mae_bayes:.4f}, Improvement = {100*(vanilla_mae_bayes-best_mae_bayes)/vanilla_mae_bayes:.1f}%")
        logger.info(f"Best Bayesian Ridge params: {best_params_bayes}")
        best_params_dict[stat]['BayesianRidge'] = {'vanilla_mae': vanilla_mae_bayes, 'best_mae': best_mae_bayes, 'best_params': best_params_bayes}

        # Neural Network
        best_mae_nn, best_params_nn = run_optuna_sweep('Neural Network', objective_nn, X_train, X_test, y_train, y_test)
        logger.info(f"\nNeural Network: Optuna Best MAE = {best_mae_nn:.4f}")
        logger.info(f"Best Neural Network params: {best_params_nn}")
        best_params_dict[stat]['NeuralNetwork'] = {'best_mae': best_mae_nn, 'best_params': best_params_nn}

    # Save all best params to JSON
    with open('optuna_best_params.json', 'w') as f:
        json.dump(best_params_dict, f, indent=2)
    logger.info("\nBest parameters for all models and stats saved to optuna_best_params.json")

if __name__ == "__main__":
    main() 