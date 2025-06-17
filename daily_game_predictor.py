#!/usr/bin/env python3
"""
WNBA Daily Game Stats Predictor
Predicts Points, Rebounds, Assists for each player in each game

Based on research of successful models:
- XGBoost with synergy features
- Neural networks with attention
- Bayesian approaches with uncertainty
- Context-aware features (opponent, rest, pace)
"""

import os
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# PyTorch for neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class WNBADailyPredictor:
    def __init__(self, game_logs_file):
        """Initialize with game logs data"""
        self.game_logs_df = pd.read_csv(game_logs_file)
        self.models = {
            'points': {},
            'rebounds': {},
            'assists': {}
        }
        self.scalers = {}
        self.feature_columns = []
        self.label_encoders = {}
        
        # Target stats to predict
        self.target_stats = ['points', 'rebounds', 'assists']
        
    def engineer_features(self, df):
        """Engineer features based on successful model research"""
        print("🔧 Engineering features for daily predictions...")
        
        df = df.copy()
        
        # Debug: show available columns
        print(f"   📊 Available columns: {list(df.columns)}")
        
        # Fix column naming issues from merges
        column_fixes = {
            'team_x': 'team',
            'opponent_x': 'opponent', 
            'date_x': 'date'
        }
        
        for old_col, new_col in column_fixes.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
                print(f"   🔧 Fixed {old_col} -> {new_col}")
        
        # Remove duplicate columns
        cols_to_drop = [col for col in df.columns if col.endswith('_y') or col.endswith('_opp')]
        if cols_to_drop:
            df = df.drop(cols_to_drop, axis=1)
            print(f"   🔧 Removed duplicate columns: {cols_to_drop}")
        
        # Ensure we have essential columns
        essential_cols = ['team', 'player', 'points', 'rebounds', 'assists', 'minutes']
        missing_essential = [col for col in essential_cols if col not in df.columns]
        if missing_essential:
            print(f"   ❌ Missing essential columns: {missing_essential}")
            raise ValueError(f"Missing essential columns: {missing_essential}")
        
        # Ensure proper data types
        numeric_cols = ['minutes', 'points', 'rebounds', 'assists', 'fg_made', 'fg_attempted', 
                       'ft_made', 'ft_attempted', 'turnovers', 'steals', 'blocks', 'fouls',
                       'pts_l5', 'reb_l5', 'ast_l5', 'rest_days', 'team_pace', 'opp_def_rating']
        # Ensure proper data types for available columns only
        numeric_cols = ['minutes', 'points', 'rebounds', 'assists', 'fg_made', 'fg_attempted', 
                       'ft_made', 'ft_attempted', 'turnovers', 'steals', 'blocks', 'fouls',
                       'pts_l5', 'reb_l5', 'ast_l5', 'rest_days', 'team_pace', 'opp_def_rating']
        
        # Only convert columns that actually exist
        existing_numeric_cols = [col for col in numeric_cols if col in df.columns]
        for col in existing_numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"   📊 Converted {len(existing_numeric_cols)} numeric columns")
        
        # Fill missing values
        df = df.fillna(0)
        
        # Ensure required columns exist with defaults
        column_defaults = {
            'fg_made': df['points'] / 2.2 if 'points' in df.columns else 5.0,
            'fg_attempted': df['points'] / 1.5 if 'points' in df.columns else 8.0,
            'ft_made': df['points'] * 0.15 if 'points' in df.columns else 1.0,
            'ft_attempted': df['points'] * 0.2 if 'points' in df.columns else 1.5,
            'turnovers': df['assists'] * 0.6 if 'assists' in df.columns else 2.0,
            'steals': 1.2,
            'blocks': 0.8,
            'fouls': 2.5,
            'pts_l5': df['points'] if 'points' in df.columns else 12.0,
            'reb_l5': df['rebounds'] if 'rebounds' in df.columns else 6.0,
            'ast_l5': df['assists'] if 'assists' in df.columns else 3.0,
            'rest_days': 1,
            'team_pace': 80.0,
            'opp_def_rating': 105.0
        }
        
        for col, default_val in column_defaults.items():
            if col not in df.columns:
                if hasattr(default_val, 'fillna'):  # It's a Series
                    df[col] = default_val.fillna(default_val.mean() if len(default_val) > 0 else 0)
                else:  # It's a scalar
                    df[col] = default_val
                print(f"   ➕ Added missing column '{col}' with defaults")
        # Create season averages if they don't exist (using current stats)
        if 'season_avg_pts' not in df.columns:
            df['season_avg_pts'] = df['points']
        if 'season_avg_reb' not in df.columns:
            df['season_avg_reb'] = df['rebounds'] 
        if 'season_avg_ast' not in df.columns:
            df['season_avg_ast'] = df['assists']
        
        print(f"   📊 Ensured season averages exist")
        
        # 1. RECENT FORM FEATURES (last 5 games vs season average)
        # Ensure season averages exist before calculating momentum
        if 'season_avg_pts' in df.columns:
            df['pts_momentum'] = df['pts_l5'] - df['season_avg_pts'] 
        else:
            df['pts_momentum'] = 0.0
            
        if 'season_avg_reb' in df.columns:
            df['reb_momentum'] = df['reb_l5'] - df['season_avg_reb']
        else:
            df['reb_momentum'] = 0.0
            
        if 'season_avg_ast' in df.columns:
            df['ast_momentum'] = df['ast_l5'] - df['season_avg_ast']
        else:
            df['ast_momentum'] = 0.0
        
        # 2. USAGE AND EFFICIENCY FEATURES
        df['usage_rate'] = np.where(df['minutes'] > 0, 
                                   (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) / df['minutes'],
                                   0.2)
        
        df['scoring_efficiency'] = np.where(df['fg_attempted'] > 0,
                                          df['points'] / df['fg_attempted'],
                                          1.0)
        
        df['assist_rate'] = np.where(df['minutes'] > 0,
                                   df['assists'] / df['minutes'],
                                   0.1)
        
        df['rebound_rate'] = np.where(df['minutes'] > 0,
                                    df['rebounds'] / df['minutes'],
                                    0.2)
        
        # 3. GAME CONTEXT FEATURES
        df['pace_factor'] = df['team_pace'] / 80.0  # Normalize to league average
        df['rest_advantage'] = (df['rest_days'] >= 2).astype(int)
        df['fatigue_factor'] = (df['rest_days'] == 0).astype(int)  # Back-to-back
        df['home_boost'] = (df['home_away'] == 'H').astype(int)
        
        # 4. OPPONENT FEATURES (key insight from research)
        df['opp_strength'] = (df['opp_def_rating'] - 100) / 10  # Normalize
        df['matchup_advantage'] = np.where(df['opp_strength'] < 0, 1, 0)  # Playing weak defense
        
        # 5. SITUATIONAL FEATURES
        df['high_usage_game'] = (df['usage_rate'] > 0.25).astype(int)
        
        # Blowout risk based on opponent strength
        df['blowout_risk'] = (np.abs(df['opp_strength']) > 1.5).astype(int)
        
        # 6. PLAYER CLUSTERING FEATURES (inspired by synergy research)
        # Create simple position-based clusters
        position_usage = {
            'PG': {'ast_weight': 1.5, 'reb_weight': 0.7, 'pts_weight': 1.0},
            'SG': {'ast_weight': 0.8, 'reb_weight': 0.8, 'pts_weight': 1.3},
            'SF': {'ast_weight': 1.0, 'reb_weight': 1.0, 'pts_weight': 1.1},
            'PF': {'ast_weight': 0.7, 'reb_weight': 1.4, 'pts_weight': 1.0},
            'C': {'ast_weight': 0.5, 'reb_weight': 1.6, 'pts_weight': 0.9}
        }
        
        # Get position from your existing data or infer from stats
        if 'pos' not in df.columns and 'position' not in df.columns:
            # Infer position from stats if not available
            df['position'] = 'SF'  # Default
            # Simple position inference based on rebounds and assists
            df.loc[df['assists'] > 4, 'position'] = 'PG'
            df.loc[df['rebounds'] > 8, 'position'] = 'C'
            df.loc[(df['points'] > 15) & (df['assists'] < 3), 'position'] = 'SG'
            df.loc[(df['rebounds'] > 6) & (df['rebounds'] < 9), 'position'] = 'PF'
        else:
            df['position'] = df.get('pos', df.get('position', 'SF'))
        
        for pos, weights in position_usage.items():
            df[f'pos_is_{pos}'] = (df['position'] == pos).astype(int)
            df[f'{pos}_ast_expectation'] = df[f'pos_is_{pos}'] * weights['ast_weight']
            df[f'{pos}_reb_expectation'] = df[f'pos_is_{pos}'] * weights['reb_weight'] 
            df[f'{pos}_pts_expectation'] = df[f'pos_is_{pos}'] * weights['pts_weight']
        
        # 7. TEMPORAL FEATURES
        df['game_number_season'] = df['game_num']
        df['season_progress'] = df['game_num'] / 34.0  # Normalize to 0-1
        df['early_season'] = (df['game_num'] <= 10).astype(int)
        df['late_season'] = (df['game_num'] >= 25).astype(int)
        
        # 8. TEAM COMPOSITION FEATURES (basic version)
        # Calculate team averages, handling missing season averages
        team_stats_cols = []
        for stat in ['points', 'rebounds', 'assists']:
            season_col = f'season_avg_{stat}'
            if season_col in df.columns:
                team_stats_cols.append(season_col)
            else:
                # Use current game stats if season averages don't exist
                team_stats_cols.append(stat)
                df[season_col] = df[stat]
        
        team_avg = df.groupby('team')[['season_avg_pts', 'season_avg_reb', 'season_avg_ast']].mean()
        team_avg.columns = ['team_avg_pts', 'team_avg_reb', 'team_avg_ast']
        df = df.merge(team_avg, left_on='team', right_index=True, how='left')
        
        # Fill any remaining NaN values
        df['team_avg_pts'] = df['team_avg_pts'].fillna(15.0)
        df['team_avg_reb'] = df['team_avg_reb'].fillna(8.0)
        df['team_avg_ast'] = df['team_avg_ast'].fillna(4.0)
        
        df['player_share_pts'] = df['season_avg_pts'] / (df['team_avg_pts'] + 0.1)
        df['player_share_reb'] = df['season_avg_reb'] / (df['team_avg_reb'] + 0.1)
        df['player_share_ast'] = df['season_avg_ast'] / (df['team_avg_ast'] + 0.1)
        
        return df
    
    def prepare_data(self):
        """Prepare data for modeling with time-series awareness"""
        print("📊 Preparing data for modeling...")
        
        # Engineer features
        df = self.engineer_features(self.game_logs_df)
        
        # Select features (excluding targets and identifiers)
        base_features = [
            'minutes', 'usage_rate', 'scoring_efficiency', 'assist_rate', 'rebound_rate',
            'pace_factor', 'rest_advantage', 'fatigue_factor', 'home_boost',
            'opp_strength', 'matchup_advantage', 'high_usage_game',
            'pts_momentum', 'reb_momentum', 'ast_momentum',
            'season_progress', 'early_season', 'late_season',
            'player_share_pts', 'player_share_reb', 'player_share_ast'
        ]
        
        # Add position features
        position_features = [col for col in df.columns if 'pos_is_' in col or '_expectation' in col]
        
        self.feature_columns = base_features + position_features
        
        # Remove any features that don't exist
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Remove rows with missing target values
        df = df.dropna(subset=self.target_stats)
        
        # Sort by date for time series split
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['date', 'player'])
        
        self.processed_df = df
        
        print(f"✅ Data prepared: {len(df)} games, {len(self.feature_columns)} features")
        print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    def create_synergy_features(self, df):
        """Create synergy features based on teammate composition"""
        print("🤝 Creating synergy features...")
        
        # Use season averages if available, otherwise current game stats
        pts_col = 'season_avg_pts' if 'season_avg_pts' in df.columns else 'points'
        ast_col = 'season_avg_ast' if 'season_avg_ast' in df.columns else 'assists'
        
        # Group by team and game to get teammate context
        team_games = df.groupby(['team', 'date']).agg({
            'usage_rate': ['mean', 'std'],
            pts_col: 'sum',
            ast_col: 'sum',
            'minutes': 'sum'
        }).reset_index()
        
        # Flatten column names
        team_games.columns = ['team', 'date', 'team_usage_mean', 'team_usage_std', 
                             'team_total_scoring', 'team_total_assists', 'team_total_minutes']
        
        # Fill NaN values that might occur from std calculation
        team_games['team_usage_std'] = team_games['team_usage_std'].fillna(0.05)
        
        # Merge back to get team context for each player
        df = df.merge(team_games, on=['team', 'date'], how='left')
        
        # Fill any remaining NaN values with sensible defaults
        df['team_usage_mean'] = df['team_usage_mean'].fillna(0.2)
        df['team_total_scoring'] = df['team_total_scoring'].fillna(75.0)
        df['team_total_assists'] = df['team_total_assists'].fillna(20.0)
        
        # Create synergy indicators
        df['usage_above_team'] = df['usage_rate'] - df['team_usage_mean']
        df['primary_scorer'] = (df[pts_col] / (df['team_total_scoring'] + 0.1) > 0.25).astype(int)
        df['primary_facilitator'] = (df[ast_col] / (df['team_total_assists'] + 0.1) > 0.3).astype(int)
        
        # Add these to feature columns
        synergy_features = ['usage_above_team', 'primary_scorer', 'primary_facilitator']
        self.feature_columns.extend(synergy_features)
        
        print(f"   ✅ Added {len(synergy_features)} synergy features")
        
        return df
    
    def train_models_for_stat(self, stat_name):
        """Train ensemble of models for a specific stat"""
        print(f"🤖 Training models for {stat_name}...")
        
        df = self.processed_df
        X = df[self.feature_columns]
        y = df[stat_name]
        
        # Time series split (more realistic for sports prediction)
        tscv = TimeSeriesSplit(n_splits=3)
        split_idx = list(tscv.split(X))[-1]  # Use last split
        train_idx, test_idx = split_idx
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx] 
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[stat_name] = scaler
        
        models_to_train = {
            'xgboost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=0
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                random_state=42, verbosity=-1
            ),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=42
            ),
            'hist_gradient': HistGradientBoostingRegressor(
                max_iter=100, max_depth=6, random_state=42
            ),
            'bayesian_ridge': BayesianRidge(
                alpha_1=1e-6, alpha_2=1e-6, lambda_1=1e-6, lambda_2=1e-6
            )
        }
        
        stat_models = {}
        results = {}
        
        for name, model in models_to_train.items():
            try:
                if name == 'bayesian_ridge':
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                else:
                    model.fit(X_train, y_train) 
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                stat_models[name] = model
                results[name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
                
                print(f"  {name}: R² = {r2:.3f}, MAE = {mae:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name} for {stat_name}: {e}")
                
        self.models[stat_name] = stat_models
        return results
    
    def train_neural_network(self, stat_name):
        """Train PyTorch neural network for a stat with uncertainty quantification"""
        print(f"🧠 Training neural network for {stat_name}...")
        
        try:
            df = self.processed_df
            X = df[self.feature_columns].values
            y = df[stat_name].values
            
            # Time series split
            tscv = TimeSeriesSplit(n_splits=3)
            split_idx = list(tscv.split(X))[-1]
            train_idx, test_idx = split_idx
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Create datasets
            train_dataset = GameStatsDataset(X_train_scaled, y_train)
            test_dataset = GameStatsDataset(X_test_scaled, y_test)
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Create model
            model = StatsNeuralNetwork(X_train_scaled.shape[1])
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # Training loop with early stopping
            best_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(100):
                model.train()
                epoch_loss = 0.0
                
                for batch_X, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in test_loader:
                        outputs = model(batch_X)
                        val_loss += criterion(outputs.squeeze(), batch_y).item()
                
                val_loss /= len(test_loader)
                
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break
            
            # Load best model and save
            model.load_state_dict(best_model_state)
            self.models[stat_name]['neural_network'] = model
            self.scalers[f'{stat_name}_nn'] = scaler
            
            # Calculate final metrics
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test_scaled)
                y_pred = model(X_test_tensor).squeeze().numpy()
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print(f"  Neural Network: R² = {r2:.3f}, MAE = {mae:.3f}")
            return {'MSE': mse, 'MAE': mae, 'R2': r2}
            
        except Exception as e:
            print(f"  ❌ Error training neural network for {stat_name}: {e}")
            return None
    
    def predict_game_stats(self, player_features, confidence_level=0.95):
        """Predict points, rebounds, assists for a player in a specific game"""
        predictions = {}
        uncertainties = {}
        
        for stat in self.target_stats:
            if stat not in self.models or not self.models[stat]:
                predictions[stat] = 0.0
                uncertainties[stat] = 1.0
                continue
            
            stat_predictions = []
            
            # Get predictions from each model
            for model_name, model in self.models[stat].items():
                try:
                    if model_name == 'bayesian_ridge':
                        scaler = self.scalers[stat]
                        X_scaled = scaler.transform([player_features])
                        pred = model.predict(X_scaled)[0]
                    elif model_name == 'neural_network':
                        scaler = self.scalers[f'{stat}_nn']
                        X_scaled = scaler.transform([player_features])
                        model.eval()
                        with torch.no_grad():
                            X_tensor = torch.FloatTensor(X_scaled)
                            pred = model(X_tensor).item()
                    else:
                        pred = model.predict([player_features])[0]
                    
                    stat_predictions.append(max(0, pred))  # Non-negative stats
                    
                except Exception as e:
                    print(f"Warning: {model_name} failed for {stat}: {e}")
                    continue
            
            if stat_predictions:
                # Ensemble prediction (mean) and uncertainty (std)
                predictions[stat] = np.mean(stat_predictions)
                uncertainties[stat] = np.std(stat_predictions) if len(stat_predictions) > 1 else 1.0
            else:
                predictions[stat] = 0.0
                uncertainties[stat] = 1.0
        
        return predictions, uncertainties
    
    def predict_daily_games(self, games_schedule):
        """Predict stats for all players in today's games"""
        print("🔮 Making daily game predictions...")
        
        daily_predictions = []
        
        for game in games_schedule:
            home_team = game['home_team']
            away_team = game['away_team']
            
            # Get recent player data for both teams
            home_players = self.get_team_recent_form(home_team)
            away_players = self.get_team_recent_form(away_team)
            
            # Predict for each player
            for player_data in home_players + away_players:
                features = self.extract_prediction_features(player_data, game)
                predictions, uncertainties = self.predict_game_stats(features)
                
                prediction_entry = {
                    'game_id': game['game_id'],
                    'player': player_data['player'],
                    'team': player_data['team'],
                    'opponent': away_team if player_data['team'] == home_team else home_team,
                    'home_away': 'H' if player_data['team'] == home_team else 'A',
                    'predicted_points': round(predictions['points'], 1),
                    'predicted_rebounds': round(predictions['rebounds'], 1),
                    'predicted_assists': round(predictions['assists'], 1),
                    'points_uncertainty': round(uncertainties['points'], 2),
                    'rebounds_uncertainty': round(uncertainties['rebounds'], 2),
                    'assists_uncertainty': round(uncertainties['assists'], 2),
                    # Confidence intervals
                    'points_ci_lower': round(max(0, predictions['points'] - 1.96 * uncertainties['points']), 1),
                    'points_ci_upper': round(predictions['points'] + 1.96 * uncertainties['points'], 1),
                    'rebounds_ci_lower': round(max(0, predictions['rebounds'] - 1.96 * uncertainties['rebounds']), 1),
                    'rebounds_ci_upper': round(predictions['rebounds'] + 1.96 * uncertainties['rebounds'], 1),
                    'assists_ci_lower': round(max(0, predictions['assists'] - 1.96 * uncertainties['assists']), 1),
                    'assists_ci_upper': round(predictions['assists'] + 1.96 * uncertainties['assists'], 1)
                }
                
                daily_predictions.append(prediction_entry)
        
        return pd.DataFrame(daily_predictions)
    
    def get_team_recent_form(self, team):
        """Get recent form data for all players on a team"""
        team_players = self.processed_df[self.processed_df['team'] == team]
        
        # Get most recent stats for each player
        latest_players = team_players.sort_values('game_num').groupby('player').tail(1)
        
        return latest_players.to_dict('records')
    
    def extract_prediction_features(self, player_data, game_info):
        """Extract features needed for prediction from player and game data"""
        features = []
        
        for feature in self.feature_columns:
            if feature in player_data:
                features.append(player_data[feature])
            else:
                # Default values for missing features
                defaults = {
                    'minutes': 25.0,
                    'usage_rate': 0.2,
                    'home_boost': 1 if game_info.get('home_team') == player_data['team'] else 0,
                    'pace_factor': 1.0,
                    'opp_strength': 0.0,
                    'rest_advantage': 1,
                    'fatigue_factor': 0
                }
                features.append(defaults.get(feature, 0.0))
        
        return features
    
    def train_all_models(self):
        """Train models for all target statistics"""
        print("\n🚀 Training models for all statistics...")
        print("=" * 50)
        
        # Prepare data
        self.prepare_data()
        
        # Add synergy features
        self.processed_df = self.create_synergy_features(self.processed_df)
        
        all_results = {}
        
        # Train traditional models for each stat
        for stat in self.target_stats:
            print(f"\n📊 Training models for {stat}:")
            results = self.train_models_for_stat(stat)
            all_results[stat] = results
            
            # Train neural network
            nn_results = self.train_neural_network(stat)
            if nn_results:
                all_results[stat]['neural_network'] = nn_results
        
        print("\n✅ All models trained successfully!")
        return all_results


class GameStatsDataset(Dataset):
    """PyTorch Dataset for game stats data"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class StatsNeuralNetwork(nn.Module):
    """Neural Network for predicting game stats with uncertainty"""
    def __init__(self, input_dim, dropout_rate=0.3):
        super(StatsNeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    """Main execution function"""
    print("🏀 WNBA Daily Game Stats Predictor")
    print("🎯 Predicting Points, Rebounds, Assists for each game")
    print("=" * 60)
    
    # Check for game logs data
    game_logs_files = [
        'wnba_game_data/player_game_logs_2025_*.csv',
        'player_game_logs.csv'
    ]
    
    game_logs_file = None
    for pattern in game_logs_files:
        import glob
        files = glob.glob(pattern)
        if files:
            game_logs_file = files[0]
            break
    
    if not game_logs_file:
        print("❌ No game logs file found!")
        print("💡 Run the enhanced data fetcher first to generate game logs")
        return
    
    print(f"📊 Loading game logs from: {game_logs_file}")
    
    # Initialize predictor
    predictor = WNBADailyPredictor(game_logs_file)
    
    # Train all models
    results = predictor.train_all_models()
    
    # Example daily prediction
    sample_schedule = [
        {
            'game_id': '2025-06-16_ATL_CHI',
            'home_team': 'ATL',
            'away_team': 'CHI',
            'game_time': '7:00 PM ET'
        },
        {
            'game_id': '2025-06-16_GSV_LAS',
            'home_team': 'GSV',
            'away_team': 'LAS',
            'game_time': '10:00 PM ET'
        }
    ]
    
    print("\n🔮 Generating sample daily predictions...")
    daily_preds = predictor.predict_daily_games(sample_schedule)
    
    if len(daily_preds) > 0:
        print("\n📋 Sample Daily Predictions:")
        print(daily_preds[['player', 'team', 'predicted_points', 'predicted_rebounds', 
                          'predicted_assists', 'points_ci_lower', 'points_ci_upper']].head(10))
        
        # Save predictions
        os.makedirs('wnba_predictions', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        daily_preds.to_csv(f'wnba_predictions/daily_predictions_{timestamp}.csv', index=False)
        print(f"\n✅ Daily predictions saved to wnba_predictions/daily_predictions_{timestamp}.csv")
    
    print("\n🎉 Daily prediction system ready!")
    print("💡 Use predictor.predict_daily_games(schedule) for live predictions")


if __name__ == "__main__":
    main()