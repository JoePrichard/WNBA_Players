# main_application.py - WNBA Daily Game Prediction System (Improved Schedule Handling)
#!/usr/bin/env python3
"""
WNBA Daily Game Prediction System - Improved Schedule Handling Version

CRITICAL IMPROVEMENTS:
- Better handling of schedule data validation
- Clear warnings when sample data is used instead of real schedules
- Improved error handling for schedule fetching failures
- Enhanced logging for debugging schedule issues
- Validation of schedule data before generating predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import argparse
import sys
import os
from pathlib import Path
import json
import random

# Add project directory to path
try:
    from utils import setup_project_structure, add_project_to_path, setup_logging
    add_project_to_path()
    setup_project_structure()
except ImportError:
    # Fallback if utils not available
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))

# Fixed imports with proper error handling
try:
    from data_models import (
        GameSchedule, PlayerPrediction, PredictionConfig, HomeAway,
        WNBADataError, WNBAModelError, WNBAPredictionError
    )
    from data_fetcher import WNBAStatsScraper  # For historical data only
    from schedule_fetcher import WNBAScheduleFetcher  # For upcoming games
    from prediction_models import WNBAPredictionModel
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please ensure all Python files are in the same directory")
    sys.exit(1)


class WNBADailyPredictor:
    """
    WNBA Daily Predictor - Improved Schedule Handling Version
    
    This version includes better validation and handling of schedule data,
    with clear warnings when sample data is used instead of real game schedules.
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        data_dir: str = "wnba_game_data",
        output_dir: str = "wnba_predictions"
    ):
        """Initialize the WNBA daily predictor with improved schedule handling."""
        self.config = config or PredictionConfig()
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Initialize components with proper separation of concerns
        try:
            # For historical training data
            self.data_fetcher = WNBAStatsScraper()
        except Exception as e:
            self.data_fetcher = None
            logging.warning(f"Could not initialize data fetcher: {e}")
        
        try:
            # For upcoming game schedules (CORRECT approach)
            self.schedule_fetcher = WNBAScheduleFetcher()
        except Exception as e:
            self.schedule_fetcher = None
            logging.warning(f"Could not initialize schedule fetcher: {e}")
        
        try:
            self.prediction_model = WNBAPredictionModel(config=self.config)
        except Exception as e:
            self.prediction_model = None
            logging.error(f"Could not initialize prediction model: {e}")
        
        # Create directories
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup logging
        setup_logging(log_level="INFO", log_file="logs/wnba_main.log")
        self.logger = logging.getLogger(__name__)

    def check_data_availability(self, year: int) -> Dict[str, bool]:
        """Check what data is available for a given year."""
        self.logger.info(f"Checking data availability for {year}...")
        local_files = {
            'schedule': False,
            'player_stats': False,
            'player_game_logs': False,
            'team_stats': False
        }
        
        try:
            data_path = Path(self.data_dir)
            if data_path.exists():
                for file_path in data_path.glob(f"*{year}*.csv"):
                    filename = file_path.name.lower()
                    if 'schedule' in filename:
                        local_files['schedule'] = True
                    elif 'player' in filename and ('game' in filename or 'log' in filename):
                        local_files['player_game_logs'] = True
                    elif 'player' in filename:
                        local_files['player_stats'] = True
                    elif 'team' in filename:
                        local_files['team_stats'] = True
        except Exception as e:
            self.logger.error(f"Error checking data availability: {e}")
        
        self.logger.info(f"Data availability for {year}: {local_files}")
        return local_files

    def fetch_season_data(self, year: int, force_refresh: bool = False) -> Dict[str, str]:
        """
        Fetch and save season data (HISTORICAL data for training).
        
        IMPORTANT: This is for training data, not for getting today's schedule.
        """
        self.logger.info(f"Fetching season data for {year}...")
        file_paths = {}
        
        if not self.data_fetcher:
            raise WNBADataError("Data fetcher not initialized")
        
        try:
            # Check existing files first
            if not force_refresh:
                existing_files = list(Path(self.data_dir).glob(f"*{year}*.csv"))
                if existing_files:
                    self.logger.info(f"Found {len(existing_files)} existing data files for {year}")
                    for file_path in existing_files:
                        filename = file_path.name.lower()
                        if 'schedule' in filename:
                            file_paths["schedule"] = str(file_path)
                        elif 'player' in filename and ('game' in filename or 'log' in filename):
                            file_paths["player_game_logs"] = str(file_path)
                        elif 'player' in filename:
                            file_paths["player_stats"] = str(file_path)
                        elif 'team' in filename:
                            file_paths["team_stats"] = str(file_path)
                    if file_paths:
                        return file_paths
            
            # Fetch new player stats for the season (HISTORICAL data)
            start_date = f"{year}-05-01"  # Typical WNBA season start
            end_date = f"{year}-09-30"    # Typical WNBA season end
            
            try:
                player_stats_df = self.data_fetcher.scrape_date_range(start_date, end_date)
                if not player_stats_df.empty:
                    player_path = self.data_fetcher.save_to_csv(player_stats_df, f"player_stats_{year}.csv")
                    file_paths["player_stats"] = player_path
                    self.logger.info(f"✅ Player stats: {len(player_stats_df)} records")
                else:
                    self.logger.warning(f"No player stats found for {year}")
            except Exception as e:
                self.logger.error(f"Failed to fetch player stats: {e}")
                # Continue without failing completely
            
            return file_paths
            
        except Exception as e:
            raise WNBADataError(f"Season data fetch failed: {e}")

    def load_game_logs(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """Load game logs from file with improved validation and error handling."""
        try:
            if file_path and os.path.exists(file_path):
                target_file = file_path
            else:
                # Prefer wnba_game_data/wnba_stats.csv if it exists
                default_stats_path = os.path.join('wnba_game_data', 'wnba_stats.csv')
                if os.path.exists(default_stats_path):
                    target_file = default_stats_path
                else:
                    # Fallback to finding any CSV file
                    data_files = list(Path(self.data_dir).glob("*.csv"))
                    if not data_files:
                        raise WNBADataError(f"No data files found in {self.data_dir}")
                    
                    # Prioritize files by name
                    for pattern in ['*player*log*', '*player*stat*', '*game*', '*.csv']:
                        matching_files = list(Path(self.data_dir).glob(pattern))
                        if matching_files:
                            target_file = max(matching_files, key=lambda x: x.stat().st_mtime)
                            break
                    else:
                        target_file = max(data_files, key=lambda x: x.stat().st_mtime)
                    
                    self.logger.info(f"Using data file: {target_file.name}")
            
            # Load and validate data
            df = pd.read_csv(target_file)
            self.logger.info(f"Loaded data: {len(df)} records from {target_file}")
            
            if df.empty:
                raise WNBADataError("Data file is empty")
            
            # Validate and convert data format
            df = self._validate_and_convert_data(df)
            
            return df
            
        except Exception as e:
            raise WNBADataError(f"Failed to load data: {e}")

    def _validate_and_convert_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and convert data to expected format for player game logs."""
        self.logger.info("Validating and converting data format...")
        
        # Import utilities for data conversion
        try:
            from utils import standardize_team_name, clean_player_name, safe_float_conversion
        except ImportError:
            # Fallback functions if utils not available
            def standardize_team_name(team): return str(team).upper()
            def clean_player_name(name): return str(name).strip()
            def safe_float_conversion(val, default=0.0):
                try: return float(val) if pd.notna(val) else default
                except: return default
        
        # Enhanced column mapping for Basketball Reference data
        column_mapping = {
            # Standard mappings
            'player': ['player', 'Player', 'name', 'Name', 'player_name'],
            'team': ['team', 'Team', 'tm', 'Tm', 'team_id'],
            'date': ['date', 'Date', 'game_date', 'Date_game'],
            'opponent': ['opponent', 'Opponent', 'opp', 'Opp', 'vs'],
            'home_away': ['home_away', 'Home_Away', 'location', 'venue'],
            'minutes': ['minutes', 'Minutes', 'mp', 'MP', 'min'],
            'points': ['points', 'Points', 'pts', 'PTS'],
            'total_rebounds': ['rebounds', 'Rebounds', 'trb', 'TRB', 'reb', 'total_rebounds'],
            'assists': ['assists', 'Assists', 'ast', 'AST'],
            
            # Additional Basketball Reference columns
            'fg_made': ['fg', 'FG', 'fg_made'],
            'fg_attempted': ['fga', 'FGA', 'fg_attempted'],
            'fg_pct': ['fg%', 'FG%', 'fg_pct'],
            'fg3_made': ['3p', '3P', 'fg3_made'],
            'fg3_attempted': ['3pa', '3PA', 'fg3_attempted'],
            'fg3_pct': ['3p%', '3P%', 'fg3_pct'],
            'ft_made': ['ft', 'FT', 'ft_made'],
            'ft_attempted': ['fta', 'FTA', 'ft_attempted'],
            'ft_pct': ['ft%', 'FT%', 'ft_pct'],
            'off_rebounds': ['orb', 'ORB', 'off_rebounds'],
            'def_rebounds': ['drb', 'DRB', 'def_rebounds'],
            'steals': ['stl', 'STL', 'steals'],
            'blocks': ['blk', 'BLK', 'blocks'],
            'turnovers': ['tov', 'TOV', 'turnovers'],
            'fouls': ['pf', 'PF', 'fouls'],
            'plus_minus': ['+/-', 'plus_minus', 'pm']
        }
        
        # Apply column mapping
        for standard_col, possible_cols in column_mapping.items():
            for col in possible_cols:
                if col in df.columns and standard_col not in df.columns:
                    df = df.rename(columns={col: standard_col})
                    break
        
        # Required columns for basic functionality
        required_columns = ['player', 'team', 'date', 'minutes', 'points']
        
        # Check and create missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            for col in missing_columns:
                if col == 'date':
                    df[col] = datetime.now().date()
                    self.logger.warning("Missing date column, using current date")
                elif col in ['minutes', 'points']:
                    df[col] = 0.0
                    self.logger.warning(f"Missing {col} column, using 0")
                else:
                    raise WNBADataError(f"Cannot create required column: {col}")
        
        # Add missing optional columns with defaults
        optional_columns = {
            'opponent': 'UNK',
            'home_away': 'H',
            'total_rebounds': 0.0,
            'assists': 0.0,
            'fg_made': 0.0,
            'fg_attempted': 0.0,
            'fg_pct': 0.0
        }
        
        for col, default_value in optional_columns.items():
            if col not in df.columns:
                df[col] = default_value
        
        # Clean and standardize data
        try:
            df['player'] = df['player'].apply(clean_player_name)
            df['team'] = df['team'].apply(standardize_team_name)
            
            # Convert date column
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.date
            
            # Convert numeric columns
            numeric_columns = ['minutes', 'points', 'total_rebounds', 'assists']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: safe_float_conversion(x, 0.0))
            
        except Exception as e:
            self.logger.error(f"Error cleaning data: {e}")
            # Continue with raw data if cleaning fails
        
        # Add game number if missing
        if 'game_num' not in df.columns:
            df['game_num'] = df.groupby('player').cumcount() + 1
        
        # Add result if missing
        if 'result' not in df.columns:
            df['result'] = 'W'  # Default to wins
        
        self.logger.info(f"Data converted successfully: {len(df)} records for {df['player'].nunique()} players")
        return df

    def train_prediction_models(
        self, 
        game_logs_df: Optional[pd.DataFrame] = None,
        save_models: bool = True
    ) -> Dict[str, Dict[str, any]]:
        """Train prediction models with improved feature engineering validation."""
        self.logger.info("Training prediction models...")
        
        if not self.prediction_model:
            raise WNBAModelError("Prediction model not initialized")
        
        try:
            # Load data if not provided
            if game_logs_df is None:
                game_logs_df = self.load_game_logs()
            
            # Validate minimum data requirements
            if len(game_logs_df) < 50:
                raise WNBAModelError(f"Insufficient data for training: {len(game_logs_df)} records (need at least 50)")
            
            if game_logs_df['player'].nunique() < 5:
                raise WNBAModelError(f"Insufficient players for training: {game_logs_df['player'].nunique()} players (need at least 5)")
            
            # Train models with improved feature engineering
            metrics = self.prediction_model.train_all_models(game_logs_df)
            
            # Save models if requested
            if save_models:
                try:
                    model_path = self.prediction_model.save_models()
                    self.logger.info(f"Models saved to: {model_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save models: {e}")
            
            # Log performance summary with data leakage warnings
            self.logger.info("📊 Training Results:")
            for stat, stat_metrics in metrics.items():
                if stat_metrics:
                    avg_r2 = np.mean([m.r2_score for m in stat_metrics.values()])
                    avg_mae = np.mean([m.mae for m in stat_metrics.values()])
                    
                    self.logger.info(f"  {stat}: R²={avg_r2:.3f}, MAE={avg_mae:.3f}")
                    
                    # CRITICAL: Warn if scores are too good (indicating data leakage)
                    if avg_r2 > 0.95:
                        self.logger.warning(f"  ⚠️  R² = {avg_r2:.3f} is suspiciously high - possible data leakage!")
                else:
                    self.logger.warning(f"  {stat}: No models trained successfully")
            
            return metrics
            
        except Exception as e:
            raise WNBAModelError(f"Model training failed: {e}")

    def get_todays_schedule(self) -> Tuple[List[GameSchedule], bool]:
        """
        Get today's game schedule with validation and real data indication.
        
        Returns:
            Tuple of (games_list, is_real_data)
        """
        try:
            if self.schedule_fetcher:
                today = date.today()
                games_data = self.schedule_fetcher.get_games_for_date(today)
                
                # Check if any games are sample data
                has_sample_data = any(not game.get('is_real_data', True) for game in games_data)
                is_real_data = not has_sample_data
                
                if has_sample_data:
                    self.logger.warning("🔶 Some or all games are sample data - not real schedule!")
                
                # Convert to GameSchedule objects
                schedule = []
                for game_data in games_data:
                    try:
                        game_schedule = GameSchedule(
                            game_id=f"{game_data['date']}_{game_data['away_team']}_{game_data['home_team']}",
                            date=datetime.strptime(game_data['date'], '%Y-%m-%d').date(),
                            home_team=game_data['home_team'],
                            away_team=game_data['away_team'],
                            game_time=game_data.get('game_time', 'TBD'),
                            status=game_data.get('status', 'scheduled')
                        )
                        schedule.append(game_schedule)
                    except Exception as e:
                        self.logger.warning(f"Error converting game data: {e}")
                        continue
                
                return schedule, is_real_data
            else:
                # Fallback to sample schedule
                schedule = self._create_sample_schedule(date.today())
                return schedule, False
                
        except Exception as e:
            self.logger.error(f"Could not get today's schedule: {e}")
            schedule = self._create_sample_schedule(date.today())
            return schedule, False

    def predict_daily_games(
        self, 
        target_date: Optional[date] = None,
        schedule: Optional[List[GameSchedule]] = None
    ) -> Tuple[List[PlayerPrediction], bool]:
        """
        Generate predictions for games with improved schedule validation.
        
        Returns:
            Tuple of (predictions_list, is_real_data)
        """
        if target_date is None:
            target_date = date.today()
        
        self.logger.info(f"Generating predictions for {target_date}")
        
        try:
            is_real_data = True
            
            # Get schedule if not provided
            if schedule is None:
                if target_date == date.today():
                    schedule, is_real_data = self.get_todays_schedule()
                else:
                    # For other dates, also use schedule fetcher
                    if self.schedule_fetcher:
                        games_data = self.schedule_fetcher.get_games_for_date(target_date)
                        
                        # Check if data is real
                        has_sample_data = any(not game.get('is_real_data', True) for game in games_data)
                        is_real_data = not has_sample_data
                        
                        schedule = []
                        for game_data in games_data:
                            try:
                                game_schedule = GameSchedule(
                                    game_id=f"{game_data['date']}_{game_data['away_team']}_{game_data['home_team']}",
                                    date=target_date,
                                    home_team=game_data['home_team'],
                                    away_team=game_data['away_team'],
                                    game_time=game_data.get('game_time', 'TBD'),
                                    status=game_data.get('status', 'scheduled')
                                )
                                schedule.append(game_schedule)
                            except Exception as e:
                                self.logger.warning(f"Error converting game data: {e}")
                                continue
                    else:
                        schedule = self._create_sample_schedule(target_date)
                        is_real_data = False
            
            if not schedule:
                self.logger.warning(f"No games scheduled for {target_date}")
                return [], is_real_data
            
            # Warn user if using sample data
            if not is_real_data:
                self.logger.warning("🔶 USING SAMPLE SCHEDULE DATA - PREDICTIONS ARE FOR FAKE GAMES!")
                self.logger.warning("🔶 Real schedule data sources failed - these predictions are for development only")
            
            # Check if models are trained
            if not self.prediction_model or not self.prediction_model.is_trained:
                self.logger.info("Models not trained, attempting to load existing models...")
                try:
                    self._load_latest_models()
                except Exception as e:
                    self.logger.error(f"Could not load models: {e}")
                    # Generate sample predictions instead of failing
                    return self._generate_sample_predictions(schedule, target_date), is_real_data
            
            # Generate predictions using trained models
            self.logger.info("Generating predictions using trained models")
            all_predictions = self._generate_model_predictions(schedule, target_date)
            
            self.logger.info(f"Generated {len(all_predictions)} player predictions")
            return all_predictions, is_real_data
            
        except Exception as e:
            self.logger.error(f"Prediction generation failed: {e}")
            # Fallback to sample predictions
            return self._generate_sample_predictions(schedule or [], target_date), False

    def _create_sample_schedule(self, target_date: date) -> List[GameSchedule]:
        """Create sample game schedule for development with clear indication."""
        self.logger.warning("🔶 Creating sample schedule - not real games!")
        
        teams = ['LV', 'NY', 'CHI', 'CONN', 'IND', 'PHX', 'SEA', 'ATL', 'DAL', 'MIN', 'WAS']
        
        # Create 2-3 sample games (realistic for WNBA)
        games = []
        for i in range(min(3, len(teams) // 2)):
            home_team = teams[i * 2]
            away_team = teams[i * 2 + 1]
            
            game = GameSchedule(
                game_id=f"{target_date}_{away_team}_{home_team}",
                date=target_date,
                home_team=home_team,
                away_team=away_team,
                game_time="7:00 PM",
                status="scheduled"
            )
            games.append(game)
        
        return games

    def _generate_model_predictions(self, schedule: List[GameSchedule], target_date: date) -> List[PlayerPrediction]:
        """Generate predictions for scheduled games using trained models."""
        predictions: List[PlayerPrediction] = []
        
        try:
            # Load historical game logs for feature creation
            game_logs_df = self.load_game_logs()
            if game_logs_df.empty:
                self.logger.warning("No historical game logs available – falling back to sample predictions")
                return self._generate_sample_predictions(schedule, target_date)
            
            # Create features with improved feature engineering
            try:
                features_df = self.prediction_model.feature_engineer.create_all_features(game_logs_df)
            except Exception as fe_err:
                self.logger.error(f"Feature engineering failed: {fe_err}")
                return self._generate_sample_predictions(schedule, target_date)
            
            # Get latest features for each player
            latest_features = (
                features_df.sort_values('date')
                .groupby('player', as_index=False)
                .tail(1)
            )
            
            # Generate predictions for each game
            for game in schedule:
                for team, is_home in [(game.home_team, True), (game.away_team, False)]:
                    opponent = game.away_team if is_home else game.home_team
                    
                    # Get players for this team
                    team_players_df = latest_features[latest_features['team'] == team]
                    
                    if team_players_df.empty:
                        self.logger.warning(f"No player data found for team {team}")
                        continue
                    
                    # Select top players by minutes
                    team_players_df = team_players_df.sort_values('minutes', ascending=False).head(8)
                    
                    for _, player_row in team_players_df.iterrows():
                        try:
                            # Generate prediction
                            player_pred_raw = self.prediction_model.predict_player_stats(player_row)
                            
                            # Create final prediction object
                            pred = PlayerPrediction(
                                game_id=game.game_id,
                                player=str(player_row['player']),
                                team=team,
                                opponent=opponent,
                                home_away=HomeAway.HOME if is_home else HomeAway.AWAY,
                                predicted_points=round(player_pred_raw.predicted_points, 1),
                                predicted_rebounds=round(player_pred_raw.predicted_rebounds, 1),
                                predicted_assists=round(player_pred_raw.predicted_assists, 1),
                                points_uncertainty=round(player_pred_raw.points_uncertainty, 1),
                                rebounds_uncertainty=round(player_pred_raw.rebounds_uncertainty, 1),
                                assists_uncertainty=round(player_pred_raw.assists_uncertainty, 1),
                                confidence_score=round(player_pred_raw.confidence_score, 2),
                                model_version=self.prediction_model.model_version
                            )
                            predictions.append(pred)
                            
                        except Exception as pred_err:
                            self.logger.warning(f"Prediction failed for {player_row.get('player', 'Unknown')}: {pred_err}")
                            continue
        
        except Exception as e:
            self.logger.error(f"Model-based prediction generation failed: {e}")
            return self._generate_sample_predictions(schedule, target_date)
        
        return predictions

    def _generate_sample_predictions(self, schedule: List[GameSchedule], target_date: date) -> List[PlayerPrediction]:
        """Generate sample predictions for development and fallback."""
        self.logger.warning("🔶 Generating sample predictions - not real player predictions!")
        
        # Sample players for each team
        team_players = {
            'LV': ['A\'ja Wilson', 'Kelsey Plum', 'Jackie Young'],
            'NY': ['Sabrina Ionescu', 'Breanna Stewart', 'Jonquel Jones'],
            'CHI': ['Chennedy Carter', 'Angel Reese', 'Dana Evans'],
            'CONN': ['Alyssa Thomas', 'DeWanna Bonner', 'DiJonai Carrington'],
            'IND': ['Caitlin Clark', 'Aliyah Boston', 'Kelsey Mitchell'],
            'PHX': ['Diana Taurasi', 'Brittney Griner', 'Kahleah Copper'],
            'SEA': ['Jewell Loyd', 'Nneka Ogwumike', 'Skylar Diggins-Smith'],
            'ATL': ['Rhyne Howard', 'Tina Charles', 'Allisha Gray'],
            'DAL': ['Arike Ogunbowale', 'Satou Sabally', 'Teaira McCowan'],
            'MIN': ['Napheesa Collier', 'Kayla McBride', 'Bridget Carleton'],
            'WAS': ['Ariel Atkins', 'Elena Delle Donne', 'Aaliyah Edwards']
        }
        
        predictions = []
        
        for game in schedule:
            # Generate predictions for players from both teams
            for team, is_home in [(game.home_team, True), (game.away_team, False)]:
                opponent = game.away_team if is_home else game.home_team
                
                players = team_players.get(team, [f"Player 1", f"Player 2", f"Player 3"])
                
                for player in players:
                    # Generate realistic predictions based on player type
                    if any(star in player for star in ['Wilson', 'Stewart', 'Clark', 'Ionescu', 'Taurasi']):
                        # Star players
                        points = random.normalvariate(22, 3)
                        rebounds = random.normalvariate(8, 2)
                        assists = random.normalvariate(6, 2)
                        confidence = random.uniform(0.75, 0.95)
                    else:
                        # Role players
                        points = random.normalvariate(12, 3)
                        rebounds = random.normalvariate(5, 2)
                        assists = random.normalvariate(3, 1.5)
                        confidence = random.uniform(0.6, 0.8)
                    
                    # Ensure non-negative values
                    points = max(0, points)
                    rebounds = max(0, rebounds)
                    assists = max(0, assists)
                    
                    prediction = PlayerPrediction(
                        game_id=game.game_id,
                        player=player,
                        team=team,
                        opponent=opponent,
                        home_away=HomeAway.HOME if is_home else HomeAway.AWAY,
                        predicted_points=round(points, 1),
                        predicted_rebounds=round(rebounds, 1),
                        predicted_assists=round(assists, 1),
                        points_uncertainty=round(points * 0.2, 1),
                        rebounds_uncertainty=round(rebounds * 0.25, 1),
                        assists_uncertainty=round(assists * 0.3, 1),
                        confidence_score=round(confidence, 2),
                        model_version="sample_v1.0"
                    )
                    
                    predictions.append(prediction)
        
        return predictions

    def _load_latest_models(self) -> None:
        """Load the most recent trained models."""
        try:
            model_dirs = list(Path("wnba_models").glob("models_*"))
            
            if not model_dirs:
                raise WNBAModelError("No trained models found. Run training first.")
            
            # Get most recent model
            latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
            
            if self.prediction_model:
                self.prediction_model.load_models(str(latest_model_dir))
                self.logger.info(f"Loaded models from: {latest_model_dir}")
            else:
                raise WNBAModelError("Prediction model not initialized")
                
        except Exception as e:
            raise WNBAModelError(f"Failed to load models: {e}")

    def export_predictions(
        self, 
        predictions: List[PlayerPrediction],
        target_date: Optional[date] = None,
        is_real_data: bool = True
    ) -> str:
        """Export predictions to CSV file with data source indication."""
        if target_date is None:
            target_date = date.today()
        
        # Convert to DataFrame
        prediction_dicts = [pred.to_dict() for pred in predictions]
        df = pd.DataFrame(prediction_dicts)
        
        # Add data source information
        df['data_source'] = 'Real Schedule' if is_real_data else 'Sample Data'
        df['is_real_data'] = is_real_data
        
        # Export file with clear indication of data type
        timestamp = datetime.now().strftime("%H%M%S")
        data_type = "real" if is_real_data else "sample"
        filename = f"predictions_{target_date.strftime('%Y%m%d')}_{data_type}_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        
        df.to_csv(filepath, index=False)
        
        if is_real_data:
            self.logger.info(f"✅ Exported {len(predictions)} real predictions to: {filepath}")
        else:
            self.logger.warning(f"🔶 Exported {len(predictions)} sample predictions to: {filepath}")
            self.logger.warning(f"🔶 File contains predictions for fake games - for development only!")
        
        return filepath

    def run_full_pipeline(
        self, 
        year: int,
        train_models: bool = True,
        predict_today: bool = True
    ) -> Dict[str, any]:
        """Run the complete prediction pipeline with improved schedule validation."""
        results = {}
        
        try:
            self.logger.info("🏀 Starting WNBA Prediction Pipeline (Improved Version)")
            self.logger.info("=" * 50)
            
            # 1. Check data availability
            availability = self.check_data_availability(year)
            results['data_availability'] = availability
            
            # 2. Fetch season data (for training)
            try:
                file_paths = self.fetch_season_data(year)
                results['data_files'] = file_paths
                self.logger.info(f"✅ Data files: {list(file_paths.keys())}")
            except WNBADataError as e:
                self.logger.error(f"Data fetching failed: {e}")
                results['data_error'] = str(e)
            
            # 3. Train models if requested and data available
            if train_models and 'data_files' in results:
                try:
                    metrics = self.train_prediction_models()
                    results['training_metrics'] = metrics
                    self.logger.info("✅ Model training completed")
                    
                    # Check for data leakage in results
                    for stat, stat_metrics in metrics.items():
                        if stat_metrics:
                            avg_r2 = np.mean([m.r2_score for m in stat_metrics.values()])
                            if avg_r2 > 0.95:
                                self.logger.warning(f"⚠️  {stat}: R² = {avg_r2:.3f} - suspiciously high!")
                                results['data_leakage_warning'] = f"High R² scores detected - possible data leakage"
                    
                except WNBAModelError as e:
                    self.logger.error(f"Model training failed: {e}")
                    results['training_error'] = str(e)
            
            # 4. Generate today's predictions if requested
            if predict_today:
                try:
                    predictions, is_real_data = self.predict_daily_games()
                    
                    if predictions:
                        export_path = self.export_predictions(predictions, is_real_data=is_real_data)
                        results['predictions_file'] = export_path
                        results['num_predictions'] = len(predictions)
                        results['is_real_schedule'] = is_real_data
                        
                        if is_real_data:
                            self.logger.info(f"✅ Generated {len(predictions)} predictions from real schedule")
                        else:
                            self.logger.warning(f"🔶 Generated {len(predictions)} predictions from sample schedule")
                            results['sample_data_warning'] = "Predictions based on sample/fake game schedule"
                    else:
                        results['predictions_note'] = "No games today"
                        self.logger.info("📅 No games scheduled for today")
                        
                except WNBAPredictionError as e:
                    self.logger.error(f"Daily prediction failed: {e}")
                    results['prediction_error'] = str(e)
            
            # Summary
            errors = [key for key in results.keys() if key.endswith('_error')]
            warnings = [key for key in results.keys() if key.endswith('_warning')]
            
            if not errors and not warnings:
                self.logger.info("🎉 Pipeline completed successfully!")
            elif warnings and not errors:
                self.logger.info(f"⚠️  Pipeline completed with {len(warnings)} warnings")
                for warning_key in warnings:
                    self.logger.info(f"   - {warning_key}: {results[warning_key]}")
            else:
                self.logger.info(f"⚠️ Pipeline completed with {len(errors)} issues")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            results['pipeline_error'] = str(e)
            return results


def main():
    """Main function with improved command line interface and error handling."""
    parser = argparse.ArgumentParser(
        description="WNBA Daily Game Prediction System - Improved Schedule Handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_application.py --check-data 2025
  python main_application.py --fetch-data 2025
  python main_application.py --train 2025
  python main_application.py --predict
  python main_application.py --full-pipeline 2025
        """
    )
    
    parser.add_argument(
        '--check-data',
        type=int,
        metavar='YEAR',
        help='Check data availability for year'
    )
    
    parser.add_argument(
        '--fetch-data',
        type=int,
        metavar='YEAR',
        help='Fetch season data for year'
    )
    
    parser.add_argument(
        '--train',
        type=int,
        metavar='YEAR',
        help='Train models using data from year'
    )
    
    parser.add_argument(
        '--predict',
        action='store_true',
        help='Generate predictions for today or a specific date (use --date YYYY-MM-DD)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        help='Date for prediction in YYYY-MM-DD format (used with --predict)'
    )
    
    parser.add_argument(
        '--full-pipeline',
        type=int,
        metavar='YEAR',
        help='Run complete pipeline for year'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file (optional)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        try:
            # Try to load from TOML if config_loader available
            try:
                from config_loader import ConfigLoader
                config = ConfigLoader.load_config(args.config)
            except ImportError:
                # Fallback to default config
                config = PredictionConfig()
                logging.warning("Using default configuration (config_loader not available)")
        except Exception as e:
            logging.error(f"Failed to load configuration: {e}")
            config = PredictionConfig()
    
    # Initialize predictor
    try:
        # If config is a WNBAConfig, extract the PredictionConfig
        if config is not None:
            from config_loader import WNBAConfig as LoaderWNBAConfig, PredictionConfig as LoaderPredictionConfig
            from data_models import PredictionConfig as ModelsPredictionConfig
            if isinstance(config, LoaderWNBAConfig):
                loader_pred = config.prediction
                # Convert LoaderPredictionConfig to ModelsPredictionConfig
                models_pred = ModelsPredictionConfig(
                    target_stats=list(loader_pred.target_stats),
                    min_games_for_prediction=loader_pred.min_games_for_prediction,
                    confidence_threshold=loader_pred.confidence_threshold,
                    max_uncertainty=loader_pred.max_uncertainty,
                    model_weights=dict(getattr(loader_pred, 'model_weights', {})),
                    feature_importance_threshold=getattr(loader_pred, 'feature_importance_threshold', 0.01),
                    validation_split=getattr(loader_pred, 'validation_split', 0.2),
                    random_state=getattr(loader_pred, 'random_state', 42),
                    n_cross_validation_folds=getattr(loader_pred, 'n_cross_validation_folds', 5),
                    max_training_time_minutes=getattr(loader_pred, 'max_training_time_minutes', 60)
                )
                predictor = WNBADailyPredictor(config=models_pred)
            elif isinstance(config, LoaderPredictionConfig):
                # Convert LoaderPredictionConfig to ModelsPredictionConfig
                models_pred = ModelsPredictionConfig(
                    target_stats=list(config.target_stats),
                    min_games_for_prediction=config.min_games_for_prediction,
                    confidence_threshold=config.confidence_threshold,
                    max_uncertainty=config.max_uncertainty,
                    model_weights=dict(getattr(config, 'model_weights', {})),
                    feature_importance_threshold=getattr(config, 'feature_importance_threshold', 0.01),
                    validation_split=getattr(config, 'validation_split', 0.2),
                    random_state=getattr(config, 'random_state', 42),
                    n_cross_validation_folds=getattr(config, 'n_cross_validation_folds', 5),
                    max_training_time_minutes=getattr(config, 'max_training_time_minutes', 60)
                )
                predictor = WNBADailyPredictor(config=models_pred)
            else:
                predictor = WNBADailyPredictor(config=config)
        else:
            predictor = WNBADailyPredictor()
    except Exception as e:
        print(f"❌ Failed to initialize predictor: {e}")
        sys.exit(1)
    
    print("🏀 WNBA Daily Game Prediction System (Improved Schedule Handling)")
    print("=" * 60)
    
    try:
        if args.check_data:
            availability = predictor.check_data_availability(args.check_data)
            print(f"\n📊 Data Availability for {args.check_data}:")
            for data_type, available in availability.items():
                status = "✅ Available" if available else "❌ Not Available"
                print(f"  {data_type}: {status}")
        
        elif args.fetch_data:
            try:
                file_paths = predictor.fetch_season_data(args.fetch_data)
                print(f"\n📁 Data Files for {args.fetch_data}:")
                for data_type, path in file_paths.items():
                    print(f"  {data_type}: {path}")
            except Exception as e:
                print(f"\n❌ Data fetch failed: {e}")
        
        elif args.train:
            try:
                metrics = predictor.train_prediction_models()
                print(f"\n🤖 Training Complete for {args.train}")
                print("Performance Summary:")
                for stat, stat_metrics in metrics.items():
                    if stat_metrics:
                        print(f"  {stat}:")
                        for model, metric in stat_metrics.items():
                            r2_warning = " ⚠️ SUSPICIOUSLY HIGH" if metric.r2_score > 0.95 else ""
                            print(f"    {model}: R²={metric.r2_score:.3f}, MAE={metric.mae:.3f}{r2_warning}")
                    else:
                        print(f"  {stat}: No models trained successfully")
            except Exception as e:
                print(f"\n❌ Training failed: {e}")
        
        elif args.predict:
            try:
                # Parse date argument if provided
                target_date = None
                if args.date:
                    try:
                        from datetime import datetime
                        target_date = datetime.strptime(args.date, '%Y-%m-%d').date()
                    except Exception:
                        print(f"\n❌ Invalid date format: {args.date}. Use YYYY-MM-DD.")
                        sys.exit(1)
                predictions, is_real_data = predictor.predict_daily_games(target_date=target_date)
                
                if predictions:
                    export_path = predictor.export_predictions(predictions, is_real_data=is_real_data)
                    
                    data_type_indicator = "✅ REAL" if is_real_data else "🔶 SAMPLE"
                    print(f"\n🔮 Generated {len(predictions)} predictions ({data_type_indicator} schedule data)")
                    print(f"   Exported to: {export_path}")
                    
                    if not is_real_data:
                        print(f"\n🔶 WARNING: Using sample schedule data!")
                        print(f"🔶 Real schedule sources failed - predictions are for fake games")
                        print(f"🔶 This is intended for development/testing only")
                    
                    # Show sample predictions
                    print(f"\n📋 Sample predictions:")
                    for i, pred in enumerate(predictions[:3]):
                        print(f"  {i+1}. {pred.player} ({pred.team}): {pred.predicted_points:.1f} PTS, {pred.predicted_rebounds:.1f} REB, {pred.predicted_assists:.1f} AST")
                else:
                    print(f"\n📅 No games scheduled for today")
            except Exception as e:
                print(f"\n❌ Prediction failed: {e}")
        
        elif args.full_pipeline:
            results = predictor.run_full_pipeline(args.full_pipeline)
            print(f"\n🎉 Pipeline Results:")
            
            # Highlight important warnings
            if results.get('sample_data_warning'):
                print(f"\n🔶 IMPORTANT: {results['sample_data_warning']}")
            if results.get('data_leakage_warning'):
                print(f"\n⚠️ WARNING: {results['data_leakage_warning']}")
            
            print(json.dumps(results, indent=2, default=str))
        
        else:
            parser.print_help()
    
    except (WNBADataError, WNBAModelError, WNBAPredictionError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        print("💡 Try running with --verbose for more details")
        sys.exit(1)


if __name__ == "__main__":
    main()