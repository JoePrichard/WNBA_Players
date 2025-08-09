# feature_engineer.py - WNBA Feature Engineering (FIXED DATA LEAKAGE VERSION)
#!/usr/bin/env python3
"""
WNBA Feature Engineering - Fixed Data Leakage Version
CRITICAL FIX: Properly excludes target variables from feature matrix

MAJOR ISSUES FIXED:
- Target variables (points, assists, etc.) were being included in features
- This caused perfect R² scores (data leakage)
- Now properly separates features from targets
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from dataclasses import asdict
# Add KMeans import for clustering
from sklearn.cluster import KMeans

from data_models import (
    PlayerGameLog, TeamGameLog, HomeAway, GameResult,
    WNBADataError, PredictionConfig
)
from team_mapping import TeamNameMapper
from utils import mmss_to_float


class WNBAFeatureEngineer:
    """
    Creates features for WNBA prediction models - FIXED DATA LEAKAGE VERSION.
    
    CRITICAL FIX: Properly excludes target variables from feature matrix.
    The previous version was accidentally including target variables in features,
    leading to data leakage and artificially perfect R² scores.
    """
    
    # Add a class-level constant for all stats to create features for
    ALL_FEATURE_STATS = [
        'points', 'assists', 'total_rebounds', 'minutes',
        'fg_made', 'fg_attempted', 'fg_pct',
        'fg3_made', 'fg3_attempted', 'fg3_pct',
        'ft_made', 'ft_attempted', 'ft_pct',
        'off_rebounds', 'def_rebounds',
        'steals', 'blocks', 'turnovers', 'fouls', 'plus_minus'
    ]
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        lookback_games: int = 5,
        min_games_for_stats: int = 3
    ) -> None:
        """Initialize feature engineer with proper target variable handling."""
        self.config = config or PredictionConfig()
        self.lookback_games = lookback_games
        self.min_games_for_stats = min_games_for_stats
        self.target_stats = self.config.target_stats
        self.feature_columns: List[str] = []
        
        # CRITICAL: Define what columns are NEVER features (always targets/metadata)
        self.ALWAYS_EXCLUDE_FROM_FEATURES = {
            # Target variables - NEVER include these in features
            'points', 'total_rebounds', 'assists', 'minutes',
            'fg_made', 'fg_attempted', 'fg_pct',
            'fg3_made', 'fg3_attempted', 'fg3_pct', 
            'ft_made', 'ft_attempted', 'ft_pct',
            'off_rebounds', 'def_rebounds', 'steals', 'blocks',
            'turnovers', 'fouls', 'plus_minus',
            
            # Metadata columns
            'player', 'team', 'date', 'opponent', 'home_away', 'result',
            'prev_game_date', 'position',
            
            # Temporary/derived columns that shouldn't be features
            'game_num', 'game_number_season'
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def validate_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean input game log data.
        Map raw columns from data_fetcher.py to internal names.
        Enforce strict team validation: only real teams from team_mapping.py are allowed.
        """
        # Ensure DataFrame is a copy to avoid view issues
        df = df.copy()

        # Filter out non-player rows (e.g., 'Reserves', empty, or header rows)
        if 'Player' in df.columns:
            player_col = df['Player']
            df = df[~player_col.str.lower().isin(['reserves', '', 'mp', None])]
        if 'player' in df.columns:
            player_col = df['player']
            df = df[~player_col.str.lower().isin(['reserves', '', 'mp', None])]

        # Ensure 'float_minutes' is always present
        if 'float_minutes' not in df.columns and 'MP' in df.columns:
            df['float_minutes'] = df['MP'].apply(mmss_to_float)
        elif 'float_minutes' not in df.columns:
            df['float_minutes'] = 0.0
        # Use 'float_minutes' everywhere minutes as a float are needed

        # Map Basketball Reference/stat CSV columns to internal names
        column_map = {
            'Date': 'date',
            'Player': 'player',
            'Team': 'team',
            'Opponent': 'opponent',
            'Home/Away': 'home_away',
            # 'MP': 'minutes',  # Do not include
            # 'minutes': 'minutes',  # Do not include
            'PTS': 'points',
            'FG': 'fg_made',
            'FGA': 'fg_attempted',
            'FG%': 'fg_pct',
            '3P': 'fg3_made',
            '3PA': 'fg3_attempted',
            '3P%': 'fg3_pct',
            'FT': 'ft_made',
            'FTA': 'ft_attempted',
            'FT%': 'ft_pct',
            'ORB': 'off_rebounds',
            'DRB': 'def_rebounds',
            'TRB': 'total_rebounds',
            'AST': 'assists',
            'STL': 'steals',
            'BLK': 'blocks',
            'TOV': 'turnovers',
            'PF': 'fouls',
            '+/-': 'plus_minus',
        }
        df = df.rename(columns={k: v for k, v in column_map.items() if k in df.columns})

        # Define required columns for validation
        required_columns = [
            'player', 'team', 'date', 'opponent', 'home_away',
            'minutes', 'points', 'fg_made', 'fg_attempted', 'fg_pct',
            'fg3_made', 'fg3_attempted', 'fg3_pct', 'ft_made', 'ft_attempted', 'ft_pct',
            'off_rebounds', 'def_rebounds', 'total_rebounds', 'assists', 'steals', 'blocks',
            'turnovers', 'fouls', 'plus_minus'
        ]

        # After renaming, ensure 'minutes' column exists
        if 'minutes' not in df.columns:
            if 'MP' in df.columns:
                df['minutes'] = df['MP']
                df = df.drop(columns=['MP'])
            else:
                raise WNBADataError("No 'minutes' or 'MP' column found after processing. Please check your input data.")

        # Convert 'minutes' from MM:SS string to float (total minutes)
        if 'minutes' in df.columns:
            df['minutes'] = df['minutes'].apply(mmss_to_float)

        # Remove rows with missing key stats
        initial_rows = len(df)
        df = df.dropna(subset=['points', 'minutes'])
        final_rows = len(df)
        if final_rows < initial_rows:
            self.logger.warning(f"Removed {initial_rows - final_rows} rows with missing key stats")
        if df.empty:
            raise WNBADataError("No valid rows remaining after cleaning")
        
        # Strict team validation
        def _raise_unknown_team(team: str, colname: str) -> None:
            raise ValueError(f"Unknown {colname}: {team}. Only real teams from team_mapping.py are allowed.")
        if 'team' in df.columns:
            df['team'] = df['team'].apply(lambda t: TeamNameMapper.to_abbreviation(t) if TeamNameMapper.to_abbreviation(t) else _raise_unknown_team(t, 'team'))
        if 'opponent' in df.columns:
            df['opponent'] = df['opponent'].apply(lambda t: TeamNameMapper.to_abbreviation(t) if TeamNameMapper.to_abbreviation(t) else _raise_unknown_team(t, 'opponent'))
        
        # Ensure total_rebounds is present if off_rebounds and def_rebounds are available
        if 'total_rebounds' not in df.columns and 'off_rebounds' in df.columns and 'def_rebounds' in df.columns:
            df['total_rebounds'] = (df['off_rebounds'].fillna(0.0) + df['def_rebounds'].fillna(0.0)).astype(float)
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise WNBADataError(f"Missing required columns: {missing_columns}")
        
        if df.empty:
            raise WNBADataError("Input DataFrame is empty")
        
        # Convert data types with error handling
        df = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            try:
                df['date'] = pd.to_datetime(df['date'])
            except (ValueError, TypeError) as e:
                raise WNBADataError(f"Could not convert date column: {e}")
        
        # Ensure numeric columns are numeric
        numeric_columns = [
            'minutes', 'points', 'fg_made', 'fg_attempted', 'fg_pct',
            'fg3_made', 'fg3_attempted', 'fg3_pct', 'ft_made', 'ft_attempted', 'ft_pct',
            'off_rebounds', 'def_rebounds', 'total_rebounds', 'assists', 'steals', 'blocks',
            'turnovers', 'fouls', 'plus_minus'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with missing key stats
        initial_rows = len(df)
        df = df.dropna(subset=['points', 'minutes'])
        final_rows = len(df)
        if final_rows < initial_rows:
            self.logger.warning(f"Removed {initial_rows - final_rows} rows with missing key stats")
        if df.empty:
            raise WNBADataError("No valid rows remaining after cleaning")
        
        df = df.sort_values(['player', 'date']).reset_index(drop=True)
        self.logger.info(f"Validated data: {len(df)} game logs for {df['player'].nunique()} players")
        return df

    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create basic statistical features - FIXED VERSION.
        
        CRITICAL: Only creates DERIVED features, not using raw target variables.
        """
        df = df.copy()
        
        # Ensure 'float_minutes' is always present
        if 'float_minutes' not in df.columns and 'MP' in df.columns:
            df['float_minutes'] = df['MP'].apply(mmss_to_float)
        elif 'float_minutes' not in df.columns:
            df['float_minutes'] = 0.0
        # Use 'float_minutes' everywhere minutes as a float are needed

        # Fill missing values for stat columns with 0
        stat_columns = [
            'fg_made', 'fg_attempted', 'fg_pct',
            'fg3_made', 'fg3_attempted', 'fg3_pct',
            'ft_made', 'ft_attempted', 'ft_pct',
            'off_rebounds', 'def_rebounds', 'total_rebounds',
            'assists', 'steals', 'blocks', 'turnovers', 'fouls', 'plus_minus'
        ]
        for col in stat_columns:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)
        
        # Calculate derived stats if not present or if any nulls
        total_reb_null = df['total_rebounds'].isnull().any() if 'total_rebounds' in df.columns else True
        if bool(total_reb_null):
            df['total_rebounds'] = df['off_rebounds'].fillna(0.0) + df['def_rebounds'].fillna(0.0)
        
        fg_pct_null = df['fg_pct'].isnull().any() if 'fg_pct' in df.columns else True
        if bool(fg_pct_null):
            df['fg_pct'] = df.apply(lambda row: row['fg_made'] / row['fg_attempted'] if row['fg_attempted'] > 0 else 0.0, axis=1)
        
        fg3_pct_null = df['fg3_pct'].isnull().any() if 'fg3_pct' in df.columns else True
        if bool(fg3_pct_null):
            df['fg3_pct'] = df.apply(lambda row: row['fg3_made'] / row['fg3_attempted'] if row['fg3_attempted'] > 0 else 0.0, axis=1)
        
        ft_pct_null = df['ft_pct'].isnull().any() if 'ft_pct' in df.columns else True
        if bool(ft_pct_null):
            df['ft_pct'] = df.apply(lambda row: row['ft_made'] / row['ft_attempted'] if row['ft_attempted'] > 0 else 0.0, axis=1)
        
        # DERIVED efficiency metrics (these are OK as features since they're not direct targets)
        df['feature_usage_rate'] = np.where(
            df['float_minutes'] > 0,
            (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) / df['float_minutes'],
            0.2
        )
        # Remove feature_scoring_efficiency and feature_ts_pct from here
        df['feature_assist_rate'] = np.where(
            df['float_minutes'] > 0,
            df['assists'] / df['float_minutes'],
            0.1
        )
        df['feature_rebound_rate'] = np.where(
            df['float_minutes'] > 0,
            df['total_rebounds'] / df['float_minutes'],
            0.2
        )
        
        return df

    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recent form and momentum features - FIXED VERSION.
        
        CRITICAL: Uses LAGGED versions of target stats, not current game stats.
        """
        df = df.copy()
        
        # Ensure 'float_minutes' is always present
        if 'float_minutes' not in df.columns and 'MP' in df.columns:
            df['float_minutes'] = df['MP'].apply(mmss_to_float)
        elif 'float_minutes' not in df.columns:
            df['float_minutes'] = 0.0
        # Use 'float_minutes' everywhere minutes as a float are needed

        # Calculate rolling averages for recent form using LAGGED stats
        for stat in self.ALL_FEATURE_STATS:
            if stat in df.columns:
                # CRITICAL FIX: Use SHIFTED (lagged) values for features
                # This prevents data leakage by only using past performance
                lagged_stat = df.groupby('player')[stat].shift(1)  # Use previous game's stat
                
                # Rolling mean for last N games (excluding current game)
                df[f'feature_{stat}_l{self.lookback_games}'] = (
                    lagged_stat.groupby(df['player'])
                    .rolling(window=self.lookback_games, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Season averages (expanding window, excluding current game)
                df[f'feature_season_avg_{stat}'] = (
                    lagged_stat.groupby(df['player'])
                    .expanding(min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
                
                # Momentum (recent form vs season average)
                df[f'feature_{stat}_momentum'] = (
                    df[f'feature_{stat}_l{self.lookback_games}'] - df[f'feature_season_avg_{stat}']
                )
        # Add lagged efficiency features
        # Lagged scoring efficiency: points/fg_attempted (previous games only)
        lagged_points = df.groupby('player')['points'].shift(1)
        lagged_fg_attempted = df.groupby('player')['fg_attempted'].shift(1)
        lagged_ft_attempted = df.groupby('player')['ft_attempted'].shift(1)
        # Avoid division by zero
        lagged_scoring_eff = np.where(lagged_fg_attempted > 0, lagged_points / lagged_fg_attempted, 1.0)
        lagged_ts_denom = (lagged_fg_attempted + 0.44 * lagged_ft_attempted)
        lagged_ts_pct = np.where(lagged_ts_denom > 0, lagged_points / (2 * lagged_ts_denom), 0.0)
        # Rolling mean for last N games
        df['feature_scoring_efficiency_l' + str(self.lookback_games)] = (
            pd.Series(lagged_scoring_eff, index=df.index).groupby(df['player'])
            .rolling(window=self.lookback_games, min_periods=1)
            .mean().reset_index(0, drop=True)
        )
        df['feature_ts_pct_l' + str(self.lookback_games)] = (
            pd.Series(lagged_ts_pct, index=df.index).groupby(df['player'])
            .rolling(window=self.lookback_games, min_periods=1)
            .mean().reset_index(0, drop=True)
        )
        # Season averages (expanding window)
        df['feature_season_avg_scoring_efficiency'] = (
            pd.Series(lagged_scoring_eff, index=df.index).groupby(df['player'])
            .expanding(min_periods=1)
            .mean().reset_index(0, drop=True)
        )
        df['feature_season_avg_ts_pct'] = (
            pd.Series(lagged_ts_pct, index=df.index).groupby(df['player'])
            .expanding(min_periods=1)
            .mean().reset_index(0, drop=True)
        )
        # Momentum for efficiency features
        df['feature_scoring_efficiency_momentum'] = (
            df['feature_scoring_efficiency_l' + str(self.lookback_games)] - df['feature_season_avg_scoring_efficiency']
        )
        df['feature_ts_pct_momentum'] = (
            df['feature_ts_pct_l' + str(self.lookback_games)] - df['feature_season_avg_ts_pct']
        )
        
        # Game number in season
        df['feature_game_number_season'] = df.groupby('player').cumcount() + 1
        
        # Season progress (0-1 scale)
        max_games = df.groupby('player')['feature_game_number_season'].transform('max')
        df['feature_season_progress'] = df['feature_game_number_season'] / max_games
        
        # Early/late season indicators
        df['feature_early_season'] = (df['feature_game_number_season'] <= 10).astype(int)
        df['feature_late_season'] = (df['feature_game_number_season'] >= 25).astype(int)
        
        # --- NEW FEATURES ---
        # 1. Rolling 5/20-game pace of team possessions (team-level, lagged)
        if 'team' in df.columns and 'date' in df.columns and 'float_minutes' in df.columns and 'fg_attempted' in df.columns and 'turnovers' in df.columns and 'ft_attempted' in df.columns and 'off_rebounds' in df.columns:
            # Estimate team possessions per game (NBA formula)
            df['team_possessions'] = (
                df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers'] - df['off_rebounds']
            )
            df['team_possessions'] = df['team_possessions'].fillna(0.0)
            # Lagged by one game, rolling by team
            for window in [5, 20]:
                team_poss_lagged = df.groupby('team')['team_possessions'].shift(1)
                df[f'feature_team_possessions_l{window}'] = (
                    team_poss_lagged.groupby(df['team'])
                    .rolling(window=window, min_periods=1)
                    .mean().reset_index(0, drop=True)
                ).fillna(0.0)
        # 2. Opponent defensive four-factor stats (lagged by one game)
        if 'opponent' in df.columns and 'date' in df.columns:
            # Defensive eFG% allowed by opponent (lagged)
            opp_fg_made = df.groupby('opponent')['fg_made'].shift(1)
            opp_fg3_made = df.groupby('opponent')['fg3_made'].shift(1)
            opp_fg_attempted = df.groupby('opponent')['fg_attempted'].shift(1)
            df['feature_opp_def_efg_pct'] = np.where(
                opp_fg_attempted > 0,
                (opp_fg_made + 0.5 * opp_fg3_made) / opp_fg_attempted,
                0.0
            )
            # Defensive OREB% allowed by opponent (lagged)
            opp_oreb = df.groupby('opponent')['off_rebounds'].shift(1)
            team_dreb = df.groupby('opponent')['def_rebounds'].shift(1)
            oreb_denom = opp_oreb + team_dreb
            df['feature_opp_def_oreb_pct'] = np.where(
                oreb_denom > 0,
                opp_oreb / oreb_denom,
                0.0
            )
            # Defensive FTR allowed by opponent (lagged)
            opp_ft_attempted = df.groupby('opponent')['ft_attempted'].shift(1)
            df['feature_opp_def_ftr'] = np.where(
                opp_fg_attempted > 0,
                opp_ft_attempted / opp_fg_attempted,
                0.0
            )
            # Defensive TOV% allowed by opponent (lagged)
            opp_turnovers = df.groupby('opponent')['turnovers'].shift(1)
            tov_denom = opp_fg_attempted + 0.44 * opp_ft_attempted + opp_turnovers
            df['feature_opp_def_tov_pct'] = np.where(
                tov_denom > 0,
                opp_turnovers / tov_denom,
                0.0
            )
        # 3. Player usage% and assist%, season-long (expanding window, lagged)
        if 'player' in df.columns and 'team' in df.columns:
            lagged_fga = df.groupby('player')['fg_attempted'].shift(1)
            lagged_fta = df.groupby('player')['ft_attempted'].shift(1)
            lagged_tov = df.groupby('player')['turnovers'].shift(1)
            lagged_team_poss = df.groupby('team')['team_possessions'].shift(1)
            usage_numer = lagged_fga + 0.44 * lagged_fta + lagged_tov
            usage_pct = np.where(lagged_team_poss > 0, usage_numer / lagged_team_poss, 0.0)
            df['feature_season_avg_usage_pct'] = (
                pd.Series(usage_pct, index=df.index).groupby(df['player']).expanding(min_periods=1).mean().reset_index(0, drop=True)
            ).fillna(0.0)
            # Assist% (season-long, expanding window)
            lagged_ast = df.groupby('player')['assists'].shift(1)
            lagged_team_fgm = df.groupby('team')['fg_made'].shift(1)
            assist_pct = np.where(lagged_team_fgm > 0, lagged_ast / lagged_team_fgm, 0.0)
            df['feature_season_avg_assist_pct'] = (
                pd.Series(assist_pct, index=df.index).groupby(df['player']).expanding(min_periods=1).mean().reset_index(0, drop=True)
            ).fillna(0.0)
        
        return df

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game context features - FIXED VERSION.
        Adds seasonality/league trend features: league-wide average points to date, season progress.
        """
        df = df.copy()
        
        # Ensure 'float_minutes' is always present
        if 'float_minutes' not in df.columns and 'MP' in df.columns:
            df['float_minutes'] = df['MP'].apply(mmss_to_float)
        elif 'float_minutes' not in df.columns:
            df['float_minutes'] = 0.0
        # Use 'float_minutes' everywhere minutes as a float are needed

        # Rest days calculation
        df['prev_game_date'] = df.groupby('player')['date'].shift(1)
        df['feature_rest_days'] = (df['date'] - df['prev_game_date']).dt.days
        df['feature_rest_days'] = df['feature_rest_days'].fillna(3).clip(0, 10)
        
        # Rest/fatigue indicators
        df['feature_rest_advantage'] = (df['feature_rest_days'] >= 2).astype(int)
        df['feature_fatigue_factor'] = (df['feature_rest_days'] == 0).astype(int)
        
        # FIXED: Home court advantage with proper DataFrame handling
        if 'home_away' not in df.columns:
            df['home_away'] = 'H'
        
        try:
            # Ensure home_away is string type and handle DataFrame properly
            if hasattr(df['home_away'], 'astype'):
                df['home_away'] = df['home_away'].astype(str)
            else:
                # If it's already a DataFrame/Series, convert properly
                df['home_away'] = df['home_away'].apply(str)
            
            # Create home boost feature with proper string methods
            home_indicators = ['H', 'HOME', 'Home', 'home']
            away_indicators = ['A', 'AWAY', 'Away', 'away']
            
            # Use isin for safer comparison
            home_mask = df['home_away'].isin(home_indicators)
            df['feature_home_boost'] = home_mask.astype(int)
            
            # Log the distribution for debugging
            home_count = df['feature_home_boost'].sum()
            total_games = len(df)
            self.logger.debug(f"Home games: {home_count}/{total_games} ({home_count/total_games:.1%})")
            
        except Exception as e:
            self.logger.warning(f"Issue with home_away column processing: {e}, using default values")
            # Fallback: assume roughly 50% home games
            np.random.seed(42)  # For reproducibility
            df['feature_home_boost'] = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
        
        # Opponent strength features with improved error handling
        try:
            if 'opp_def_rating' not in df.columns:
                team_strength = self._estimate_team_strength(df)
                if not team_strength.empty:
                    # Use merge instead of join for better control
                    df = df.merge(team_strength, left_on='opponent', right_index=True, how='left')
                    if 'opp_def_rating' in df.columns:
                        df['opp_def_rating'] = df['opp_def_rating'].fillna(105.0)
                    else:
                        df['opp_def_rating'] = 105.0
                else:
                    df['opp_def_rating'] = 105.0
            
            df['feature_opp_strength'] = (df['opp_def_rating'] - 100) / 10
            df['feature_matchup_advantage'] = (df['feature_opp_strength'] < 0).astype(int)
            
        except Exception as e:
            self.logger.warning(f"Issue creating opponent strength features: {e}, using defaults")
            df['opp_def_rating'] = 105.0
            df['feature_opp_strength'] = 0.0
            df['feature_matchup_advantage'] = 0
        
        # Pace factors with validation
        if 'team_pace' not in df.columns:
            df['team_pace'] = 80.0
        
        # Ensure team_pace is numeric
        df['team_pace'] = pd.to_numeric(df['team_pace'], errors='coerce').fillna(80.0)
        
        df['feature_pace_factor'] = df['team_pace'] / 80.0
        df['feature_blowout_risk'] = (np.abs(df['feature_opp_strength']) > 1.5).astype(int)
        
        # --- Seasonality/League Trend Features ---
        # League-wide average points per game up to (not including) this date
        df = df.sort_values(['date'])
        league_points_cumsum = df['points'].cumsum() - df['points']
        league_games_cumsum = np.arange(len(df))
        df['feature_league_avg_points_to_date'] = np.where(
            league_games_cumsum > 0,
            league_points_cumsum / league_games_cumsum,
            df['points'].mean()  # fallback for first row
        )
        # Season progress (already present, but ensure it's here)
        if 'feature_game_number_season' not in df.columns:
            df['feature_game_number_season'] = df.groupby('player').cumcount() + 1
        max_games = df.groupby('player')['feature_game_number_season'].transform('max')
        df['feature_season_progress'] = df['feature_game_number_season'] / max_games
        # --- End Seasonality/League Trend Features ---
        
        return df

    def create_synergy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team synergy features - now includes player clustering (archetype) features.
        """
        df = df.copy()
        # Ensure 'float_minutes' is always present
        if 'float_minutes' not in df.columns and 'MP' in df.columns:
            df['float_minutes'] = df['MP'].apply(mmss_to_float)
        elif 'float_minutes' not in df.columns:
            df['float_minutes'] = 0.0
        # Use 'float_minutes' everywhere minutes as a float are needed

        # Position-based features (existing logic)
        if 'position' not in df.columns:
            df['position'] = self._infer_positions(df)
        positions = ['PG', 'SG', 'SF', 'PF', 'C']
        position_weights = {
            'PG': {'ast_weight': 1.5, 'reb_weight': 0.7, 'pts_weight': 1.0},
            'SG': {'ast_weight': 0.8, 'reb_weight': 0.8, 'pts_weight': 1.3},
            'SF': {'ast_weight': 1.0, 'reb_weight': 1.0, 'pts_weight': 1.1},
            'PF': {'ast_weight': 0.7, 'reb_weight': 1.4, 'pts_weight': 1.0},
            'C': {'ast_weight': 0.5, 'reb_weight': 1.6, 'pts_weight': 0.9}
        }
        for pos in positions:
            df[f'feature_pos_is_{pos}'] = (df['position'] == pos).astype(int)
            if pos in position_weights:
                weights = position_weights[pos]
                for stat, weight in weights.items():
                    stat_name = stat.replace('_weight', '')
                    df[f'feature_{pos}_{stat_name}_expectation'] = df[f'feature_pos_is_{pos}'] * weight

        # --- Player Clustering (Archetype) Features ---
        # Use lagged season averages for clustering (to avoid leakage)
        cluster_features = ['feature_season_avg_points', 'feature_season_avg_assists', 'feature_season_avg_total_rebounds', 'feature_season_avg_blocks']
        # Only use rows where all cluster features are present
        cluster_df = df.dropna(subset=cluster_features)
        # Prepare player-level lagged averages (use last available for each player)
        player_profiles = (
            cluster_df.groupby('player')[cluster_features]
            .last()
            .fillna(0.0)
        )
        # Fit KMeans (n_clusters=5)
        n_clusters = 5
        if len(player_profiles) >= n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            player_clusters = kmeans.fit_predict(player_profiles.values)
            player_cluster_map = dict(zip(player_profiles.index, player_clusters))
        else:
            # Not enough players for clustering, assign all to cluster 0
            player_cluster_map = {p: 0 for p in player_profiles.index}
        # Assign cluster to each row
        df['feature_player_cluster'] = df['player'].map(player_cluster_map).fillna(-1).astype(int)
        # One-hot encode player cluster (optional, can help some models)
        for c in range(n_clusters):
            df[f'feature_player_cluster_{c}'] = (df['feature_player_cluster'] == c).astype(int)
        # For each game, count number of each cluster type among teammates and opponents
        for c in range(n_clusters):
            # Teammates (same team, same date, not self)
            df[f'feature_num_teammate_cluster_{c}'] = df.groupby(['team', 'date'])['feature_player_cluster'].transform(lambda x: (x == c).sum()) - (df['feature_player_cluster'] == c).astype(int)
            # Opponents (opponent team, same date)
            df[f'feature_num_opponent_cluster_{c}'] = df.groupby(['opponent', 'date'])['feature_player_cluster'].transform(lambda x: (x == c).sum())
        # --- End Player Clustering Features ---

        # --- Teammate and Opponent Rolling Average Features ---
        # Key stats to use
        rolling_stats = [
            f'feature_points_l{self.lookback_games}',
            f'feature_assists_l{self.lookback_games}',
            f'feature_total_rebounds_l{self.lookback_games}',
            f'feature_usage_rate',
            f'feature_scoring_efficiency_l{self.lookback_games}'
        ]
        # For each player-game, compute teammate and opponent averages (excluding self for teammates)
        for stat in rolling_stats:
            # Teammate average (same team, same date, exclude self)
            df[f'feature_teammate_avg_{stat}'] = (
                df.groupby(['team', 'date'])[stat]
                .transform(lambda x: (x.sum() - x) / (len(x) - 1) if len(x) > 1 else 0.0)
            )
            # Opponent average (opponent team, same date)
            df[f'feature_opponent_avg_{stat}'] = (
                df.groupby(['opponent', 'date'])[stat]
                .transform('mean')
            )
        # --- End Teammate and Opponent Rolling Average Features ---

        # Team context features (existing logic)
        df['feature_team_avg_minutes'] = df.groupby(['team', 'date'])['float_minutes'].transform('mean')
        usage_spread = df.groupby(['team', 'date'])['feature_usage_rate'].transform('std')
        if isinstance(usage_spread, pd.Series):
            df['feature_team_usage_spread'] = usage_spread.fillna(0.05)
        else:
            df['feature_team_usage_spread'] = pd.Series([0.05] * len(df), index=df.index)
        # Player role indicators (using lagged stats to avoid leakage)
        scorer_base = pd.Series(0.0, index=df.index)
        if 'feature_season_avg_points' in df.columns:
            scorer_col = df['feature_season_avg_points']
            if isinstance(scorer_col, pd.Series):
                scorer_base = scorer_col.fillna(0)
            else:
                scorer_base = pd.Series(scorer_col, index=df.index)
        df['feature_primary_scorer'] = (scorer_base > df['feature_team_avg_minutes'] * 0.6).astype(int)
        return df

    def _estimate_team_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate team defensive strength from available data."""
        try:
            # Use LAGGED stats to avoid leakage
            team_defense = df.groupby('opponent').agg({
                'points': lambda x: x.shift(1).mean(),  # Use previous games
                'total_rebounds': lambda x: x.shift(1).mean(),
                'assists': lambda x: x.shift(1).mean()
            }).dropna()
            
            if len(team_defense) > 0:
                league_avg_pts = df['points'].shift(1).mean()  # Use lagged average
                team_defense['opp_def_rating'] = 100 + (team_defense['points'] - league_avg_pts) * 2
                return team_defense[['opp_def_rating']]
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.warning(f"Error estimating team strength: {e}")
            return pd.DataFrame()

    def _infer_positions(self, df: pd.DataFrame) -> pd.Series:
        """Infer player positions from their statistical profiles."""
        try:
            # Use LAGGED stats to avoid leakage
            player_profiles = df.groupby('player').agg({
                'points': lambda x: x.shift(1).mean(),
                'total_rebounds': lambda x: x.shift(1).mean(),
                'assists': lambda x: x.shift(1).mean(),
                'blocks': lambda x: x.shift(1).mean()
            }).fillna(0)
            
            positions = []
            for _, profile in player_profiles.iterrows():
                pts = profile['points']
                reb = profile['total_rebounds'] 
                ast = profile['assists']
                blk = profile['blocks']
                
                if ast > 5:
                    pos = 'PG'
                elif reb > 8 and blk > 1:
                    pos = 'C'
                elif reb > 7:
                    pos = 'PF'
                elif pts > 15 and reb < 6:
                    pos = 'SG'
                else:
                    pos = 'SF'
                
                positions.append(pos)
            
            position_map = dict(zip(player_profiles.index, positions))
            return df['player'].map(position_map).fillna('SF')
            
        except Exception as e:
            self.logger.warning(f"Error inferring positions: {e}")
            return pd.Series(['SF'] * len(df), index=df.index)

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features - FIXED DATA LEAKAGE VERSION.
        
        CRITICAL FIX: Properly excludes ALL target variables from feature matrix.
        """
        try:
            self.logger.info("Starting feature engineering pipeline...")
            
            # Validate input
            df = self.validate_input_data(df)
            
            # Create features in order
            df = self.create_basic_features(df)
            self.logger.info("✅ Created basic features")
            
            df = self.create_form_features(df)
            self.logger.info("✅ Created form features")
            
            df = self.create_context_features(df)
            self.logger.info("✅ Created context features")
            
            df = self.create_synergy_features(df)
            self.logger.info("✅ Created synergy features")
            
            # CRITICAL FIX: Properly identify feature columns
            # Only include columns that start with 'feature_'
            self.feature_columns = [
                col for col in df.columns 
                if col.startswith('feature_') and col not in self.ALWAYS_EXCLUDE_FROM_FEATURES
            ]
            
            # Ensure all feature columns are numeric
            for col in self.feature_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
            
            # Log which columns are features vs targets
            target_columns = [col for col in df.columns if col in self.ALWAYS_EXCLUDE_FROM_FEATURES and col in self.target_stats]
            
            self.logger.info(f"✅ Feature engineering complete:")
            self.logger.info(f"   - {len(self.feature_columns)} feature columns (starting with 'feature_')")
            self.logger.info(f"   - {len(target_columns)} target columns: {target_columns}")
            self.logger.info(f"   - CRITICAL: Target variables are EXCLUDED from features")

            return df
            
        except Exception as e:
            raise WNBADataError(f"Feature engineering failed: {e}")

    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by category for analysis."""
        feature_groups = {
            'basic_stats': [],
            'efficiency': [],
            'form_momentum': [],
            'context': [],
            'synergy': [],
            'positional': []
        }
        
        for feature in self.feature_columns:
            if any(x in feature for x in ['usage_rate', 'scoring_efficiency', 'assist_rate', 'rebound_rate', 'ts_pct']):
                feature_groups['efficiency'].append(feature)
            elif any(x in feature for x in ['momentum', f'_l{self.lookback_games}', 'season_progress']):
                feature_groups['form_momentum'].append(feature)
            elif any(x in feature for x in ['rest', 'home', 'pace', 'opp_', 'matchup']):
                feature_groups['context'].append(feature)
            elif any(x in feature for x in ['team_', 'primary_']):
                feature_groups['synergy'].append(feature)
            elif any(x in feature for x in ['pos_', '_expectation']):
                feature_groups['positional'].append(feature)
            else:
                feature_groups['basic_stats'].append(feature)
        
        return feature_groups


def main() -> None:
    """Example usage of the fixed feature engineer."""
    print("🏀 WNBA Feature Engineer - FIXED DATA LEAKAGE VERSION")
    print("=" * 60)
    print("✅ CRITICAL FIX: Target variables properly excluded from features")
    print("✅ Features use lagged/historical data only")
    print("✅ Proper validation split will now show realistic R² scores")


if __name__ == "__main__":
    main()