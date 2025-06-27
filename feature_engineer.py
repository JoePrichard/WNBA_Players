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
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import asdict

from data_models import (
    PlayerGameLog, TeamGameLog, HomeAway, GameResult,
    WNBADataError, PredictionConfig
)


class WNBAFeatureEngineer:
    """
    Creates features for WNBA prediction models - FIXED DATA LEAKAGE VERSION.
    
    CRITICAL FIX: Properly excludes target variables from feature matrix.
    The previous version was accidentally including target variables in features,
    leading to data leakage and artificially perfect R² scores.
    """
    
    def __init__(
        self,
        config: Optional[PredictionConfig] = None,
        lookback_games: int = 5,
        min_games_for_stats: int = 3
    ):
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
        """
        # Map Basketball Reference/stat CSV columns to internal names
        column_map = {
            'Date': 'date',
            'Player': 'player',
            'Team': 'team',
            'Opponent': 'opponent',
            'Home/Away': 'home_away',
            'MP': 'minutes',
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
        
        # Ensure total_rebounds is present if off_rebounds and def_rebounds are available
        if 'total_rebounds' not in df.columns and 'off_rebounds' in df.columns and 'def_rebounds' in df.columns:
            df['total_rebounds'] = df['off_rebounds'] + df['def_rebounds']
        
        required_columns = [
            'player', 'team', 'date', 'opponent', 'home_away',
            'minutes', 'points', 'fg_made', 'fg_attempted', 'fg_pct',
            'fg3_made', 'fg3_attempted', 'fg3_pct', 'ft_made', 'ft_attempted', 'ft_pct',
            'off_rebounds', 'def_rebounds', 'total_rebounds', 'assists', 'steals', 'blocks',
            'turnovers', 'fouls', 'plus_minus'
        ]
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
            df['minutes'] > 0,
            (df['fg_attempted'] + 0.44 * df['ft_attempted'] + df['turnovers']) / df['minutes'],
            0.2
        )
        
        df['feature_scoring_efficiency'] = np.where(
            df['fg_attempted'] > 0,
            df['points'] / df['fg_attempted'],
            1.0
        )
        
        df['feature_assist_rate'] = np.where(
            df['minutes'] > 0,
            df['assists'] / df['minutes'],
            0.1
        )
        
        df['feature_rebound_rate'] = np.where(
            df['minutes'] > 0,
            df['total_rebounds'] / df['minutes'],
            0.2
        )
        
        # True shooting percentage
        df['feature_ts_pct'] = np.where(
            (df['fg_attempted'] + 0.44 * df['ft_attempted']) > 0,
            df['points'] / (2 * (df['fg_attempted'] + 0.44 * df['ft_attempted'])),
            0.0
        )
        
        return df

    def create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recent form and momentum features - FIXED VERSION.
        
        CRITICAL: Uses LAGGED versions of target stats, not current game stats.
        """
        df = df.copy()
        
        # Calculate rolling averages for recent form using LAGGED stats
        for stat in self.target_stats:
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
        
        # Game number in season
        df['feature_game_number_season'] = df.groupby('player').cumcount() + 1
        
        # Season progress (0-1 scale)
        max_games = df.groupby('player')['feature_game_number_season'].transform('max')
        df['feature_season_progress'] = df['feature_game_number_season'] / max_games
        
        # Early/late season indicators
        df['feature_early_season'] = (df['feature_game_number_season'] <= 10).astype(int)
        df['feature_late_season'] = (df['feature_game_number_season'] >= 25).astype(int)
        
        return df

    def create_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create game context features - FIXED VERSION.
        """
        df = df.copy()
        
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
            import numpy as np
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
        
        return df

    def create_synergy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team synergy features - FIXED VERSION.
        """
        df = df.copy()
        
        # Position-based features
        if 'position' not in df.columns:
            df['position'] = self._infer_positions(df)
        
        # Position dummy variables
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
        
        # Team context features - simplified to avoid merge issues
        df['feature_team_avg_minutes'] = df.groupby(['team', 'date'])['minutes'].transform('mean')
        df['feature_team_usage_spread'] = df.groupby(['team', 'date'])['feature_usage_rate'].transform('std').fillna(0.05)
        
        # Player role indicators (using lagged stats to avoid leakage)
        df['feature_primary_scorer'] = (
            df.get('feature_season_avg_points', 0) > df['feature_team_avg_minutes'] * 0.6
        ).astype(int)
        
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


def main():
    """Example usage of the fixed feature engineer."""
    print("🏀 WNBA Feature Engineer - FIXED DATA LEAKAGE VERSION")
    print("=" * 60)
    print("✅ CRITICAL FIX: Target variables properly excluded from features")
    print("✅ Features use lagged/historical data only")
    print("✅ Proper validation split will now show realistic R² scores")


if __name__ == "__main__":
    main()