# dashboard.py - WNBA Daily Game Prediction Dashboard
#!/usr/bin/env python3
"""
WNBA Daily Game Prediction Dashboard (Fixed Version)
Streamlit application for interactive predictions and model insights.

This fixed version includes:
- Corrected import statements
- Better error handling for missing modules
- Graceful fallbacks when components aren't available
- Improved user experience with clear error messages
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Add project directory to path
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Page configuration
st.set_page_config(
    page_title="WNBA Daily Game Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Try to import optional visualization libraries
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Plotly not available - some visualizations will be disabled")
    PLOTLY_AVAILABLE = False

# Custom imports with comprehensive error handling
try:
    from data_models import (
        PredictionConfig, PlayerPrediction, HomeAway,
        WNBADataError, WNBAModelError, WNBAPredictionError
    )
    DATA_MODELS_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Cannot import data_models: {e}")
    DATA_MODELS_AVAILABLE = False

try:
    from main_application import WNBADailyPredictor
    MAIN_APP_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ Cannot import main_application: {e}")
    MAIN_APP_AVAILABLE = False

try:
    from utils import setup_project_structure, get_project_directories
    setup_project_structure()
    UTILS_AVAILABLE = True
except ImportError:
    st.warning("⚠️ Utils module not available - using basic functionality")
    UTILS_AVAILABLE = False

# Check if we have minimum required components
if not (DATA_MODELS_AVAILABLE and MAIN_APP_AVAILABLE):
    st.error("""
    ## ❌ Missing Required Components
    
    The dashboard cannot start because required modules are missing:
    
    **Required Files:**
    - `data_models.py` - Data structures and configuration
    - `main_application.py` - Main prediction system
    
    **Please ensure:**
    1. All Python files are in the same directory
    2. Dependencies are installed: `pip install -r requirements.txt`
    3. Run from the project directory
    
    **Quick Fix:**
    ```bash
    # From your project directory:
    pip install streamlit pandas numpy
    streamlit run dashboard.py
    ```
    """)
    st.stop()

# --- Custom CSS: Modern, Theme-Aware, High-Contrast ---
st.markdown("""
<style>
    html, body, [class^="css"] {
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        background: var(--background-color, #f4f6fa);
        color: var(--text-color, #22223b);
    }
    .main-header {
        font-size: 2.7rem;
        font-weight: 800;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 1.2rem;
        letter-spacing: 1px;
        text-shadow: 0 2px 8px #fff3, 0 1px 0 #fff;
    }
    .metric-card {
        background: linear-gradient(90deg, #f0f2f6 80%, #ffe5d0 100%);
        padding: 1.1rem 1rem;
        border-radius: 0.7rem;
        border-left: 5px solid #FF6B35;
        margin: 0.7rem 0;
        box-shadow: 0 2px 8px #0001;
        transition: box-shadow 0.2s;
    }
    .metric-card:hover {
        box-shadow: 0 4px 16px #ff6b3533;
    }
    .prediction-card {
        background: linear-gradient(90deg, #f8f9fa 80%, #e0e7ff 100%);
        padding: 1rem 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.4rem 0;
        box-shadow: 0 1px 4px #0001;
        transition: box-shadow 0.2s;
    }
    .prediction-card:hover {
        box-shadow: 0 4px 16px #667eea33;
    }
    .error-box {
        background: #fff5f5;
        border: 1.5px solid #fed7d7;
        border-radius: 0.7rem;
        padding: 1.1rem;
        margin: 1.2rem 0;
        color: #c53030;
        font-weight: 500;
    }
    .info-box {
        background: #ebf8ff;
        border: 1.5px solid #bee3f8;
        border-radius: 0.7rem;
        padding: 1.1rem;
        margin: 1.2rem 0;
        color: #2a69ac;
        font-weight: 500;
    }
    .sidebar-section {
        background: #fff8f0;
        border-radius: 0.7rem;
        padding: 1rem 0.7rem 0.7rem 0.7rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 1px 4px #ff6b3511;
    }
    .st-emotion-cache-1v0mbdj p {
        color: var(--text-color, #22223b) !important;
    }
    .st-emotion-cache-1v0mbdj h4 {
        color: #FF6B35 !important;
    }
    .st-emotion-cache-1v0mbdj small {
        color: #6b7280 !important;
    }
    /* Theme-aware tweaks */
    @media (prefers-color-scheme: dark) {
        html, body, [class^="css"] {
            background: #181a1b;
            color: #f4f6fa;
        }
        .main-header {
            color: #ffb385;
            text-shadow: 0 2px 8px #0008, 0 1px 0 #222;
        }
        .metric-card {
            background: linear-gradient(90deg, #23272e 80%, #3a2c1a 100%);
            border-left: 5px solid #FF6B35;
            color: #f4f6fa;
        }
        .prediction-card {
            background: linear-gradient(90deg, #23272e 80%, #2d365f 100%);
            border-left: 4px solid #667eea;
            color: #f4f6fa;
        }
        .error-box {
            background: #2d1a1a;
            border: 1.5px solid #c53030;
            color: #ffb4b4;
        }
        .info-box {
            background: #1a2633;
            border: 1.5px solid #2a69ac;
            color: #bee3f8;
        }
        .sidebar-section {
            background: #23272e;
            box-shadow: 0 1px 4px #2228;
        }
    }
</style>
""", unsafe_allow_html=True)


class WNBADashboard:
    """
    Simplified dashboard class for WNBA predictions with error handling.
    
    This version gracefully handles missing components and provides
    clear feedback to users about what's available.
    """
    
    def __init__(self):
        """Initialize the dashboard with error handling."""
        self.predictor: Optional[Any] = None
        self.config: Optional[Any] = None
        
        # Initialize session state
        self._init_session_state()
        
        # Try to load components
        self._load_components()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'predictions_data' not in st.session_state:
            st.session_state.predictions_data = None
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
        
        if 'system_status' not in st.session_state:
            st.session_state.system_status = None
    
    def _load_components(self) -> bool:
        """
        Load dashboard components with error handling.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if DATA_MODELS_AVAILABLE:
                self.config = PredictionConfig()
            
            if MAIN_APP_AVAILABLE and self.config:
                self.predictor = WNBADailyPredictor(config=self.config)
                return True
            elif MAIN_APP_AVAILABLE:
                self.predictor = WNBADailyPredictor()
                return True
        except Exception as e:
            st.error(f"Failed to load components: {e}")
        
        return False
    
    def load_predictions_data(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load predictions data from file.
        
        Args:
            file_path: Specific file to load, or None for most recent
            
        Returns:
            DataFrame with predictions or None if failed
        """
        try:
            if file_path and os.path.exists(file_path):
                df = pd.read_csv(file_path)
                st.session_state.predictions_data = df
                st.session_state.last_update = datetime.now()
                return df
            
            # Look for recent prediction files
            pred_dir = Path("wnba_predictions")
            if pred_dir.exists():
                pred_files = list(pred_dir.glob("predictions_*.csv"))
                
                if pred_files:
                    # Get most recent file
                    latest_file = max(pred_files, key=lambda x: x.stat().st_mtime)
                    df = pd.read_csv(latest_file)
                    st.session_state.predictions_data = df
                    st.session_state.last_update = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    return df
            
            return None
            
        except Exception as e:
            st.error(f"Failed to load predictions: {e}")
            return None
    
    def check_system_status(self) -> Dict[str, Any]:
        """
        Check the status of the prediction system.
        
        Returns:
            Dictionary with system status information
        """
        status = {
            'data_available': False,
            'models_trained': False,
            'predictions_available': False,
            'last_prediction': None,
            'data_files': 0,
            'model_files': 0,
            'directories_exist': True
        }
        
        try:
            # Check for data files
            data_dir = Path("wnba_game_data")
            if data_dir.exists():
                data_files = list(data_dir.glob("*.csv"))
                status['data_files'] = len(data_files)
                status['data_available'] = len(data_files) > 0
            else:
                status['directories_exist'] = False
            
            # Check for model files
            model_dir = Path("wnba_models")
            if model_dir.exists():
                model_dirs = list(model_dir.glob("models_*"))
                status['model_files'] = len(model_dirs)
                status['models_trained'] = len(model_dirs) > 0
            
            # Check for predictions
            pred_dir = Path("wnba_predictions")
            if pred_dir.exists():
                pred_files = list(pred_dir.glob("predictions_*.csv"))
                status['predictions_available'] = len(pred_files) > 0
                
                if pred_files:
                    latest_pred = max(pred_files, key=lambda x: x.stat().st_mtime)
                    status['last_prediction'] = datetime.fromtimestamp(latest_pred.stat().st_mtime)
        
        except Exception as e:
            st.warning(f"Status check failed: {e}")
        
        return status


def render_header():
    """Render the main dashboard header."""
    st.markdown('<h1 class="main-header">🏀 WNBA Daily Game Predictions</h1>', unsafe_allow_html=True)
    st.markdown("### *Machine Learning Predictions for Points, Rebounds & Assists*")
    
    # System status indicators
    dashboard = st.session_state.get('dashboard')
    if dashboard:
        status = dashboard.check_system_status()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if status['data_available']:
                st.success(f"📊 Data ({status['data_files']} files)")
            else:
                st.error("📊 No Data")
        
        with col2:
            if status['models_trained']:
                st.success(f"🤖 Models ({status['model_files']} versions)")
            else:
                st.error("🤖 No Models")
        
        with col3:
            if status['predictions_available']:
                st.success("🔮 Predictions Ready")
            else:
                st.warning("🔮 No Predictions")
        
        with col4:
            if status['last_prediction']:
                time_ago = datetime.now() - status['last_prediction']
                hours_ago = int(time_ago.total_seconds() / 3600)
                st.info(f"🕐 Updated {hours_ago}h ago")
            else:
                st.info("🕐 No Updates")


def render_sidebar():
    """Render the sidebar navigation and controls."""
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.title("🎛️ Navigation")
        page = st.selectbox(
            "Choose Page:",
            [
                "🏀 Today's Games",
                "👤 Player Analysis", 
                "📊 System Status",
                "⚙️ Management"
            ],
            help="Navigate between dashboard sections"
        )
        st.markdown("---")
        st.subheader("Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True, help="Reload predictions and update dashboard"):
            st.session_state.predictions_data = None
            st.rerun()
        st.markdown("---")
        st.subheader("Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Minimum confidence for displaying predictions"
        )
        show_uncertainties = st.checkbox(
            "Show Uncertainties",
            value=True,
            help="Display confidence intervals"
        )
        st.markdown("---")
        st.subheader("Model Info")
        st.info(
            "**Models Used:**\n"
            "• XGBoost (30%)\n"
            "• LightGBM (25%)\n"
            "• Random Forest (25%)\n"
            "• Neural Network (20%)\n\n"
            "**Target:** Accurate predictions with uncertainty quantification"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    return page, confidence_threshold, show_uncertainties


def render_todays_games(dashboard: WNBADashboard, confidence_threshold: float, show_uncertainties: bool):
    """Render today's games page."""
    st.header("🏀 Today's Game Predictions")
    st.write(f"📅 **{date.today().strftime('%A, %B %d, %Y')}**")
    st.markdown(
        "<div style='margin-bottom:1.2rem; color:#666;'>Predictions for all scheduled WNBA games today. Adjust confidence threshold and uncertainty display in the sidebar.</div>",
        unsafe_allow_html=True
    )
    
    # Load predictions
    predictions_df = dashboard.load_predictions_data()
    
    if predictions_df is None or predictions_df.empty:
        st.markdown("""
        <div class="error-box">
            <h4>❌ No Predictions Available</h4>
            <p>No prediction data found. This could mean:</p>
            <ul>
                <li>No games are scheduled for today</li>
                <li>Models haven't been trained yet</li>
                <li>Data hasn't been fetched recently</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Show instructions
        st.markdown("""
        <div class="info-box">
            <h4>💡 Getting Started</h4>
            <p><strong>To generate predictions:</strong></p>
            <ol>
                <li>Go to "Management" page</li>
                <li>Fetch current season data</li>
                <li>Train prediction models</li>
                <li>Generate daily predictions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Filter by confidence if needed
    if confidence_threshold > 0:
        filtered_df = predictions_df[predictions_df['confidence_score'] >= confidence_threshold]
        if len(filtered_df) < len(predictions_df):
            st.warning(f"Filtered {len(predictions_df) - len(filtered_df)} predictions below {confidence_threshold:.1%} confidence")
        predictions_df = filtered_df
    
    if predictions_df.empty:
        st.warning(f"No predictions meet confidence threshold of {confidence_threshold:.1%}")
        return
    
    # Show summary
    st.subheader("📊 Summary")
    with st.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Predictions", int(len(predictions_df)))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            avg_confidence = float(predictions_df['confidence_score'].mean())
            st.metric("Average Confidence", f"{avg_confidence:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            unique_games = int(pd.Series(predictions_df['game_id']).nunique())
            st.metric("Games", unique_games)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Group by game and display
    games = predictions_df.groupby('game_id')
    
    for game_id, game_predictions in games:
        st.subheader(f"�� Game: {game_id}")
        # Try to identify teams
        home_players = game_predictions[game_predictions['home_away'] == 'H']
        away_players = game_predictions[game_predictions['home_away'] == 'A']
        # Ensure DataFrame type for home_players and away_players
        if isinstance(home_players, pd.Series):
            home_players = home_players.to_frame().T
        if isinstance(away_players, pd.Series):
            away_players = away_players.to_frame().T
        col1, col2 = st.columns(2)
        with col1:
            if isinstance(home_players, pd.DataFrame) and not home_players.empty:
                home_team = home_players['team'].iloc[0]
                from team_mapping import TeamNameMapper
                if not TeamNameMapper.is_valid_abbreviation(home_team):
                    st.error(f"Invalid home team: {home_team}. Only real teams from team_mapping.py are allowed.")
                else:
                    st.markdown(f"**🏠 {home_team} (Home)**")
                    render_team_predictions(home_players, show_uncertainties)
        with col2:
            if isinstance(away_players, pd.DataFrame) and not away_players.empty:
                away_team = away_players['team'].iloc[0]
                from team_mapping import TeamNameMapper
                if not TeamNameMapper.is_valid_abbreviation(away_team):
                    st.error(f"Invalid away team: {away_team}. Only real teams from team_mapping.py are allowed.")
                else:
                    st.markdown(f"**✈️ {away_team} (Away)**")
                    render_team_predictions(away_players, show_uncertainties)
        st.markdown("---")


def render_team_predictions(team_predictions: pd.DataFrame, show_uncertainties: bool):
    """Render predictions for a team."""
    if team_predictions.empty:
        st.write("No player predictions available")
        return
    for _, player in team_predictions.iterrows():
        confidence_class = "high" if player['confidence_score'] >= 0.8 else "medium" if player['confidence_score'] >= 0.6 else "low"
        color = '#4ade80' if confidence_class == 'high' else '#fbbf24' if confidence_class == 'medium' else '#f87171'
        st.markdown(f"""
        <div class="prediction-card">
            <strong style='font-size:1.1em'>{player['player']}</strong> 
            <span style="background: {color}; color: white; padding: 2px 8px; border-radius: 3px; font-size: 0.9em; margin-left: 0.5em; font-weight:600;">
                {player['confidence_score']:.1%}
            </span>
            <br>
            <div style="margin-top: 0.5rem;">
                <span style="background: #3b82f6; color: white; padding: 2px 8px; border-radius: 3px; margin: 2px; font-size: 1em; font-weight:500;">
                    {player['predicted_points']:.1f} PTS
                </span>
                <span style="background: #10b981; color: white; padding: 2px 8px; border-radius: 3px; margin: 2px; font-size: 1em; font-weight:500;">
                    {player['predicted_rebounds']:.1f} REB
                </span>
                <span style="background: #f59e0b; color: white; padding: 2px 8px; border-radius: 3px; margin: 2px; font-size: 1em; font-weight:500;">
                    {player['predicted_assists']:.1f} AST
                </span>
            </div>
        """, unsafe_allow_html=True)
        if show_uncertainties:
            st.markdown(f"""
            <small style="color: #6b7280; font-size:0.95em;">
                Uncertainty: ±{player['points_uncertainty']:.1f} PTS, 
                ±{player['rebounds_uncertainty']:.1f} REB, 
                ±{player['assists_uncertainty']:.1f} AST
            </small>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_player_analysis(dashboard: WNBADashboard):
    """Render player analysis page with improved clarity and UX."""
    st.header("👤 Individual Player Analysis")
    st.markdown("""
    <div style='margin-bottom:1.2rem; color:#666;'>
        <b>Select a player to view their predicted stats and confidence for today.<br/>
        Use the sidebar to choose a player. See summary, detailed predictions, and recent history (if available).</b>
    </div>
    """, unsafe_allow_html=True)

    predictions_df = dashboard.load_predictions_data()
    if predictions_df is None or predictions_df.empty:
        st.error("No predictions data available for player analysis")
        return

    # Player selector in sidebar expander
    with st.sidebar:
        with st.expander("👤 Select Player for Analysis", expanded=True):
            players = sorted(predictions_df['player'].unique())
            selected_player = st.selectbox("Choose a player:", players, key="player_analysis_selectbox")

    if not selected_player:
        st.info("Please select a player from the sidebar.")
        return

    player_data = predictions_df[predictions_df['player'] == selected_player]
    if player_data.empty:
        st.warning(f"No data found for {selected_player}")
        return

    # Display player summary card
    player_row = player_data.iloc[0]
    st.markdown('<div class="metric-card" style="display:flex;align-items:center;gap:2rem;">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"""
        <div style='font-size:1.3em;font-weight:700;margin-bottom:0.3em;'>🌟 {selected_player}</div>
        <div style='margin-bottom:0.2em;'><span style='font-weight:600;'>🏀 Team:</span> {player_row['team']}</div>
        <div style='margin-bottom:0.2em;'><span style='font-weight:600;'>🆚 Opponent:</span> {player_row['opponent']}</div>
        <div style='margin-bottom:0.2em;'><span style='font-weight:600;'>📍 Location:</span> {'Home' if player_row['home_away'] == 'H' else 'Away'}</div>
        <div style='margin-bottom:0.2em;'><span style='font-weight:600;'>🔒 Confidence:</span> {player_row['confidence_score']:.1%}</div>
        """, unsafe_allow_html=True)
    with col2:
        if PLOTLY_AVAILABLE:
            fig = create_player_radar_chart(player_row)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("📊 Predictions")
            st.write(f"**Points:** {player_row['predicted_points']:.1f} ± {player_row['points_uncertainty']:.1f}")
            st.write(f"**Rebounds:** {player_row['predicted_rebounds']:.1f} ± {player_row['rebounds_uncertainty']:.1f}")
            st.write(f"**Assists:** {player_row['predicted_assists']:.1f} ± {player_row['assists_uncertainty']:.1f}")
    st.markdown('</div>', unsafe_allow_html=True)

    # Modern prediction card with icons and tooltips
    st.markdown('<div class="prediction-card" style="margin-top:1.2em;">', unsafe_allow_html=True)
    st.markdown(f"""
    <div style='font-size:1.1em;font-weight:600;margin-bottom:0.5em;'>Today's Prediction</div>
    <div style='display:flex;gap:1.5em;align-items:center;'>
        <div title='Predicted Points'>
            <span style='font-size:1.2em;'>🎯</span>
            <span style='color:#3b82f6;font-weight:700;font-size:1.1em;margin-left:0.3em;'>{player_row['predicted_points']:.1f}</span>
            <span style='color:#6b7280;font-size:0.95em;'>&plusmn;{player_row['points_uncertainty']:.1f}</span>
            <span style='font-size:0.95em;color:#3b82f6;margin-left:0.2em;'>PTS</span>
        </div>
        <div title='Predicted Rebounds'>
            <span style='font-size:1.2em;'>🛡️</span>
            <span style='color:#10b981;font-weight:700;font-size:1.1em;margin-left:0.3em;'>{player_row['predicted_rebounds']:.1f}</span>
            <span style='color:#6b7280;font-size:0.95em;'>&plusmn;{player_row['rebounds_uncertainty']:.1f}</span>
            <span style='font-size:0.95em;color:#10b981;margin-left:0.2em;'>REB</span>
        </div>
        <div title='Predicted Assists'>
            <span style='font-size:1.2em;'>🎯</span>
            <span style='color:#f59e0b;font-weight:700;font-size:1.1em;margin-left:0.3em;'>{player_row['predicted_assists']:.1f}</span>
            <span style='color:#6b7280;font-size:0.95em;'>&plusmn;{player_row['assists_uncertainty']:.1f}</span>
            <span style='font-size:0.95em;color:#f59e0b;margin-left:0.2em;'>AST</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Show recent predictions for this player if available (e.g., last 5 games)
    if 'game_date' in player_data.columns:
        recent = player_data.sort_values(by='game_date', ascending=False).head(5)
        if len(recent) > 1:
            st.markdown("<div style='margin-top:1.5em;'><b>Recent Predictions (last 5 games):</b></div>", unsafe_allow_html=True)
            st.dataframe(
                recent.loc[:, ['game_date', 'predicted_points', 'predicted_rebounds', 'predicted_assists', 'confidence_score']]
                .rename(columns={
                    'game_date': 'Date',
                    'predicted_points': 'PTS',
                    'predicted_rebounds': 'REB',
                    'predicted_assists': 'AST',
                    'confidence_score': 'Confidence'
                }),
                use_container_width=True,
                hide_index=True
            )


def create_player_radar_chart(player_data):
    """Create radar chart for player predictions (only if Plotly available)."""
    if not PLOTLY_AVAILABLE:
        return None
    
    categories = ['Points', 'Rebounds', 'Assists']
    values = [
        player_data['predicted_points'],
        player_data['predicted_rebounds'],
        player_data['predicted_assists']
    ]
    
    # Normalize to percentage of typical max values
    max_values = [30, 15, 12]
    normalized_values = [min(v/m*100, 100) for v, m in zip(values, max_values)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=categories,
        fill='toself',
        name=player_data['player'],
        line_color='#667eea'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )),
        showlegend=False,
        title=f"{player_data['player']} - Game Projection",
        height=400
    )
    
    return fig


def render_system_status(dashboard: WNBADashboard):
    """Render system status page."""
    st.header("📊 System Status")
    st.markdown("<div style='margin-bottom:1.2rem; color:#666;'>Overview of data, models, predictions, and system health.</div>", unsafe_allow_html=True)
    
    status = dashboard.check_system_status()
    
    # System status overview
    st.subheader("Current Status")
    
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 📁 Data")
            if status['data_available']:
                st.success(f"✅ {status['data_files']} data files available")
            else:
                st.error("❌ No data files found")
            st.markdown("### 🤖 Models")
            if status['models_trained']:
                st.success(f"✅ {status['model_files']} model versions available")
            else:
                st.error("❌ No trained models found")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("### 🔮 Predictions")
            if status['predictions_available']:
                st.success("✅ Predictions available")
                if status['last_prediction']:
                    st.info(f"Last updated: {status['last_prediction'].strftime('%Y-%m-%d %H:%M')}")
            else:
                st.warning("⚠️ No predictions available")
            st.markdown("### 📂 Directories")
            if status['directories_exist']:
                st.success("✅ All directories exist")
            else:
                st.warning("⚠️ Some directories missing")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Component availability
    st.subheader("Component Status")
    
    components = {
        "Data Models": DATA_MODELS_AVAILABLE,
        "Main Application": MAIN_APP_AVAILABLE,
        "Plotly Visualization": PLOTLY_AVAILABLE,
        "Utilities": UTILS_AVAILABLE
    }
    
    with st.container():
        for component, available in components.items():
            st.markdown('<div class="metric-card" style="margin-bottom:0.5rem;">', unsafe_allow_html=True)
            if available:
                st.success(f"✅ {component}")
            else:
                st.error(f"❌ {component}")
            st.markdown('</div>', unsafe_allow_html=True)


def render_management(dashboard: WNBADashboard):
    """Render system management page."""
    st.header("⚙️ System Management")
    st.markdown("<div style='margin-bottom:1.2rem; color:#666;'>Manage data, models, and predictions. Use the actions below to update the system.</div>", unsafe_allow_html=True)
    
    if not dashboard.predictor:
        st.error("❌ Predictor not available - cannot perform management actions")
        return
    
    st.markdown("### Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Fetch Data", use_container_width=True):
            with st.spinner("Fetching season data..."):
                try:
                    current_year = datetime.now().year
                    file_paths = dashboard.predictor.fetch_season_data(current_year)
                    st.success(f"✅ Fetched data: {list(file_paths.keys())}")
                except Exception as e:
                    st.error(f"❌ Data fetch failed: {e}")
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            with st.spinner("Training prediction models..."):
                try:
                    metrics = dashboard.predictor.train_prediction_models()
                    st.success("✅ Models trained successfully")
                except Exception as e:
                    st.error(f"❌ Model training failed: {e}")
    
    with col3:
        if st.button("🔮 Generate Predictions", use_container_width=True):
            with st.spinner("Generating daily predictions..."):
                try:
                    predictions = dashboard.predictor.predict_daily_games()
                    if predictions:
                        export_path = dashboard.predictor.export_predictions(predictions)
                        st.success(f"✅ Generated {len(predictions)} predictions")
                        st.session_state.predictions_data = None  # Force reload
                    else:
                        st.info("📅 No games scheduled for today")
                except Exception as e:
                    st.error(f"❌ Prediction generation failed: {e}")
    
    # Instructions
    st.markdown("---")
    st.markdown("### 💡 Getting Started")
    
    st.markdown("""
    <div class="info-box">
    <strong>If this is your first time using the system:</strong>
    <ol>
        <li><b>Fetch Data:</b> Download current season game logs and player stats</li>
        <li><b>Train Models:</b> Train ML models on the fetched data</li>
        <li><b>Generate Predictions:</b> Create predictions for today's games</li>
    </ol>
    <span style='color:#888'>Note: The system requires real WNBA data from Basketball Reference. Initial setup may take several minutes depending on data availability.</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = WNBADashboard()
    
    dashboard = st.session_state.dashboard
    
    # Render header
    render_header()
    
    # Render sidebar and get selections
    page, confidence_threshold, show_uncertainties = render_sidebar()
    
    # Render selected page
    if page == "🏀 Today's Games":
        render_todays_games(dashboard, confidence_threshold, show_uncertainties)
    
    elif page == "👤 Player Analysis":
        render_player_analysis(dashboard)
    
    elif page == "📊 System Status":
        render_system_status(dashboard)
    
    elif page == "⚙️ Management":
        render_management(dashboard)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏀 <strong>WNBA Daily Game Predictions</strong> | Powered by Machine Learning</p>
        <p>📊 Predictions include Points, Rebounds, Assists with Confidence Intervals</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()