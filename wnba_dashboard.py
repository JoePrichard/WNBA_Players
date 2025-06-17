#!/usr/bin/env python3
"""
WNBA Daily Game Prediction Dashboard
Streamlit application for interactive predictions and model insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import json

# Custom imports
try:
    from wnba_data_models import (
        PredictionConfig, PlayerPrediction, HomeAway,
        WNBADataError, WNBAModelError, WNBAPredictionError
    )
    from wnba_main_application import WNBADailyPredictor
    from wnba_model_validator import WNBAModelValidator
except ImportError as e:
    st.error(f"Missing required modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="WNBA Daily Game Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .stat-badge {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 0.5rem 1rem;
        border-radius: 1rem;
        color: white;
        font-weight: bold;
        margin: 0.2rem;
        display: inline-block;
    }
    
    .confidence-high { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .confidence-medium { background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%); color: #333; }
    .confidence-low { background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%); color: #333; }
    
    .error-box {
        background: #fff5f5;
        border: 1px solid #fed7d7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #c53030;
    }
    
    .info-box {
        background: #ebf8ff;
        border: 1px solid #bee3f8;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
        color: #2a69ac;
    }
</style>
""", unsafe_allow_html=True)


class WNBADashboard:
    """
    Main dashboard class for WNBA predictions.
    
    Manages state, data loading, and page routing.
    
    Attributes:
        predictor: Main prediction application
        validator: Model validation component
        config: Configuration for the dashboard
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        self.predictor: Optional[WNBADailyPredictor] = None
        self.validator: Optional[WNBAModelValidator] = None
        self.config = PredictionConfig()
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'predictions_data' not in st.session_state:
            st.session_state.predictions_data = None
        
        if 'model_metrics' not in st.session_state:
            st.session_state.model_metrics = None
        
        if 'last_update' not in st.session_state:
            st.session_state.last_update = None
    
    def load_components(self) -> bool:
        """
        Load dashboard components.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.predictor = WNBADailyPredictor(config=self.config)
            self.validator = WNBAModelValidator(config=self.config)
            return True
        except Exception as e:
            st.error(f"Failed to load components: {e}")
            return False
    
    def load_predictions_data(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load predictions data from file or generate new.
        
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
            pred_files = list(Path("wnba_predictions").glob("predictions_*.csv"))
            
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
            'model_files': 0
        }
        
        try:
            # Check for data files
            data_dir = Path("wnba_game_data")
            if data_dir.exists():
                data_files = list(data_dir.glob("*.csv"))
                status['data_files'] = len(data_files)
                status['data_available'] = len(data_files) > 0
            
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
    st.markdown("### *Advanced ML Ensemble for Points, Rebounds & Assists*")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    dashboard = st.session_state.get('dashboard')
    if dashboard:
        status = dashboard.check_system_status()
        
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
    st.sidebar.title("🎛️ Navigation")
    
    # Page selection
    page = st.sidebar.selectbox(
        "Choose Page:",
        [
            "🏀 Today's Games",
            "👤 Player Analysis", 
            "📊 Team Comparison",
            "🎯 Model Performance",
            "⚙️ System Management"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Quick actions
    st.sidebar.subheader("Quick Actions")
    
    if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
        st.session_state.predictions_data = None
        st.rerun()
    
    if st.sidebar.button("📈 Generate Report", use_container_width=True):
        st.sidebar.success("Report generation would start here")
    
    # Settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for displaying predictions"
    )
    
    show_uncertainties = st.sidebar.checkbox(
        "Show Uncertainties",
        value=True,
        help="Display confidence intervals"
    )
    
    # Model info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Model Info")
    st.sidebar.info(
        "**Models Used:**\n"
        "• XGBoost (30%)\n"
        "• LightGBM (25%)\n"
        "• Random Forest (25%)\n"
        "• Neural Network (20%)\n\n"
        "**Target:** Brier Score < 0.12"
    )
    
    return page, confidence_threshold, show_uncertainties


def render_todays_games(dashboard: WNBADashboard, confidence_threshold: float, show_uncertainties: bool):
    """Render today's games page."""
    st.header("🏀 Today's Game Predictions")
    st.write(f"📅 **{date.today().strftime('%A, %B %d, %Y')}**")
    
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
            <p><strong>To generate predictions:</strong></p>
            <ol>
                <li>Go to "System Management" page</li>
                <li>Fetch current season data</li>
                <li>Train prediction models</li>
                <li>Generate daily predictions</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Filter by confidence if needed
    if confidence_threshold > 0:
        predictions_df = predictions_df[predictions_df['confidence_score'] >= confidence_threshold]
    
    if predictions_df.empty:
        st.warning(f"No predictions meet confidence threshold of {confidence_threshold:.1%}")
        return
    
    # Group by game
    games = predictions_df.groupby('game_id')
    
    if len(games) == 0:
        st.info("📅 No games found in predictions data")
        return
    
    # Display each game
    for game_id, game_predictions in games:
        st.subheader(f"🎮 {game_id}")
        
        # Game summary
        home_team = game_predictions[game_predictions['home_away'] == 'H']['team'].iloc[0] if 'H' in game_predictions['home_away'].values else "TBD"
        away_team = game_predictions[game_predictions['home_away'] == 'A']['team'].iloc[0] if 'A' in game_predictions['home_away'].values else "TBD"
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🏠 {home_team}**")
            home_players = game_predictions[game_predictions['home_away'] == 'H']
            render_team_predictions(home_players, show_uncertainties)
        
        with col2:
            st.markdown(f"**✈️ {away_team}**")
            away_players = game_predictions[game_predictions['home_away'] == 'A']
            render_team_predictions(away_players, show_uncertainties)
        
        # Game totals
        total_points_home = home_players['predicted_points'].sum() if not home_players.empty else 0
        total_points_away = away_players['predicted_points'].sum() if not away_players.empty else 0
        
        st.markdown(f"""
        <div class="metric-card">
            <h4>Game Projection</h4>
            <p><strong>{away_team}:</strong> {total_points_away:.1f} points</p>
            <p><strong>{home_team}:</strong> {total_points_home:.1f} points</p>
            <p><strong>Projected Winner:</strong> {home_team if total_points_home > total_points_away else away_team}</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")


def render_team_predictions(team_predictions: pd.DataFrame, show_uncertainties: bool):
    """Render predictions for a team."""
    if team_predictions.empty:
        st.write("No player predictions available")
        return
    
    for _, player in team_predictions.iterrows():
        confidence_class = get_confidence_class(player['confidence_score'])
        
        uncertainty_text = ""
        if show_uncertainties:
            uncertainty_text = f"""
            <small>
                Points: {player['predicted_points']:.1f} ± {player['points_uncertainty']:.1f}<br>
                Rebounds: {player['predicted_rebounds']:.1f} ± {player['rebounds_uncertainty']:.1f}<br>
                Assists: {player['predicted_assists']:.1f} ± {player['assists_uncertainty']:.1f}
            </small>
            """
        
        st.markdown(f"""
        <div class="prediction-card">
            <strong>{player['player']}</strong>
            <span class="stat-badge confidence-{confidence_class}">
                {player['confidence_score']:.1%} confidence
            </span>
            <br>
            <div style="margin-top: 0.5rem;">
                <span class="stat-badge">{player['predicted_points']:.1f} PTS</span>
                <span class="stat-badge">{player['predicted_rebounds']:.1f} REB</span>
                <span class="stat-badge">{player['predicted_assists']:.1f} AST</span>
            </div>
            {uncertainty_text}
        </div>
        """, unsafe_allow_html=True)


def get_confidence_class(confidence: float) -> str:
    """Get CSS class for confidence level."""
    if confidence >= 0.8:
        return "high"
    elif confidence >= 0.6:
        return "medium"
    else:
        return "low"


def render_player_analysis(dashboard: WNBADashboard):
    """Render player analysis page."""
    st.header("👤 Individual Player Analysis")
    
    predictions_df = dashboard.load_predictions_data()
    
    if predictions_df is None or predictions_df.empty:
        st.error("No predictions data available for player analysis")
        return
    
    # Player selector
    players = sorted(predictions_df['player'].unique())
    selected_player = st.selectbox("Choose a player:", players)
    
    if not selected_player:
        return
    
    player_data = predictions_df[predictions_df['player'] == selected_player]
    
    if player_data.empty:
        st.warning(f"No data found for {selected_player}")
        return
    
    # Display player info
    player_row = player_data.iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f"🌟 {selected_player}")
        st.write(f"**Team:** {player_row['team']}")
        st.write(f"**Opponent:** {player_row['opponent']}")
        st.write(f"**Location:** {'Home' if player_row['home_away'] == 'H' else 'Away'}")
        st.write(f"**Confidence:** {player_row['confidence_score']:.1%}")
    
    with col2:
        # Create radar chart
        fig = create_player_radar_chart(player_row)
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed predictions
    st.subheader("📊 Detailed Predictions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Points",
            f"{player_row['predicted_points']:.1f}",
            delta=f"±{player_row['points_uncertainty']:.1f}"
        )
        st.write(f"**95% CI:** {player_row.get('points_ci_lower', 0):.1f} - {player_row.get('points_ci_upper', 0):.1f}")
    
    with col2:
        st.metric(
            "Rebounds", 
            f"{player_row['predicted_rebounds']:.1f}",
            delta=f"±{player_row['rebounds_uncertainty']:.1f}"
        )
        st.write(f"**95% CI:** {player_row.get('rebounds_ci_lower', 0):.1f} - {player_row.get('rebounds_ci_upper', 0):.1f}")
    
    with col3:
        st.metric(
            "Assists",
            f"{player_row['predicted_assists']:.1f}",
            delta=f"±{player_row['assists_uncertainty']:.1f}"
        )
        st.write(f"**95% CI:** {player_row.get('assists_ci_lower', 0):.1f} - {player_row.get('assists_ci_upper', 0):.1f}")


def create_player_radar_chart(player_data: pd.Series) -> go.Figure:
    """Create radar chart for player predictions."""
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


def render_system_management(dashboard: WNBADashboard):
    """Render system management page."""
    st.header("⚙️ System Management")
    
    status = dashboard.check_system_status()
    
    # System status overview
    st.subheader("📊 System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if status['data_available']:
            st.success(f"✅ Data Files\n({status['data_files']} available)")
        else:
            st.error("❌ No Data Files")
    
    with col2:
        if status['models_trained']:
            st.success(f"✅ Trained Models\n({status['model_files']} versions)")
        else:
            st.error("❌ No Trained Models")
    
    with col3:
        if status['predictions_available']:
            st.success("✅ Predictions Available")
        else:
            st.warning("⚠️ No Predictions")
    
    with col4:
        if status['last_prediction']:
            st.info(f"🕐 Last Update\n{status['last_prediction'].strftime('%m/%d %H:%M')}")
        else:
            st.info("🕐 No Updates")
    
    # Action buttons
    st.subheader("🔧 Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Fetch Data", use_container_width=True):
            with st.spinner("Fetching season data..."):
                try:
                    if dashboard.predictor:
                        current_year = datetime.now().year
                        file_paths = dashboard.predictor.fetch_season_data(current_year)
                        st.success(f"✅ Fetched data: {list(file_paths.keys())}")
                    else:
                        st.error("Predictor not initialized")
                except Exception as e:
                    st.error(f"Data fetch failed: {e}")
    
    with col2:
        if st.button("🤖 Train Models", use_container_width=True):
            with st.spinner("Training prediction models..."):
                try:
                    if dashboard.predictor:
                        metrics = dashboard.predictor.train_prediction_models()
                        st.success("✅ Models trained successfully")
                        st.session_state.model_metrics = metrics
                    else:
                        st.error("Predictor not initialized")
                except Exception as e:
                    st.error(f"Model training failed: {e}")
    
    with col3:
        if st.button("🔮 Generate Predictions", use_container_width=True):
            with st.spinner("Generating daily predictions..."):
                try:
                    if dashboard.predictor:
                        predictions = dashboard.predictor.predict_daily_games()
                        if predictions:
                            export_path = dashboard.predictor.export_predictions(predictions)
                            st.success(f"✅ Generated {len(predictions)} predictions")
                            st.session_state.predictions_data = None  # Force reload
                        else:
                            st.info("No games scheduled for today")
                    else:
                        st.error("Predictor not initialized")
                except Exception as e:
                    st.error(f"Prediction generation failed: {e}")
    
    # Configuration section
    st.subheader("⚙️ Configuration")
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 Getting Started</h4>
        <p>If this is your first time using the system:</p>
        <ol>
            <li><strong>Fetch Data:</strong> Download current season game logs and player stats</li>
            <li><strong>Train Models:</strong> Train ML models on the fetched data</li>
            <li><strong>Generate Predictions:</strong> Create predictions for today's games</li>
        </ol>
        <p>The system requires real WNBA data from Basketball Reference or similar sources.</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main dashboard application."""
    # Initialize dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = WNBADashboard()
    
    dashboard = st.session_state.dashboard
    
    # Load components if needed
    if dashboard.predictor is None:
        if not dashboard.load_components():
            st.error("Failed to initialize dashboard components")
            st.stop()
    
    # Render header
    render_header()
    
    # Render sidebar and get selections
    page, confidence_threshold, show_uncertainties = render_sidebar()
    
    # Render selected page
    if page == "🏀 Today's Games":
        render_todays_games(dashboard, confidence_threshold, show_uncertainties)
    
    elif page == "👤 Player Analysis":
        render_player_analysis(dashboard)
    
    elif page == "📊 Team Comparison":
        st.header("📊 Team Comparison")
        st.info("Team comparison features would be implemented here")
    
    elif page == "🎯 Model Performance":
        st.header("🎯 Model Performance")
        st.info("Model performance analytics would be implemented here")
    
    elif page == "⚙️ System Management":
        render_system_management(dashboard)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>🏀 <strong>WNBA Daily Game Predictions</strong> | Powered by Advanced ML Ensemble</p>
        <p>🎯 Based on research from DARKO, XGBoost Synergy Models, and Neural Networks</p>
        <p>📊 Predictions include Points, Rebounds, Assists with 95% Confidence Intervals</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()