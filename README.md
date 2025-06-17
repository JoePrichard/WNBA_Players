# 🏀 WNBA Daily Game Prediction System

Advanced machine learning system for predicting WNBA player statistics (Points, Rebounds, Assists) with uncertainty quantification.

## 🎯 Overview

This system implements state-of-the-art sports prediction techniques based on research from successful models like DARKO, XGBoost Synergy approaches, and Neural Networks with attention mechanisms. It provides daily predictions with confidence intervals and comprehensive model validation.

### Key Features

- **🤖 Ensemble ML Models**: XGBoost, LightGBM, Random Forest, Neural Networks
- **📊 Advanced Features**: Recent form, opponent strength, team synergy, positional expectations
- **🎯 Research-Based**: Targets Brier score < 0.12 (research benchmark)
- **📈 Uncertainty Quantification**: 95% confidence intervals using Monte Carlo methods
- **🕐 Time Series Validation**: Walk-forward validation simulating real-world usage
- **🚀 Interactive Dashboard**: Streamlit-based interface for predictions and insights
- **🔧 Modular Design**: Clean, documented, type-hinted Python codebase

## 📋 Prerequisites

- **Python 3.8+** (required)
- **Internet connection** (for data fetching)
- **~2GB free space** (for models and data)

## 🚀 Quick Start

### Option 1: Automated Setup

```bash
# Clone or download the project files
# Run automated setup
python setup.py

# Follow the setup prompts, then:
python wnba_main_application.py --full-pipeline 2025
streamlit run wnba_dashboard.py