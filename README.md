# WNBA Players Prediction & Analytics

A comprehensive pipeline for scraping, processing, and predicting WNBA player statistics using machine learning. This project fetches detailed player game logs, engineers features, trains ensemble models, and provides daily game predictions.

## Features
- Scrapes full box score stats for all WNBA games (minutes, points, FG, 3P, FT, rebounds, assists, steals, blocks, turnovers, fouls, plus/minus, and more)
- Cleans and engineers advanced features for modeling
- Trains ensemble ML models (XGBoost, LightGBM, Random Forest, Neural Network)
- Predicts player stats for upcoming games with uncertainty estimates
- Exports predictions to CSV for further analysis or dashboarding

## Installation
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd WNBA_Players
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(If requirements.txt is missing, install: pandas, numpy, scikit-learn, xgboost, lightgbm, torch, beautifulsoup4, requests, streamlit, plotly)*

## File Structure
- `data_fetcher.py` — Scrapes WNBA player stats from Basketball Reference
- `data_models.py` — Data classes and configuration for stats, predictions, and validation
- `feature_engineer.py` — Feature engineering for ML models
- `prediction_models.py` — ML model training and prediction logic
- `main_application.py` — Main pipeline: data fetch, train, predict, export
- `dashboard.py` — (Optional) Streamlit dashboard for visualization
- `model_validator.py` — Model evaluation and validation utilities
- `config_loader.py` — Loads and validates configuration
- `setup.py` — Project setup and utility scripts

## Basic Usage
All commands assume you are in the project root directory.

### 1. **Scrape WNBA Player Data**
Fetch all player stats for a season (e.g., 2024):
```bash
python main_application.py --fetch-data 2024
```

### 2. **Train Prediction Models**
Train models on the latest available data:
```bash
python main_application.py --train 2024
```

### 3. **Generate Predictions for Today**
Predict player stats for today's scheduled games:
```bash
python main_application.py --predict
```

### 4. **Run the Full Pipeline**
Fetch data, train models, and predict in one step:
```bash
python main_application.py --full-pipeline 2024
```

### 5. **Check Data Availability**
See what data is available locally for a season:
```bash
python main_application.py --check-data 2024
```

### 6. **Launch the Dashboard (Optional)**
If you want to visualize predictions:
```bash
streamlit run dashboard.py
```

## Multi-Season Training

To train models on multiple seasons, specify the years in your config file under the [data] section:

```toml
[data]
train_years = [2021, 2022, 2023, 2024]
```

The pipeline will automatically fetch, load, and train on all specified years.

## Notes
- Data and model files are saved in `wnba_game_data/` and `wnba_models/` by default.
- All outputs (predictions) are saved in `wnba_predictions/`.
- The scraper uses Basketball Reference and may be subject to their request limits.

## License
MIT License 