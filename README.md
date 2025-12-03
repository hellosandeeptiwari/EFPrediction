# Intercept AI - Enrollment Form Prediction

## Enterprise Machine Learning Platform for Rare Disease Pharmaceutical

This project provides an end-to-end machine learning pipeline for predicting enrollment forms in a rare disease pharmaceutical context.

## Project Structure

```
Intercept EF Prediction/
├── Reporting Tables/          # Source data files (13 CSV files)
│   ├── icpt_ai_hcp_universe_*.csv
│   ├── monthly_base_kpis__c_*.csv
│   ├── daily_base_kpis__c_*.csv
│   └── ... (other data sources)
│
├── src/                       # Python ML Pipeline
│   ├── __init__.py           # Package initialization
│   ├── config.py             # Configuration settings
│   ├── data_discovery.py     # Data loading & profiling
│   ├── eda.py                # Exploratory Data Analysis
│   ├── feature_engineering.py # Feature creation
│   ├── feature_selection.py  # Feature selection methods
│   ├── target_engineering.py # Target variable engineering
│   ├── model_training.py     # Model training (XGBoost, LightGBM, RF, LR)
│   ├── scoring.py            # Batch & real-time inference
│   ├── pptx_generator.py     # PowerPoint report generation
│   ├── main.py               # Pipeline orchestrator
│   └── api_server.py         # Flask REST API
│
├── ui/                        # Node.js Web Dashboard
│   ├── package.json          # Node.js dependencies
│   ├── server.js             # Express server
│   ├── .env                  # Environment configuration
│   ├── views/                # EJS templates
│   │   ├── layout.ejs
│   │   ├── dashboard.ejs
│   │   ├── predict.ejs
│   │   ├── batch.ejs
│   │   ├── eda.ejs
│   │   └── model-info.ejs
│   └── public/               # Static assets
│       ├── css/styles.css
│       └── js/
│           ├── common.js
│           ├── dashboard.js
│           ├── predict.js
│           ├── batch.js
│           ├── eda.js
│           └── model-info.js
│
├── outputs/                   # Generated outputs (created at runtime)
│   ├── models/               # Trained model artifacts
│   ├── reports/              # EDA reports & PowerPoint
│   ├── predictions/          # Prediction results
│   └── figures/              # Visualization outputs
│
└── README.md                  # This file
```

## ML Pipeline Steps

### 1. Data Discovery
- Loads all 13 data sources from Reporting Tables
- Generates comprehensive data profiling report
- Analyzes column types, missing values, distributions

### 2. Exploratory Data Analysis (EDA)
- Statistical analysis (Mann-Whitney, Chi-square tests)
- Target variable distribution analysis
- Feature correlation analysis
- Visualization generation

### 3. Feature Engineering
- Temporal features (month, quarter, year trends)
- Aggregate features (rolling statistics, cumulative sums)
- Engagement scores (calls, meetings, emails)
- Territorial and segment-based features

### 4. Feature Selection
- Correlation-based filtering
- Variance threshold filtering
- Model-based importance (Random Forest, XGBoost)
- Recursive Feature Elimination (RFE)

### 5. Target Engineering
- Binary classification target (enrollment vs. non-enrollment)
- Class imbalance handling with SMOTE
- Stratified train/test splitting

### 6. Model Training
- Multiple algorithms: Logistic Regression, Random Forest, XGBoost, LightGBM
- Hyperparameter tuning with cross-validation
- Model comparison and selection

### 7. Scoring
- Batch prediction capability
- Real-time single prediction
- Probability calibration
- SHAP explanations

## Installation & Setup

### Prerequisites
- Python 3.10+ with pip
- Node.js 18+ with npm

### Python Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm shap flask flask-cors imbalanced-learn matplotlib seaborn plotly python-pptx statsmodels scipy joblib openpyxl
```

### Run ML Pipeline

```bash
# Run the full pipeline
python src/main.py
```

### Start Flask API

```bash
# Start API server (runs on http://localhost:5000)
python src/api_server.py
```

### Node.js UI Setup

```bash
# Navigate to UI folder
cd ui

# Install dependencies
npm install

# Start development server
npm run dev

# Or production
npm start
```

The UI will be available at http://localhost:3000

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | API health check |
| `/api/predict` | POST | Single prediction |
| `/api/batch_predict` | POST | Batch predictions |
| `/api/model_info` | GET | Model information & metrics |
| `/api/eda_results` | GET | EDA results & statistics |

## Dashboard Features

### Dashboard
- Model performance KPIs
- Quick prediction form
- Feature importance visualization
- Pipeline status tracker

### Predict
- Detailed prediction form with all features
- Real-time prediction results
- Confidence scores and probabilities

### Batch Prediction
- CSV file upload (drag & drop)
- Bulk prediction processing
- Results download capability

### EDA
- Dataset statistics
- Target distribution
- Regional analysis
- Monthly trends
- Feature correlations

### Model Info
- Model performance metrics
- Confusion matrix
- Hyperparameters
- Model comparison table

## Data Sources

| File | Records | Description |
|------|---------|-------------|
| icpt_ai_hcp_universe | 20,089 | HCP master data |
| monthly_base_kpis__c | 527,160 | Monthly enrollments |
| daily_base_kpis__c | 1,241,155 | Daily enrollment events |
| territory_hierarchy__c | 127 | Territory structure |
| monthly_calls_by_territory__c | 7,618 | Call metrics |
| monthly_meetings__c | 1,791 | Meeting counts |
| writers_prescribers_count__c | 2,264 | Writer metrics |
| tbm_goals__c | 521 | Territory goals |
| And more... | | |

## Key Technologies

- **ML Framework**: scikit-learn, XGBoost, LightGBM
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Explainability**: SHAP
- **API**: Flask, Flask-CORS
- **Frontend**: Node.js, Express, EJS, Bootstrap 5, Chart.js

## License

MIT License - Intercept AI Team © 2025
