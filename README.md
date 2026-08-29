# 🎓 Student Performance Prediction System

A complete end-to-end Machine Learning and PyTorch Deep Learning system for predicting student performance, assessing academic risk, providing SHAP explainability, serving FastAPI REST predictions, and displaying interactive analytics via a Streamlit Dashboard.

---

## 🌟 Key Features

1. **Synthetic Data Generator & Preprocessing Pipeline**: Standardizes schema, validates boundaries, handles missing values, label encodes categoricals, scales numerical features.
2. **Feature Engineering**: Interaction terms (`Attendance x Study Hours`), coursework averages, score improvement metrics.
3. **ML & PyTorch Deep Learning**:
   - Classification: Logistic Regression, Random Forest, XGBoost / Gradient Boosting, PyTorch Multi-Layer Perceptron (Neural Network with Dropout & Early Stopping).
   - Metrics: Accuracy, Precision, Recall, F1 Score, ROC-AUC, MAE, MSE, RMSE, R² Score.
4. **FastAPI REST API**:
   - `POST /predict`: Single student performance prediction & risk assessment.
   - `POST /predict/batch`: Bulk student file processing.
   - `GET /metrics`: Model evaluation summary & feature importances.
   - `POST /retrain`: Automated model retraining trigger.
5. **Streamlit Interactive Dashboard**:
   - Dark theme glassmorphic design.
   - Executive KPIs & correlation charts.
   - Live interactive student predictor with instant recommendations.
   - Batch CSV uploader with report exporter.
   - ML vs DL benchmark comparison.
   - Model Explainability (SHAP & feature weights).
   - Continuous learning feedback loop.
6. **Containerized Production**: Complete Docker & Docker Compose configuration.

---

## 🚀 Quick Start Guide

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
```bash
python -m src.data_generator
python -m src.train_ml
python -m src.train_dl
```

### 3. Launch Streamlit Dashboard
```bash
streamlit run app.py
```

### 4. Launch FastAPI REST Server
```bash
uvicorn src.api.main:app --reload --port 8000
```
FastAPI interactive docs will be available at: `http://localhost:8000/docs`.

### 5. Run Automated Tests
```bash
pytest tests/
```

### 6. Docker Deployment
```bash
docker-compose -f docker/docker-compose.yml up --build
```
