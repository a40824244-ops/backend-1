import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from src.data_generator import generate_student_dataset
from src.preprocessing import DataPreprocessor, CATEGORICAL_COLS, NUMERICAL_COLS
from src.feature_engineering import add_engineered_features

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


FEATURE_COLUMNS = [
    "Gender", "Age", "Study_Hours", "Attendance", "Assignment_Score",
    "Quiz_Score", "Midterm_Score", "Internet_Access", "Parent_Education",
    "Family_Income", "Extra_Activities", "Previous_GPA",
    "Attendance_Study_Interaction", "Avg_Quiz_Assignment",
    "Coursework_Average", "Score_Improvement", "GPA_Midterm_Ratio"
]


def train_ml_models(data_path: str = None, models_dir: str = "models/saved_models") -> Dict[str, Any]:
    os.makedirs(models_dir, exist_ok=True)
    
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        df = generate_student_dataset(n_samples=1500)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/student_data.csv", index=False)

    # 1. Feature Engineering
    df_feat = add_engineered_features(df)
    
    # Target variables
    y_cls = (df_feat["Target"] == "Pass").astype(int)
    y_reg = df_feat["Final_Score"]
    
    # Preprocessing
    preprocessor = DataPreprocessor()
    df_proc = preprocessor.fit_transform(df_feat)
    preprocessor.save(os.path.join(models_dir, "preprocessor.joblib"))
    
    X = df_proc[FEATURE_COLUMNS]
    
    # Train test split
    X_train, X_test, y_train_cls, y_test_cls, y_train_reg, y_test_reg = train_test_split(
        X, y_cls, y_reg, test_size=0.2, random_state=42, stratify=y_cls
    )
    
    results = {"classification": {}, "regression": {}}
    
    # --- CLASSIFICATION MODELS ---
    cls_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
    }
    
    if HAS_XGBOOST:
        cls_models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42, eval_metric="logloss")
    else:
        cls_models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=100, random_state=42)
        
    best_cls_model = None
    best_cls_f1 = -1.0
    best_cls_name = ""
    
    for name, model in cls_models.items():
        model.fit(X_train, y_train_cls)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else preds
        
        acc = accuracy_score(y_test_cls, preds)
        prec = precision_score(y_test_cls, preds, zero_division=0)
        rec = recall_score(y_test_cls, preds, zero_division=0)
        f1 = f1_score(y_test_cls, preds, zero_division=0)
        auc = roc_auc_score(y_test_cls, probs)
        
        results["classification"][name] = {
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1 Score": round(f1, 4),
            "ROC-AUC": round(auc, 4)
        }
        
        if f1 > best_cls_f1:
            best_cls_f1 = f1
            best_cls_model = model
            best_cls_name = name
            
    # Save best classification model
    joblib.dump(best_cls_model, os.path.join(models_dir, "best_classification_model.joblib"))
    results["best_classification_model"] = best_cls_name
    
    # --- REGRESSION MODELS ---
    reg_models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=8)
    }
    
    if HAS_XGBOOST:
        reg_models["XGBoost Regressor"] = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05, random_state=42)
    else:
        reg_models["Gradient Boosting Regressor"] = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
    best_reg_model = None
    best_reg_r2 = -999.0
    best_reg_name = ""
    
    for name, model in reg_models.items():
        model.fit(X_train, y_train_reg)
        preds = model.predict(X_test)
        
        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)
        
        results["regression"][name] = {
            "MAE": round(mae, 4),
            "MSE": round(mse, 4),
            "RMSE": round(rmse, 4),
            "R2 Score": round(r2, 4)
        }
        
        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_model = model
            best_reg_name = name
            
    # Save best regression model
    joblib.dump(best_reg_model, os.path.join(models_dir, "best_regression_model.joblib"))
    results["best_regression_model"] = best_reg_name
    
    # Save feature names
    joblib.dump(FEATURE_COLUMNS, os.path.join(models_dir, "feature_columns.joblib"))
    
    print(f"ML Training Complete! Best Cls: {best_cls_name} (F1: {best_cls_f1:.4f}), Best Reg: {best_reg_name} (R2: {best_reg_r2:.4f})")
    return results

if __name__ == "__main__":
    metrics = train_ml_models()
    print("Metrics Summary:", metrics)
