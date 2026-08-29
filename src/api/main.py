import os
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    StudentInputSchema, PredictionResponseSchema,
    BatchPredictionRequestSchema, BatchPredictionResponseSchema,
    RetrainResponseSchema
)
from src.preprocessing import DataPreprocessor
from src.feature_engineering import add_engineered_features
from src.train_ml import train_ml_models, FEATURE_COLUMNS
from src.train_dl import train_dl_model
from src.explainability import get_feature_importances
from src.db import check_db_status, save_single_prediction, save_batch_predictions

MODELS_DIR = "models/saved_models"

app = FastAPI(
    title="Student Performance Prediction System API",
    description="Production REST API serving Machine Learning & Deep Learning predictions for student academic performance, risk assessment, and explainability.",
    version="1.0.0"
)

# CORS middleware enabling cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_artifacts():
    cls_path = os.path.join(MODELS_DIR, "best_classification_model.joblib")
    reg_path = os.path.join(MODELS_DIR, "best_regression_model.joblib")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    
    if not (os.path.exists(cls_path) and os.path.exists(reg_path) and os.path.exists(prep_path)):
        # Auto train models if not present
        train_ml_models(models_dir=MODELS_DIR)
        
    cls_model = joblib.load(cls_path)
    reg_model = joblib.load(reg_path)
    preprocessor = DataPreprocessor.load(prep_path)
    
    return cls_model, reg_model, preprocessor


def get_recommendations(prob: float, score: float, risk: str, attendance: float, study_hours: float) -> List[str]:
    recs = []
    if risk == "High":
        recs.append("Trigger Immediate Educator Intervention & One-on-One Tutoring.")
    if attendance < 75.0:
        recs.append(f"Improve attendance rate (Currently {attendance}%). Target >= 85%.")
    if study_hours < 4.0:
        recs.append(f"Increase weekly study regimen by at least 2.5 hours per day (Currently {study_hours} hrs/day).")
    if score < 60.0:
        recs.append("Schedule remedial coursework for midterm concepts.")
    if not recs:
        recs.append("Student is performing strongly across all indicators. Encourage academic honors pursuit.")
    return recs


def predict_single_student(student_dict: Dict[str, Any], cls_model, reg_model, preprocessor) -> PredictionResponseSchema:
    df_raw = pd.DataFrame([student_dict])
    df_feat = add_engineered_features(df_raw)
    df_proc = preprocessor.transform(df_feat)
    
    X = df_proc[FEATURE_COLUMNS]
    
    # Classification prediction
    prob = float(cls_model.predict_proba(X)[:, 1][0]) if hasattr(cls_model, "predict_proba") else 0.8
    target = "Pass" if prob >= 0.5 else "Fail"
    
    # Regression prediction
    pred_score = float(reg_model.predict(X)[0])
    pred_score = round(max(0.0, min(100.0, pred_score)), 1)
    
    # Grade assignment
    if pred_score >= 85:
        grade = "A"
    elif pred_score >= 75:
        grade = "B"
    elif pred_score >= 65:
        grade = "C"
    elif pred_score >= 50:
        grade = "D"
    else:
        grade = "F"
        
    # Risk assessment
    attendance = float(student_dict.get("Attendance", 80))
    midterm = float(student_dict.get("Midterm_Score", 70))
    
    if prob < 0.45 or attendance < 60 or midterm < 45 or pred_score < 50:
        risk_level = "High"
        risk_color = "#ef4444"
    elif prob < 0.70 or attendance < 75 or pred_score < 65:
        risk_level = "Medium"
        risk_color = "#f59e0b"
    else:
        risk_level = "Low"
        risk_color = "#10b981"
        
    recs = get_recommendations(
        prob=prob, score=pred_score, risk=risk_level,
        attendance=attendance, study_hours=float(student_dict.get("Study_Hours", 4.0))
    )
    
    return PredictionResponseSchema(
        prediction_status="Success",
        pass_probability=round(prob, 4),
        predicted_target=target,
        predicted_final_score=pred_score,
        predicted_grade=grade,
        risk_level=risk_level,
        risk_color=risk_color,
        recommendations=recs
    )


@app.get("/health", tags=["Health"])
def health_check():
    cls_model, reg_model, preprocessor = load_artifacts()
    db_ok, db_msg = check_db_status()
    return {
        "status": "healthy",
        "database": "connected" if db_ok else db_msg,
        "system": "Student Performance Prediction API",
        "models_loaded": {
            "classification_model": str(type(cls_model).__name__),
            "regression_model": str(type(reg_model).__name__)
        }
    }


@app.post("/predict", response_model=PredictionResponseSchema, tags=["Prediction"])
def predict(student: StudentInputSchema):
    try:
        cls_model, reg_model, preprocessor = load_artifacts()
        res = predict_single_student(student.model_dump(), cls_model, reg_model, preprocessor)
        save_single_prediction(student.model_dump(), res.model_dump())
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponseSchema, tags=["Prediction"])
def predict_batch(request: BatchPredictionRequestSchema):
    try:
        cls_model, reg_model, preprocessor = load_artifacts()
        preds = []
        for student in request.students:
            res = predict_single_student(student.model_dump(), cls_model, reg_model, preprocessor)
            preds.append(res)
            
        save_batch_predictions([p.model_dump() for p in preds])
        return BatchPredictionResponseSchema(
            total_processed=len(preds),
            predictions=preds
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", tags=["Metrics & Analysis"])
def get_metrics():
    try:
        df_imp = get_feature_importances(MODELS_DIR)
        return {
            "feature_importance_ranking": df_imp.to_dict(orient="records")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrain", response_model=RetrainResponseSchema, tags=["Administration"])
def retrain_pipeline():
    try:
        ml_results = train_ml_models(models_dir=MODELS_DIR)
        dl_results = train_dl_model(models_dir=MODELS_DIR)
        return RetrainResponseSchema(
            status="Success",
            message="Machine Learning & Deep Learning pipelines retrained and updated successfully.",
            ml_results=ml_results,
            dl_results=dl_results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
