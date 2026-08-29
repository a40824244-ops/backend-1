import os
import pytest
import pandas as pd
import numpy as np

from src.data_generator import generate_student_dataset
from src.preprocessing import DataPreprocessor, CATEGORICAL_COLS, NUMERICAL_COLS
from src.feature_engineering import add_engineered_features
from src.train_ml import train_ml_models, FEATURE_COLUMNS
from src.train_dl import train_dl_model


def test_data_generation():
    df = generate_student_dataset(n_samples=100)
    assert len(df) == 100
    assert "Student_ID" in df.columns
    assert "Final_Score" in df.columns
    assert "Target" in df.columns
    assert "Risk_Level" in df.columns


def test_preprocessing():
    df = generate_student_dataset(n_samples=50)
    preprocessor = DataPreprocessor()
    is_valid, msg = preprocessor.validate_data(df)
    assert is_valid, msg
    
    df_proc = preprocessor.fit_transform(df)
    assert preprocessor.is_fitted
    for col in CATEGORICAL_COLS:
        assert np.issubdtype(df_proc[col].dtype, np.integer)


def test_feature_engineering():
    df = generate_student_dataset(n_samples=50)
    df_feat = add_engineered_features(df)
    assert "Attendance_Study_Interaction" in df_feat.columns
    assert "Coursework_Average" in df_feat.columns
    assert "Score_Improvement" in df_feat.columns


def test_ml_and_dl_training(tmp_path):
    models_dir = os.path.join(tmp_path, "saved_models")
    ml_results = train_ml_models(models_dir=models_dir)
    assert "classification" in ml_results
    assert "regression" in ml_results
    assert os.path.exists(os.path.join(models_dir, "best_classification_model.joblib"))
    
    dl_results = train_dl_model(models_dir=models_dir, epochs=5)
    assert "Accuracy" in dl_results
    
    # Check that either PyTorch or Scikit-Learn MLP Neural Network model artifact exists
    pt_exists = os.path.exists(os.path.join(models_dir, "pytorch_student_model.pt"))
    mlp_exists = os.path.exists(os.path.join(models_dir, "mlp_neural_network.joblib"))
    assert pt_exists or mlp_exists
