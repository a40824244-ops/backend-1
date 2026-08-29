import os
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any

CATEGORICAL_COLS = ["Gender", "Internet_Access", "Parent_Education", "Family_Income", "Extra_Activities"]
NUMERICAL_COLS = ["Age", "Study_Hours", "Attendance", "Assignment_Score", "Quiz_Score", "Midterm_Score", "Previous_GPA"]

COLUMN_ALIASES = {
    "gender": "Gender", "sex": "Gender",
    "age": "Age",
    "study_hours": "Study_Hours", "studyhours": "Study_Hours", "study hours": "Study_Hours",
    "attendance": "Attendance", "attendance_rate": "Attendance", "attendance %": "Attendance",
    "assignment_score": "Assignment_Score", "assignment score": "Assignment_Score", "assignments": "Assignment_Score",
    "quiz_score": "Quiz_Score", "quiz score": "Quiz_Score", "quizzes": "Quiz_Score",
    "midterm_score": "Midterm_Score", "midterm score": "Midterm_Score", "midterm": "Midterm_Score",
    "internet_access": "Internet_Access", "internet access": "Internet_Access",
    "parent_education": "Parent_Education", "parent education": "Parent_Education",
    "family_income": "Family_Income", "family income": "Family_Income",
    "extra_activities": "Extra_Activities", "extracurricular": "Extra_Activities", "extracurriculars": "Extra_Activities",
    "previous_gpa": "Previous_GPA", "previous gpa": "Previous_GPA", "gpa": "Previous_GPA"
}

DEFAULT_VALUES = {
    "Gender": "Female",
    "Age": 18,
    "Study_Hours": 5.0,
    "Attendance": 80.0,
    "Assignment_Score": 70.0,
    "Quiz_Score": 70.0,
    "Midterm_Score": 70.0,
    "Internet_Access": "Yes",
    "Parent_Education": "High School",
    "Family_Income": "Medium",
    "Extra_Activities": "Yes",
    "Previous_GPA": 3.0
}

class DataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoders: Dict[str, LabelEncoder] = {}
        self.is_fitted = False

    @staticmethod
    def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names by stripping spaces and matching aliases."""
        df_norm = df.copy()
        new_cols = {}
        for col in df_norm.columns:
            clean_name = str(col).strip()
            lower_name = clean_name.lower().replace("-", "_")
            if lower_name in COLUMN_ALIASES:
                new_cols[col] = COLUMN_ALIASES[lower_name]
            else:
                new_cols[col] = clean_name
        return df_norm.rename(columns=new_cols)

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Validates dataframe schema and range boundaries."""
        df_norm = self.normalize_columns(df)
        required = set(NUMERICAL_COLS + CATEGORICAL_COLS)
        missing = required - set(df_norm.columns)
        if missing:
            return False, f"Missing required columns: {sorted(list(missing))}"
        return True, "Validation successful."

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes duplicates, normalizes columns, fills missing columns/values."""
        df_clean = self.normalize_columns(df)
        df_clean = df_clean.drop_duplicates()
        
        # Ensure all required input columns exist with default fallbacks
        for col, default_val in DEFAULT_VALUES.items():
            if col not in df_clean.columns:
                df_clean[col] = default_val

        # Fill numerical missing with median or default
        for col in NUMERICAL_COLS:
            if col in df_clean.columns and df_clean[col].isnull().any():
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val if pd.notnull(median_val) else DEFAULT_VALUES[col])
                
        # Fill categorical missing with mode or default
        for col in CATEGORICAL_COLS:
            if col in df_clean.columns and df_clean[col].isnull().any():
                mode_vals = df_clean[col].mode()
                df_clean[col] = df_clean[col].fillna(mode_vals[0] if len(mode_vals) > 0 else DEFAULT_VALUES[col])
                
        return df_clean

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits encoders and scaler, returns transformed features dataframe."""
        df_proc = self.clean_data(df)
        
        # Fit LabelEncoders for categorical features
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df_proc[col] = le.fit_transform(df_proc[col].astype(str))
            self.encoders[col] = le
            
        # Fit StandardScaler for numerical features
        df_proc[NUMERICAL_COLS] = self.scaler.fit_transform(df_proc[NUMERICAL_COLS])
        
        self.is_fitted = True
        return df_proc

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms new input data using fitted encoders and scaler."""
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet!")
            
        df_proc = self.clean_data(df)
        
        for col in CATEGORICAL_COLS:
            if col in df_proc.columns:
                le = self.encoders[col]
                # Handle unseen categories gracefully
                df_proc[col] = df_proc[col].astype(str).map(
                    lambda s: le.transform([s])[0] if s in le.classes_ else 0
                )
                
        if set(NUMERICAL_COLS).issubset(df_proc.columns):
            df_proc[NUMERICAL_COLS] = self.scaler.transform(df_proc[NUMERICAL_COLS])
            
        return df_proc

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @staticmethod
    def load(filepath: str) -> "DataPreprocessor":
        return joblib.load(filepath)
