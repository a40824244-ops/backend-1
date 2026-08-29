import pandas as pd
import numpy as np

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates useful interaction terms, ratios, and aggregated metrics
    to improve machine learning & deep learning model accuracy.
    """
    df_feat = df.copy()
    
    # 1. Attendance x Study Hours interaction
    if "Attendance" in df_feat.columns and "Study_Hours" in df_feat.columns:
        df_feat["Attendance_Study_Interaction"] = df_feat["Attendance"] * df_feat["Study_Hours"]
        
    # 2. Average Quiz & Assignment score
    if "Quiz_Score" in df_feat.columns and "Assignment_Score" in df_feat.columns:
        df_feat["Avg_Quiz_Assignment"] = (df_feat["Quiz_Score"] + df_feat["Assignment_Score"]) / 2.0
        
    # 3. Coursework Average
    if all(col in df_feat.columns for col in ["Assignment_Score", "Quiz_Score", "Midterm_Score"]):
        df_feat["Coursework_Average"] = (df_feat["Assignment_Score"] + df_feat["Quiz_Score"] + df_feat["Midterm_Score"]) / 3.0
        
    # 4. Score Improvement (Midterm vs early coursework average)
    if "Midterm_Score" in df_feat.columns and "Avg_Quiz_Assignment" in df_feat.columns:
        df_feat["Score_Improvement"] = df_feat["Midterm_Score"] - df_feat["Avg_Quiz_Assignment"]
        
    # 5. Previous GPA to Midterm ratio
    if "Previous_GPA" in df_feat.columns and "Midterm_Score" in df_feat.columns:
        # Scale GPA (1-4) to 100 scale for comparison ratio
        df_feat["GPA_Midterm_Ratio"] = (df_feat["Previous_GPA"] * 25.0) / (df_feat["Midterm_Score"] + 1e-5)
        
    return df_feat
