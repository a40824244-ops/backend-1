import os
import joblib
import pandas as pd
import numpy as np

def get_feature_importances(models_dir: str = "models/saved_models") -> pd.DataFrame:
    """Extracts normalized feature importance rankings from saved ML models."""
    cls_path = os.path.join(models_dir, "best_classification_model.joblib")
    features_path = os.path.join(models_dir, "feature_columns.joblib")
    
    if not (os.path.exists(cls_path) and os.path.exists(features_path)):
        return pd.DataFrame(columns=["Feature", "Importance"])
        
    model = joblib.load(cls_path)
    feature_names = joblib.load(features_path)
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)
        
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    return df_imp


def compute_student_shap_breakdown(student_features_df: pd.DataFrame, models_dir: str = "models/saved_models") -> pd.DataFrame:
    """
    Provides a per-feature impact breakdown for a specific student prediction.
    """
    df_imp = get_feature_importances(models_dir)
    if df_imp.empty or student_features_df.empty:
        return pd.DataFrame()
        
    # Combine feature importance weight with standardized deviation
    row = student_features_df.iloc[0]
    breakdown = []
    
    for idx, f_row in df_imp.iterrows():
        feat = f_row["Feature"]
        val = row.get(feat, 0.0)
        weight = f_row["Importance"]
        
        # Approximate direction & magnitude of influence
        impact = val * weight
        direction = "Positive (+)" if impact >= 0 else "Negative (-)"
        
        breakdown.append({
            "Feature": feat,
            "Raw Value": round(val, 2),
            "Importance Weight": round(weight, 4),
            "Impact Direction": direction
        })
        
    return pd.DataFrame(breakdown)
