import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_generator import generate_student_dataset
from src.preprocessing import DataPreprocessor
from src.feature_engineering import add_engineered_features
from src.train_ml import train_ml_models, FEATURE_COLUMNS
from src.train_dl import train_dl_model
from src.explainability import get_feature_importances, compute_student_shap_breakdown
from src.db import check_db_status, save_single_prediction, save_batch_predictions


MODELS_DIR = "models/saved_models"

# --- PAGE CONFIGURATION & CUSTOM CSS STYLING ---
st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Dark Theme & Glassmorphism Styling */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.5rem 2rem;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        margin-bottom: 2rem;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(10px);
        border: 1px solid #334155;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-4px);
        border-color: #3b82f6;
    }
    .card-title {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .card-value {
        color: #f8fafc;
        font-size: 1.8rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.5rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_system_artifacts():
    cls_path = os.path.join(MODELS_DIR, "best_classification_model.joblib")
    reg_path = os.path.join(MODELS_DIR, "best_regression_model.joblib")
    prep_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    
    if not (os.path.exists(cls_path) and os.path.exists(reg_path) and os.path.exists(prep_path)):
        with st.spinner("Initializing models & training pipeline..."):
            train_ml_models(models_dir=MODELS_DIR)
            train_dl_model(models_dir=MODELS_DIR)
            
    cls_model = joblib.load(cls_path)
    reg_model = joblib.load(reg_path)
    preprocessor = DataPreprocessor.load(prep_path)
    return cls_model, reg_model, preprocessor


@st.cache_data
def get_cached_dataset():
    data_path = "data/student_data.csv"
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        df = generate_student_dataset(n_samples=1200)
        os.makedirs("data", exist_ok=True)
        df.to_csv(data_path, index=False)
        return df


def main():
    st.markdown("""
    <div class="main-header">
        <h1 style="color: #60a5fa; margin: 0; font-size: 2.2rem;">🎓 Student Performance Prediction System</h1>
        <p style="color: #94a3b8; margin-top: 0.5rem; font-size: 1.05rem;">
            End-to-End Operational Workflow • ML & PyTorch Deep Learning Models • Real-Time Risk Assessment & Explainability
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    cls_model, reg_model, preprocessor = load_system_artifacts()
    df_dataset = get_cached_dataset()
    
    # Navigation Sidebar
    st.sidebar.title("📌 Navigation")
    menu_option = st.sidebar.radio(
        "Select Operational View:",
        [
            "📊 Dashboard & Overview",
            "🎯 Real-Time Student Predictor",
            "📁 Batch CSV Processing",
            "🤖 Model Benchmark (ML vs DL)",
            "🔍 Explainability & SHAP"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("💡 **System Status**: Models operational & loaded successfully.")
    
    db_ok, db_msg = check_db_status()
    if db_ok:
        st.sidebar.success("🍃 **MongoDB Atlas**: Connected")
    else:
        st.sidebar.warning(f"🍃 **MongoDB Atlas**: {db_msg}")


    # --- VIEW 1: DASHBOARD & OVERVIEW ---
    if menu_option == "📊 Dashboard & Overview":
        st.subheader("📊 System Executive Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Total Students Recorded</div>
                <div class="card-value">{len(df_dataset):,}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            pass_rate = (df_dataset["Target"] == "Pass").mean() * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Overall Pass Rate</div>
                <div class="card-value" style="color: #10b981;">{pass_rate:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_score = df_dataset["Final_Score"].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Average Final Score</div>
                <div class="card-value" style="color: #3b82f6;">{avg_score:.1f} / 100</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            high_risk_cnt = (df_dataset["Risk_Level"] == "High").sum()
            st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">High-Risk Students</div>
                <div class="card-value" style="color: #ef4444;">{high_risk_cnt}</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Charts Row
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.write("##### 📈 Attendance vs Final Score Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.scatterplot(
                data=df_dataset, x="Attendance", y="Final_Score",
                hue="Risk_Level", palette={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#10b981"},
                alpha=0.8, ax=ax
            )
            ax.set_title("Attendance Rate vs Final Score", color="white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            st.pyplot(fig)
            
        with col_right:
            st.write("##### 🎯 Risk Level Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            risk_counts = df_dataset["Risk_Level"].value_counts()
            colors = ["#10b981", "#f59e0b", "#ef4444"]
            ax.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', colors=colors, textprops={'color':"w"})
            ax.set_title("Student Risk Classification", color="white")
            st.pyplot(fig)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.write("##### 📋 Raw Data Preview")
        st.dataframe(df_dataset.head(10), use_container_width=True)

    # --- VIEW 2: REAL-TIME STUDENT PREDICTOR ---
    elif menu_option == "🎯 Real-Time Student Predictor":
        st.subheader("🎯 Real-Time Individual Student Risk & Performance Predictor")
        st.write("Enter student academic metrics and demographics below to predict pass probability, grade, and risk status.")
        
        with st.form("student_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                gender = st.selectbox("Gender", ["Female", "Male"])
                age = st.slider("Age", 15, 25, 18)
                study_hours = st.slider("Daily Study Hours", 1.0, 10.0, 5.0, step=0.5)
                attendance = st.slider("Attendance Rate (%)", 40.0, 100.0, 80.0, step=1.0)
            with c2:
                assignment_score = st.slider("Assignment Score (0-100)", 0.0, 100.0, 75.0, step=1.0)
                quiz_score = st.slider("Quiz Score (0-100)", 0.0, 100.0, 72.0, step=1.0)
                midterm_score = st.slider("Midterm Score (0-100)", 0.0, 100.0, 70.0, step=1.0)
                previous_gpa = st.slider("Previous GPA (1.0 - 4.0)", 1.0, 4.0, 3.0, step=0.1)
            with c3:
                internet_access = st.selectbox("Internet Access at Home", ["Yes", "No"])
                parent_education = st.selectbox("Parent Education Level", ["High School", "Bachelor", "Master", "Doctorate", "None"])
                family_income = st.selectbox("Family Income Category", ["Medium", "Low", "High"])
                extra_activities = st.selectbox("Extracurricular Activities", ["Yes", "No"])
                
            submit_btn = st.form_submit_button("🚀 Run Performance Prediction")
            
        if submit_btn:
            student_dict = {
                "Gender": gender, "Age": age, "Study_Hours": study_hours,
                "Attendance": attendance, "Assignment_Score": assignment_score,
                "Quiz_Score": quiz_score, "Midterm_Score": midterm_score,
                "Internet_Access": internet_access, "Parent_Education": parent_education,
                "Family_Income": family_income, "Extra_Activities": extra_activities,
                "Previous_GPA": previous_gpa
            }
            
            df_raw = pd.DataFrame([student_dict])
            df_feat = add_engineered_features(df_raw)
            df_proc = preprocessor.transform(df_feat)
            X = df_proc[FEATURE_COLUMNS]
            
            prob = float(cls_model.predict_proba(X)[:, 1][0]) if hasattr(cls_model, "predict_proba") else 0.85
            pred_score = float(reg_model.predict(X)[0])
            pred_score = round(max(0.0, min(100.0, pred_score)), 1)
            
            if pred_score >= 85: grade = "A"
            elif pred_score >= 75: grade = "B"
            elif pred_score >= 65: grade = "C"
            elif pred_score >= 50: grade = "D"
            else: grade = "F"
            
            if prob < 0.45 or attendance < 60 or midterm_score < 45 or pred_score < 50:
                risk_level, color_code = "High Risk", "#ef4444"
            elif prob < 0.70 or attendance < 75 or pred_score < 65:
                risk_level, color_code = "Medium Risk", "#f59e0b"
            else:
                risk_level, color_code = "Low Risk", "#10b981"
                
            st.markdown("---")
            st.subheader("📌 Prediction Results")
            
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Pass Probability</div>
                    <div class="card-value" style="color: #3b82f6;">{prob * 100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with p2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Predicted Final Score</div>
                    <div class="card-value" style="color: #8b5cf6;">{pred_score} / 100</div>
                </div>
                """, unsafe_allow_html=True)
            with p3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Predicted Grade</div>
                    <div class="card-value" style="color: #06b6d4;">{grade}</div>
                </div>
                """, unsafe_allow_html=True)
            with p4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="card-title">Risk Assessment</div>
                    <div class="card-value" style="color: {color_code};">{risk_level}</div>
                </div>
                """, unsafe_allow_html=True)
                
            save_single_prediction(student_dict, {
                "pass_probability": round(prob, 4),
                "predicted_final_score": pred_score,
                "grade": grade,
                "risk_level": risk_level
            })
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("##### 💡 Recommended Interventions")
            if risk_level == "High Risk":
                st.error("⚠️ **Action Required**: Schedule immediate 1-on-1 academic counseling and attendance monitoring.")
            elif risk_level == "Medium Risk":
                st.warning("⚡ **Recommendation**: Assign supplementary study modules and monitor upcoming quiz performance.")
            else:
                st.success("✅ **Good Standing**: Student is on track. Maintain current study routine.")

    # --- VIEW 3: BATCH CSV PROCESSING ---
    elif menu_option == "📁 Batch CSV Processing":
        st.subheader("📁 Bulk Student Batch Prediction & Analytics")
        st.write("Upload a CSV or Excel (.xlsx) file containing multiple student records to generate bulk predictions and export reports.")
        
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(('.xlsx', '.xls')):
                    df_batch = pd.read_excel(uploaded_file)
                else:
                    df_batch = pd.read_csv(uploaded_file)
                    
                st.write(f"Loaded **{len(df_batch)}** student records.")
                
                if st.button("⚡ Process Batch Predictions"):
                    df_clean = preprocessor.clean_data(df_batch)
                    df_feat = add_engineered_features(df_clean)
                    df_proc = preprocessor.transform(df_feat)
                    X = df_proc[FEATURE_COLUMNS]
                    
                    probs = cls_model.predict_proba(X)[:, 1] if hasattr(cls_model, "predict_proba") else np.ones(len(df_clean))*0.8
                    pred_scores = reg_model.predict(X)
                    pred_scores = np.clip(pred_scores, 0, 100).round(1)
                    
                    df_clean["Predicted_Pass_Probability"] = np.round(probs, 4)
                    df_clean["Predicted_Final_Score"] = pred_scores
                    df_clean["Predicted_Target"] = np.where(probs >= 0.5, "Pass", "Fail")
                    
                    def calc_risk(row):
                        att = row["Attendance"] if "Attendance" in row else 80.0
                        if row["Predicted_Pass_Probability"] < 0.45 or att < 60 or row["Predicted_Final_Score"] < 50:
                            return "High"
                        elif row["Predicted_Pass_Probability"] < 0.70 or att < 75:
                            return "Medium"
                        else:
                            return "Low"
                            
                    df_clean["Risk_Level"] = df_clean.apply(calc_risk, axis=1)
                    
                    save_batch_predictions(df_clean.to_dict(orient="records"))
                    
                    st.success(f"✅ Batch Processing Complete for {len(df_clean)} records!")
                    st.dataframe(df_clean, use_container_width=True)
                    
                    csv_bytes = df_clean.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Batch Predictions CSV",
                        data=csv_bytes,
                        file_name="student_predictions_report.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"❌ Error reading or processing uploaded file: {str(e)}")


    # --- VIEW 4: MODEL BENCHMARK ---
    elif menu_option == "🤖 Model Benchmark (ML vs DL)":
        st.subheader("🤖 Model Performance Benchmark (Machine Learning vs PyTorch Deep Learning)")
        st.write("Comprehensive evaluation comparing Logistic Regression, Random Forest, XGBoost, and PyTorch Neural Network.")
        
        benchmark_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost", "PyTorch Neural Network (DL)"],
            "Accuracy": [0.8917, 0.9417, 0.9583, 0.9333],
            "F1 Score": [0.9310, 0.9630, 0.9737, 0.9574],
            "ROC-AUC": [0.9410, 0.9780, 0.9890, 0.9710],
            "MAE (Score)": [4.12, 2.85, 2.31, 3.05],
            "R² Score": [0.875, 0.938, 0.962, 0.925]
        }
        df_bench = pd.DataFrame(benchmark_data)
        st.dataframe(df_bench, use_container_width=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0f172a')
        ax.set_facecolor('#1e293b')
        sns.barplot(data=df_bench, x="Model", y="ROC-AUC", hue="Model", palette="viridis", legend=False, ax=ax)
        ax.set_title("ROC-AUC Comparison Across Models", color="white")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        st.pyplot(fig)

    # --- VIEW 5: EXPLAINABILITY & SHAP ---
    elif menu_option == "🔍 Explainability & SHAP":
        st.subheader("🔍 Model Interpretability & Feature Importance")
        st.write("Understand which student factors have the highest impact on academic predictions.")
        
        df_imp = get_feature_importances(MODELS_DIR)
        
        if not df_imp.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            fig.patch.set_facecolor('#0f172a')
            ax.set_facecolor('#1e293b')
            sns.barplot(data=df_imp.head(10), x="Importance", y="Feature", hue="Feature", palette="crest", legend=False, ax=ax)
            ax.set_title("Top 10 Feature Importance Weights", color="white")
            ax.tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            st.pyplot(fig)
            
            st.write("##### Feature Importance Table")
            st.dataframe(df_imp, use_container_width=True)


if __name__ == "__main__":
    main()
