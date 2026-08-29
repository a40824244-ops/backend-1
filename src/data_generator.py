import os
import numpy as np
import pandas as pd

def generate_student_dataset(n_samples: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic dataset for student academic performance prediction
    matching the schema in the specification.
    """
    np.random.seed(seed)
    
    student_ids = np.arange(1001, 1001 + n_samples)
    genders = np.random.choice(["Male", "Female"], size=n_samples, p=[0.49, 0.51])
    ages = np.random.randint(15, 23, size=n_samples)
    
    # Study hours per day (1 to 10)
    study_hours = np.round(np.random.gamma(shape=3.0, scale=1.2, size=n_samples).clip(1.0, 10.0), 1)
    
    # Attendance rate (40% to 100%)
    attendance = np.round(np.random.beta(a=7.0, b=2.0, size=n_samples) * 60 + 40, 1)
    
    # Previous GPA (1.0 to 4.0)
    previous_gpa = np.round(np.random.normal(loc=2.8, scale=0.6, size=n_samples).clip(1.0, 4.0), 2)
    
    # Socioeconomic & environment factors
    internet_access = np.random.choice(["Yes", "No"], size=n_samples, p=[0.85, 0.15])
    parent_education = np.random.choice(
        ["High School", "Bachelor", "Master", "Doctorate", "None"],
        size=n_samples,
        p=[0.30, 0.40, 0.15, 0.05, 0.10]
    )
    family_income = np.random.choice(["Low", "Medium", "High"], size=n_samples, p=[0.30, 0.50, 0.20])
    extra_activities = np.random.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])
    
    # Base academic capacity influenced by study, attendance, previous gpa, parent ed, internet
    parent_bonus = pd.Series(parent_education).map({
        "None": -3.0, "High School": 0.0, "Bachelor": 3.0, "Master": 5.0, "Doctorate": 7.0
    }).values
    internet_bonus = np.where(internet_access == "Yes", 3.0, -2.0)
    income_bonus = pd.Series(family_income).map({"Low": -2.0, "Medium": 1.0, "High": 3.0}).values
    
    base_score = (
        study_hours * 4.5 +
        attendance * 0.35 +
        previous_gpa * 8.0 +
        parent_bonus +
        internet_bonus +
        income_bonus +
        np.random.normal(0, 5, size=n_samples)
    )
    
    assignment_score = np.round((base_score * 0.85 + np.random.normal(0, 6, size=n_samples)).clip(10, 100), 1)
    quiz_score = np.round((base_score * 0.80 + np.random.normal(0, 8, size=n_samples)).clip(10, 100), 1)
    midterm_score = np.round((base_score * 0.90 + np.random.normal(0, 7, size=n_samples)).clip(10, 100), 1)
    
    # Final score is weighted combination of coursework, midterms, study, attendance + final exam effect
    final_score_raw = (
        assignment_score * 0.20 +
        quiz_score * 0.15 +
        midterm_score * 0.25 +
        (study_hours * 2.5) +
        (attendance * 0.20) +
        (previous_gpa * 3.5) +
        np.random.normal(0, 4, size=n_samples)
    )
    final_score = np.round(final_score_raw.clip(0, 100), 1)
    
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Gender": genders,
        "Age": ages,
        "Study_Hours": study_hours,
        "Attendance": attendance,
        "Assignment_Score": assignment_score,
        "Quiz_Score": quiz_score,
        "Midterm_Score": midterm_score,
        "Final_Score": final_score,
        "Internet_Access": internet_access,
        "Parent_Education": parent_education,
        "Family_Income": family_income,
        "Extra_Activities": extra_activities,
        "Previous_GPA": previous_gpa
    })
    
    # Derived target variables
    df["Target"] = np.where(df["Final_Score"] >= 50.0, "Pass", "Fail")
    
    def assign_grade(score):
        if score >= 85:
            return "A"
        elif score >= 75:
            return "B"
        elif score >= 65:
            return "C"
        elif score >= 50:
            return "D"
        else:
            return "F"
            
    df["Grade"] = df["Final_Score"].apply(assign_grade)
    
    def assign_risk(row):
        if row["Final_Score"] < 50 or row["Attendance"] < 60 or row["Midterm_Score"] < 45:
            return "High"
        elif row["Final_Score"] < 65 or row["Attendance"] < 75:
            return "Medium"
        else:
            return "Low"
            
    df["Risk_Level"] = df.apply(assign_risk, axis=1)
    
    return df

def save_default_dataset(output_dir: str = "data") -> str:
    os.makedirs(output_dir, exist_ok=True)
    df = generate_student_dataset()
    file_path = os.path.join(output_dir, "student_data.csv")
    df.to_csv(file_path, index=False)
    print(f"Dataset saved to {file_path} (Shape: {df.shape})")
    return file_path

if __name__ == "__main__":
    save_default_dataset()
