from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class StudentInputSchema(BaseModel):
    Gender: str = Field(..., json_schema_extra={"example": "Female"})
    Age: int = Field(..., ge=10, le=50, json_schema_extra={"example": 18})
    Study_Hours: float = Field(..., ge=0.0, le=24.0, json_schema_extra={"example": 6.5})
    Attendance: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 85.0})
    Assignment_Score: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 78.0})
    Quiz_Score: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 82.0})
    Midterm_Score: float = Field(..., ge=0.0, le=100.0, json_schema_extra={"example": 75.0})
    Internet_Access: str = Field(..., json_schema_extra={"example": "Yes"})
    Parent_Education: str = Field(..., json_schema_extra={"example": "Bachelor"})
    Family_Income: str = Field(..., json_schema_extra={"example": "Medium"})
    Extra_Activities: str = Field(..., json_schema_extra={"example": "Yes"})
    Previous_GPA: float = Field(..., ge=0.0, le=4.0, json_schema_extra={"example": 3.2})


class PredictionResponseSchema(BaseModel):
    prediction_status: str
    pass_probability: float
    predicted_target: str
    predicted_final_score: float
    predicted_grade: str
    risk_level: str
    risk_color: str
    recommendations: List[str]


class BatchPredictionRequestSchema(BaseModel):
    students: List[StudentInputSchema]


class BatchPredictionResponseSchema(BaseModel):
    total_processed: int
    predictions: List[PredictionResponseSchema]


class RetrainResponseSchema(BaseModel):
    status: str
    message: str
    ml_results: Dict[str, Any]
    dl_results: Optional[Dict[str, Any]] = None
