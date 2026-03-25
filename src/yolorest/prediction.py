from pydantic import BaseModel


class Prediction(BaseModel):
    label: str
    confidence: float
    y_min: float
    x_min: float
    y_max: float
    x_max: float


class Predictions(BaseModel):
    predictions: list[Prediction]
    success: bool
