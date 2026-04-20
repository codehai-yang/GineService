from pydantic import BaseModel
from typing import List

class PredictRequest(BaseModel):
    x:          List[List[float]]
    edge_index: List[List[int]]
    edge_attr:  List[List[float]]

class PredictResponse(BaseModel):
    predicted_cost: float
    elapsed_ms:     float