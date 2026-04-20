from fastapi import APIRouter, Request
from app.models.predict_model import PredictRequest, PredictResponse
from app.services.predict_service import PredictService

router = APIRouter()

# json接口
@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    return await PredictService.get_instance().predict(req)

#二进制接口
@router.post("/predict_binary")
async def predict_binary(request: Request):
    body = await request.body()
    return await PredictService.get_instance().predict_binary(body)