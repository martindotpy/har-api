from fastapi import APIRouter

from model.har_model import HarRequest, HarResponse
from service.har_service import predict_har

har_router = APIRouter(tags=["har"])


@har_router.post("/har", summary="Get HAR data")
def post_predict_har(har_data: HarRequest) -> HarResponse:
    """Predict the type of HAR activity based on accelerometer readings."""
    return predict_har(har_data)
