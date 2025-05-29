from pathlib import Path

import joblib
from sklearn.neural_network import MLPClassifier

from model.har_model import HarRequest, HarResponse, HarType

build_path = Path.cwd() / "build"
model: MLPClassifier = joblib.load(build_path / "mlp_model.pkl")


def predict_har(har_data: HarRequest) -> HarResponse:
    """Predict the type of HAR activity based on accelerometer readings."""
    features = [
        har_data.back_x,
        har_data.back_y,
        har_data.back_z,
        har_data.thigh_x,
        har_data.thigh_y,
        har_data.thigh_z,
    ]

    prediction = model.predict([features])

    return HarResponse(type=HarType(prediction[0]))
