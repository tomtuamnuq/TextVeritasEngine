import os
import joblib
import json
import pandas as pd

from utils.text_processing import preprocess_dataset


def model_fn(model_dir):
    """
    Load model when container starts.
    Called once when the container starts up.
    """
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and preprocess input data.
    Called for each prediction request.
    """
    if request_content_type == "application/json":
        # Parse JSON
        input_data = json.loads(request_body)

        # Create DataFrame with required columns
        df = pd.DataFrame([input_data])

        # Apply same preprocessing as during training
        return preprocess_dataset(df)
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Generate prediction from preprocessed input.
    Called after input_fn with its return value.
    """
    probability = model.predict_proba(input_data)[0]
    prediction = True if probability[1] >= 0.5 else False

    return {
        "fake": prediction,
        "probability": {"fake": float(probability[1]), "real": float(probability[0])},
    }


def output_fn(prediction, response_content_type):
    """
    Serialize prediction output.
    Called with the return value from predict_fn.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    raise ValueError(f"Unsupported content type: {response_content_type}")
