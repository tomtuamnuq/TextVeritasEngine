import os
import joblib
import logging
from typing import Dict, Any, Union
import pandas as pd
from flask import Flask, request, jsonify
from werkzeug.exceptions import BadRequest
from utils.namings import MODEL_FILENAME
from utils.text_processing import preprocess_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)


class InferenceError(Exception):
    """Custom exception for inference-related errors"""

    pass


class ModelService:
    def __init__(self, model_dir: str):
        """
        Initialize the model service.

        Args:
            model_dir: Directory containing the model file
        """
        self.model = self._load_model(model_dir)

    def _load_model(self, model_dir: str) -> Any:
        """
        Load the model from disk.

        Args:
            model_dir: Directory containing the model file

        Returns:
            Loaded model object

        Raises:
            InferenceError: If model loading fails
        """
        try:
            model_path = os.path.join(model_dir, MODEL_FILENAME)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}")

            logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise InferenceError(f"Failed to load model: {str(e)}")

    def validate_input(self, input_data: Dict[str, Any]) -> None:
        """
        Validate input data structure and content.

        Args:
            input_data: Dictionary containing input data

        Raises:
            ValueError: If input validation fails
        """
        required_fields = ["title", "text"]

        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")
            if not isinstance(input_data[field], str):
                raise ValueError(f"Field {field} must be a string")
            if not input_data[field].strip():
                raise ValueError(f"Field {field} cannot be empty")

    def preprocess(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data.

        Args:
            input_data: Dictionary containing input data

        Returns:
            Preprocessed DataFrame ready for inference
        """
        df = pd.DataFrame([input_data])
        return preprocess_dataset(df)

    def predict(
        self, input_data: pd.DataFrame
    ) -> Dict[str, Union[bool, Dict[str, float]]]:
        """
        Generate prediction from preprocessed input.

        Args:
            input_data: Preprocessed input DataFrame

        Returns:
            Dictionary containing prediction and probabilities
        """
        try:
            probability = self.model.predict_proba(input_data)[0]
            prediction = probability[1] >= 0.5

            result = {
                "fake": int(prediction),
                "probability": {
                    "fake": float(probability[1]),
                    "real": float(probability[0]),
                },
            }

            logger.info(f"Prediction generated successfully: {result}")
            return result

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise InferenceError(f"Failed to generate prediction: {str(e)}")


# Initialize model service
model_service = ModelService(model_dir="/opt/ml/model")


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    """Handle bad request errors."""
    return jsonify(error=str(e)), 400


@app.errorhandler(InferenceError)
def handle_inference_error(e):
    """Handle inference-specific errors."""
    return jsonify(error=str(e)), 500


@app.errorhandler(Exception)
def handle_general_error(e):
    """Handle general errors."""
    logger.error(f"Unexpected error: {str(e)}")
    return jsonify(error="Internal server error"), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(status="healthy"), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Prediction endpoint.

    Expected JSON input:
    {
        "title": "News article title",
        "text": "News article content"
    }
    """
    try:
        if not request.is_json:
            raise BadRequest("Content-Type must be application/json")

        input_data = request.get_json()

        # Validate input
        model_service.validate_input(input_data)

        # Preprocess input
        preprocessed_data = model_service.preprocess(input_data)

        # Generate prediction
        prediction = model_service.predict(preprocessed_data)

        return jsonify(prediction)

    except ValueError as e:
        raise BadRequest(str(e))


if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("PORT", 8080))

    # Run Flask app
    app.run(host="0.0.0.0", port=port)
