import sys

sys.path.append("/opt/ml/code")
import inference
import json


def test_inference():
    # Load model
    model = inference.model_fn("/opt/ml/model")

    # Test input
    test_input = {"title": "Breaking news headline", "text": "Article content here"}

    # Run inference pipeline
    processed = inference.input_fn(json.dumps(test_input), "application/json")
    prediction = inference.predict_fn(processed, model)
    result = inference.output_fn(prediction, "application/json")
    print(f"Prediction: {result}")


if __name__ == "__main__":
    test_inference()
