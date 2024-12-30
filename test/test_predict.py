import requests

# Health check
response = requests.get("http://localhost:8080/health")
print(response.json())

# Make prediction
data = {"title": "Your news title", "text": "Your news content"}

response = requests.post(
    "http://localhost:8080/predict",
    json=data,
    headers={"Content-Type": "application/json"},
)

prediction = response.json()
print(prediction)
