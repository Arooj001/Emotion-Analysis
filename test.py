from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_positive_sentiment():
    response = client.post("/analyze/", json={"text": "I'm thrilled with the results of this service!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "joy"}

def test_negative_sentiment():
    response = client.post("/analyze/", json={"text": "What a disappointment! It didn't meet my expectations at all."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "anger"}

def test_neutral_sentiment():
    response = client.post("/analyze/", json={"text": "The product does exactly what it promises. Nothing more, nothing less."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "neutral"}

def test_anger_sentiment():
    response = client.post("/analyze/", json={"text": "This app is frustrating and annoying to use!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "anger"}

def test_disgust_sentiment():
    response = client.post("/analyze/", json={"text": "The taste of this food is disgusting!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "disgust"}

def test_fear_sentiment():
    response = client.post("/analyze/", json={"text": "I'm worried that this might be unsafe."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "fear"}

def test_sadness_sentiment():
    response = client.post("/analyze/", json={"text": "It's sad to see it perform so poorly."})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "sadness"}

def test_surprise_sentiment():
    response = client.post("/analyze/", json={"text": "Wow, I didn't expect it to be this good!"})
    assert response.status_code == 200
    assert response.json() == {"sentiment": "surprise"}
