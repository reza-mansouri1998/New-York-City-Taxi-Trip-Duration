import pytest
from app import app

@pytest.fixture
def client():
    # Configure Flask for testing
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    """Test that the home page loads successfully."""
    response = client.get('/')
    assert response.status_code == 200
    assert b"NYC Taxi Predictor" in response.data

def test_predict_page_get(client):
    """Test that the prediction form loads successfully."""
    response = client.get('/predict')
    assert response.status_code == 200
    assert b"Trip Details" in response.data