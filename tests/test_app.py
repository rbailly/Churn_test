import pytest
from app import app

@pytest.fixture
def client():
    """Fixture pour initialiser le client Flask en mode test"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    """Vérifie que la route principale (/) retourne un statut 200"""
    response = client.get('/')
    assert response.status_code == 200, "La route principale (/) ne retourne pas un statut 200."

def test_predict_route_valid(client):
    """Vérifie que la route /predict retourne une prédiction pour des données valides"""
    data = {
        'Age': 45,
        'Account_Manager': 1,
        'Years': 5,
        'Num_Sites': 3
    }
    # Envoyer les données en tant que formulaire encodé
    response = client.post('/predict', data=data)
    assert response.status_code == 200, "La route /predict ne retourne pas un statut 200."

    # Vérifiez que la réponse contient une clé 'prédiction'
    json_data = response.get_json()
    assert 'prédiction' in json_data, "La réponse ne contient pas de clé 'prédiction'."


    # Vérifiez que la réponse contient une clé 'prédiction'
    json_data = response.get_json()
    assert 'prédiction' in json_data, "La réponse ne contient pas de clé 'prédiction'."