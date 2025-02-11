import requests

# URL de votre API (ajustez si nécessaire)
url = "http://127.0.0.1:5000/predict"

# Données de test
data = {
    "Age": 70,
    "Account_Manager": 0,
    "Years": 20,
    "Num_Sites": 23
}

# Envoi de la requête POST
try:
    response = requests.post(url, data=data)
    response.raise_for_status()  # Vérifie s'il y a des erreurs HTTP
    print("Réponse de l'API :", response.json())
except requests.exceptions.RequestException as e:
    print("Erreur lors de la requête :", e)