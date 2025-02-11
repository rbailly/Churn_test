
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

def test_train_model_file_exists():
    """Vérifie que le fichier logreg_modelpkl est créé après exécution de train.py"""
    assert os.path.exists('data/logreg_model.pkl'), (
        "Le fichier logreg_model.pkl n'existe pas après l'exécution de train.py."
    )

def test_train_model_loading():
    """Vérifie que le fichier sauvegardé contient un modèle Random Forest"""
    model = joblib.load('data/logreg_model.pkl')
    assert isinstance(model, LogisticRegression), (
        "Le fichier logreg_model.pkl ne contient pas un modèle LogisticRegression."
    )

def test_train_model_prediction():
    """Vérifie que le modèle entraîné peut prédire sur un sous-ensemble des données"""
    model = joblib.load('data/logreg_model.pkl')
    data = pd.read_csv('data/customer_churn.csv')
    X = data[['Age', 'Years', 'Account_Manager', 'Num_Sites']]

    prediction = model.predict(X[:1])
    assert prediction is not None, "Le modèle n'a pas retourné de prédiction."
