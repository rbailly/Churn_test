# image de python légère
FROM python:3.8-slim

#Répertoire de travail dans le conteneur
WORKDIR /app

#Copie des requirements
COPY requirements.txt requirements.txt

#Installation des dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier les fichiers nécessaires dans l'image Docker
COPY train.py train.py
COPY validate.py validate.py
COPY app.py app.py
COPY data data
COPY tests tests
COPY templates/ templates/

# Exposer le port utilisé par Flask
EXPOSE 5001

# Exécuter le script pour entraîner le modèle et générer churn_model_clean.pkl
RUN python train.py

ENV PYTHONPATH=/app

# Commande pour démarrer l'application Flask
CMD ["python", "app.py"]
