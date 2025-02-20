from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np 

##Charger le modèle
model = joblib.load('data/logreg_model.pkl')

# Initialiser l'application Flask
app = Flask(__name__)

# Route pour afficher la page HTML
@app.route('/')
def index():
    return render_template('index.html')

#Route pour les prédictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        #Récupération des données du formulaire
        Age = float(request.form['Age'])
        Account_Manager = int(request.form['Account_Manager'])
        Years = float(request.form['Years'])
        Num_Sites = int(request.form['Num_Sites'])

        #Créer le tableau de données
        features = np.array([[Age,Years,Account_Manager,Num_Sites]])

        #faire la prédiction
        prediction = model.predict(features)
        result = int(prediction[0])

        #renvoyer la réponse sous la forme de json
        return jsonify({'prédiction' : result}) 
    except Exception as e:
        return jsonify({'error: ' : str(e)})

    # run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)