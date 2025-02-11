from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

import pandas as pd
dataset = pd.read_csv('data/customer_churn.csv')

# prompt: Code pour diviser les données en données de test et données d'entrainement avec une répartition 20 / 80. Utiliser uniquement les variables Age, Years, Num_Site, et State, Account_manager. Convertir State en integer au préalable. La cible est Churn

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming 'dataset' is your DataFrame and already loaded
# If not, replace this with: dataset = pd.read_csv('customer_churn.csv')


# Select features and target variable
features = ['Age', 'Years', 'Account_Manager','Num_Sites']
target = 'Churn'

X = dataset[features]
y = dataset[target]

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # set random_state for reproducibility

# Initialize and train the Logistic Regression model using scikit-learn
logreg_sklearn = LogisticRegression()  # You can adjust hyperparameters here
logreg_sklearn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = logreg_sklearn.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)


# Save the trained model using joblib
#
joblib.dump(logreg_sklearn, 'data/logreg_model.pkl')

