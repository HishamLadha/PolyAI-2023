import pandas as pd
import joblib
from sklearn.metrics import f1_score
from fairlearn.metrics import demographic_parity_difference

# Load the preprocessed data
X_test = pd.read_csv('data/X_test.csv')

# Load the trained model
best_classifier = joblib.load('models/trained_model.pkl')

# Make predictions on the test set
y_pred = best_classifier.predict(X_test).astype(int)  # Convert predictions to integers

# Load the true labels and skip the header
y_test = pd.read_csv('data/y_test.csv', header=None, skiprows=1).values.ravel()

# Assuming data_participant.csv has an order that matches X_test.csv
sensitive_attribute = pd.read_csv('data/data_participant.csv')['SEX'].iloc[:len(y_test)]

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Calculate fairness score (demographic parity difference)
fairness_score = demographic_parity_difference(y_true=y_test, y_pred=y_pred, sensitive_features=sensitive_attribute)  # <--- Fixed this line

# Print the evaluation scores
print(f'F1 Score: {f1}')
print(f'Fairness Score (Demographic Parity Difference): {fairness_score}')
