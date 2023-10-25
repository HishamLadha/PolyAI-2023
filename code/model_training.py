import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV

# Load the preprocessed data while skipping the header row
X_train = pd.read_csv('data/X_train.csv', header=None, skiprows=1)
y_train = pd.read_csv('data/y_train.csv', header=None, skiprows=1)

# Convert y_train to a NumPy array
y_train = np.array(y_train)

# Add the code to check the number of samples
print("Number of samples in X_train:", X_train.shape[0])
print("Number of samples in y_train:", y_train.shape[0])

print("X_train head:")
print(X_train.head())

print("\ny_train head:")
print(y_train[:5])  # Print the first 5 elements of y_train

# Initialize the classifier (you can choose a different one)
classifier = RandomForestClassifier(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
}

# Define custom scorer for F1 score
scorer = make_scorer(f1_score)

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(classifier, param_grid, scoring=scorer, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train.ravel())  # Use ravel() to flatten y_train

# Get the best model from the grid search
best_classifier = grid_search.best_estimator_

# Save the best model
joblib.dump(best_classifier, 'models/trained_model.pkl')
