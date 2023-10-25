import pandas as pd
import joblib

# Load the preprocessed data (including rows with NaN in target)
data = pd.read_csv('data/data_participant.csv')

# Load the label encoders for categorical variables
label_encoders = joblib.load('models/label_encoders.pkl')

# Encode categorical variables
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in categorical_columns:
    le = label_encoders[col]
    data[col] = le.transform(data[col])

print(f"Total number of rows with NaN values: {data['default-payment-next-month'].isna().sum()}")

# Separate rows with non-NaN and NaN values in the target column
data_with_labels = data[data['default-payment-next-month'].notna()]
data_to_predict = data[data['default-payment-next-month'].isna()]

# If there are no rows with NaN values in the target column, then exit
if data_to_predict.shape[0] == 0:
    print("There are no rows with NaN values in the target column. Exiting...")
    exit()

# Load the trained model
best_classifier = joblib.load('models/trained_model.pkl')

# Predict the NaN values
X_to_predict = data_to_predict.drop(['ID', 'SEX','default-payment-next-month'], axis=1)
predicted_values = best_classifier.predict(X_to_predict).astype(int)


# Update the original dataset with predicted values
data_to_predict['default-payment-next-month'] = predicted_values

# Combine the data back together
final_data = pd.concat([data_with_labels, data_to_predict])

# Ensure 'default-payment-next-month' column is of integer type
final_data['default-payment-next-month'] = final_data['default-payment-next-month'].astype(int)

# Save the final dataset with predicted values
final_data.to_csv('models/predictions.csv', index=False)


