import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
data = pd.read_csv('data/data_participant.csv')

# Handle missing values by imputing or dropping them
# For now, we'll drop rows with missing values in the target column
data = data.dropna(subset=['default-payment-next-month'])

# Encode categorical variables (e.g., SEX, EDUCATION, MARRIAGE)
label_encoders = {}
categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split the data into features and target
X = data.drop(['ID', 'SEX', 'default-payment-next-month'], axis=1)
y = data['default-payment-next-month']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the preprocessed data
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

# Save the label encoders for later use
import joblib
joblib.dump(label_encoders, 'models/label_encoders.pkl')

