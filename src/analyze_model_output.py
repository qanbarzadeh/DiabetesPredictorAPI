import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

# Load the calibrated model
model_path = './calibrated_best_model.joblib'
calibrated_model = joblib.load(model_path)

# Define your data loading and preprocessing steps
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)

    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    preprocessing_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
    ])

    X_preprocessed = preprocessing_pipeline.fit_transform(X)

    return X_preprocessed, y

# Adjust this path to where your feature-engineered dataset is located
dataset_path = '../data/processed/diabetes_processed.csv'
X_preprocessed, y = load_and_preprocess_data(dataset_path)

# Generate predicted probabilities
predicted_probabilities = calibrated_model.predict_proba(X_preprocessed)[:, 1]

# Analyze the probability distribution
plt.hist(predicted_probabilities, bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Predicted Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.show()


# Categorize into risk levels
risk_levels = ['Low Risk' if prob < 0.33 else 'Medium Risk' if prob < 0.66 else 'High Risk' for prob in predicted_probabilities]

# Print the categorized risk levels
print("\nCategorized Risk Levels:")
for i, risk_level in enumerate(risk_levels):
    print(f"Instance {i+1}: {risk_level}")

