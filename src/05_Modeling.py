
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import sys
sys.path.append('../Helpers')  # Ensure this path is correct for your project structure
from data_helpers import load_config, load_data


def load_feature_engineered_data(config_path):
    """
    Loads the feature-engineered dataset
    """
    try:
        config = load_config(config_path)
        processed_data_path = config['processed_data_path']
        df = pd.read_csv(processed_data_path)
        print("Feature-engineered dataset successfully loaded.")
        return df
    except Exception as e:
        print(f"Failed to load feature-engineered dataset: {e}")
        return None


    
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model's performance and plots evaluation metrics.
    """
    predictions = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_auc = auc(fpr, tpr)

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("\nConfusion Matrix:")
    sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.show()
    
    print("\nROC AUC Score:", roc_auc)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    config_path = '../config/config.json'  # Adjust as necessary
    df = load_feature_engineered_data(config_path)
    
    if df is not None:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        evaluate_model(model, X_test, y_test)




