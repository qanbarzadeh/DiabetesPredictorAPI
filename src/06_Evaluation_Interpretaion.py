
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from joblib import dump
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import logging
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Add the Helpers directory to the path
helpers_path = Path("../Helpers").resolve()
sys.path.append(str(helpers_path))

from data_helpers import load_config, load_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_feature_engineered_data(config_path):
    """Loads feature-engineered dataset from the specified configuration path."""
    try:
        config = load_config(config_path)
        processed_data_path = config['processed_data_path']
        df = pd.read_csv(processed_data_path)
        logging.info("Feature-engineered dataset successfully loaded.")
        return df
    except Exception as e:
        logging.error(f"Failed to load feature-engineered dataset: {e}")
        return None

def evaluate_model_performance(model, X_test, y_test):
    """Evaluates the model's performance and prints various metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics/
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Log metrics
    logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    #plot confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    

def plot_roc_curve(y_test, y_pred_proba):
    """Plots the ROC curve for a given set of true labels and predictions."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_confusion_matrix(y_test, y_pred):
    """Plots and prints the confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    sns.set(font_scale=1.4) # Adjust to make sure text fits inside boxes
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Calculate and print the confusion matrix values
    TN, FP, FN, TP = cm.ravel()
    print(f"True Negatives (TN): {TN}")
    print(f"False Positives (FP): {FP}")
    print(f"False Negatives (FN): {FN}")
    print(f"True Positives (TP): {TP}")



    

if __name__ == "__main__":
    config_path = '../config/config.json'
    df = load_feature_engineered_data(config_path)
    if df is not None:
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Perform hyperparameter tuning
        best_xgb_model = perform_hyperparameter_tuning(X_train, y_train)
        
        # Evaluate the best model
        evaluate_model_performance(best_xgb_model, X_test, y_test)
        
        # Serialize the model
        dump(best_xgb_model, 'best_model_xgboost.joblib')
        
        