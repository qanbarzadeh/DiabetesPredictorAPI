import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from joblib import dump
from pathlib import Path
import sys
import logging
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Add the Helpers directory to the path
sys.path.append(str(Path("../Helpers").resolve()))

from data_helpers import load_config, load_data

def load_feature_engineered_data(config_path):
    try:
        config = load_config(config_path)
        processed_data_path = config['processed_data_path']
        df = pd.read_csv(processed_data_path)
        logging.info("Feature-engineered dataset successfully loaded.")
        return df
    except Exception as e:
        logging.error(f"Failed to load feature-engineered dataset: {e}")
        return None

def evaluate_model_performance(model, X_test, y_test, model_name='Model'):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    logging.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}")

    # Plotting ROC Curve and Confusion Matrix
    plot_evaluation_charts(y_test, y_pred, y_pred_proba, model_name)

def plot_evaluation_charts(y_test, y_pred, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")

    cm = confusion_matrix(y_test, y_pred)
    plt.subplot(1, 2, 2)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load and preprocess the data
    config_path = '../config/config.json'
    df = load_feature_engineered_data(config_path)
    columns_to_replace = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[columns_to_replace] = df[columns_to_replace].replace(0, np.nan)


    # Prepare the data
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define and apply the preprocessing pipeline
    preprocessing_pipeline = Pipeline([
        ('imputer', KNNImputer(n_neighbors=5)),
        ('scaler', StandardScaler()),
    ])
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    # Define models
    models = {
        'RandomForest': RandomForestClassifier(random_state=42),
        'GradientBoosting': GradientBoostingClassifier(random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'SVC': SVC(probability=True, random_state=42),
        'MLP': MLPClassifier(random_state=42)
    }

    # Train, calibrate models, and evaluate
    best_model_score = 0
    best_model_name = ''
    best_calibrated_model = None

    for model_name, model in models.items():
        logging.info(f"Training and calibrating {model_name}")
        model.fit(X_train_preprocessed, y_train)
        calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=5)
        calibrated_model.fit(X_train_preprocessed, y_train)

        score = roc_auc_score(y_test, calibrated_model.predict_proba(X_test_preprocessed)[:, 1])
        if score > best_model_score:
            best_model_score = score
            best_model_name = model_name
            best_calibrated_model = calibrated_model

        evaluate_model_performance(calibrated_model, X_test_preprocessed, y_test, model_name)

    logging.info(f"Best model is {best_model_name} with ROC AUC {best_model_score:.4f}")

    # Save the best calibrated model
    model_path = 'calibrated_best_model.joblib'
    dump(best_calibrated_model, model_path)
    logging.info(f"Successfully saved the best calibrated model to {model_path}")
