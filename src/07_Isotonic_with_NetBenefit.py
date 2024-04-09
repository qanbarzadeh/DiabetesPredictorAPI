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
import matplotlib.pyplot as plt
from sklearn.calibration import IsotonicRegression
from sklearn.calibration import calibration_curve
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import log_loss

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
    
    # Removed logging here to prevent duplication
    plot_evaluation_charts(y_test, y_pred, y_pred_proba, model_name)
    
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    }

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

#Calibration Curve Function Updated for Isotonic Regression
def plot_calibration_curve(y_true, y_pred, n_bins=10):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, strategy='uniform', n_bins=10)    
    print("\nCalibration values:")
    print("Binned Probability | Fraction of Positives")
    for prob_predicted, prob_actual in zip(prob_pred, prob_true):
        print(f"{prob_predicted:19} | {prob_actual}")

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(111)
    ax1.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    ax1.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated', color='gray')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Plot')
    ax1.legend()
    plt.show()

def calculate_net_benefit(y_true, y_proba, threshold_prob):
    """
    Calculate the net benefit for a specific threshold probability.

    Parameters:
    - y_true: array-like, true binary labels.
    - y_proba: array-like, predicted probabilities.
    - threshold_prob: float, the probability threshold for making a decision.

    Returns:
    - Net benefit as a float.
    """
    TP = ((y_proba >= threshold_prob) & (y_true == 1)).sum()
    FP = ((y_proba >= threshold_prob) & (y_true == 0)).sum()
    num_total = len(y_true)
    num_positive = y_true.sum()

    # Calculate the benefit and harm
    benefit = TP / num_total
    harm = FP / num_total * (threshold_prob / (1 - threshold_prob))

    return benefit - harm

def decision_curve_analysis(y_true, y_proba, thresholds=np.linspace(0.01, 0.99, 50)):
    """
    Perform decision curve analysis.

    Parameters:
    - y_true: array-like, true binary labels.
    - y_proba: array-like, predicted probabilities.
    - thresholds: array-like, a range of threshold probabilities.

    Returns:
    - A plot of net benefit vs. threshold probabilities.
    """
    net_benefits = [calculate_net_benefit(y_true, y_proba, thr) for thr in thresholds]

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, net_benefits, label='Model')
    plt.plot(thresholds, thresholds, linestyle='--', label='Treat All', color='gray')
    plt.plot(thresholds, np.zeros_like(thresholds), linestyle='--', label='Treat None', color='black')
    plt.xlabel('Threshold Probability')
    plt.ylabel('Net Benefit')
    plt.title('Decision Curve Analysis')
    plt.legend()
    plt.show()


def log_and_return_metrics(model, X_test, y_test, model_name='Model'):
    metrics = evaluate_model_performance(model, X_test, y_test, model_name)
    logging.info(f"{model_name} - Accuracy: {metrics['Accuracy']:.4f}, Precision: {metrics['Precision']:.4f}, "
                 f"Recall: {metrics['Recall']:.4f}, F1 Score: {metrics['F1 Score']:.4f}, ROC AUC: {metrics['ROC AUC']:.4f}")
    return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    config_path = '../config/config.json'
    df = load_feature_engineered_data(config_path)
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    preprocessing_pipeline = Pipeline([('imputer', KNNImputer(n_neighbors=5)), ('scaler', StandardScaler())])
    X_train_preprocessed = preprocessing_pipeline.fit_transform(X_train)
    X_test_preprocessed = preprocessing_pipeline.transform(X_test)

    models = {'RandomForest': RandomForestClassifier(random_state=42), 'GradientBoosting': GradientBoostingClassifier(random_state=42),
              'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42), 'SVC': SVC(probability=True, random_state=42),
              'MLP': MLPClassifier(random_state=42)}

    scores_auc = {}
    scores_log_loss = {}
    calibrated_models = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        cv_probs = cross_val_predict(model, X_train_preprocessed, y_train, cv=skf, method='predict_proba')[:, 1]
        model.fit(X_train_preprocessed, y_train)
        test_probs = model.predict_proba(X_test_preprocessed)[:, 1]

        roc_auc = roc_auc_score(y_test, test_probs)
        log_loss_val = log_loss(y_test, test_probs)

        scores_auc[model_name] = roc_auc
        scores_log_loss[model_name] = log_loss_val

        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=skf)
        calibrated_model.fit(X_train_preprocessed, y_train)
        calibrated_models[model_name] = calibrated_model

        test_probs_calibrated = calibrated_model.predict_proba(X_test_preprocessed)[:, 1]
        roc_auc_calibrated = roc_auc_score(y_test, test_probs_calibrated)
        log_loss_calibrated = log_loss(y_test, test_probs_calibrated)
                        
        logging.info(f"{model_name}: ROC AUC = {roc_auc:.4f}, Log Loss = {log_loss_val:.4f}")
        logging.info(f"{model_name} (Calibrated): ROC AUC = {roc_auc_calibrated:.4f}, Log Loss = {log_loss_calibrated:.4f}")
        #new
        plot_evaluation_charts(y_test, calibrated_model.predict(X_test_preprocessed), test_probs, f"{model_name} (Calibrated)")

    best_model_name = max(scores_auc, key=scores_auc.get)
    best_calibrated_model = calibrated_models[best_model_name]

    logging.info(f"Best model based on ROC AUC: {best_model_name}")

    y_pred_proba_best = best_calibrated_model.predict_proba(X_test_preprocessed)[:, 1]
    plot_calibration_curve(y_test, y_pred_proba_best, n_bins=10)
    decision_curve_analysis(y_test, y_pred_proba_best)

    dump(best_calibrated_model, f'{best_model_name}_best_calibrated_model.joblib')
    logging.info(f"Best calibrated model ({best_model_name}) saved.")