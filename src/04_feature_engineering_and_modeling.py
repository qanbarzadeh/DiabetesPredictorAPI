import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV  # For calibration
import sys 

sys.path.append('../helpers')
from data_helpers import load_config, load_data

# Data loading and preprocessing
def load_and_preprocess_data(config_path):
    
    config = load_config(config_path)

    if not config:
        raise Exception("Failed to load configuration.")

    
    data_path = config['data_path']
    df = load_data(data_path)

    if df is None:
        raise Exception("Failed to load the data.")

    # Handling missing values
    # Adjusted code without using inplace=True
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df[col].replace(0, np.nan)   
        df['BMI_Age_Interaction'] = df['BMI'] * df['Age']
    
    return df

# Splitting dataset
def split_dataset(df):
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    return train_test_split(X, y, test_size=0.2, random_state=42)


#model pipeline construction with class weight handling
def build_model_pipeline():
    imputer = KNNImputer(n_neighbors=5)
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    classifier = RandomForestClassifier(random_state=42, class_weight='balanced')  

    pipeline = Pipeline(steps=[('imputer', imputer),
                               ('scaler', scaler),
                               ('poly', poly),
                               ('classifier', classifier)])
    return pipeline

# Hyperparameter tuning using grid search
def hyperparameter_tuning(X_train, y_train):
    pipeline = build_model_pipeline()
    parameter_grid = {
        'classifier__n_estimators': [100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    cv = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(pipeline, parameter_grid, cv=cv, scoring='recall')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_



# find the optimal threshold for classification
def find_optimal_threshold(precision, recall, thresholds):
    # Convert to a f1 score and find the index of the highest score
    f1_scores = 2 * recall * precision / (recall + precision)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

def evaluate_model_with_calibration(best_estimator, X_train, y_train, X_test, y_test, method='sigmoid'):
    """
    Evaluates the best estimator after calibrating it on a validation set.
    Args:
        best_estimator: The best uncalibrated pipeline or model from hyperparameter tuning.
        X_train (DataFrame): Training feature data.
        y_train (Series): Training target data.
        X_test (DataFrame): Test feature data.
        y_test (Series): Test target data.
        method (str): Calibration method, 'sigmoid' or 'isotonic'.
    """
    # Split the training data for calibration purpose (15% of the original training data)
    X_train_cal, X_val_cal, y_train_cal, y_val_cal = train_test_split(X_train, y_train, test_size=0.15, random_state=42)
    
    # Fit the best model pipeline on the remaining training data
    best_estimator.fit(X_train_cal, y_train_cal)
    
    # Calibrate the model on the validation set
    calibrator = CalibratedClassifierCV(best_estimator, method=method, cv='prefit')
    calibrator.fit(X_val_cal, y_val_cal)
    
    # Use the calibrated model for predictions
    probabilities = calibrator.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    fpr, tpr, roc_thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, probabilities)
    pr_auc = average_precision_score(y_test, probabilities)
    optimal_threshold = find_optimal_threshold(precision, recall, pr_thresholds)
    adjusted_predictions = [1 if prob > optimal_threshold else 0 for prob in probabilities]
    
    # Output evaluation metrics
    print("\nAdjusted Classification Report:")
    print(classification_report(y_test, adjusted_predictions))
    print("\nROC AUC Score:", roc_auc)
    print("\nPR AUC Score:", pr_auc)

    # Visualization
    plot_confusion_matrix(confusion_matrix(y_test, adjusted_predictions))
    plot_roc_curve(y_test, probabilities)
    plot_precision_recall_curve(recall, precision, pr_auc)



def save_processed_data(df, config_path):
    """
    Saves the processed DataFrame to a CSV file and handles errors.
    
    Parameters:
    - df: DataFrame to save
    - config_path: Path to the configuration file
    """
    try:
        config = load_config(config_path)
        processed_data_path = config['processed_data_path']
        df.to_csv(processed_data_path, index=False)
        print(f"Processed data successfully saved to {processed_data_path}")
    except Exception as e:
        print(f"Failed to save processed data: {e}")

# plot the Precision-Recall curve
def plot_precision_recall_curve(recall, precision, pr_auc):
    plt.figure()
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.show(block = False)
    input("Press Enter to continue...")  


def plot_confusion_matrix(cm):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show(block = False)
    input("Press Enter to continue...")  
    
    
    # Function to plot the ROC curve
def plot_roc_curve(y_test, probabilities):
    fpr, tpr, thresholds = roc_curve(y_test, probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show(block = False)
    input("Press Enter to continue...")  


if __name__ == "__main__":
    config_path = '../config/config.json'
    processed_data_path = '../data/processed_data.csv'  # Specify the path for saving processed data
    
    try: 
        df = load_and_preprocess_data(config_path)
        print(df.head()) 

        # Save the processed data
        save_processed_data(df, processed_data_path)
        
        X, y = df.drop('Outcome', axis=1), df['Outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Ensure class_weight is correctly set up; this should be a parameter, not a set
        class_weight = 'balanced'  # If dynamic, consider calculating based on y_train
        
        best_pipeline = hyperparameter_tuning(X_train, y_train)  
        evaluate_model_with_calibration(best_pipeline, X_train, y_train, X_test, y_test, method='sigmoid')
    except Exception as e:
        print(f"An error occurred: {e}")