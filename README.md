## Diabetes Risk Prediction Model

**Overview**

This model, developed for "Arad Madar Rasa" as part of a broader health analytics initiative, predicts the risk of diabetes using patient health data. It emphasizes model calibration for precise risk assessment, offering actionable insights into diabetes management.

**Problem Statement**

Early detection and management of diabetes can significantly alter the disease's trajectory, enhancing patient outcomes. This model aims to identify at-risk individuals early, allowing for timely intervention and preventive care.

**Dataset**

**Proprietary Dataset**

The project utilizes a proprietary dataset from "Arad Madar Rasa," comprising detailed health metrics for accurate diabetes risk prediction. Due to confidentiality, specifics are not disclosed.

**Alternative Dataset for Demonstration**

* **Source:** [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
* **Description:** 768 entries with features like Glucose, BloodPressure, BMI, Age, and Insulin levels.
* **Preprocessing:** Missing values in critical fields were imputed using KNNImputer. StandardScaler was applied for normalization.

**Model Development**

Tested algorithms include RandomForest, GradientBoosting, XGBClassifier, SVC, and MLPClassifier. Calibration used CalibratedClassifierCV with the 'sigmoid' method to enhance prediction reliability.

**Evaluation**

The model's performance was evaluated on metrics such as Accuracy, Precision, Recall, F1 Score, and ROC AUC. Calibration notably improved prediction probabilities, offering nuanced risk assessments.

**How to Use**

The Flask API (`app.py`) serves the model for easy integration and use. Detailed usage instructions, including API endpoints and example requests, are available in the project documentation.
To make a prediction using the API, you can use the following `curl` example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H 'Content-Type: application/json' \
     -d '{"Pregnancies": 2, "Glucose": 138, "BloodPressure": 62, "SkinThickness": 35, "Insulin": 0, "BMI": 33.6, "DiabetesPedigreeFunction": 0.127, "Age": 47}'
```
**Results and Interpretation**

The RandomForest model achieved the highest ROC AUC of 0.8215, demonstrating strong predictive accuracy. While effective for preliminary assessments, professional medical evaluation remains paramount.

**Project Structure**

* `data_preprocessing.py`: Data cleaning and preparation.
* `model_training.py`: Model training and evaluation scripts.
* `calibration.py`: Calibration of models to enhance predictions.
* `app.py`: Flask API for deploying the model.

**Future Work**

Explorations into advanced algorithms, additional features, and a broader dataset are planned to refine the model's accuracy further.

**Contributing**

Contributions towards model enhancement, feature expansion, or dataset augmentation are welcomed.
Please note that this code is not intended for production use.

