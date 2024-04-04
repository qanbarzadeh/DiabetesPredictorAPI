import joblib
import xgboost as xgb

model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\best_model_xgboost.joblib'

# Use joblib to load the model correctly
loaded_model = joblib.load(model_path)

# Assuming the loaded model is an XGBClassifier, we get the Booster object
booster = loaded_model.get_booster()

new_model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\Updated_model_xgboost.json'

# Use the save_model method from the Booster object to save the model in a compatible format
booster.save_model(new_model_path)

print(f"Model re-exported as {new_model_path}")
