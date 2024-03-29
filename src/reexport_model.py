import joblib
import xgboost as xgb

# Correct the approach for loading the model
# Adjust the path to where your model is currently saved
model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\best_model_xgboost.joblib'

# Use joblib to load the model correctly
loaded_model = joblib.load(model_path)

# Assuming the loaded model is an XGBClassifier, we get the Booster object
booster = loaded_model.get_booster()

# Specify the new path for the re-exported model, changing the extension to .json for compatibility
new_model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\Updated_model_xgboost.json'

# Use the save_model method from the Booster object to save the model in a compatible format
booster.save_model(new_model_path)

print(f"Model re-exported as {new_model_path}")
