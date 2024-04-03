from flask import Flask, request, jsonify, render_template
from flask_restx import Api, Resource, fields
import joblib
import numpy as np
import xgboost

# Initialize the Flask application
app = Flask(__name__)
api = Api(app, version='1.0', title='Diabetes Prediction API',description='A simple API for predicting diabetes risk')

# Load the trained model
model_path = './calibrated_best_model.joblib'
model = joblib.load(model_path)

# Define the input and output data model for Swagger documentation
input_model = api.model('InputModel', {
    'Pregnancies': fields.Integer(required=True, description='Number of pregnancies'),
    'Glucose': fields.Float(required=True, description='Plasma glucose concentration'),
    'BloodPressure': fields.Float(required=True, description='Diastolic blood pressure (mm Hg)'),
    'SkinThickness': fields.Float(required=True, description='Triceps skinfold thickness (mm)'),
    'Insulin': fields.Float(required=True, description='2-Hour serum insulin (mu U/ml)'),
    'BMI': fields.Float(required=True, description='Body mass index (weight in kg/(height in m)^2)'),
    'DiabetesPedigreeFunction': fields.Float(required=True, description='Diabetes pedigree function'),
    'Age': fields.Integer(required=True, description='Age (years)'),
})


output_model = api.model('OutputModel', {'risk_level': fields.String(description='Risk level (Low Risk, Medium Risk, High Risk)')
})

def preprocess_input(data):
    # Calculate BMI_Age_Interaction
    data['BMI_Age_Interaction'] = data['BMI'] * data['Age']
    # Prepare the features in the order expected by the model
    features = np.array([data['Pregnancies'], data['Glucose'], data['BloodPressure'],
                         data['SkinThickness'], data['Insulin'], data['BMI'],
                         data['DiabetesPedigreeFunction'], data['Age'], 
                         data['BMI_Age_Interaction']]).reshape(1, -1)
    return features

@api.route('/predict')
class Predict(Resource):
    @api.expect(input_model, validate=True)
    @api.response(200, 'Success', output_model)
    def post(self):
        # Extract features from the POST request's body
        data = request.get_json()
        processed_features = preprocess_input(data)
        
        # Make prediction
        prediction_probability = model.predict_proba(processed_features)[0][1] # Probability of being diabetic
        
        # Determine risk level based on prediction probability
        if prediction_probability < 0.25:
            risk_level = 'Low Risk'
        elif prediction_probability < 0.55:
            risk_level = 'Medium Risk'
        else:
            risk_level = 'High Risk'
        
        # Respond with the predicted value
        return {'risk_level': risk_level}

@app.route('/')
def index():
    # This route serves the HTML page
    return render_template('/index.html')

if __name__ == '__main__':
    app.run(debug=True)