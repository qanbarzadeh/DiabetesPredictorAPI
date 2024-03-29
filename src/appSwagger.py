from flask import Flask, request, jsonify, render_template
from flask_restx import Api, Resource, fields
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)
api = Api(app, version='1.0', title='Diabetes Prediction API',
          description='A simple API for predicting diabetes risk')

# Load the trained model (ensure the path is correct)
model_path = 'C:\\Users\\ali\\diabetes_risk_prediction\\notebooks\\best_model_xgboost.joblib'
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

output_model = api.model('OutputModel', {
    'prediction': fields.Integer(description='Predicted class (0 for non-diabetic, 1 for diabetic)')
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
        prediction = model.predict(processed_features)[0]
        
        # Respond with the predicted value
        return {'prediction': int(prediction)}

@app.route('/')
def index():
    # This route serves the HTML page
    return render_template('/index.html')  # Ensure the HTML file exists in the templates folder

if __name__ == '__main__':
    app.run(debug=True)
