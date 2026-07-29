import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
# Make sure 'logistic_regression_model.sav' is in the same directory as this app.py file
try:
    model = joblib.load('logistic_regression_model.sav')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None # Handle case where model might not load

@app.route('/')
def home():
    return "Welcome to the Diabetes Prediction API! Send POST requests to /predict."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        # Get data from the POST request
        data = request.get_json(force=True)
        
        # Convert dictionary to DataFrame
        # The order of columns must match the training data
        feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        input_df = pd.DataFrame([data], columns=feature_names)
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Convert prediction to Python native types for JSON serialization
        output = {'prediction': int(prediction[0]), 'probability_no_diabetes': prediction_proba[0][0].item(), 'probability_diabetes': prediction_proba[0][1].item()}
        
        return jsonify(output)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Run the app locally
    # For production deployment, you might use a WSGI server like Gunicorn
    app.run(debug=True, host='0.0.0.0', port=5000)
