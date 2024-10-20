import pandas as pd
import numpy as np
from flask import Flask, request, render_template
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('random_forest_model.joblib')
scaler = joblib.load('scaler.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input from form
    age = float(request.form['age'])
    family_history = int(request.form['family_history'])
    breast_density = int(request.form['breast_density'])
    bmi = float(request.form['bmi'])

    # Prepare input data
    input_data = pd.DataFrame([[age, family_history, breast_density, bmi]],
                              columns=['age', 'family_history', 'breast_density', 'bmi'])
    
    # Apply one-hot encoding
    input_data = pd.get_dummies(input_data, columns=['family_history'], drop_first=True)
    
    # Ensure all columns from training are present
    expected_columns = ['age', 'breast_density', 'bmi', 'family_history_1']
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    # Reorder columns to match training data
    input_data = input_data[expected_columns]

    # Scale the input data
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # Prepare result
    result = "High" if prediction == 1 else "Low"
    probability = f"{probability:.2%}"

    return render_template('result.html', result=result, probability=probability)

if __name__ == '__main__':
    app.run(debug=True)
