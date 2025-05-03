from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the model and label encoders
model = joblib.load('meta_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Create a DataFrame with the input data
        input_data = pd.DataFrame([{
            'Age': int(data['age']),
            'Gender': data['gender'],
            'Smoking_Status': data['smoking_status'],
            'Residence': data['residence'],
            'Air_Pollution_Exposure': data['air_pollution'],
            'Biomass_Fuel_Use': data['biomass_fuel'],
            'Factory_Exposure': data['factory_exposure'],
            'Family_History': data['family_history'],
            'Diet_Habit': data['diet_habit'],
            'Symptoms': data['symptoms'],
            'Tumor_Size_mm': float(data['tumor_size']),
            'Histology_Type': data['histology_type'],
            'Stage': data['stage'],
            'Treatment': data['treatment'],
            'Hospital_Type': data['hospital_type']
        }])

        # Encode categorical variables
        for column in input_data.select_dtypes(include=['object']).columns:
            if column in label_encoders:
                input_data[column] = label_encoders[column].transform(input_data[column])

        # Make prediction
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[0]

        return jsonify({
            'prediction': 'Yes' if prediction[0] == 1 else 'No',
            'probability': float(max(probability))
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 