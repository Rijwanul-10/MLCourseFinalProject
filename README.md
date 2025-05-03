# Lung Cancer Survival Predictor

This web application uses a meta model to predict 1-year survival probability for lung cancer patients based on various clinical and demographic factors.

## Features

- User-friendly interface for inputting patient data
- Real-time prediction of 1-year survival probability
- Support for all relevant clinical and demographic factors
- Responsive design that works on both desktop and mobile devices

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Make sure you have the following files in your project directory:
   - `app.py`
   - `meta_model.pkl` (your trained model)
   - `label_encoders.pkl` (your label encoders)
   - `templates/index.html`

4. Run the application:
```bash
python app.py
```

5. Open your web browser and navigate to:
```
http://localhost:5000
```

## Input Parameters

The application accepts the following parameters:
- Age
- Gender
- Smoking Status
- Residence
- Air Pollution Exposure
- Biomass Fuel Use
- Factory Exposure
- Family History
- Diet Habit
- Symptoms
- Tumor Size
- Histology Type
- Stage
- Treatment
- Hospital Type

## Note

Make sure you have your trained model (`meta_model.pkl`) and label encoders (`label_encoders.pkl`) in the same directory as `app.py` before running the application. 