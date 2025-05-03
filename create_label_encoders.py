import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Read the dataset
df = pd.read_csv('Lung.csv')

# Create a dictionary to store label encoders
label_encoders = {}

# List of categorical columns
categorical_columns = [
    'Gender',
    'Smoking_Status',
    'Residence',
    'Air_Pollution_Exposure',
    'Biomass_Fuel_Use',
    'Factory_Exposure',
    'Family_History',
    'Diet_Habit',
    'Symptoms',
    'Histology_Type',
    'Stage',
    'Treatment',
    'Hospital_Type'
]

# Create and fit label encoders for each categorical column
for column in categorical_columns:
    le = LabelEncoder()
    le.fit(df[column])
    label_encoders[column] = le

# Save the label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

print("Label encoders have been created and saved to 'label_encoders.pkl'")
print("\nCategories for each feature:")
for column in categorical_columns:
    print(f"\n{column}:")
    print(label_encoders[column].classes_) 