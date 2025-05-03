import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import joblib

# Read the dataset
df = pd.read_csv('Lung.csv')

# Create label encoders for categorical variables
label_encoders = {}
categorical_columns = [
    'Gender', 'Smoking_Status', 'Residence', 'Air_Pollution_Exposure',
    'Biomass_Fuel_Use', 'Factory_Exposure', 'Family_History', 'Diet_Habit',
    'Symptoms', 'Histology_Type', 'Stage', 'Treatment', 'Hospital_Type'
]

# Encode categorical variables
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Save label encoders
joblib.dump(label_encoders, 'label_encoders.pkl')

# Prepare features and target
X = df.drop(['Patient_ID', 'Survival_1_Year'], axis=1)
y = df['Survival_1_Year'].map({'Yes': 1, 'No': 0})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create base models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
svc = SVC(probability=True, random_state=42)

# Create the voting classifier
meta_model = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svc', svc)
    ],
    voting='soft'
)

# Train the meta model
print("Training the meta model...")
meta_model.fit(X_train, y_train)

# Evaluate the model
train_score = meta_model.score(X_train, y_train)
test_score = meta_model.score(X_test, y_test)
print(f"\nTraining accuracy: {train_score:.4f}")
print(f"Testing accuracy: {test_score:.4f}")

# Save the trained model
joblib.dump(meta_model, 'meta_model.pkl')
print("\nModel has been saved as 'meta_model.pkl'") 