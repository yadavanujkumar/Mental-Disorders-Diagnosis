#!/usr/bin/env python3
"""
Quick Start Example - Mental Disorders Diagnosis
This script demonstrates how to use a trained model to make predictions on new patient data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib
import os

print("="*80)
print("QUICK START: Training and Using the Best Model")
print("="*80)

# Load dataset
df = pd.read_csv('mental_disorders_dataset.csv')
print(f"\n✓ Loaded dataset with {df.shape[0]} patients")

# Prepare data
X = df.drop(['Patient Number', 'Expert Diagnose'], axis=1)
y = df['Expert Diagnose']

# Encode features
X_encoded = X.copy()
feature_encoders = {}
for column in X_encoded.columns:
    if X_encoded[column].dtype == 'object':
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
        feature_encoders[column] = le

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train best model (LightGBM)
print("\n🤖 Training LightGBM model...")
model = LGBMClassifier(random_state=42, verbose=-1)
model.fit(X_train_scaled, y_train)

# Evaluate
accuracy = model.score(X_test_scaled, y_test)
print(f"✓ Model trained successfully!")
print(f"  Test Accuracy: {accuracy:.2%}")

# Save the model and preprocessors
print("\n💾 Saving model and preprocessors...")
os.makedirs('saved_models', exist_ok=True)
joblib.dump(model, 'saved_models/lightgbm_model.pkl')
joblib.dump(scaler, 'saved_models/scaler.pkl')
joblib.dump(le_target, 'saved_models/label_encoder.pkl')
joblib.dump(feature_encoders, 'saved_models/feature_encoders.pkl')
print("✓ Model saved to 'saved_models/' directory")

# Example: Make predictions on new data
print("\n" + "="*80)
print("EXAMPLE: Making Predictions on New Patient Data")
print("="*80)

# Get a sample from test set
sample_idx = 0
sample_features = X_test.iloc[sample_idx:sample_idx+1]
sample_features_scaled = X_test_scaled[sample_idx:sample_idx+1]
actual_diagnosis = le_target.inverse_transform([y_test[sample_idx]])[0]

print("\n📋 Sample Patient Features:")
for col, val in sample_features.iloc[0].items():
    # Decode the encoded value back to original
    if col in feature_encoders:
        original_val = feature_encoders[col].inverse_transform([val])[0]
        print(f"   {col}: {original_val}")

# Make prediction
prediction = model.predict(sample_features_scaled)
predicted_diagnosis = le_target.inverse_transform(prediction)[0]

# Get prediction probabilities
proba = model.predict_proba(sample_features_scaled)[0]

print(f"\n🎯 Actual Diagnosis: {actual_diagnosis}")
print(f"🔮 Predicted Diagnosis: {predicted_diagnosis}")
print(f"{'✓ CORRECT!' if predicted_diagnosis == actual_diagnosis else '✗ Incorrect'}")

print("\n📊 Prediction Probabilities:")
for i, class_name in enumerate(le_target.classes_):
    print(f"   {class_name}: {proba[i]:.2%}")

# Instructions for using saved model
print("\n" + "="*80)
print("HOW TO USE THE SAVED MODEL FOR NEW PREDICTIONS")
print("="*80)

print("""
To make predictions on completely new patient data:

1. Load the saved model and preprocessors:
   ```python
   import joblib
   model = joblib.load('saved_models/lightgbm_model.pkl')
   scaler = joblib.load('saved_models/scaler.pkl')
   le_target = joblib.load('saved_models/label_encoder.pkl')
   feature_encoders = joblib.load('saved_models/feature_encoders.pkl')
   ```

2. Prepare new patient data with the same features:
   ```python
   new_patient = {
       'Sadness': 'Usually',
       'Euphoric': 'Seldom',
       'Exhausted': 'Sometimes',
       # ... all other features ...
   }
   ```

3. Encode and scale the features:
   ```python
   for col, value in new_patient.items():
       if col in feature_encoders:
           new_patient[col] = feature_encoders[col].transform([value])[0]
   
   features_scaled = scaler.transform([list(new_patient.values())])
   ```

4. Make prediction:
   ```python
   prediction = model.predict(features_scaled)
   diagnosis = le_target.inverse_transform(prediction)[0]
   print(f"Predicted Diagnosis: {diagnosis}")
   ```
""")

print("="*80)
print("✅ Quick Start Complete!")
print("="*80)
