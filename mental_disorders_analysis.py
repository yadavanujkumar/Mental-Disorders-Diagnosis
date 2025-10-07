#!/usr/bin/env python3
"""
Mental Disorders Diagnosis - Complete Analysis
Performs EDA, AI, ML, and DL on the mental disorders dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE

# Deep Learning Libraries
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MENTAL DISORDERS DIAGNOSIS - COMPREHENSIVE ANALYSIS")
print("EDA, AI, ML, and DL Implementation")
print("="*80)

# ============================================================================
# 1. DATA LOADING AND INITIAL EXPLORATION (EDA)
# ============================================================================
print("\n" + "="*80)
print("1. DATA LOADING AND INITIAL EXPLORATION")
print("="*80)

# Load the dataset
df = pd.read_csv('mental_disorders_dataset.csv')

print("\n📊 Dataset Shape:", df.shape)
print(f"   Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n📋 First Few Rows:")
print(df.head())

print("\n📝 Dataset Information:")
print(df.info())

print("\n📊 Statistical Summary:")
print(df.describe(include='all'))

print("\n🔍 Missing Values:")
print(df.isnull().sum())

print("\n🎯 Target Variable Distribution:")
print(df['Expert Diagnose'].value_counts())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
print("\n" + "="*80)
print("2. EXPLORATORY DATA ANALYSIS (EDA)")
print("="*80)

# Create visualizations directory
import os
os.makedirs('visualizations', exist_ok=True)

# Target distribution plot
plt.figure(figsize=(10, 6))
df['Expert Diagnose'].value_counts().plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Mental Disorder Diagnoses', fontsize=14, fontweight='bold')
plt.xlabel('Diagnosis', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: visualizations/target_distribution.png")

# Correlation analysis for numeric features
print("\n🔗 Analyzing Feature Correlations...")

# Create a copy for encoding
df_encoded = df.copy()

# Encode categorical features for correlation analysis
label_encoders = {}
for column in df_encoded.columns:
    if df_encoded[column].dtype == 'object' and column != 'Patient Number':
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
        label_encoders[column] = le

# Correlation heatmap
plt.figure(figsize=(16, 12))
correlation_matrix = df_encoded.drop('Patient Number', axis=1).corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
            linewidths=0.5, cbar_kws={'label': 'Correlation'})
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('visualizations/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/correlation_heatmap.png")

# Feature distributions by diagnosis
print("\n📊 Creating feature distribution visualizations...")
categorical_features = ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(categorical_features):
    pd.crosstab(df[feature], df['Expert Diagnose']).plot(
        kind='bar', ax=axes[idx], stacked=False, 
        colormap='Set2', edgecolor='black'
    )
    axes[idx].set_title(f'{feature} by Diagnosis', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel(feature, fontsize=10)
    axes[idx].set_ylabel('Count', fontsize=10)
    axes[idx].legend(title='Diagnosis', fontsize=8)
    axes[idx].tick_params(labelsize=8)
    
plt.tight_layout()
plt.savefig('visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/feature_distributions.png")

# ============================================================================
# 3. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*80)
print("3. DATA PREPROCESSING")
print("="*80)

# Drop patient number column
X = df.drop(['Patient Number', 'Expert Diagnose'], axis=1)
y = df['Expert Diagnose']

print(f"\n📦 Features shape: {X.shape}")
print(f"🎯 Target shape: {y.shape}")

# Encode categorical features
print("\n🔄 Encoding categorical features...")
X_encoded = X.copy()
feature_encoders = {}

for column in X_encoded.columns:
    if X_encoded[column].dtype == 'object':
        le = LabelEncoder()
        X_encoded[column] = le.fit_transform(X_encoded[column].astype(str))
        feature_encoders[column] = le
        print(f"   ✓ Encoded: {column}")

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\n✓ Target classes: {list(le_target.classes_)}")
print(f"   Encoded as: {list(range(len(le_target.classes_)))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\n✂️ Data Split:")
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Handle class imbalance with SMOTE
print("\n⚖️ Handling class imbalance with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"   Original training samples: {len(y_train)}")
print(f"   Balanced training samples: {len(y_train_balanced)}")
print(f"   Class distribution after SMOTE:")
unique, counts = np.unique(y_train_balanced, return_counts=True)
for cls, cnt in zip(unique, counts):
    print(f"      Class {le_target.classes_[cls]}: {cnt} samples")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

print("\n✅ Preprocessing completed!")

# ============================================================================
# 4. MACHINE LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("4. MACHINE LEARNING MODELS")
print("="*80)

# Dictionary to store model results
ml_results = {}

# Define models
ml_models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes': GaussianNB(),
    'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42, verbose=-1)
}

print("\n🤖 Training and evaluating ML models...")
print("-" * 80)

for name, model in ml_models.items():
    print(f"\n📌 Training {name}...")
    
    # Train model
    model.fit(X_train_scaled, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Store results
    ml_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'model': model
    }
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_scaled, y_train_balanced, cv=5)
    print(f"   CV Score:  {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Best ML model
best_ml_model = max(ml_results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n🏆 Best ML Model: {best_ml_model[0]}")
print(f"   Accuracy: {best_ml_model[1]['accuracy']:.4f}")

# Classification report for best model
print(f"\n📊 Detailed Classification Report for {best_ml_model[0]}:")
y_pred_best = best_ml_model[1]['model'].predict(X_test_scaled)
print(classification_report(y_test, y_pred_best, 
                          target_names=le_target.classes_,
                          zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            cbar_kws={'label': 'Count'})
plt.title(f'Confusion Matrix - {best_ml_model[0]}', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_best_ml.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/confusion_matrix_best_ml.png")

# ============================================================================
# 5. DEEP LEARNING MODELS
# ============================================================================
print("\n" + "="*80)
print("5. DEEP LEARNING MODELS")
print("="*80)

print("\n🧠 Building Deep Neural Network...")

# Convert to one-hot encoding for DL
num_classes = len(np.unique(y_train_balanced))
y_train_dl = keras.utils.to_categorical(y_train_balanced, num_classes)
y_test_dl = keras.utils.to_categorical(y_test, num_classes)

# Build neural network
def create_neural_network(input_dim, num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(16, activation='relu'),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create and train model
dl_model = create_neural_network(X_train_scaled.shape[1], num_classes)

print("\n📋 Model Architecture:")
dl_model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

print("\n🚀 Training Deep Learning Model...")
history = dl_model.fit(
    X_train_scaled, y_train_dl,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

# Evaluate DL model
print("\n📊 Evaluating Deep Learning Model...")
dl_loss, dl_accuracy = dl_model.evaluate(X_test_scaled, y_test_dl, verbose=0)
print(f"   Test Loss: {dl_loss:.4f}")
print(f"   Test Accuracy: {dl_accuracy:.4f}")

# DL predictions
y_pred_dl = dl_model.predict(X_test_scaled, verbose=0)
y_pred_dl_classes = np.argmax(y_pred_dl, axis=1)

# DL metrics
dl_precision = precision_score(y_test, y_pred_dl_classes, average='weighted', zero_division=0)
dl_recall = recall_score(y_test, y_pred_dl_classes, average='weighted', zero_division=0)
dl_f1 = f1_score(y_test, y_pred_dl_classes, average='weighted', zero_division=0)

print(f"   Precision: {dl_precision:.4f}")
print(f"   Recall:    {dl_recall:.4f}")
print(f"   F1-Score:  {dl_f1:.4f}")

# Plot training history
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Accuracy plot
axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)

# Loss plot
axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/dl_training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("\n✅ Saved: visualizations/dl_training_history.png")

# DL Confusion matrix
cm_dl = confusion_matrix(y_test, y_pred_dl_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm_dl, annot=True, fmt='d', cmap='Purples',
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Deep Learning Model', fontsize=14, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig('visualizations/confusion_matrix_dl.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/confusion_matrix_dl.png")

# ============================================================================
# 6. MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("6. MODEL COMPARISON")
print("="*80)

# Add DL results to comparison
ml_results['Deep Learning (Neural Network)'] = {
    'accuracy': dl_accuracy,
    'precision': dl_precision,
    'recall': dl_recall,
    'f1_score': dl_f1
}

# Create comparison dataframe
comparison_df = pd.DataFrame(ml_results).T[['accuracy', 'precision', 'recall', 'f1_score']]
comparison_df = comparison_df.sort_values('accuracy', ascending=False)

print("\n📊 Model Performance Comparison:")
print(comparison_df.to_string())

# Save comparison to CSV
comparison_df.to_csv('model_comparison.csv')
print("\n✅ Saved: model_comparison.csv")

# Visualize comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
    ax = axes[idx // 2, idx % 2]
    comparison_df[metric].plot(kind='barh', ax=ax, color='steelblue', edgecolor='black')
    ax.set_title(f'Model Comparison - {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel(title, fontsize=10)
    ax.set_ylabel('Model', fontsize=10)
    ax.set_xlim([0, 1])
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, v in enumerate(comparison_df[metric]):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/model_comparison.png")

# Feature importance (for tree-based models)
print("\n🔍 Feature Importance Analysis...")
rf_model = ml_results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 Top 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'], color='teal', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Top 15 Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("✅ Saved: visualizations/feature_importance.png")

# ============================================================================
# 7. FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("7. FINAL SUMMARY")
print("="*80)

best_overall_model = comparison_df.index[0]
best_accuracy = comparison_df.iloc[0]['accuracy']

print(f"\n🏆 BEST MODEL: {best_overall_model}")
print(f"   Accuracy:  {comparison_df.iloc[0]['accuracy']:.4f}")
print(f"   Precision: {comparison_df.iloc[0]['precision']:.4f}")
print(f"   Recall:    {comparison_df.iloc[0]['recall']:.4f}")
print(f"   F1-Score:  {comparison_df.iloc[0]['f1_score']:.4f}")

print("\n📁 Generated Files:")
print("   ✓ visualizations/target_distribution.png")
print("   ✓ visualizations/correlation_heatmap.png")
print("   ✓ visualizations/feature_distributions.png")
print("   ✓ visualizations/confusion_matrix_best_ml.png")
print("   ✓ visualizations/confusion_matrix_dl.png")
print("   ✓ visualizations/dl_training_history.png")
print("   ✓ visualizations/model_comparison.png")
print("   ✓ visualizations/feature_importance.png")
print("   ✓ model_comparison.csv")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)
print("\n🎯 Key Insights:")
print(f"   • Dataset contains {df.shape[0]} patients with {df.shape[1]-1} features")
print(f"   • {len(le_target.classes_)} diagnosis classes: {', '.join(le_target.classes_)}")
print(f"   • Best performing model: {best_overall_model}")
print(f"   • Achieved {best_accuracy:.2%} accuracy on test data")
print(f"   • SMOTE used to handle class imbalance")
print(f"   • {len(ml_models)} ML models and 1 DL model evaluated")
print("\n")
