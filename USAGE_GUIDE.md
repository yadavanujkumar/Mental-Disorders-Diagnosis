# Usage Guide - Mental Disorders Diagnosis Analysis

## Table of Contents
1. [Installation](#installation)
2. [Running the Analysis](#running-the-analysis)
3. [Understanding the Output](#understanding-the-output)
4. [Interpreting Results](#interpreting-results)
5. [Customization](#customization)

## Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yadavanujkumar/Mental-Disorders-Diagnosis.git
cd Mental-Disorders-Diagnosis
```

### Step 2: Install Required Packages
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn tensorflow keras xgboost lightgbm imbalanced-learn plotly jupyter
```

## Running the Analysis

### Basic Execution
Simply run the Python script:
```bash
python mental_disorders_analysis.py
```

### Expected Runtime
- On a typical machine: 2-5 minutes
- Majority of time is spent on:
  - Training 8 ML models
  - Training the Deep Learning neural network
  - Generating visualizations

## Understanding the Output

### Console Output

The script provides detailed console output organized in sections:

1. **Data Loading** - Dataset dimensions and preview
2. **EDA** - Statistical summaries and distributions
3. **Preprocessing** - Encoding and data preparation
4. **ML Models** - Performance metrics for each model
5. **Deep Learning** - Neural network training and evaluation
6. **Comparison** - Side-by-side model comparison
7. **Summary** - Best model and key insights

### Generated Files

#### Visualizations (in `visualizations/` folder)

1. **target_distribution.png**
   - Bar chart showing distribution of diagnoses
   - Helps understand class balance

2. **correlation_heatmap.png**
   - Shows relationships between all features
   - Red = positive correlation, Blue = negative correlation

3. **feature_distributions.png**
   - 4 subplots showing key features by diagnosis
   - Helps identify diagnostic patterns

4. **confusion_matrix_best_ml.png**
   - Shows prediction accuracy for best ML model
   - Diagonal = correct predictions
   - Off-diagonal = misclassifications

5. **confusion_matrix_dl.png**
   - Confusion matrix for deep learning model
   - Compare with ML models

6. **dl_training_history.png**
   - Two plots: accuracy and loss over training epochs
   - Shows learning progress and convergence

7. **model_comparison.png**
   - 4 subplots comparing all models
   - Metrics: Accuracy, Precision, Recall, F1-Score

8. **feature_importance.png**
   - Top 15 most important features from Random Forest
   - Higher values = more important for prediction

#### Data Files

1. **model_comparison.csv**
   - Tabular comparison of all models
   - Easy to import into Excel or other tools

## Interpreting Results

### Performance Metrics Explained

- **Accuracy**: Percentage of correct predictions (0-1, higher is better)
- **Precision**: When model predicts positive, how often is it correct?
- **Recall**: Of all actual positives, how many did the model find?
- **F1-Score**: Harmonic mean of precision and recall

### Typical Results

Expected performance range:
- Best models: 80-90% accuracy
- Average models: 70-80% accuracy
- Poor models: <70% accuracy

### Key Insights from Analysis

1. **Most Important Features**:
   - Mood Swing (highest importance)
   - Optimism levels
   - Sexual Activity
   - Suicidal thoughts

2. **Best Models**:
   - LightGBM typically achieves highest accuracy
   - Random Forest and XGBoost also perform well
   - Deep Learning may underperform on small datasets

3. **Class Distribution**:
   - Dataset is relatively balanced
   - SMOTE is used to ensure equal representation

## Customization

### Modifying the Analysis

#### Change Train/Test Split
```python
# In the script, find this line:
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Change test_size to desired proportion (e.g., 0.3 for 30% test)
```

#### Add More Models
```python
# In the ml_models dictionary, add new models:
ml_models = {
    # ... existing models ...
    'Your Model': YourClassifier(parameters)
}
```

#### Adjust Neural Network Architecture
```python
# In the create_neural_network function, modify layers:
def create_neural_network(input_dim, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),  # Increase neurons
        # ... add or remove layers ...
    ])
```

#### Change Hyperparameters
```python
# Example: Increase Random Forest trees
'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
```

### Using Your Own Dataset

1. Replace `mental_disorders_dataset.csv` with your data
2. Ensure your CSV has:
   - Patient identifier column
   - Feature columns
   - Target column named 'Expert Diagnose'
3. Update column names in the script if needed

## Troubleshooting

### Common Issues

**Issue**: ImportError for TensorFlow
```bash
# Solution: Install TensorFlow
pip install tensorflow
```

**Issue**: Memory errors
```bash
# Solution: Reduce batch size in DL training
batch_size=8  # Change from 16 to 8
```

**Issue**: Plots not displaying
```bash
# Solution: Ensure matplotlib backend is set
import matplotlib
matplotlib.use('Agg')  # For non-interactive backend
```

**Issue**: SMOTE fails due to small class size
```bash
# Solution: Disable SMOTE or reduce k_neighbors
smote = SMOTE(random_state=42, k_neighbors=2)
```

## Advanced Usage

### Running Specific Sections

Comment out sections you don't need:
```python
# Skip EDA visualizations
# plt.figure(...)
# ... visualization code ...
```

### Saving Models

Add this code after training:
```python
import joblib

# Save best model
joblib.dump(best_ml_model[1]['model'], 'best_model.pkl')

# Load later
loaded_model = joblib.load('best_model.pkl')
```

### Making Predictions on New Data

```python
# After training, use the model:
new_patient_data = [[...]]  # Your new patient features
new_patient_scaled = scaler.transform(new_patient_data)
prediction = best_model.predict(new_patient_scaled)
diagnosis = le_target.inverse_transform(prediction)
print(f"Predicted diagnosis: {diagnosis[0]}")
```

## Support

For issues or questions:
1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the dataset file is in the correct location
4. Review this guide for common solutions

## Next Steps

After running the analysis:
1. Review all visualizations in the `visualizations/` folder
2. Examine the model comparison CSV
3. Identify the best performing model
4. Analyze feature importance to understand key diagnostic factors
5. Consider tuning hyperparameters for better performance
