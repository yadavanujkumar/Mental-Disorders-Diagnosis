# Mental-Disorders-Diagnosis

A comprehensive machine learning and deep learning project for diagnosing mental disorders based on patient symptoms and behavioral patterns.

## 📋 Project Overview

This project implements a complete data science pipeline including:
- **Exploratory Data Analysis (EDA)** - Understanding the dataset through statistical analysis and visualizations
- **Artificial Intelligence (AI)** - Intelligent pattern recognition and diagnosis prediction
- **Machine Learning (ML)** - Multiple traditional ML algorithms for classification
- **Deep Learning (DL)** - Neural networks for advanced pattern recognition

## 📊 Dataset

The dataset contains 120 patient records with 18 features including:
- Behavioral symptoms (Sadness, Euphoric, Exhausted, etc.)
- Sleep patterns
- Mood indicators
- Social behaviors
- Concentration levels
- Target variable: Expert Diagnose (Bipolar Type-1, Bipolar Type-2, Depression, Normal)

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yadavanujkumar/Mental-Disorders-Diagnosis.git
cd Mental-Disorders-Diagnosis
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Analysis

Execute the main analysis script:
```bash
python mental_disorders_analysis.py
```

This will:
- Load and explore the dataset
- Generate visualizations
- Train multiple ML models
- Train a deep learning neural network
- Compare all models
- Save results and visualizations

## 🔍 Analysis Components

### 1. Exploratory Data Analysis (EDA)
- Data loading and initial exploration
- Statistical summaries
- Missing value analysis
- Target variable distribution
- Feature correlation analysis
- Feature distributions by diagnosis

### 2. Data Preprocessing
- Categorical feature encoding
- Target variable encoding
- Train-test split (80-20)
- Class imbalance handling with SMOTE
- Feature scaling with StandardScaler

### 3. Machine Learning Models

The following ML algorithms are implemented:
- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- Naive Bayes
- XGBoost
- LightGBM

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- Cross-validation scores

### 4. Deep Learning Model

A neural network architecture with:
- 5 layers (128, 64, 32, 16, output)
- Batch Normalization
- Dropout for regularization
- Adam optimizer
- Early stopping
- Learning rate reduction

### 5. Model Comparison

Comprehensive comparison of all models based on:
- Classification metrics
- Confusion matrices
- Feature importance analysis
- Performance visualizations

## 📈 Results

The analysis generates the following outputs:

### Visualizations (saved in `visualizations/` folder):
- `target_distribution.png` - Distribution of diagnoses
- `correlation_heatmap.png` - Feature correlations
- `feature_distributions.png` - Feature patterns by diagnosis
- `confusion_matrix_best_ml.png` - Best ML model confusion matrix
- `confusion_matrix_dl.png` - Deep learning confusion matrix
- `dl_training_history.png` - Neural network training progress
- `model_comparison.png` - Performance comparison across all models
- `feature_importance.png` - Most important features

### Data Files:
- `model_comparison.csv` - Detailed metrics for all models

## 🛠️ Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms
- **TensorFlow/Keras** - Deep learning framework
- **XGBoost** - Gradient boosting
- **LightGBM** - Light gradient boosting
- **Matplotlib/Seaborn** - Data visualization
- **SMOTE** - Handling imbalanced datasets

## 📊 Model Performance

All models are evaluated on:
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy
- **Recall** - Ability to find all positive cases
- **F1-Score** - Harmonic mean of precision and recall

The best performing model is automatically identified and reported.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Anuj Kumar Yadav**

## 🙏 Acknowledgments

- Mental health dataset for enabling this analysis
- Open source community for the amazing ML/DL libraries
- Healthcare professionals for domain expertise