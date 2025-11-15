# Loan Default Prediction Model

## Overview

This project implements a machine learning model to predict loan default risk using historical loan application data. The model helps financial institutions assess borrower risk and make informed lending decisions.

## Problem Statement

Financial institutions need to predict loan defaults to manage risk and maintain profitability. Traditional methods using credit scores alone are insufficient. This model considers multiple factors including income, employment, credit history, and behavioral patterns.

## Dataset

- **Source**: Loan application data with 121,856 records
- **Features**: 40+ variables including demographic, financial, and behavioral data
- **Target**: Binary classification (Default: 1=Yes, 0=No)
- **Class Distribution**: 8.08% default rate (highly imbalanced)

## Methodology

### 1. Exploratory Data Analysis (EDA)

- Analyzed data distributions and correlations
- Identified missing values and data types
- Visualized target variable distribution

### 2. Data Preprocessing

- **Missing Values**: Imputed numerical with median, categorical with mode
- **Feature Engineering**: Created age groups, employment stability categories, debt-to-income ratios
- **Encoding**: Label encoding for categorical variables
- **Scaling**: StandardScaler for numerical features

### 3. Handling Class Imbalance

- **Technique**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Result**: Balanced training set with 179,216 samples

### 4. Model Selection

- **Algorithm**: Random Forest Classifier
- **Rationale**: Handles mixed data types, provides feature importance, robust to overfitting
- **Hyperparameters**: 100 trees, random state for reproducibility

### 5. Model Evaluation

- **Metrics**: Precision, Recall, F1-Score, AUC-ROC
- **AUC-ROC**: 0.77 (good discriminatory power)
- **Precision**: 0.72 (for default predictions)
- **Recall**: 0.16 (identifies 16% of actual defaults)

## Key Findings

### Feature Importance (Top 10)

1. Credit_Bureau (8.81%) - Number of recent credit inquiries
2. Score_Source_2 (7.72%) - External credit score
3. Employment_Stability (6.35%) - Employment duration categories
4. Application_Process_Day (4.61%) - Day of week applied
5. Client_Occupation (4.55%) - Job type
6. Score_Source_3 (4.26%) - Another external score
7. Age_Group (3.91%) - Age categories
8. Client_Education (3.89%) - Education level
9. Client_Gender (3.44%) - Gender
10. Phone_Change (3.03%) - Days since phone change

### Business Insights

- Credit bureau activity is the strongest predictor of default risk
- External scoring systems provide valuable risk signals
- Employment stability and age are important demographic factors
- Application timing shows unexpected predictive power

## Production Deployment Architecture

### System Components

- **Data Pipeline**: ETL processes for batch and real-time data
- **Model Serving**: REST API with Flask/FastAPI
- **Monitoring**: Performance tracking and drift detection
- **CI/CD**: Automated deployment pipelines

### Deployment Strategies

- **Canary Deployment**: Gradual rollout with traffic splitting
- **Load Testing**: Performance validation under various loads
- **Model Monitoring**: Continuous performance and drift monitoring

### Infrastructure

- **Cloud Platform**: AWS/GCP/Azure
- **Containerization**: Docker
- **Orchestration**: Kubernetes for scalability
- **Storage**: Data lake + Redis caching

## Usage

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook main.ipynb
```

### Model Training

```python
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load and preprocess data
# ... (see main.ipynb for details)

# Handle imbalance
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_sm, y_train_sm)
```

### Model Prediction

```python
# Make predictions
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]
```

## Files Structure

```
├── main.ipynb              # Main analysis and modeling notebook
├── data/
│   ├── Dataset.csv         # Training data
│   └── Data_Dictionary.csv # Feature descriptions
├── problem_statement.txt   # Project requirements
└── README.md              # This documentation
```

## Evaluation Criteria Met

### ✓ EDA and Pre-processing

- Comprehensive data exploration
- Missing value handling
- Feature engineering

### ✓ Feature Importance

- Random Forest feature importance analysis
- Business interpretation of key drivers

### ✓ Modelling and Results

- Random Forest implementation
- SMOTE for imbalance handling
- Comprehensive evaluation metrics

### ✓ Business Solution/Interpretation

- Risk stratification recommendations
- Model limitations and use cases
- Actionable business insights

### ✓ Handling Imbalanced Dataset

- SMOTE implementation
- Alternative approaches discussed
- Performance impact analysis

### ✓ System Design

- Production architecture
- Deployment strategies (canary, monitoring)
- Infrastructure considerations

## Future Improvements

1. **Model Enhancement**

   - Try XGBoost/LightGBM for better performance
   - Hyperparameter tuning with grid/random search
   - Ensemble methods

2. **Feature Engineering**

   - Interaction features
   - Time-based features
   - External data integration

3. **Advanced Techniques**

   - Neural networks for complex patterns
   - AutoML for automated model selection
   - Explainable AI (SHAP/LIME)

4. **Production Features**
   - Model versioning and A/B testing
   - Real-time feature engineering
   - Automated retraining pipelines

## Dependencies

- pandas
- numpy
- scikit-learn
- imbalanced-learn
- matplotlib
- seaborn

## License

This project is for educational and demonstration purposes.
