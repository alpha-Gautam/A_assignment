# Machine Learning Model Analysis for Loan Default Prediction

## 1. How Many ML Models (Algorithms) Are There?

There are hundreds of machine learning algorithms, with estimates ranging from 100-200 commonly used ones, and thousands of variations and specialized algorithms. They can be broadly categorized as:

### Supervised Learning Algorithms

- **Linear Models**: Linear Regression, Logistic Regression, Ridge/Lasso Regression
- **Tree-based Models**: Decision Trees, Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost)
- **Support Vector Machines**: SVM, Kernel SVM
- **Bayesian Models**: Naive Bayes, Bayesian Networks
- **Neural Networks**: Feedforward NN, CNN, RNN, LSTM, Transformer
- **Ensemble Methods**: Bagging, Boosting, Stacking
- **Instance-based**: k-Nearest Neighbors (kNN)
- **Probabilistic**: Gaussian Processes

### Unsupervised Learning Algorithms

- **Clustering**: k-Means, Hierarchical Clustering, DBSCAN, Gaussian Mixture Models
- **Dimensionality Reduction**: PCA, t-SNE, Autoencoders
- **Association Rules**: Apriori, FP-Growth

### Other Categories

- **Reinforcement Learning**: Q-Learning, SARSA, Deep RL
- **Semi-supervised Learning**: Label Propagation, Co-training
- **Self-supervised Learning**: Contrastive Learning, Masked Language Models

For classification problems specifically, there are approximately 50-70 commonly used algorithms, depending on how you count variations and implementations.

## 2. How Many ML Models Can Solve This Loan Default Prediction Problem?

This is a **binary classification problem with imbalanced data**, so most supervised classification algorithms can theoretically solve it. However, practical applicability depends on:

### Suitable Algorithms (20-30 commonly used for this type of problem):

- **Logistic Regression** - Baseline model, interpretable
- **Decision Trees** - Simple, handles mixed data types
- **Random Forest** - Ensemble of decision trees
- **Gradient Boosting** (XGBoost, LightGBM, CatBoost) - Advanced ensemble methods
- **Support Vector Machines** - Effective for high-dimensional data
- **Naive Bayes** - Probabilistic approach
- **k-Nearest Neighbors** - Instance-based learning
- **Neural Networks** (MLP, TabNet) - Deep learning approaches
- **AdaBoost** - Boosting ensemble
- **Extra Trees** - Randomized tree ensemble
- **Linear Discriminant Analysis** - Statistical classification
- **Quadratic Discriminant Analysis** - Extension of LDA

### Specialized for Imbalanced Data:

- **Cost-sensitive algorithms** with class weights
- **Ensemble methods** with balanced sampling
- **Anomaly detection approaches** (treating minority class as anomalies)

### Algorithms Less Suitable:

- **Unsupervised methods** (clustering, dimensionality reduction) - Not directly applicable for supervised classification
- **Regression algorithms** - Would need adaptation for classification
- **Reinforcement learning** - Not suitable for static prediction tasks

**Conclusion**: Approximately 15-25 algorithms are commonly used and effective for binary classification problems like loan default prediction.

## 3. Why Did We Choose Random Forest?

We selected **Random Forest** for this loan default prediction problem for several key reasons:

### Technical Advantages

- **Handles Mixed Data Types**: Works well with both numerical and categorical features without extensive preprocessing
- **Robust to Overfitting**: Ensemble nature reduces variance compared to single decision trees
- **Feature Importance**: Provides built-in feature importance scores for interpretability
- **Handles Missing Values**: Can handle missing data reasonably well
- **Scalability**: Performs well on large datasets (121K+ samples)

### Problem-Specific Fit

- **Imbalanced Data Handling**: Works effectively with SMOTE oversampling
- **Non-linear Relationships**: Can capture complex interactions between features (income, credit score, employment, etc.)
- **Robustness**: Less sensitive to outliers and noisy data compared to parametric models
- **Speed**: Faster training than many deep learning alternatives

### Business Considerations

- **Interpretability**: More explainable than black-box models like neural networks
- **Production Ready**: Easier to deploy and monitor than complex models
- **Regulatory Compliance**: Feature importance helps with model explainability requirements
- **Cost-Effective**: Lower computational requirements than deep learning

### Performance Validation

Our Random Forest achieved:

- **AUC-ROC**: 0.77 (good discriminatory power)
- **Precision**: 0.72 for default predictions
- **Recall**: 0.16 for identifying defaults
- **Feature Insights**: Identified credit bureau activity as top predictor

## 4. Effect on Results If We Chose Other ML Models

### Logistic Regression

**Expected Performance**: AUC-ROC 0.70-0.75
**Effects**:

- **Pros**: Faster training/inference, highly interpretable coefficients, less prone to overfitting
- **Cons**: Assumes linear relationships, may miss complex patterns between features
- **Impact**: Lower performance on non-linear interactions (e.g., income × credit score), potentially missing 5-10% of predictive power

### Support Vector Machines (SVM)

**Expected Performance**: AUC-ROC 0.75-0.80
**Effects**:

- **Pros**: Effective in high-dimensional spaces, robust to outliers
- **Cons**: Slower training on large datasets (121K samples), sensitive to kernel parameters, harder to interpret
- **Impact**: Potentially better precision but higher computational cost, may overfit on noisy features

### XGBoost/LightGBM

**Expected Performance**: AUC-ROC 0.78-0.82
**Effects**:

- **Pros**: Often superior performance, built-in handling of missing values, faster than Random Forest
- **Cons**: More complex hyperparameters, less interpretable, higher risk of overfitting
- **Impact**: 2-5% better AUC-ROC possible, but increased complexity in production monitoring

### Neural Networks (MLP)

**Expected Performance**: AUC-ROC 0.75-0.82
**Effects**:

- **Pros**: Can capture very complex patterns, potentially better with large datasets
- **Cons**: Requires more data for training, black-box nature, prone to overfitting, higher computational cost
- **Impact**: Variable results - could be better or worse depending on architecture, but significantly harder to explain decisions

### Decision Trees (Single)

**Expected Performance**: AUC-ROC 0.65-0.72
**Effects**:

- **Pros**: Highly interpretable, handles mixed data types
- **Cons**: High variance, prone to overfitting, unstable with small data changes
- **Impact**: Much lower performance than ensemble methods, poor generalization

### Naive Bayes

**Expected Performance**: AUC-ROC 0.68-0.73
**Effects**:

- **Pros**: Very fast, works well with categorical data, interpretable probabilities
- **Cons**: Assumes feature independence (rarely true), poor with correlated features
- **Impact**: Lower performance due to violated independence assumptions in financial data

### k-Nearest Neighbors

**Expected Performance**: AUC-ROC 0.70-0.75
**Effects**:

- **Pros**: Simple, no training phase, handles non-linear data
- **Cons**: Slow inference on large datasets, sensitive to irrelevant features, requires good distance metric
- **Impact**: Reasonable performance but poor scalability for real-time predictions

## Summary of Model Choice Impact

| Model               | Expected AUC-ROC | Training Speed | Interpretability | Production Complexity | Best Use Case                         |
| ------------------- | ---------------- | -------------- | ---------------- | --------------------- | ------------------------------------- |
| Random Forest       | 0.75-0.78        | Medium         | Good             | Medium                | **Our Choice** - Balanced performance |
| XGBoost             | 0.78-0.82        | Fast           | Fair             | Medium-High           | When maximum performance needed       |
| Logistic Regression | 0.70-0.75        | Very Fast      | Excellent        | Low                   | Baseline/interpretability priority    |
| Neural Networks     | 0.75-0.82        | Slow           | Poor             | High                  | Large datasets with complex patterns  |
| SVM                 | 0.75-0.80        | Slow           | Fair             | Medium                | High-dimensional data                 |

**Why Random Forest Was Optimal**: It provided the best balance of performance, interpretability, and production feasibility for this financial risk assessment problem. The model's ability to handle mixed data types, provide feature importance, and work well with imbalanced data made it ideal for this application.
