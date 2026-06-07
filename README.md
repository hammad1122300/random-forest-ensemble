# Random Forest Ensemble Learning - Iris Flower Classification

## Project Overview
This project demonstrates the implementation of **Random Forest**, a powerful ensemble learning technique using the **Bagging method** for multi-class classification of iris flowers. The model achieves **91.11% accuracy** on the test dataset.

## Dataset Information
- **Name**: Iris Flower Dataset
- **Total Samples**: 150 (105 training, 45 testing)
- **Features**: 4 numerical features
  - Sepal Length (cm)
  - Sepal Width (cm)
  - Petal Length (cm)
  - Petal Width (cm)
- **Classes**: 3 (Setosa, Versicolor, Virginica)
- **Split Ratio**: 70% training, 30% testing

## Algorithm: Random Forest
Random Forest is an ensemble learning method that:
1. Creates multiple decision trees using bootstrap samples
2. Trains each tree independently on random data subsets
3. Uses majority voting for classification predictions
4. Combines predictions to produce robust, accurate results

### Model Configuration
```python
RandomForestClassifier(
    n_estimators=100,      # 100 decision trees
    max_depth=10,          # Maximum tree depth
    min_samples_split=5,   # Minimum samples to split
    min_samples_leaf=2,    # Minimum samples per leaf
    random_state=42
)
```

## Results & Performance

### Overall Metrics
| Metric | Score |
|--------|-------|
| **Accuracy** | 91.11% |
| **Precision** | 91.55% |
| **Recall** | 91.11% |
| **F1-Score** | 91.07% |

### Per-Class Performance
| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Setosa** | 100.00% | 100.00% | 100.00% | 15 |
| **Versicolor** | 82.35% | 93.33% | 87.50% | 15 |
| **Virginica** | 92.31% | 80.00% | 85.71% | 15 |

### Feature Importance
1. **Petal Width**: 0.4620 (46.20%) - Most Important
2. **Petal Length**: 0.4032 (40.32%)
3. **Sepal Length**: 0.1205 (12.05%)
4. **Sepal Width**: 0.0144 (1.44%) - Least Important

## Key Findings
- ✓ Perfect classification (100%) for Setosa flowers
- ✓ Petal measurements are dominant classification factors
- ✓ Effective ensemble reduces overfitting
- ✓ High confidence predictions (most > 90%)
- ✓ Robust handling of multi-class classification

## Project Structure
```
├── iris_random_forest_project.py    # Main Python source code
├── ML_Project_Random_Forest.docx    # Complete project report
├── random_forest_results.png        # Visualization (6 subplots)
├── model_predictions.csv            # Detailed predictions
├── feature_importance.csv           # Feature rankings
└── README.md                         # This file
```

## Files Description

### 1. iris_random_forest_project.py
Complete Python implementation including:
- Dataset loading and exploration
- Train-test split (70-30)
- Random Forest model training
- Predictions and evaluation
- Visualization generation (6 subplots)
- CSV output files

### 2. ML_Project_Random_Forest.docx
Professional project report containing:
- Project overview and objectives
- Dataset information
- Algorithm explanation
- Performance results and metrics
- Feature importance analysis
- Comprehensive visualization
- Conclusions and recommendations
- Implementation details

### 3. random_forest_results.png
Comprehensive visualization with 6 subplots:
1. **Confusion Matrix**: Shows classification accuracy per class
2. **Feature Importance**: Bar chart of feature contributions
3. **Model Performance**: Comparison of evaluation metrics
4. **Prediction Distribution**: Actual vs predicted class counts
5. **Confidence Distribution**: Histogram of prediction confidence
6. **Model Summary**: Text summary of key metrics

### 4. model_predictions.csv
CSV file with detailed predictions including:
- Actual species labels
- Predicted species labels
- Prediction confidence scores
- Correct/Incorrect classification

### 5. feature_importance.csv
CSV file ranking features by importance score

## Technologies Used

### Programming Language
- Python 3.8+

### Libraries
- **scikit-learn**: Machine learning algorithms
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization

### Installation
```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Python Script
```bash
python iris_random_forest_project.py
```

### 3. Output Files Generated
```
✓ random_forest_results.png - Comprehensive visualization
✓ model_predictions.csv - Detailed predictions
✓ feature_importance.csv - Feature importance rankings
```

## Advantages of Random Forest for This Project
✓ Achieves high accuracy on iris dataset
✓ Automatically handles feature importance
✓ Robust to outliers and noisy data
✓ No need for feature scaling
✓ Parallel training capability
✓ Reduces overfitting through ensemble voting

## Disadvantages Considered
✗ Computational cost with 100 trees
✗ Black box model (less interpretable)
✗ Memory usage during inference
✗ Slower inference than single tree

## Learning Objectives Met
1. ✓ Understand ensemble learning concepts
2. ✓ Implement bagging (Bootstrap Aggregating)
3. ✓ Train Random Forest classifier
4. ✓ Evaluate model performance
5. ✓ Analyze feature importance
6. ✓ Create professional visualizations
7. ✓ Document project comprehensively

## Conclusion
This project successfully demonstrates Random Forest ensemble learning for multi-class iris flower classification with 91.11% accuracy. The implementation showcases best practices in machine learning including proper data splitting, comprehensive evaluation, feature importance analysis, and professional visualization.

## Future Improvements
- Test with other ensemble methods (Gradient Boosting, XGBoost)
- Implement hyperparameter tuning using GridSearchCV
- Add cross-validation for more robust evaluation
- Test on other datasets for generalization
- Implement model persistence (pickle/joblib)

## References
- Scikit-learn Random Forest Documentation
- Ensemble Learning Overview (from course materials)
- Iris Dataset: https://archive.ics.uci.edu/ml/datasets/iris

---

**Project Date**: 2026  
**Author**: Machine Learning Student  
**Course**: Machine Learning Fundamentals  
**Grade Target**: A+ (91%+ accuracy achieved)
