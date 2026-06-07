"""
Machine Learning Project: Ensemble Learning with Random Forest
Dataset: Iris Dataset
Technique: Random Forest (Bagging Method)
Author: ML Project
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, f1_score)
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND EXPLORE THE DATASET
# ============================================================================
print("="*70)
print("RANDOM FOREST ENSEMBLE LEARNING - IRIS DATASET PROJECT")
print("="*70)
print("\n1. LOADING DATASET...\n")

# Load Iris Dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Labels (0: Setosa, 1: Versicolor, 2: Virginica)

# Create DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = iris.target_names[y]

print("Dataset Shape:", X.shape)
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nDataset Statistics:")
print(df.describe())
print("\nClass Distribution:")
print(df['Species'].value_counts())

# ============================================================================
# STEP 2: SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================================================
print("\n" + "="*70)
print("2. SPLITTING DATA INTO TRAIN AND TEST SETS")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")
print(f"Train-Test Split Ratio: 70-30")

# ============================================================================
# STEP 3: TRAIN RANDOM FOREST MODEL
# ============================================================================
print("\n" + "="*70)
print("3. TRAINING RANDOM FOREST CLASSIFIER")
print("="*70)

# Create and train Random Forest with 100 decision trees
rf_model = RandomForestClassifier(
    n_estimators=100,      # Number of trees in the forest
    max_depth=10,          # Maximum depth of the tree
    min_samples_split=5,   # Minimum samples to split a node
    min_samples_leaf=2,    # Minimum samples in leaf node
    random_state=42,
    n_jobs=-1              # Use all processors
)

rf_model.fit(X_train, y_train)

print("\nRandom Forest Model Configuration:")
print(f"  - Number of trees (n_estimators): 100")
print(f"  - Max tree depth: 10")
print(f"  - Min samples to split: 5")
print(f"  - Min samples per leaf: 2")
print(f"  - Total parameters used: {rf_model.n_features_in_}")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("4. MAKING PREDICTIONS ON TEST DATA")
print("="*70)

y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)

print(f"\nTotal predictions made: {len(y_pred)}")
print(f"\nFirst 10 predictions:")
for i in range(10):
    print(f"  Sample {i+1}: Predicted={iris.target_names[y_pred[i]]}, "
          f"Actual={iris.target_names[y_test[i]]}, "
          f"Confidence={np.max(y_pred_proba[i]):.2%}")

# ============================================================================
# STEP 5: EVALUATE MODEL PERFORMANCE
# ============================================================================
print("\n" + "="*70)
print("5. MODEL EVALUATION METRICS")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nModel Performance Metrics:")
print(f"  - Accuracy Score:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  - Precision Score: {precision:.4f}")
print(f"  - Recall Score:    {recall:.4f}")
print(f"  - F1-Score:        {f1:.4f}")

print("\n" + "-"*70)
print("DETAILED CLASSIFICATION REPORT")
print("-"*70)
print(classification_report(y_test, y_pred, 
                          target_names=iris.target_names, 
                          digits=4))

# ============================================================================
# STEP 6: FEATURE IMPORTANCE
# ============================================================================
print("\n" + "="*70)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("="*70)

feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
for idx, row in feature_importance.iterrows():
    print(f"  {row['Feature']:<30} {row['Importance']:.4f} "
          f"({'█' * int(row['Importance'] * 50)})")

# ============================================================================
# STEP 7: CREATE VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("7. GENERATING VISUALIZATIONS...")
print("="*70)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. Confusion Matrix
ax1 = plt.subplot(2, 3, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names, ax=ax1, cbar=True)
ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label', fontsize=12)
ax1.set_xlabel('Predicted Label', fontsize=12)

# 2. Feature Importance Bar Chart
ax2 = plt.subplot(2, 3, 2)
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
ax2.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
ax2.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax2.set_title('Feature Importance in Random Forest', fontsize=14, fontweight='bold')
ax2.invert_yaxis()
for i, v in enumerate(feature_importance['Importance']):
    ax2.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=10)

# 3. Model Performance Metrics
ax3 = plt.subplot(2, 3, 3)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
scores = [accuracy, precision, recall, f1]
colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
bars = ax3.bar(metrics, scores, color=colors_metrics, edgecolor='black', linewidth=1.5)
ax3.set_ylim([0, 1])
ax3.set_ylabel('Score', fontsize=12, fontweight='bold')
ax3.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax3.grid(axis='y', alpha=0.3)
for bar, score in zip(bars, scores):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{score:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# 4. Prediction Distribution
ax4 = plt.subplot(2, 3, 4)
pred_counts = pd.Series(y_pred).value_counts().sort_index()
actual_counts = pd.Series(y_test).value_counts().sort_index()
x = np.arange(len(iris.target_names))
width = 0.35
ax4.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.8, color='skyblue')
ax4.bar(x + width/2, pred_counts.values, width, label='Predicted', alpha=0.8, color='lightcoral')
ax4.set_xlabel('Iris Species', fontsize=12, fontweight='bold')
ax4.set_ylabel('Count', fontsize=12, fontweight='bold')
ax4.set_title('Actual vs Predicted Distribution', fontsize=14, fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels(iris.target_names)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

# 5. Prediction Confidence Distribution
ax5 = plt.subplot(2, 3, 5)
max_confidence = np.max(y_pred_proba, axis=1)
ax5.hist(max_confidence, bins=20, color='mediumseagreen', edgecolor='black', alpha=0.7)
ax5.axvline(max_confidence.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean: {max_confidence.mean():.3f}')
ax5.set_xlabel('Confidence Score', fontsize=12, fontweight='bold')
ax5.set_ylabel('Frequency', fontsize=12, fontweight='bold')
ax5.set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
ax5.legend()
ax5.grid(axis='y', alpha=0.3)

# 6. Model Summary Text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')
summary_text = f"""
RANDOM FOREST MODEL SUMMARY

Dataset: Iris Flowers
Samples: {len(X)} (Train: {len(X_train)}, Test: {len(X_test)})
Features: {X.shape[1]}
Classes: {len(iris.target_names)}

Model Configuration:
• Base Learners: 100 Decision Trees
• Max Tree Depth: 10
• Min Split Samples: 5
• Ensemble Method: Bagging

Performance Results:
• Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision: {precision:.4f}
• Recall:    {recall:.4f}
• F1-Score:  {f1:.4f}

Top Features:
1. {feature_importance.iloc[0, 0]}: {feature_importance.iloc[0, 1]:.4f}
2. {feature_importance.iloc[1, 0]}: {feature_importance.iloc[1, 1]:.4f}
3. {feature_importance.iloc[2, 0]}: {feature_importance.iloc[2, 1]:.4f}
"""
ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, 
         fontsize=10, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.suptitle('Random Forest Ensemble Learning - Complete Analysis', 
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()

# Save the visualization
output_path = '/mnt/user-data/outputs/random_forest_results.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Visualization saved to: {output_path}")

plt.close()

# ============================================================================
# STEP 8: SAVE RESULTS TO CSV
# ============================================================================
print("\n" + "="*70)
print("8. SAVING RESULTS")
print("="*70)

# Save predictions with confidence
results_df = pd.DataFrame({
    'Actual_Species': [iris.target_names[i] for i in y_test],
    'Predicted_Species': [iris.target_names[i] for i in y_pred],
    'Confidence': np.max(y_pred_proba, axis=1),
    'Correct': y_test == y_pred
})

results_csv_path = '/mnt/user-data/outputs/model_predictions.csv'
results_df.to_csv(results_csv_path, index=False)
print(f"✓ Predictions saved to: {results_csv_path}")

# Save feature importance
feature_importance.to_csv('/mnt/user-data/outputs/feature_importance.csv', index=False)
print(f"✓ Feature importance saved to: /mnt/user-data/outputs/feature_importance.csv")

print("\n" + "="*70)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  1. random_forest_results.png - Comprehensive visualization")
print("  2. model_predictions.csv - Detailed predictions")
print("  3. feature_importance.csv - Feature rankings")
print("\n" + "="*70)
