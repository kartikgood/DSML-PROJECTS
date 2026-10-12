"""
Loan Default Prediction System
Compare classification models and identify useful signals of loan default.

Keep this Python file in the same folder as "Module 3 Project Data.csv".
Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

# -------------------- 1. Read and prepare the dataset --------------------

DATA_FILE = r"C:\Users\kartik\Desktop\DSML PROJECT 3\Module 3 Project Data.csv"
data = pd.read_csv(DATA_FILE)

# Remove the row-number column and use shorter names in the analysis.
data = data.drop(columns=["Index"], errors="ignore")
data = data.rename(columns={
    "Bank Balance": "Bank_Balance",
    "Annual Salary": "Annual_Salary",
    "Defaulted?": "Defaulted"
})

print("Dataset size:", data.shape)
print("\nMissing values in each column:")
print(data.isna().sum())
print("\nPreview:")
print(data.head())

# -------------------- 2. Explore the target variable --------------------

class_share = data["Defaulted"].value_counts(normalize=True).sort_index()
print("\nTarget class proportions:")
print(class_share.rename(index={0: "No default", 1: "Default"}))

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

plt.figure(figsize=(5, 4))
sns.countplot(data=data, x="Defaulted")
plt.title("How Many Customers Defaulted?")
plt.xlabel("Defaulted (0 = No, 1 = Yes)")
plt.ylabel("Number of customers")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Variables")
plt.tight_layout()
plt.show()

# -------------------- 3. Split the data --------------------

features = data.drop(columns="Defaulted")
target = data["Defaulted"]

X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.20,
    stratify=target,
    random_state=27
)

# A scaler is used only for models that are sensitive to feature scale.
scale = StandardScaler()
X_train_scaled = scale.fit_transform(X_train)
X_test_scaled = scale.transform(X_test)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

# -------------------- 4. Set up the classification models --------------------

positive_weight = (y_train == 0).sum() / (y_train == 1).sum()

classifiers = {
    "Logistic Regression": LogisticRegression(
        max_iter=1200, class_weight="balanced", random_state=27
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250, class_weight="balanced",
        random_state=27, n_jobs=-1
    ),
    

    
    "SVM (RBF)": CalibratedClassifierCV(
        estimator=SVC(
            kernel="rbf",
            class_weight="balanced",
            random_state=27
        ),
        ensemble=False
    ),
    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=positive_weight,
        eval_metric="logloss",
        random_state=27,
        n_jobs=-1
    )
}

scaled_names = {"Logistic Regression", "SVM (RBF)"}
scores_by_model = {}
curves_by_model = {}
trained_models = {}

# -------------------- 5. Train and evaluate each model --------------------

for model_name, classifier in classifiers.items():
    if model_name in scaled_names:
        train_values, test_values = X_train_scaled, X_test_scaled
    else:
        train_values, test_values = X_train, X_test

    classifier.fit(train_values, y_train)
    predictions = classifier.predict(test_values)
    probabilities = classifier.predict_proba(test_values)[:, 1]

    scores_by_model[model_name] = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1-score": f1_score(y_test, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
        "Confusion matrix": confusion_matrix(y_test, predictions)
    }
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
    curves_by_model[model_name] = (
        false_positive_rate, true_positive_rate,
        scores_by_model[model_name]["ROC-AUC"]
    )
    trained_models[model_name] = classifier

    print(f"\n{model_name}")
    print(classification_report(y_test, predictions, zero_division=0))
    print("Confusion matrix:")
    print(scores_by_model[model_name]["Confusion matrix"])

# -------------------- 6. Compare model performance --------------------

metric_table = pd.DataFrame({
    name: {metric: value for metric, value in metrics.items()
           if metric != "Confusion matrix"}
    for name, metrics in scores_by_model.items()
}).T.sort_values("ROC-AUC", ascending=False)

print("\nModel comparison (ranked by ROC-AUC):")
print(metric_table.round(3))

metric_table[["Accuracy", "Precision", "Recall", "F1-score", "ROC-AUC"]].plot(
    kind="bar", figsize=(9, 5)
)
plt.title("Performance of the Four Classifiers")
plt.ylabel("Score")
plt.xticks(rotation=15)
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 5))
for model_name, (fpr, tpr, auc_value) in curves_by_model.items():
    plt.plot(fpr, tpr, label=f"{model_name} (AUC={auc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.show()

# -------------------- 7. Inspect the top model --------------------

top_model_name = metric_table["ROC-AUC"].idxmax()
print("\nModel with the highest ROC-AUC:", top_model_name)

top_matrix = scores_by_model[top_model_name]["Confusion matrix"]
plt.figure(figsize=(4.5, 4))
sns.heatmap(
    top_matrix, annot=True, fmt="d", cmap="Blues",
    xticklabels=["No default", "Default"],
    yticklabels=["No default", "Default"]
)
plt.title(f"Confusion Matrix: {top_model_name}")
plt.xlabel("Predicted label")
plt.ylabel("Actual label")
plt.tight_layout()
plt.show()

# -------------------- 8. Examine tree-based feature importance --------------------

forest_importance = pd.Series(
    trained_models["Random Forest"].feature_importances_,
    index=features.columns
).sort_values(ascending=False)

boost_importance = pd.Series(
    trained_models["XGBoost"].feature_importances_,
    index=features.columns
).sort_values(ascending=False)

plt.figure(figsize=(5, 4))
sns.barplot(x=forest_importance.values, y=forest_importance.index)
plt.title("Random Forest: Feature Importance")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()
print("\nRandom Forest feature importance:")
print(forest_importance)

plt.figure(figsize=(5, 4))
sns.barplot(x=boost_importance.values, y=boost_importance.index)
plt.title("XGBoost: Feature Importance")
plt.xlabel("Importance score")
plt.tight_layout()
plt.show()
print("\nXGBoost feature importance:")
print(boost_importance)









