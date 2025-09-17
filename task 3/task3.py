import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, classification_report
)

# ========================
# Load dataset
# ========================
data = pd.read_csv('covtype.csv')

# ========================
# Convert Soil_Type one-hot → single column
# ========================
soil_cols = [f'Soil_Type{i}' for i in range(1, 41)]
data_reversed = data[soil_cols].idxmax(axis=1)  # find which Soil_Type column is 1
data['Soil_Type'] = data_reversed.str.replace('Soil_Type', '', regex=False).astype(int)

# Drop original Soil_Type dummy columns
data = data.drop(columns=soil_cols)

# ========================
# Features / target
# ========================
X = data.drop(columns=['Cover_Type'])
y = data['Cover_Type']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# ========================
# Train models
# ========================
dt = DecisionTreeClassifier(random_state=42)
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

dt.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Predictions
y_pred_dt = dt.predict(X_test)
y_pred_rf = rf.predict(X_test)

# ========================
# Metrics
# ========================
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred_dt)*100:.2f}%")
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)*100:.2f}%")

print(f"Decision Tree ROC: {roc_auc_score(y_test, dt.predict_proba(X_test), multi_class='ovo'):.3f}")
print(f"Random Forest ROC: {roc_auc_score(y_test, rf.predict_proba(X_test), multi_class='ovo'):.3f}")

# ========================
# Confusion Matrices
# ========================
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_dt)).plot(
    ax=axes[0], values_format="d", cmap="Blues"
)
axes[0].set_title("Decision Tree Confusion Matrix")

ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_rf)).plot(
    ax=axes[1], values_format="d", cmap="Greens"
)
axes[1].set_title("Random Forest Confusion Matrix")

plt.tight_layout()
plt.show()

# ========================
# Classification Reports
# ========================
print("Decision Tree Report:\n", classification_report(y_test, y_pred_dt))
print("Random Forest Report:\n", classification_report(y_test, y_pred_rf))

# ========================
# Feature Importance
# ========================
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Decision Tree': dt.feature_importances_,
    'Random Forest': rf.feature_importances_
}).set_index('Feature')

importance_df.plot(kind='bar', figsize=(14, 6))
plt.title('Feature Importance - Decision Tree vs Random Forest')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.xticks(rotation=90)
plt.legend(title='Model')
plt.tight_layout()
plt.show()
