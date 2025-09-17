# ===============================
# 1. Imports
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from sklearn.ensemble import RandomForestClassifier


# ===============================
# 2. Load Dataset from CSV
# ===============================
data = pd.read_csv("covtype.csv")

# Features + target
X = data.drop(columns=["Cover_Type"])
y = data["Cover_Type"]

print("Original shape:", X.shape, "Target shape:", y.shape)


# ===============================
# 3. Reverse one-hot encodings
# ===============================

# --- Soil_Type (40 one-hot → single column) ---
soil_cols = [col for col in X.columns if col.startswith("Soil_Type")]
soil_reversed = X[soil_cols].idxmax(axis=1).str.replace("Soil_Type", "", regex=False).astype(int)
X["Soil_Type"] = soil_reversed
X = X.drop(columns=soil_cols)

# --- Wilderness_Area (4 one-hot → single column) ---
wilderness_cols = [col for col in X.columns if col.startswith("Wilderness_Area")]
wilderness_reversed = X[wilderness_cols].idxmax(axis=1).str.replace("Wilderness_Area", "", regex=False).astype(int)
X["Wilderness_Area"] = wilderness_reversed
X = X.drop(columns=wilderness_cols)

print("New shape after reversing one-hot:", X.shape)


# ===============================
# 4. Scale continuous features
# ===============================
continuous_features = X.columns[:10]  # first 10 are continuous

scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])


# ===============================
# 5. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


# ===============================
# 6. Train Model (Random Forest)
# ===============================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)


# ===============================
# 7. Evaluation
# ===============================
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y))
disp.plot(cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix - Random Forest")
plt.show()


# ===============================
# 8. Feature Importance
# ===============================
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(12,6))
sns.barplot(x=importances[indices][:15], y=X.columns[indices][:15], palette="viridis")
plt.title("Top 15 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()
