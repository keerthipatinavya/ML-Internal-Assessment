
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = ROOT
OUT = os.path.join(ROOT, "results")
os.makedirs(OUT, exist_ok=True)

# ============================================================
# 1. CLUSTERING — Mall Customers
# ============================================================
mall = pd.read_csv(os.path.join(DATA, "Mall_Customers.csv"))

# CustomerID is an identifier, not a meaningful clustering feature.
X_cluster = mall[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Elbow method
inertias = []
silhouettes = []
K = range(2, 11)
for k in K:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(7, 5))
plt.plot(list(K), inertias, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method — Mall Customers")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "clustering_elbow.png"), dpi=200)
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(list(K), silhouettes, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Scores — Mall Customers")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "clustering_silhouette.png"), dpi=200)
plt.close()

best_k = list(K)[int(np.argmax(silhouettes))]
# For the standard Mall Customers dataset, k=5 is a conventional segmentation.
# Use the best silhouette unless it gives an obviously poor segmentation.
if best_k < 2:
    best_k = 5

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
mall["Cluster"] = kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, mall["Cluster"])
profile = mall.groupby("Cluster")[["Age", "Annual Income (k$)", "Spending Score (1-100)"]].mean().round(2)
profile.to_csv(os.path.join(OUT, "cluster_profile.csv"))

plt.figure(figsize=(8, 6))
for c in sorted(mall["Cluster"].unique()):
    part = mall[mall["Cluster"] == c]
    plt.scatter(part["Annual Income (k$)"], part["Spending Score (1-100)"], label=f"Cluster {c}")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title(f"Customer Segmentation using K-Means (k={best_k})")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT, "customer_clusters.png"), dpi=200)
plt.close()

# ============================================================
# 2. CLASSIFICATION — Social Network Ads
# ============================================================
social = pd.read_csv(os.path.join(DATA, "Social_Network_Ads.csv"))

# User ID is an identifier and is excluded.
X = social[["Gender", "Age", "EstimatedSalary"]]
y = social["Purchased"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), ["Gender"]),
    ("num", StandardScaler(), ["Age", "EstimatedSalary"])
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

clf = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(random_state=42))
])
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

metrics = {
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred),
    "Recall": recall_score(y_test, pred),
    "F1 Score": f1_score(y_test, pred)
}
pd.DataFrame([metrics]).to_csv(os.path.join(OUT, "classification_metrics.csv"), index=False)

cm = confusion_matrix(y_test, pred)
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix — Social Network Ads")
plt.xlabel("Predicted")
plt.ylabel("Actual")
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.xticks([0, 1], ["Not Purchased", "Purchased"])
plt.yticks([0, 1], ["Not Purchased", "Purchased"])
plt.tight_layout()
plt.savefig(os.path.join(OUT, "classification_confusion_matrix.png"), dpi=200)
plt.close()

# ============================================================
# 3. BOOSTING — Insurance Charges
# ============================================================
insurance = pd.read_csv(os.path.join(DATA, "insurance_pre.csv"))

X = insurance.drop(columns=["charges"])
y = insurance["charges"]

cat_cols = ["sex", "smoker"]
num_cols = ["age", "bmi", "children"]

preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

boost = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])
boost.fit(X_train, y_train)
pred = boost.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
boost_metrics = {
    "MAE": mean_absolute_error(y_test, pred),
    "RMSE": rmse,
    "R2 Score": r2_score(y_test, pred)
}
pd.DataFrame([boost_metrics]).to_csv(os.path.join(OUT, "boosting_metrics.csv"), index=False)

plt.figure(figsize=(7, 5))
plt.scatter(y_test, pred, alpha=0.7)
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
plt.plot(lims, lims)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Gradient Boosting — Actual vs Predicted Insurance Charges")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "boosting_actual_vs_predicted.png"), dpi=200)
plt.close()

# ============================================================
# Summary
# ============================================================
summary = pd.DataFrame([
    {"Task": "Clustering", "Dataset": "Mall_Customers.csv", "Algorithm": f"K-Means (k={best_k})",
     "Primary Metric": "Silhouette Score", "Score": final_silhouette},
    {"Task": "Classification", "Dataset": "Social_Network_Ads.csv", "Algorithm": "Logistic Regression",
     "Primary Metric": "Accuracy", "Score": metrics["Accuracy"]},
    {"Task": "Boosting", "Dataset": "insurance_pre.csv", "Algorithm": "Gradient Boosting Regressor",
     "Primary Metric": "R2 Score", "Score": boost_metrics["R2 Score"]}
])
summary.to_csv(os.path.join(OUT, "overall_results.csv"), index=False)

print("\n=== ML INTERNAL ASSESSMENT RESULTS ===")
print(summary.to_string(index=False))
print("\nClassification metrics:", metrics)
print("Boosting metrics:", boost_metrics)
print("Clustering k:", best_k, "Silhouette:", round(final_silhouette, 4))
print("\nResults saved in:", OUT)
