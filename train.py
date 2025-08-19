
# Train two models on the Iris dataset, compare via CV, select the best, and save artifacts.
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ART_DIR = Path(".")
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1) Load data
iris = load_iris(as_frame=True)
df = iris.frame.copy()
df.rename(columns={"target": "target"}, inplace=True)

# 2) Train-test split
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3) Define pipelines
numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("num", StandardScaler(), numeric_features)],
    remainder="drop"
)

logreg = Pipeline(steps=[("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
rf = Pipeline(steps=[("prep", preprocessor), ("clf", RandomForestClassifier(random_state=42))])

models = {
    "LogisticRegression": logreg,
    "RandomForest": rf
}

# 4) Cross-validation comparison
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
compare = {"model": [], "cv_mean": [], "cv_std": []}
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=skf, scoring="accuracy")
    compare["model"].append(name)
    compare["cv_mean"].append(float(np.mean(cv_scores)))
    compare["cv_std"].append(float(np.std(cv_scores)))
with open("model_compare.json", "w") as f:
    json.dump(compare, f, indent=2)

# 5) Fit both and evaluate on test set
test_scores = {}
fitted = {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    test_scores[name] = float(acc)
    fitted[name] = pipe

# 6) Select best model by test accuracy
best_name = max(test_scores, key=test_scores.get)
best_model = fitted[best_name]

# 7) Save artifacts
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

# Save dataset to data/ for the Streamlit app
DATA_DIR.mkdir(parents=True, exist_ok=True)
df.to_csv(DATA_DIR / "dataset.csv", index=False)

# Save test set for performance page
test_df = X_test.copy()
test_df["target"] = y_test.values
test_df.to_csv(DATA_DIR / "test.csv", index=False)

# Metrics & reports
best_y_pred = best_model.predict(X_test)
report = classification_report(y_test, best_y_pred, output_dict=True)
cm = confusion_matrix(y_test, best_y_pred)
metrics = {
    "best_model_name": best_name,
    "best_test_accuracy": float(test_scores[best_name]),
    "best_cv_mean": float(max(compare["cv_mean"])),
    "classification_report": report
}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

# Save model info (feature names, target, class names)
model_info = {
    "feature_names": numeric_features,
    "target_name": "target",
    "class_names": iris.target_names.tolist()
}
with open("model_info.json", "w") as f:
    json.dump(model_info, f, indent=2)

# Confusion matrix image
fig, ax = plt.subplots(figsize=(4, 3))
im = ax.imshow(cm, interpolation="nearest")
ax.set_title("Confusion Matrix (Test)")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
fig.colorbar(im)
for (i, j), val in np.ndenumerate(cm):
    ax.text(j, i, int(val), ha="center", va="center")
plt.tight_layout()
fig.savefig(DATA_DIR / "confusion_matrix.png")
plt.close(fig)

print("Training complete. Artifacts saved: model.pkl, metrics.json, model_compare.json, data/dataset.csv, data/test.csv, data/confusion_matrix.png, model_info.json")
