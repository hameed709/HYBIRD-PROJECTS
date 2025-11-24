import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score
)

# ---------------------- Load Dataset ----------------------
df = pd.read_csv("synthetic_student_success.csv")

# Add effort feature again
np.random.seed(42)
df["effort"] = df["hours"] * np.random.uniform(0.7, 1.2, size=len(df))

X = df[["hours", "effort"]]
y = df["label"]

# ---------------------- Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Train Logistic Regression ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------------- Predictions ----------------------
probs = model.predict_proba(X_test_scaled)[:, 1]   # Probability of class 1
pred_default = (probs >= 0.5).astype(int)

# ---------------------- Metrics ----------------------
acc = accuracy_score(y_test, pred_default)
prec = precision_score(y_test, pred_default)
rec = recall_score(y_test, pred_default)
f1 = f1_score(y_test, pred_default)

print("-------- Evaluation (Threshold = 0.5) --------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")

# ---------------------- Confusion Matrix ----------------------
cm = confusion_matrix(y_test, pred_default)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix (Threshold = 0.5)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix(classification_evaluation.py).png")
plt.close()

# ---------------------- ROC Curve ----------------------
fpr, tpr, thresholds = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# ---------------------- Save Results ----------------------
with open("day12_results.txt", "w") as f:
    f.write(f"Accuracy: {acc}\n")
    f.write(f"Precision: {prec}\n")
    f.write(f"Recall: {rec}\n")
    f.write(f"F1-Score: {f1}\n")
    f.write(f"AUC: {roc_auc}\n")
    f.write("ROC curve & confusion matrix saved.\n")

print("\nResults saved: day12_results.txt, confusion_matrix.png, roc_curve.png")
