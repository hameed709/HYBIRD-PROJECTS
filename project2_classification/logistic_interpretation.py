import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ---------------------- Load Dataset ----------------------
df = pd.read_csv("synthetic_student_success.csv")

# Add the effort feature again for consistency
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

# Coefficients
coef = model.coef_[0]
intercept = model.intercept_[0]

print("------- Logistic Regression Coefficients -------")
print(f"Coefficient for Hours : {coef[0]:.4f}")
print(f"Coefficient for Effort: {coef[1]:.4f}")
print(f"Intercept             : {intercept:.4f}")

# ---------------------- Predictions ----------------------
pred = model.predict(X_test_scaled)

# ---------------------- Evaluation ----------------------
acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("\n------- Evaluation -------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")

# ---------------------- Save Explanation ----------------------
with open("logistic_interpretation_results.txt", "w") as f:
    f.write(f"Coefficient Hours : {coef[0]}\n")
    f.write(f"Coefficient Effort: {coef[1]}\n")
    f.write(f"Intercept         : {intercept}\n")
    f.write(f"Accuracy          : {acc}\n")
    f.write(f"Precision         : {prec}\n")
    f.write(f"Recall            : {rec}\n")
    f.write(f"F1-Score          : {f1}\n")

print("\nResults saved to logistic_interpretation_results.txt")
