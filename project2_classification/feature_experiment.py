import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("synthetic_student_success.csv")

# Add effort (already exists)
import numpy as np
np.random.seed(42)
df["effort"] = df["hours"] * np.random.uniform(0.7, 1.2, size=len(df))

# ---------------- Experiment: Add ONE new feature ----------------
# Example: hours squared
df["hours_sq"] = df["hours"] ** 2

# You can change this line to your own feature idea
df["interaction"] = df["hours"] * df["effort"]


X = df[["hours", "effort", "interaction"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, pred)
prec = precision_score(y_test, pred)
rec = recall_score(y_test, pred)
f1 = f1_score(y_test, pred)

print("------- RESULTS -------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-score : {f1:.4f}")
