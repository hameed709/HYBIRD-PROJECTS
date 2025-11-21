import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ----------------------- Load Dataset -----------------------
df = pd.read_csv("synthetic_student_success.csv")

# Add new feature: Effort score
np.random.seed(42)
df['effort'] = df['hours'] * np.random.uniform(0.7, 1.2, size=len(df))

# Keep original for before-comparison
df_original = df[['hours', 'label']].copy()

# ----------------------- BEFORE Feature Engineering -----------------------
X_before = df_original[['hours']]
y = df_original['label']

X_train_b, X_test_b, y_train, y_test = train_test_split(
    X_before, y, test_size=0.3, random_state=42, stratify=y
)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train_b, y_train)
pred_before = model_before.predict(X_test_b)

acc_b = accuracy_score(y_test, pred_before)
prec_b = precision_score(y_test, pred_before)
rec_b = recall_score(y_test, pred_before)
f1_b = f1_score(y_test, pred_before)

# ----------------------- AFTER Feature Engineering -----------------------
X_after = df[['hours', 'effort']]

X_train_a, X_test_a, y_train, y_test = train_test_split(
    X_after, y, test_size=0.3, random_state=42, stratify=y
)

model_after = LogisticRegression(max_iter=1000)
model_after.fit(X_train_a, y_train)
pred_after = model_after.predict(X_test_a)

acc_a = accuracy_score(y_test, pred_after)
prec_a = precision_score(y_test, pred_after)
rec_a = recall_score(y_test, pred_after)
f1_a = f1_score(y_test, pred_after)

# ----------------------- Print Results -----------------------
print("------- BEFORE Feature Engineering -------")
print(f"Accuracy : {acc_b:.4f}")
print(f"Precision: {prec_b:.4f}")
print(f"Recall   : {rec_b:.4f}")
print(f"F1-score : {f1_b:.4f}")

print("\n------- AFTER Feature Engineering -------")
print(f"Accuracy : {acc_a:.4f}")
print(f"Precision: {prec_a:.4f}")
print(f"Recall   : {rec_a:.4f}")
print(f"F1-score : {f1_a:.4f}")

# ----------------------- Save Results -----------------------
with open("feature_engineering_results.txt", "w") as f:
    f.write("Before:\n")
    f.write(f"Accuracy: {acc_b}\nPrecision: {prec_b}\nRecall: {rec_b}\nF1: {f1_b}\n\n")
    f.write("After:\n")
    f.write(f"Accuracy: {acc_a}\nPrecision: {prec_a}\nRecall: {rec_a}\nF1: {f1_a}\n")

print("\nResults saved to feature_engineering_results.txt")
