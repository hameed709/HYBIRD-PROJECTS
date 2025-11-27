import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# ---------------------- Load Dataset ----------------------
df = pd.read_csv("synthetic_student_success.csv")

# Add effort for consistency
np.random.seed(42)
df["effort"] = df["hours"] * np.random.uniform(0.7, 1.2, size=len(df))

X = df[["hours", "effort"]]
y = df["label"]

# ---------------------- Train/Test Split ----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---------------------- Scaling ----------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- Train Model ----------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# ---------------------- Save Model & Scaler ----------------------
joblib.dump(model, "logistic_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")

# ---------------------- Load Model & Scaler ----------------------
loaded_model = joblib.load("logistic_model.pkl")
loaded_scaler = joblib.load("scaler.pkl")
print("Model and scaler loaded successfully.")

# ---------------------- Test With New Sample ----------------------
new_hours = 6
new_effort = new_hours * np.random.uniform(0.7, 1.2)

sample = np.array([[new_hours, new_effort]])
sample_scaled = loaded_scaler.transform(sample)

prediction = loaded_model.predict(sample_scaled)
probability = loaded_model.predict_proba(sample_scaled)[0][1]

print("\n------- New Sample Test -------")
print(f"Hours studied  : {new_hours}")
print(f"Effort feature : {new_effort:.2f}")
print(f"Predicted Label: {prediction[0]}")
print(f"Probability of Passing: {probability:.4f}")
