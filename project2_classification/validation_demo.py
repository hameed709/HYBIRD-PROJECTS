# 1. Import and load
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("data.csv")

X=df[["hours"]].values
y=df["label"].values

# 2. Train/test split 

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,shuffle=True)

# 3. Training the model(LogisticRegression)

model=LogisticRegression(random_state=42,max_iter=1000)
model.fit(X_train,y_train)

# 4. Prediction

y_pred=model.predict(X_test)

# 5. Evaluation Metrics

acc=accuracy_score(y_test,y_pred)
precision=precision_score(y_test,y_pred)
recall=recall_score(y_test,y_pred)
f1=f1_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)

print("--------------------Evaluation Metrics-------------------------")

print(f"Confusion Matric : \n{cm}")
print(f"Accuracy : {acc}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

cv_fold=5
CV=cross_val_score(model,X,y,cv=cv_fold,scoring="accuracy")

print(f"\n{cv_fold} - fold CV accuracy(mean +std) : {CV.mean():.4f} ± {CV.std():.4f}")