import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

df=pd.read_csv("synthetic_student_success.csv")

np.random.seed(42)
df["effort"]=df["hours"]* np.random.uniform(0.7,1.2,size=len(df)
                                            )
X=df[["hours","effort"]]
y=df["label"]

print("Missing values : \n")
print(df.isnull().sum())

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)

scalar=StandardScaler()
X_train_scalar=scalar.fit_transform(X_train)
X_test_scalar=scalar.transform(X_test)

model=LogisticRegression(max_iter=1000)
model.fit(X_train_scalar,y_train)

pred=model.predict(X_test_scalar)

acc=accuracy_score(y_test,pred)
precision=precision_score(y_test,pred)
recall=recall_score(y_test,pred)
f1=f1_score(y_test,pred)

print("--------------Results After preprocessing & scaling--------------")
print(f"Accuracy : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

with open("preprocessing_results.txt","w") as f:
    f.write(f"Accuracy : {acc}\n")
    f.write(f"Precision : {precision}\n")
    f.write(f"Recall : {recall}\n")
    f.write(f"F1-Score : {f1}\n")

print("\n Results saved in preprocessing_results.txt.")