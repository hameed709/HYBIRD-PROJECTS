import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([0,0,0,0,1,1,1,1,1,1])

model=LogisticRegression()
model.fit(X,y)

prob=model.predict_proba(X)[:,1]
pred=model.predict(X)
accuracy=accuracy_score(y,pred)

print(f"Preddicted Probabilities : {prob}")
print(f"Predicted classes : {pred}")
print(f"Accuracy : {accuracy}")

plt.figure(figsize=(8,6))

plt.scatter(X,y,color="blue",label="Actual Classes")

X_line=np.linspace(1,10,200).reshape(-1,1)
prob_line=model.predict_proba(X_line)[:,1]
plt.plot(X_line,prob_line,color="red",label="Sigmoid Probability Curve")

boundary=-(model.intercept_[0]/model.coef_[0][0])
plt.axvline(x=boundary,color="green",linestyle="--",label=f"Desicion boundary : {boundary:.2f}")

plt.xlabel("X Value")
plt.ylabel("Probability of class 1")
plt.title("Logistic Regression Visualization")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logistic_regression_plot.png")
plt.show()