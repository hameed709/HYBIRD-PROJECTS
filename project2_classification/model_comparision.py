import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#------------Dataset-------------------

X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([0,0,0,0,1,1,1,1,1,1])

#--------------Logistic_model_training------------

log_model=LogisticRegression()
log_model.fit(X,y)
log_prob=log_model.predict_proba(np.linspace(1,10,200).reshape(-1,1))[:,1]

#---------------Decisoin Tree-----------------

tree=DecisionTreeClassifier(max_depth=3)
tree.fit(X,y)
tree_pred=tree.predict(np.linspace(1,10,200).reshape(-1,1))

#--------------plotting--------------------

plt.figure(figsize=(10,6))

#Actual Data points
plt.scatter(X,y, color="black",label="Actual data (0 or 1)")

#Logistic Regression Probability curve
plt.plot(np.linspace(1,10,200),log_prob,color="red",label="Logistic Regression Probabiltiy")

#Decision Tree step Function
plt.step(np.linspace(1,10,200),tree_pred,where="mid",color="blue",label="Decision Tree Function")

plt.xlabel("X Value")
plt.ylabel("Prediction / Probability")
plt.title("Logistic Regression vs Decision Tree (Classification Boundary)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("model_comparison_plot.png")
plt.show()

#---------------------Accuracies------------------------------
print(f"Logistic Regression Accuracy : {accuracy_score(y,log_model.predict(X))}")

print(f"Decision Tree Accuracy : {accuracy_score(y,tree.predict(X))}")
