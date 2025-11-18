import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#----------------Dataset----------------------

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,1,1,1,1,1,1])

#-----------------DecisionTree(unstable model)--------------

tree=DecisionTreeClassifier(max_depth=3)
tree.fit(X,y)
tree_pred_line=tree.predict(np.linspace(1,10,200).reshape(-1,1))

#-------------------RandomForest(stable ensemble)-------------

forest=RandomForestClassifier(n_estimators=50,max_depth=3)
forest.fit(X,y)
forest_pred_line=forest.predict(np.linspace(1,10,200).reshape(-1,1))

#----------------Accuracies--------------------------

print(f"Decision Tree Accuracy : {accuracy_score(y,tree.predict(X))}")
print(f"Random Forest Accuracy : {accuracy_score(y,forest.predict(X))}")
print(f"Feature Importance : {forest.feature_importances_}")

#-----------------plotting-----------------

plt.figure(figsize=(10,6))

# Actual Points
plt.scatter(X, y, color='black', label='Actual Data')

# Decision Tree Step Function
plt.step(np.linspace(1,10,200), tree_pred_line, where='mid', 
         color='blue', label='Decision Tree', alpha=0.5)

# Random Forest Smoother Step Function
plt.step(np.linspace(1,10,200), forest_pred_line, where='mid',
         color='green', label='Random Forest', alpha=0.8)

plt.xlabel("X Value")
plt.ylabel("Predicted Class")
plt.title("Decision Tree vs Random Forest (Classification Behavior)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("random_forest_comparison.png")
plt.show()