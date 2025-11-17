import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score

#----------------Dataset----------------------------

X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([0,0,0,0,1,1,1,1,1,1])

#-------------Model_Training-------------------------

model=DecisionTreeClassifier(max_depth=3)
model.fit(X,y)

#--------------Predictions----------------------------

pred=model.predict(X)
acc=accuracy_score(y,pred)

print(f"Prediction : {pred}")
print(f"Accuracy : {acc}")

#---------------Visualizing_Tree--------------------

plt.figure(figsize=(10,6))
plot_tree(model,filled=True,rounded=True,feature_names=['hours'],class_names=['fail','pass'])
plt.title("Decision Tree Visualization")
plt.savefig("decision_tree_demo.png")
plt.show()