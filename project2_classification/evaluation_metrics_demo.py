import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

#-------------------Dataset-----------------------------

X=np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y=np.array([0,0,0,0,1,1,1,1,1,1])

#--------------Model training---------------------------

model=LogisticRegression()
model.fit(X,y)
pred=model.predict(X)

#------------------Evaluation Metrics----------------------

cm=confusion_matrix(y,pred)
precision=precision_score(y,pred)
recall=recall_score(y,pred)
f1=f1_score(y,pred)

print(f"Confusion Matrix : {cm}")
print(f"Precision : {precision}")
print(f"Recall : {recall}")
print(f"F1-Score : {f1}")

#---------------------visualizing confusion matrix-------------------------

plt.imshow(cm,cmap="Blues")
plt.title("Confusion Matrix")
plt.colorbar()

#Anotate the cells

for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha="center",va="center",color="black")

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()