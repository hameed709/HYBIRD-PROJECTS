import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import joblib

#Load dataset
df=pd.read_csv("data.csv")


X=df[["hours"]]  #2D dataframe
y=df["marks"]    #1D vector

#Train model
model=LinearRegression()
model.fit(X,y)

#Test model
y_pred=model.predict(X)

#Calculate error or Evaluation
mse=mean_squared_error(y,y_pred)
r2=r2_score(y,y_pred)

print(f"Coefficient : {model.coef_[0]:.4f}")
print(f"Intercept : {model.intercept_:.4f}")
print(f"MSE : {mse:.4f}")
print(f"R² : {r2:.4f}")

#Scatter plot
plt.scatter(X,y,color="blue",label="actual")
plt.plot(X,y,color='red',label="Predicted")
plt.title("Linear Regression Fit")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.legend()
plt.savefig("regression_fit.png")
plt.show()

#Save model
joblib.dump(model,"linear_model.joblib")
print("Model Saved.")