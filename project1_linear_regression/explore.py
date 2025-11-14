import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("data.csv")

print("\nHEAD:")
print(df.head())

print("\nINFO:")
print(df.info())

print("\nDESCRIBE:")
print(df.describe)

print("\nMISSING VALUES:")
print(df.isnull().sum())

plt.scatter(df['hours'], df['marks'], color='blue')
plt.xlabel("Hours Studied")
plt.ylabel("Marks Obtained")
plt.title("Hours vs Marks")
plt.savefig("scatter_plot.png")
plt.show()