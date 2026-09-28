import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\kartik\Desktop\DSML assignments\titanic.csv.xls")


# 1. Display first 10 records
print("1. First 10 Records:")
print(df.head(10))


# 2. Number of rows and columns
print("\n2. Number of Rows and Columns:")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])


# 3. Data types of all variables
print("\n3. Data Types:")
print(df.dtypes)


# 4. Missing values in each column
print("\n4. Missing Values:")
print(df.isnull().sum())


# 5. Mean age of passengers
print("\n5. Mean Age:")
print(df["age"].mean())


# 6. Number of male and female passengers
print("\n6. Male and Female Passengers:")
print(df["sex"].value_counts())


# 7. Survival rate
survival_rate = df["survived"].mean()

print("\n7. Survival Rate:")
print(survival_rate)

print("Survival Rate (%):", survival_rate * 100)


# 8. Bar chart showing survival by gender
survival_gender = df.groupby("sex")["survived"].mean()

plt.figure(figsize=(6, 4))

survival_gender.plot(kind="bar")

plt.title("Survival Rate by Gender")
plt.xlabel("Gender")
plt.ylabel("Survival Rate")

plt.xticks(rotation=0)

plt.show()


# 9. Histogram of passenger ages
plt.figure(figsize=(7, 5))

plt.hist(df["age"].dropna(), bins=20)

plt.title("Distribution of Passenger Ages")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")

plt.show()


# 10. Correlation between numerical variables
numerical_data = df.select_dtypes(include=np.number)

correlation = numerical_data.corr()

print("\n10. Correlation Between Numerical Variables:")
print(correlation)


# Correlation heatmap
plt.figure(figsize=(10, 7))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Variables")

plt.show()
