# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset (Fix the syntax error in pd.read_csv)
df_sal = pd.read_csv("Salary_Data.csv")  # Ensure the file is in the same directory

# Display first few rows
print(df_sal.head())

# Describe data
print(df_sal.describe())

# Data distribution plot (Fix deprecated function)
plt.figure(figsize=(8, 5))
plt.title('Salary Distribution Plot')
sns.histplot(df_sal['Salary'], kde=True, bins=10, color='lightcoral')
plt.show()

# Splitting variables
X = df_sal.iloc[:, :-1].values  # independent variable (Years of Experience)
y = df_sal.iloc[:, -1].values   # dependent variable (Salary)

# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create Linear Regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
y_pred_test = regressor.predict(X_test)   # predicted salaries for test set
y_pred_train = regressor.predict(X_train) # predicted salaries for train set

# Plot Training set results
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='lightcoral', label="Actual Salary")
plt.plot(X_train, y_pred_train, color='firebrick', linewidth=2, label="Predicted Salary")
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()

# Plot Test set results
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='lightcoral', label="Actual Salary")
plt.plot(X_train, y_pred_train, color='firebrick', linewidth=2, label="Predicted Salary")
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()