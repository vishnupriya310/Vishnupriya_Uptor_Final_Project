import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('student_performance_large_dataset.csv')

# Features (X) and Target (y)
X = df[["Time_Spent_on_Social_Media (hours/week)"]]
y = df["Final_Grade"] # Label encode the 'Final_Grade' column

# convert categorical data to numerical values using LabelEncoder
if y.dtypes == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)  # Convert categorical labels to numeric

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Scaling: Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform on training data
X_test_scaled = scaler.transform(X_test)  # Only transform on testing data

# Create and train the SVR model
svr = SVR(kernel="linear", C=1.0, epsilon=0.1)  # Use a more reasonable epsilon value
svr.fit(X_train_scaled, y_train)

# Make predictions
y_pred = svr.predict(X_test_scaled)

# Print predictions and evaluate model performance
print("Predictions:", y_pred)
print(f"Actual values for test set: {y_test}")

#performance using metrics like MSE or R^2
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared: {r2}")

# Visualize the results
plt.scatter(X_test, y_test, color="red", label="Actual")  # Plot actual values
plt.scatter(X_test, y_pred, color="blue", label="Predicted")  # Plot predicted values
plt.title("SVR Model - Actual vs Predicted")
plt.xlabel("Time Spent on Social Media (hours/week)")
plt.ylabel("Final Grade")
plt.legend()
plt.show()
