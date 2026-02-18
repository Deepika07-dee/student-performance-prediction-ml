# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# -------------------------------------------
# Step 1: Create Sample Dataset
# -------------------------------------------

data = {
    'Attendance': [85, 90, 75, 60, 95, 80, 70, 88, 92, 65],
    'Internal_Marks': [78, 85, 72, 60, 90, 76, 68, 82, 88, 58],
    'Assignment_Score': [80, 88, 70, 55, 92, 75, 65, 85, 90, 60],
    'Study_Hours': [3, 4, 2, 1, 5, 3, 2, 4, 5, 1],
    'Final_Marks': [82, 89, 74, 58, 93, 78, 69, 86, 91, 60]
}

df = pd.DataFrame(data)

# -------------------------------------------
# Step 2: Define Features and Target
# -------------------------------------------

X = df[['Attendance', 'Internal_Marks', 'Assignment_Score', 'Study_Hours']]
y = df['Final_Marks']

# -------------------------------------------
# Step 3: Split Data
# -------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------
# Step 4: Train Model (Random Forest)
# -------------------------------------------

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------------------
# Step 5: Make Predictions
# -------------------------------------------

y_pred = model.predict(X_test)

# -------------------------------------------
# Step 6: Evaluate Model
# -------------------------------------------

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("R² Score:", round(r2, 2))
print("Mean Absolute Error:", round(mae, 2))
print("Root Mean Square Error:", round(rmse, 2))

# -------------------------------------------
# Step 7: Plot Actual vs Predicted
# -------------------------------------------

plt.figure()
plt.plot(range(len(y_test)), y_test.values, marker='o')
plt.plot(range(len(y_pred)), y_pred, marker='o')
plt.xlabel("Students")
plt.ylabel("Marks")
plt.title("Actual vs Predicted Student Marks")
plt.legend(["Actual Marks", "Predicted Marks"])
plt.show()