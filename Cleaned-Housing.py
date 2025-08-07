import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('Housing.csv')

# One-hot encode 'furnishingstatus'
# Identify all categorical columns
categorical_cols = df.select_dtypes(include='object').columns

# Convert all categorical (string) columns to numbers using one-hot encoding
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df_encoded.dtypes)



# Define features (X) and target (y)
X = df_encoded.drop('price', axis=1)
y = df_encoded['price']

# Check for NaN in full dataset before splitting
print("Missing values before train-test split:")
print("X missing:", X.isnull().sum().sum())
print("y missing:", y.isnull().sum())

# Drop rows with any NaN in X or y
combined = pd.concat([X, y], axis=1)
combined = combined.dropna()

# Separate cleaned X and y
X = combined.drop('price', axis=1)
y = combined['price']

# Final check
print("Cleaned shapes:", X.shape, y.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Coefficients
coefficients = pd.DataFrame({'Feature': X.columns, 'Coefficient': lr_model.coef_})
print(coefficients)

# Make predictions
y_pred = lr_model.predict(X_test)

# Evaluate the model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

df_encoded.to_csv(r"D:\AIML PROJECTS\cleaned_house.csv", index=False)