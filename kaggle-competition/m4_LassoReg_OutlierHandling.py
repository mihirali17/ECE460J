import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_csv('kaggle-competition/zooglepredict-2023/train_final.csv')
test_data = pd.read_csv('kaggle-competition/zooglepredict-2023/test_final.csv')

# Separate the target variable from the features
X = train_data.drop(columns=['Y'])
y = train_data['Y']

# Data preprocessing: Use RobustScaler to handle outliers
scaler = RobustScaler()
X = scaler.fit_transform(X)

# Range of alpha values to try for Lasso regression
alphas = np.logspace(-4, 4, 9)

# Initialize variables to store best RMSE and corresponding alpha
best_rmse = float('inf')
best_alpha = None

# Cross-validation to find the best alpha
for alpha in alphas:
    model = Lasso(alpha=alpha)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
    avg_rmse = np.mean(rmse_scores)
    if avg_rmse < best_rmse:
        best_rmse = avg_rmse
        best_alpha = alpha

print(f"Best Alpha: {best_alpha}")
print(f"Best RMSE: {best_rmse}")

# Train the final model with the best alpha
final_model = Lasso(alpha=best_alpha)
final_model.fit(X, y)

# Make predictions on the test data
X_test = test_data.values
X_test = scaler.transform(X_test)  # Standardize test data as well
test_predictions = final_model.predict(X_test)

# Create a DataFrame for the submission
submission = pd.DataFrame({'Id': test_data['Id'], 'Y': test_predictions})

# Save the submission to a CSV file
submission.to_csv('kaggle-competition/submission4.csv', index=False)
