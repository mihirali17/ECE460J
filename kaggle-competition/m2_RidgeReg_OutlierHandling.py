import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from scipy import stats

# Load the data
train_df = pd.read_csv('kaggle-competition/zooglepredict-2023/train_final.csv')
test_df = pd.read_csv('kaggle-competition/zooglepredict-2023/test_final.csv')

# Separate the target variable from the training data
X_train = train_df.drop('Y', axis=1)
y_train = train_df['Y']

# Data Preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Calculate Z-scores
z_scores = np.abs(stats.zscore(X_train_scaled))
outlier_threshold = 3  # You can adjust this threshold as needed

# Identify and cap outliers
X_train_scaled[z_scores > outlier_threshold] = np.sign(X_train_scaled[z_scores > outlier_threshold]) * outlier_threshold

# Define a function to calculate cross-validation RMSE
def cross_val_rmse(alpha):
    model = Ridge(alpha=alpha)
    rmse_scores = np.sqrt(-cross_val_score(model, X_train_scaled, y_train, scoring='neg_mean_squared_error', cv=5))
    return rmse_scores.mean()

# Define a range of alpha values
alphas = [0.01, 0.1, 1, 10, 100]

# Find the best alpha value with the lowest RMSE
best_alpha = min(alphas, key=cross_val_rmse)
print(f'Best Alpha: {best_alpha}')

# Train the Ridge Regression model with the best alpha
ridge_model = Ridge(alpha=best_alpha)
ridge_model.fit(X_train_scaled, y_train)

# Preprocess the test data
X_test = test_df  # No need to drop 'target' column because it's not present in test data
X_test_scaled = scaler.transform(X_test)

# Predict the target variable for the test data
y_test_pred = ridge_model.predict(X_test_scaled)

# Create a new DataFrame with the predictions
submission_df = pd.DataFrame({'Id': test_df['Id'], 'Y': y_test_pred})

# Write the predictions to a CSV file
submission_df.to_csv('kaggle-competition/submission2.csv', index=False)
