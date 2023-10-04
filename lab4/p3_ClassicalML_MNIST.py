import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Here we throw the kitchen sink of classical ML (i.e. pre-deep learning) on MNIST.
#
# a) Use Random Forests to try to get the best possible test accuracy on MNIST. Use Cross
# Validation to find the best settings. How well can you do? You should use the accuracy
# metric to compare to logistic regression. What are the hyperparameters of your best model?
#
# b) Use Gradient Boosting to do the same. Try your best to tune your hyper parameters. What
# are the hyperparameters of your best model?



# Part A: Random Forests

# Load the MNIST dataset using fetch_openml
mnist = fetch_openml(name='mnist_784')

# Extract features (X) and labels (y)
X = mnist.data.astype(np.uint8)  # Convert to uint8 for memory efficiency
y = mnist.target.astype(int)

# Initialize Random Forests classifier
rf_classifier = RandomForestClassifier()

# Define hyperparameters to tune
param_grid_rf = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
}

# Perform grid search with cross-validation to find the best hyperparameters
grid_search_rf = GridSearchCV(rf_classifier, param_grid_rf, cv=3, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(X, y)

# Get the best hyperparameters and corresponding accuracy
best_rf_params = grid_search_rf.best_params_
best_rf_accuracy = grid_search_rf.best_score_

print("Part A - Random Forests:")
print(f"Best Hyperparameters: {best_rf_params}")
print(f"Best Accuracy: {best_rf_accuracy:.4f}")



# Part B: Gradient Boosting

# Load a smaller subset of the MNIST dataset (one-tenth of the original size)
mnist = fetch_openml('mnist_784', version=1)
X, y = mnist.data, mnist.target

# Downsample the dataset to one-tenth of its original size
sample_size = len(X) // 10
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=sample_size, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X_sampled, y_sampled, test_size=0.2, random_state=42)

param_combinations = [
    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.05},
    {'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.01}
]

best_accuracy = 0
best_params = None

for params in param_combinations:
    gb_classifier = GradientBoostingClassifier(**params, random_state=42)
    
    cross_val_scores = cross_val_score(gb_classifier, X_train, y_train, cv=5, scoring='accuracy')
    mean_accuracy = np.mean(cross_val_scores)
    
    if mean_accuracy > best_accuracy:
        best_accuracy = mean_accuracy
        best_params = params

best_gb_classifier = GradientBoostingClassifier(**best_params, random_state=42)
best_gb_classifier.fit(X_train, y_train)

test_accuracy = best_gb_classifier.score(X_test, y_test)

print("Best Hyperparameters:", best_params)
print("Best Cross-Validation Accuracy:", best_accuracy)
print("Test Accuracy:", test_accuracy)



# OLD CODE BELOW (took forever to run)

# # Initialize Gradient Boosting classifier
# gb_classifier = GradientBoostingClassifier()

# # Define hyperparameters to tune
# param_grid_gb = {
#     'n_estimators': [100, 200, 300],  # Number of boosting stages to be used
#     'max_depth': [3, 4, 5],          # Maximum depth of the individual trees
#     'learning_rate': [0.1, 0.01],    # Step size shrinkage for the updates
# }

# # Perform grid search with cross-validation to find the best hyperparameters
# grid_search_gb = GridSearchCV(gb_classifier, param_grid_gb, cv=3, n_jobs=-1, scoring='accuracy')
# grid_search_gb.fit(X, y)

# # Get the best hyperparameters and corresponding accuracy
# best_gb_params = grid_search_gb.best_params_
# best_gb_accuracy = grid_search_gb.best_score_

# print("\nPart B - Gradient Boosting:")
# print(f"Best Hyperparameters: {best_gb_params}")
# print(f"Best Accuracy: {best_gb_accuracy:.4f}")


