import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
import matplotlib.pyplot as plt

# MNIST is a dataset of handwritten digits, and considered one of the easiest image recognition
# problems in computer vision. We will see here how well logistic regression does, as done 
# previously in Problem 1 on the CIFAR-10 subset. We will also see that we can visualize the
# solution, and that in connection to this, sparsity can be useful.
# Visualize the dataset MNIST, and use Multi-class Logistic Regression to classify it:
#
# a) Use the fetch_openml command from sklearn.datasets to import the MNIST data set.
#
# b) Choose a reasonable train-test split, and run multi-class logistic regression on these
# using the cross entropy loss. Try to optimize the hyperparameters.
#
# c) Report your training and test loss from above.
#
# d) Choose an l1 regularizer (penalty) and see if you can get a sparse solution with almost
# as good accuracy.
#
# e) Note that in Logistic Regression, the coefficients returned (i.e. the beta's) are the same
# dimension as the data. Therefore we can pretend that the coefficients of the solution are an 
# image of the same dimension, and plot it. Do this for the 10 sets of coefficients that correspond
# to the 10 classes. You should observe that, at least for the sparse solutions, these "kind of" look
# like the digits they are classifying.



# Part A: Load the MNIST dataset using fetch_openml
mnist = fetch_openml(name='mnist_784')

# Extract features (X) and labels (y)
X = mnist.data
y = mnist.target.astype(int)



# Part B: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform multi-class logistic regression with cross-entropy loss
# Optimized hyperparameters: increased # iterations, set a higher tolerance
model = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=1000, tol=0.1)
model.fit(X_train, y_train)



# Part C: Report training and test loss
train_loss = log_loss(y_train, model.predict_proba(X_train))
test_loss = log_loss(y_test, model.predict_proba(X_test))

print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")



# Part D: Apply L1 regularization and retrain the model
l1_model = LogisticRegression(solver='saga', multi_class='multinomial', penalty='l1', max_iter=1000, tol=0.1)
l1_model.fit(X_train, y_train)

# Count non-zero coefficients to assess sparsity
non_zero_count = np.count_nonzero(l1_model.coef_)
total_coefficients = l1_model.coef_.size
sparsity_ratio = 1 - (non_zero_count / total_coefficients)  # sparse solutions will have many coefficients close to 0

# Evaluate accuracy of Part D (sparse) and Part B (non-sparse) models
y_pred_l1 = l1_model.predict(X_test)
accuracy_l1 = accuracy_score(y_test, y_pred_l1)

y_pred_b = model.predict(X_test)
accuracy_b = accuracy_score(y_test, y_pred_b)

print(f"Part D - Sparsity Ratio: {sparsity_ratio:.4f}")
print(f"Part D - Accuracy: {accuracy_l1:.4f}")
print(f"Part B - Accuracy: {accuracy_b:.4f}")



# Part E: Visualize coefficients as images for each class
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.abs(l1_model.coef_[i].reshape(28, 28)), cmap='gray')
    plt.title(f'Class {i}')
    plt.axis('off')

plt.suptitle('L1 Regularization Coefficients (Absolute Values)')
plt.show()
