import numpy as np

# Estimate the mean and covariance matrix for multi-dimensional data:
# generate 10,000 samples of 2-dim. data from the Gaussian distribution:
#      | Xi |      |-5|  |20 .8|
#      | Yi | ~ N( | 5|, |.8 30| )
#
# Then, estimate the mean and covariance matrix for this data using
# elementary numpy commands (addition, multiplication, division).
# Do not use a command that takes data and returns the mean or std dev.



# Mean and covariance matrix calculation method (standard numpy commands) below:

# Set the random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 10000

# Mean vector
mean_vector = np.array([-5, 5])

# Covariance matrix
cov_matrix = np.array([[20, 0.8],
                      [0.8, 30]])

# Generate 10,000 samples of 2-dimensional data from the Gaussian distribution
samples = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)

# Calculate the sample mean (average)
sample_mean = np.mean(samples, axis=0)

# Calculate the sample covariance matrix
sample_cov_matrix = np.cov(samples, rowvar=False)

print("Sample Mean:")
print(sample_mean)
print("\nSample Covariance Matrix:")
print(sample_cov_matrix)

print("-------------------------------")



# Mean and covariance matrix estimation method (elementary numpy commands) below:

np.random.seed(0)

num_samples = 10000

mean_vector = [-5, 5]

cov_matrix = [[20, 0.8], [0.8, 30]]

samples = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)

# Compute the sample mean vector manually
sum_x = np.sum(samples[:, 0])
sum_y = np.sum(samples[:, 1])
estimated_mean_vector = [sum_x / num_samples, sum_y / num_samples]

# Compute the sample covariance matrix vector manually
diff_x = samples[:, 0] - estimated_mean_vector[0]
diff_y = samples[:, 1] - estimated_mean_vector[1]

cov_xx = np.sum(diff_x * diff_x) / (num_samples - 1)
cov_xy = np.sum(diff_x * diff_y) / (num_samples - 1)
cov_yy = np.sum(diff_y * diff_y) / (num_samples - 1)

estimated_cov_matrix = [[cov_xx, cov_xy], [cov_xy, cov_yy]]

print("Estimated Sample Mean:")
print(estimated_mean_vector)
print("\nEstimated Sample Covariance Matrix:")
print(estimated_cov_matrix)