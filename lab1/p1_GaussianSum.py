import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Create 1000 samples from a Gaussian distribution with mean -10 and std. dev. 5.
# Create another 1000 samples from another independent Gaussian with mean 10 and 
# std dev 5.
#
# a) Take the sum of these two Gaussians by adding the two sets of 1000 points,
# point by point, and plot the histogram of the resulting 1000 points. What do
# you observe?
#
# b) Estimate the mean and variance of the sum.



# Part A
mean1 = -10
stdv1 = 5
sampleSize = 1000
sample1 = np.random.normal(mean1, stdv1, sampleSize)

mean2= 10
stdv2 = 5
sample2 = np.random.normal(mean2, stdv2, sampleSize)

sum_samples = sample1 + sample2

plt.hist(sum_samples, bins=10, density=True, alpha = 0.7, color = 'blue',
         label='Sum of Gaussians')
plt.title('Histogram of Sum of Gaussian Distributions')
plt.grid(True)
plt.show()

# We observe that the shape of the Gaussian distribution shape is maintained, with the 
# peak being around 0 and being decently evenly distributed throughout.



#Part B
mean = np.mean(sum_samples)
variance = np.var(sum_samples)

print("The estimated mean of the sum of samples is: " + str(mean))
print("The estimated variance of the sum: " + str(variance))

# Mean is estimated to be around 0.
# Variance is estimated to be around 50.