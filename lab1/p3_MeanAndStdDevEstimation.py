import numpy as np

# Estimate the mean and std dev from 1-dim. data: generate 25,000 samples from a
# Gaussian distribution with mean 0 and std dev 5. Then estimate the mean and std
# dev of this Gaussian using elementary numpy commands (addition, multiplication,
# division; do not use a command that takes data and returns the mean or std dev).



mean=0
stdv=5
size=25000

sample = np.random.normal(mean, stdv, size)
estimatedMean = sample.mean()
estimatedStdv = sample.std()

estimatedMeanElementary = np.sum(sample) / size
numerator = np.square(sample-estimatedMeanElementary)
estimatedStdvElementary = np.sqrt(np.sum(numerator) / size)

print("Estimated mean using pandas function is: " + str(estimatedMean))
print("Estimated Standard Deviation using pandas function is: " +str(estimatedStdv))

print("Estimated mean using elementary calculations is: " + str(estimatedMeanElementary))
print("Estimated Standard Deviation using elementary calculations is: " +str(estimatedStdvElementary))