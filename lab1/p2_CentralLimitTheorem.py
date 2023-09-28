import pandas as pd
import numpy as np
import matplotlib.pyplot as plot

# Xi = iid Bernoulli random variable with value {-1, 1}.
# Look at random variable Zn = (1/ROOT(n))*SUM[Xi].
# Take 1000 draws from Zn and plot histogram.
# Check that for small n (say, 5-10 range) Zn does not look that much like
# a Gaussian, but when n is bigger (n = 30 or 50), it looks much more like
# a Gaussian. Check also for much bigger n (say, n = 250), to see that
# at this point, one can really start to see the bell curve.



# Number of draws
num_draws = 1000

# Number of Bernoulli random variables for each Zn
n = 250

# Generate 1000 draws of Zn
draws = []
for _ in range(num_draws):
    Xi = np.random.binomial(1, 0.5, size=n) * 2 - 1     # binomial() generates 0 or 1 -> map those values to 1 or -1
    Zn = (1 / np.sqrt(n)) * np.sum(Xi)
    draws.append(Zn)

# Create a Pandas Series from the draws
Zn_series = pd.Series(draws)

# Plot a histogram using Pandas' plot function
Zn_series.plot(kind='hist', bins=30, density=True, alpha=0.6, color='b')

plot.title('Histogram of Zn Draws')
plot.xlabel('Value')
plot.ylabel('Probability Density')
plot.show()