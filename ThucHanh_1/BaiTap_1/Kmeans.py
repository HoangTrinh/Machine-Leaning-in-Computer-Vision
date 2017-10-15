# Apply Kmeans on random samples data with 2 Gaussians
# Author: Trinh Man Hoang - 14520320
# Last Updated: 24/09/2017


import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Random 750 samples
nSamples = 750
randomState = random.randrange(500)
X, _ = make_blobs(n_samples=nSamples, centers=2, random_state=randomState)

# Apply Kmeans
yPredict = KMeans(n_clusters=2, random_state=randomState).fit_predict(X)

# Visualize result
plt.scatter(X[:, 0], X[:, 1], c=yPredict)
plt.title("Bai Tap 1")
plt.show()

