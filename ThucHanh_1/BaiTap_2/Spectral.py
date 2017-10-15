# Apply Spectral on hand digits dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 2/10/2017


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data

# Calculate similarity matrix
similar_data = np.corrcoef(X)

# Apply Spectral
y = SpectralClustering(n_clusters=10, eigen_solver='arpack', affinity='precomputed').fit_predict(similar_data)

# Visualize result
X = PCA(n_components=2).fit_transform(X)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 2 - Spectral')

plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=digits.target)
plt.title('Bai Tap 2 - True Labels')
plt.show()
