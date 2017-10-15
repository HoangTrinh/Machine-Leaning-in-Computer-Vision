# Apply DBSCAN on hand digits dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 3/10/2017


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# Load data & reshape feature space by PCA
digits = load_digits()
X = digits.data


# Apply DBSCAN
y = DBSCAN(eps=17.6, min_samples=1).fit_predict(X)

# Visualize result
X = PCA(n_components=2).fit_transform(X)
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 2 - DBSCAN')

plt.subplot(222)
plt.scatter(X[:, 0], X[:, 1], c=digits.target)
plt.title('Bai Tap 2 - True Labels')
plt.show()
