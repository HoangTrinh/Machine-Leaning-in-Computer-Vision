# Apply Agglomerative on hand digits dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 3/10/2017

from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Load data & reshape feature space by PCA
digits = datasets.load_digits()
X = digits.data
X = PCA(n_components=2).fit_transform(X)

# Apply Agglomerative
y = AgglomerativeClustering(n_clusters=10).fit_predict(X)

# Visualize result
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 2 - Agglomerative')
plt.show()