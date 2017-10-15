# Apply DBSCAN on labeled face dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 5/10/2017


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from ThucHanh_1.BaiTap_3.PreparedData import load
from sklearn.preprocessing import scale


# Load features from prepared file & reduce feature space by PCA
X = scale(load())
X = PCA(n_components=2).fit_transform(X)

# Apply DBSCAN
y = DBSCAN(eps=0.6, min_samples=2).fit_predict(X)

# Visualize result
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 3 - DBSCAN')
plt.show()