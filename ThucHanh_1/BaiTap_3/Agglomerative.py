# Apply Agglomerative on labeled face dataset
# Author: Trinh Man Hoang - 14520320
# Last Updated: 5/10/2017


from matplotlib import pyplot as plt
from ThucHanh_1.BaiTap_3.PreparedData import load
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

# Load features from prepared file & reduce feature space by PCA
X = load()
X = PCA(n_components=2).fit_transform(X)

# Apply Agglomerative
y = AgglomerativeClustering(n_clusters=7).fit_predict(X)

# Visualize result
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.title('Bai Tap 3 - Agglomerative')
plt.show()