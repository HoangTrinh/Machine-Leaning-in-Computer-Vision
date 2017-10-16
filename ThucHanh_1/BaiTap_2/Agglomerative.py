# Apply Agglomerative
# Author: Trinh Man Hoang - 14520320
# Last Updated: 3/10/2017

from sklearn.cluster import AgglomerativeClustering

def agglomerative(data, nCluster):
    # Load data
    X = data

    # Apply Agglomerative
    y = AgglomerativeClustering(n_clusters=nCluster).fit_predict(X)

    return y


