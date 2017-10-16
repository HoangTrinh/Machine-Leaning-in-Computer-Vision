# Apply DBSCAN
# Author: Trinh Man Hoang - 14520320
# Last Updated: 3/10/2017


from sklearn.cluster import DBSCAN


def dbscan(data,eps,minSample):
    # Load data
    X = data

    # Apply DBSCAN
    y = DBSCAN(eps=eps, min_samples=minSample).fit_predict(X)

    return y

