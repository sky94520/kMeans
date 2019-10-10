"""
肘部观察法
"""
import numpy
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from kMeans import load_data_set


if __name__ == '__main__':
    data_mat = numpy.mat(load_data_set('testSet2.txt'))

    plt.scatter(data_mat[:, 0].tolist(), data_mat[:, 1].tolist())
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

    # 测试9种不同聚类中心数量下，每种情况的聚类质量，并作图
    K = range(1, 10)
    meandistortions = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data_mat)
        value = sum(numpy.min(cdist(data_mat, kmeans.cluster_centers_, 'euclidean'), axis=1) / data_mat.shape[0])
        meandistortions.append(value)
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Averange Dispersion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
