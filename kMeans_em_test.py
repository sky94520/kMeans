import numpy
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from kMeans import load_data_set, kMeans


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
        centroids, cluster_assment = kMeans(data_mat, k)
        value = sum(numpy.min(cdist(data_mat, centroids, 'euclidean'), axis=1) / data_mat.shape[0])
        meandistortions.append(value)
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Averange Dispersion')
    plt.title('Selecting k with the Elbow Method')
    plt.show()
