"""
使用sklearn提供的K均值聚类算法
"""
import numpy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from kMeans import load_data_set


def show_image(data_set, centroids, cluster_centers):
    colors = 'bgrcmykb'
    markers = 'osDv^p*+'
    for index in range(len(cluster_centers)):
        datum = data_set[index]
        j = int(cluster_centers[index])
        flag = markers[j] + colors[j]
        plt.plot(datum[:, 0], datum[:, 1], flag)
    # 质心
    plt.plot(centroids[:, 0], centroids[:, 1], '+k')


if __name__ == '__main__':
    data_set = numpy.mat(load_data_set('testSet.txt'))
    kmeans_model = KMeans(n_clusters=4).fit(data_set)
    # 显示模型
    show_image(data_set, kmeans_model.cluster_centers_, kmeans_model.labels_)
    plt.show()
