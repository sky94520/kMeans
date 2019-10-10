"""
轮廓系数相关代码
"""
import numpy
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from kMeans import load_data_set


if __name__ == '__main__':
    data_mat = numpy.mat(load_data_set('testSet2.txt'))
    # 分辨率1280 1024
    plt.figure(figsize=(12.8, 10.24))
    # 分割出3*2=6个子图，并在1号作图
    plt.subplot(3, 2, 1)

    x1_min = min(data_mat[:, 0]) - 1
    x1_max = max(data_mat[:, 0]) + 1

    x2_min = min(data_mat[:, 0]) - 1
    x2_max = max(data_mat[:, 1]) + 1

    plt.xlim([x1_min, x1_max])
    plt.ylim([x2_min, x2_max])
    plt.title('Instance')
    plt.scatter(data_mat[:, 0].tolist(), data_mat[:, 1].tolist())

    colors = 'bgrcmykb'
    markers = 'osDv^p*+'

    clusters = [2, 3, 4, 5, 8]
    subplot_counter = 1
    sc_scores = []
    for t in clusters:
        subplot_counter += 1
        plt.subplot(3, 2, subplot_counter)
        kmeans_model = KMeans(n_clusters=t).fit(data_mat)

        for i, l in enumerate(kmeans_model.labels_):
            plt.plot(data_mat[i, 0], data_mat[i, 1], color=colors[l], marker=markers[l], ls='None')
        plt.xlim([x1_min, x1_max])
        plt.ylim([x2_min, x2_max])
        sc_score = silhouette_score(data_mat, kmeans_model.labels_, metric='euclidean')
        sc_scores.append(sc_score)

        plt.title('K=%s,silhouette coefficient=%0.03f' % (t, sc_score))

    plt.figure()
    plt.plot(clusters, sc_scores, '*-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('SC Score')

    plt.show()
