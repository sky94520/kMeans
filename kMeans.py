from numpy import *
from matplotlib import pyplot as plt
import matplotlib


def load_data_set(filename):
    data_matrix = []
    with open(filename) as fp:
        for line in fp.readlines():
            # 多个浮点型的列
            data = line.strip().split('\t')
            float_data = [float(datum) for datum in data]
            data_matrix.append(float_data)
    return data_matrix


def euclidean_distance(vec_a, vec_b):
    """计算两个向量的欧式距离"""
    return sqrt(sum(power(vec_a - vec_b, 2)))


def random_centroids(data_set, k):
    """
    返回k个随机的质心 保证这些质心随机并且在整个数据集的边界之内
    :param data_set: 数据集
    :param k: 质心的数量
    :return: k个随机的质心
    """
    # 创建一个k行n列的矩阵，用于保存随机质心
    n = shape(data_set)[1]
    centroids = mat(zeros((k, n)))
    # 对n个特征进行遍历
    for j in range(n):
        minJ = min(data_set[:, j])
        rangeJ = float(max(data_set[:, j]) - minJ)
        centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
    return centroids


def kMeans(data_set, k, distance_measure=euclidean_distance, create_centroids=random_centroids):
    """
    k means 算法
    :param data_set: 数据集
    :param k: k means算法的k 即要生成几个簇
    :param distance_measure: 计算距离函数
    :param create_centroids: 创建质心的函数
    :return:
    """
    m = shape(data_set)[0]
    # 向量分配到某一个(簇索引值,误差)
    cluster_assment = mat(zeros((m, 2)))
    # 创建k个质心
    centroids = create_centroids(data_set, k)
    cluster_changed = True

    while cluster_changed:
        cluster_changed = False
        # 计算该点离哪个质心最近
        for i in range(m):
            min_index, min_dist = -1, inf
            # 遍历k个质心 获取一个最近的质心
            for j in range(k):
                # 计算该点和质心j的距离
                distJI = distance_measure(centroids[j, :], data_set[i, :])
                if distJI < min_dist:
                    min_dist, min_index = distJI, j
            # 分配质心索引发生了变化 则仍然需要迭代
            if cluster_assment[i, 0] != min_index:
                cluster_changed = True
            # 不断更新最小值
            cluster_assment[i, :] = min_index, min_dist ** 2
        # print(centroids)
        # 更新质心
        for cent in range(k):
            # 获取属于该簇的所有点
            ptsInClust = data_set[nonzero(cluster_assment[:, 0].A == cent)[0]]
            if len(ptsInClust) == 0:
                continue
            # 按矩阵的列进行均值计算
            centroids[cent, :] = mean(ptsInClust, axis=0)
        # 显示每一次迭代后的簇的情况
        # show_image(data_set, centroids, cluster_assment)

    return centroids, cluster_assment


def binary_kmeans(data_set, k, distance_measure=euclidean_distance):
    """
    二分 K-均值算法
    :param data_set: 样本集
    :param k: 要划分的簇的数量
    :param distance_measure: 距离计算函数
    :return: 返回同kMeans()函数
    """
    m = shape(data_set)[0]
    cluster_assment = mat(zeros((m, 2)))
    # 按照列求平均数并转换成列表
    centroid0 = mean(data_set, axis=0).tolist()[0]
    # 划分的簇
    cent_list = [centroid0]
    # 计算点到当前簇的误差
    for j in range(m):
        cluster_assment[j, 1] = distance_measure(mat(centroid0), data_set[j, :]) ** 2
    # 划分的簇小于选定值时 继续划分
    while len(cent_list) < k:
        lowestSSE = inf
        # 找到一个划分后SSE最小的簇
        for i in range(len(cent_list)):
            # 获取该簇的所有数据
            ptsInCurrCluster = data_set[nonzero(cluster_assment[:, 0].A == i)[0], :]
            # 经过划分 一个簇得到编号分别为0和1的两个簇
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distance_measure)
            # 计算分簇之后的Sum of Square Error
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(cluster_assment[nonzero(cluster_assment[:, 0].A != i)[0], 1])

            print('sseSplit, and not split', sseSplit, sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 重新编排编号
        bestClustAss[nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(cent_list)
        bestClustAss[nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit

        print('the bestCentToSplit is:', bestCentToSplit)
        print('the len of bestClustAss is', len(bestClustAss))
        cent_list[bestCentToSplit] = bestNewCents.tolist()[0]
        cent_list.append(bestNewCents.tolist()[1])
        cluster_assment[nonzero(cluster_assment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
        # 显示每一次迭代后的簇的情况
        show_image(data_set, mat(cent_list), cluster_assment)

    return mat(cent_list), cluster_assment


def main2():
    data_mat = mat(load_data_set('testSet2.txt'))
    centroids, clustAssing = binary_kmeans(data_mat, 8)
    print('----')
    # print(clustAssing)
    show_image(data_mat, centroids, clustAssing)


def main1():
    data_mat = mat(load_data_set('testSet.txt'))
    centroids, clustAssing = kMeans(data_mat, 4)
    plt.show()
    print('----')
    # print(clustAssing)
    show_image(data_mat, centroids, clustAssing)
    plt.show()


def show_image(data_set, centroids, clustAssing):
    colors = 'bgrcmykb'
    markers = 'osDv^p*+'
    for index in range(len(clustAssing)):
        datum = data_set[index]
        j = int(clustAssing[index, 0])
        flag = markers[j] + colors[j]
        plt.plot(datum[:, 0], datum[:, 1], flag)
    # 质心
    plt.plot(centroids[:, 0], centroids[:, 1], '+k')
    # plt.show()


def distSLC(vecA, vecB):  # Spherical Law of Cosines
    a = sin(vecA[0, 1] * pi / 180) * sin(vecB[0, 1] * pi / 180)
    b = cos(vecA[0, 1] * pi / 180) * cos(vecB[0, 1] * pi / 180) * \
        cos(pi * (vecB[0, 0] - vecA[0, 0]) / 180)
    return arccos(a + b) * 6371.0  # pi is imported with numpy


def clusterClubs(numClust=5):
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])])
    datMat = mat(datList)
    myCentroids, clustAssing = binary_kmeans(datMat, numClust, distance_measure=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', \
                      'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[nonzero(clustAssing[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0], ptsInCurrCluster[:, 1].flatten().A[0], marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0].flatten().A[0], myCentroids[:, 1].flatten().A[0], marker='+', s=300)
    plt.show()


if __name__ == '__main__':
    main1()
