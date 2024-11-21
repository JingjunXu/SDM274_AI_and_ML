import numpy as np
import matplotlib.pyplot as plt
import kmeans as km
import importlib

importlib.reload(km)

class KMeansSplitMerge:
    def __init__(self, clusters_num, max_iter1 = 300, max_iter2 = 10, tol = None, random_state = None):
        self.clusters_num = clusters_num
        self.max_iter1 = max_iter1
        self.max_iter2 = max_iter2
        self.random_state = random_state
        self.labels = None
        if tol:
            self.tol = tol
        else:
            self.tol = float('-inf')
        self.centroids = None

    def fit(self, X, init_type = 1, init_centroids = None):
        # 使用kmeans获得最初的聚类结果
        kmeans = km.KMeans(self.clusters_num, self.max_iter1, self.tol, self.random_state)
        kmeans.fit(X, init_type, init_centroids)
        self.labels = kmeans.predict(X)
        self.centroids = kmeans.return_centroids()
        # 尝试对聚类进行拆分合并去优化聚类的结果
        for k in range(0, self.max_iter2):
            previous_labels = self.labels.copy()

            # 先根据类的离散程度决定哪一类需要拆开成两类
            split_class = self._find_split_class(X)
            # 更新标签
            cluster_points = X[self.labels == split_class]
            # 为了对半分，将最远的两个点作为新的类中心
            farthest_points = self._find_farthest_points(cluster_points)
            kmeans_temp = km.KMeans(clusters_num=2, max_iter=1, tol = self.tol, random_state = self.random_state)
            kmeans_temp.fit(cluster_points, init_type= 4, init_centroids=np.array(farthest_points))
            labels_temp = kmeans_temp.predict(cluster_points)
            self._update_labels(X, labels_temp, split_class, kmeans_temp.return_centroids())
            # 更新类中心
            self.centroids = self._update_centroids(X)

            # 选取距离最近的两个类中心进行合并
            nearest_centroids = self._find_nearest_points(self.centroids)
            # 更新标签
            self.labels[self.labels == np.max(nearest_centroids) + 1] = np.min(nearest_centroids) + 1
            # 对齐标签
            self.labels[self.labels > np.max(nearest_centroids) + 1] -= 1
            # 更新类中心
            self.centroids = self._update_centroids(X)

            # 检查收敛性
            if np.all(previous_labels == self.labels):
                break
    
    def _find_farthest_points(self, points):
        if len(points) < 2:
            return None
        # 计算所有点对之间的距离矩阵
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        # 将对角线元素设置为0，因为它们是点与其自身的距离
        np.fill_diagonal(dist_matrix, 0)
        # 找到最远的点对索引
        farthest_points_idxs = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        # 返回最远的两个点
        return points[farthest_points_idxs[0]], points[farthest_points_idxs[1]]
    
    
    def _find_nearest_points(self, points):
        if len(points) < 2:
            return None
        # 计算所有点对之间的距离矩阵
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))
        # 将对角线元素设置为0，因为它们是点与其自身的距离
        np.fill_diagonal(dist_matrix, 0)
        # 找到最近的点对索引
        nearest_points_idxs = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        # 返回最近的两个点的下标
        return nearest_points_idxs

    def _find_split_class(self, features):
        largest_cluster_variance = float('-inf')
        split_class = 0
        for i in np.unique(self.labels):
            cluster_points = features[self.labels == i]
            if len(cluster_points) > 1:
                # 计算簇内方差
                cluster_variance = np.var(cluster_points, axis=0).sum()
                if cluster_variance > largest_cluster_variance:
                    largest_cluster_variance = cluster_variance
                    split_class = i
        # 返回方差最大的类的标签
        return split_class

    def _update_labels(self, X, new_cluster_labels, original_cluster_label, new_centroids):
        # 更新标签
        indices = np.where(self.labels == original_cluster_label)
        max_label = np.max(self.labels)
        transformed_labels = new_cluster_labels + max_label
        self.labels[indices] = transformed_labels
        # 更新标签使得标签的下标跟类中心的下标对齐
        self.labels[self.labels > original_cluster_label] -= 1

    def _update_centroids(self, X):
        # 根据数据点的标签来更新类中心
        centroids = []
        for i in np.unique(self.labels):
            centroids.append(np.mean(X[self.labels == i], axis=0))
        return np.array(centroids)
    
    def _cal_distances(self, X, centroids):
        # 将距离存进dist列表中，每一行对应每个点距离每个类中心的距离
        dist = np.zeros(shape = (X.shape[0], self.clusters_num))
        for i in range(0, self.clusters_num):
            dist[:,i] = np.linalg.norm(X - centroids[i], axis=1)
        return dist
    
    def _update_clusters(self, dist):
        assignments = np.zeros_like(dist)
        # 获取dist中每一行最小的值的index
        min_indices = np.argmin(dist, axis=1)
        # 给最小值赋值为1，其余为0
        assignments[np.arange(dist.shape[0]), min_indices] = 1
        return assignments
    
    def predict(self, X): # 对数据集X进行聚类
        # 计算每个点到每个类中心的距离
        dist = self._cal_distances(X, self.centroids)
        # 根据距离将数据点聚类
        assignments = self._update_clusters(dist)
        labels = np.argmax(assignments, axis=1)
        labels = labels + 1
        return labels
    
    def return_centroids(self): # 返回类中心
        return self.centroids