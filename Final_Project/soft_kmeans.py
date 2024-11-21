import numpy as np
import matplotlib as plt

class SoftKMeans:
    def __init__(self, clusters_num=3, max_iter=100, beta = 1.0, tol = None, random_state=None):
        self.clusters_num = clusters_num
        self.max_iter = max_iter
        self.beta = beta
        self.random_state = random_state
        if tol:
            self.tol = tol
        else:
            self.tol = float('-inf')

    def _init_centroids_random(self, X):
        self.feature_dim = X.shape[1]
        # 随机初始化类中心
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centroids = X[random_idx[:self.clusters_num]]
        return centroids
    
    def _init_centroids_plus(self, X): # 使用kmeans++的方法来初始化
        m, _ = X.shape
        # 随机选取第一个类中心
        np.random.seed(self.random_state)
        centroids = np.zeros(shape = (self.clusters_num, self.feature_dim))
        first_centroid_idx = np.random.choice(m)
        centroids[0] = X[first_centroid_idx]
        # 计算每个数据点到类中心的距离
        distances = np.linalg.norm(X - centroids[0], axis=1)
        for i in range(1, self.clusters_num):
            # 根据距离平方的概率去选取下一个类中心
            probabilities = distances ** 2 + 1e-10
            probabilities /= probabilities.sum()
            centroid_idx = np.random.choice(m, p=probabilities)
            centroids[i] = X[centroid_idx]
            # 计算每个数据点到新的类中心的距离
            new_distances = np.linalg.norm(X - centroids[i], axis=1)
            distances = np.minimum(distances, new_distances)
        return centroids
    
    def _init_centroids_far(self, X): # 使用最远距离的方法来初始化
        m, _ = X.shape
        # 随机选取第一个类中心
        np.random.seed(self.random_state)
        centroids = np.zeros(shape = (self.clusters_num, self.feature_dim))
        first_centroid_idx = np.random.choice(m)
        centroids[0] = X[first_centroid_idx]
        # 计算每个数据点到类中心的距离
        distances = np.linalg.norm(X - centroids[0], axis=1)
        for i in range(1, self.clusters_num):
            # 计算每个点到最近质心的距离
            distances = np.min([np.linalg.norm(X - centroid, axis=1) for centroid in centroids[:i]], axis=0)
            # 选择距离任何质心最远的点作为下一个质心
            farthest_point_idx = np.argmax(distances)
            centroids[i] = X[farthest_point_idx]
        return centroids
    
    def _cal_distances(self, X, centroids): 
        # 将距离存进dist列表中，每一行对应每个点距离每个类中心的距离
        dist = np.zeros(shape = (X.shape[0], self.clusters_num))
        for i in range(0, self.clusters_num):
            dist[:,i] = np.linalg.norm(X - centroids[i], axis=1)
        return dist
    
    def _update_clusters(self, dist): 
        exps = np.exp(-self.beta * dist)
        # 计算每一列的和，并保持输出的维度和输入的维度一致
        sums = np.sum(exps, axis=1, keepdims=True)
        # 计算softmax
        assignments = exps / sums
        return assignments

    def _update_centroids(self, X, assignments):
        centroids = np.zeros(shape = (self.clusters_num, self.feature_dim))
        # 根据数据点来更新类中心
        for i in range(0, self.clusters_num):
            centroids[i,:] = np.sum(X * np.repeat(assignments[:,i].reshape(-1, 1), self.feature_dim, axis=1), axis = 0) / np.sum(assignments[:,i])
        return centroids
    
    def fit(self, X, init_type = 1, init_centroids = None):
        _, n = X.shape
        self.feature_dim = n
        # 根据init_type去选取不同的初始化方法
        if init_type == 1:
            self.centroids = self._init_centroids_random(X)
        elif init_type == 2:
            self.centroids = self._init_centroids_plus(X)
        elif init_type == 3:
            self.centroids = self._init_centroids_far(X)
        elif init_type == 4:
            self.centroids = init_centroids
        # 迭代更新数据的标签和类中心
        for i in range(0, self.max_iter):
            # 计算每个点距离每个类中心的距离
            dist = self._cal_distances(X, self.centroids)
            # 根据距离来对每个类赋上标签
            assignments = self._update_clusters(dist)
            # 根据新的标签来更新类中心
            centroids = self._update_centroids(X, assignments)
            if np.sum(self.centroids - centroids) < self.tol:
                break
            else:
                self.centroids = centroids

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