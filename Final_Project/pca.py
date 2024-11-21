import numpy as np

class PCA:
    def __init__(self, component_num = None) -> None:
        self.component_num = component_num # 主成分的个数
        self.components = None # 主成分
        self.mean = None

    def fit(self, X):
        # 对数据做标准化
        self.mean = np.mean(X, axis = 0)
        X_standard = np.subtract(X, self.mean)
        # 计算相关系数矩阵
        cov_matrix = np.cov(X_standard.T)
        # 计算特征值和特征向量
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # 将特征向量按照特征值的大小进行排序
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        # 选取主成分
        self.components = eigenvectors[:, idxs][:, :self.component_num]
    
    def transform(self, X): # 将数据X进行降维
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)
    
    def inverse_transform(self, X): # 将降维后的数据转换回原始数据空间
        # 与主成分相乘
        inverse_X = np.dot(X, self.components.T)
        # 加上原始数据的平均值
        inverse_X += self.mean
        return inverse_X