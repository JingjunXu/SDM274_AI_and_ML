import numpy as np
from cvxopt import matrix, solvers

class SVMClassifier:
    def __init__(self, kernel_type='Polynomial', lamba = 1.):
        # lamba 是控制松弛变量的超参数
        self.kernel_type = kernel_type
        self.lamba = lamba
        self.X = None
        self.y = None

    def _ker_poly(self, X1, X2, degree = 3):
        return (np.dot(X1, X2.T) + 1.0) ** degree
    
    def _ker_gaussian(self, X1, X2, sigma = 1.0):
        # 以向量化的方式计算所有行之间的欧几里得距离的平方
        sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
        # 计算高斯核
        return np.exp(-sq_dists / (2 * sigma**2))

    def _ker_sigmoid(self, X1, X2, beta=0.0625, alpha=0.0625):
        # 计算函数的线性部分
        linear_part = beta * np.dot(X1, X2.T) + alpha
        # 对数据进行异常的处理
        linear_part_clipped = np.clip(linear_part, -20, 20)
        tanh_kernel = np.tanh(linear_part_clipped)
        tanh_kernel = np.nan_to_num(tanh_kernel, nan=0.0)
        
        return tanh_kernel
    
    def fit(self, X, y, kernel_type='Polynomial', lamba = 1.):
        self.kernel_type = kernel_type
        self.lamba = lamba
        m, n = X.shape
        self.y = y.reshape(-1, 1) * 1.
        self.mean = np.mean(X, axis = 0)
        self.std = np.std(X, axis = 0)
        self.X = (X - self.mean) / self.std

        if self.kernel_type == 'Polynomial':
            # 这里采用 d=2 的多项式
            kernel = self._ker_poly(self.X, self.X, 5)
        elif self.kernel_type == 'Gaussian':
            # 这里的 sigma = 1.
            kernel = self._ker_gaussian(self.X, self.X)
        elif self.kernel_type == 'Sigmoid':
            # 这里的 beta = 1. alpha = 0
            kernel = self._ker_sigmoid(self.X, self.X)
        else:
            print('The type of the kernel is invalid')
            return False
        H = np.dot(self.y, self.y.T) * kernel

        # 转化cvxopt格式
        P = matrix(H)
        q = matrix(-np.ones((m, 1)))
        G = matrix(np.vstack((np.eye(m)*-1, np.eye(m))))
        h = matrix(np.hstack((np.zeros(m), np.ones(m) * self.lamba)))
        A = matrix(self.y.reshape(1, -1))
        b = matrix(np.zeros(1))

        # 求解方程
        solvers.options['show_progress'] = True
        if self.kernel_type == 'Sigmoid':
            sol = solvers.qp(P, q, G, h, A, b, kktsolver='ldl', options={'kktreg':1e-9})
        else:
            sol = solvers.qp(P, q, G, h, A, b)
        alphas = np.array(sol['x']).flatten()

        # 
        sv = (alphas > 1e-4)
        self.support_vectors_ = self.X[sv]
        self.alphas = alphas[sv]
        self.sv_y = self.y[sv]
        self.b = np.mean(self.sv_y - np.sum(self.alphas * self.sv_y * kernel[sv][:, sv], axis=0))

    def project(self, X):
        X = (X - self.mean) / self.std
        if self.kernel_type == 'Polynomial':
            # 这里采用 d=2 的多项式
            kernel = self._ker_poly(self.support_vectors_, X)
        elif self.kernel_type == 'Gaussian':
            # 这里的 sigma = 1.
            kernel = self._ker_gaussian(self.support_vectors_, X)
        elif self.kernel_type == 'Sigmoid':
            # 这里的 beta = 1. alpha = 0
            kernel = self._ker_sigmoid(self.support_vectors_, X)
        else:
            print('The type of the kernel is invalid')
            return False
        return np.dot(kernel.T, self.alphas.reshape(-1, 1) * self.sv_y.reshape(-1, 1)) + self.b

    def predict(self, X):
        return np.sign(self.project(X))

