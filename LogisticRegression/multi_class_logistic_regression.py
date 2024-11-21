import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, class_num, input_feature = 1): 
        self.loss_iter = [] # 储存loss的值，便于后续画图
        self.class_num = class_num
        self.weight = [] # 维度 k x (n+1)
        # 初始化模型参数
        for i in range(0, class_num):
            self.weight.append(np.random.random(input_feature + 1)) 

    def _loss(self, outputs, targets): # outputs为softmax的结果
        _, num = np.array(outputs).shape
        return -np.sum((targets.T * np.log(outputs)).sum(axis=0)) / num
    
    def _softmax(self, X):
        exps = np.exp(X)
        # 计算每一列的和，并保持输出的维度和输入的维度一致
        sums = np.sum(exps, axis=0, keepdims=True)
        # 计算softmax
        softmax_matrix = exps / sums
        return softmax_matrix
    
    def _gradient(self, input, target): # outputs为softmax的结果，输出的结果更新一整个weight矩阵
        return np.dot((self._softmax((np.dot(self.weight, input.T))) - target)[:, np.newaxis], input[np.newaxis, :])
    
    def _batch_update(self, batch_size, learning_rate, inputs, targets):
        # 计算梯度
        m, n = np.array(self.weight).shape
        grad_total = np.zeros((m,n))
        for k in range(0, batch_size):
            grad_total = grad_total + self._gradient(inputs[k], targets[k])
        # 更新参数
        self.weight = self.weight - learning_rate / batch_size * grad_total
    
    def _stochastic_update(self, batch_size, learning_rate, inputs, targets): # 随机选取一个组中的一个数据更新一次参数，target为一个(num,1)的向量，input为一个(num,input_dim)的向量
        # 计算梯度
        k = int(np.random.randint(0, batch_size, size=1))
        grad_final = self._gradient(inputs[k], targets[k])
        # 更新参数
        self.weight = self.weight - learning_rate * grad_final
    
    def _preprocess_data(self, predata): # 将input转化numpy中的array, 同时扩展1在第一位为bias 
        predata = np.array(predata)
        # 处理 std 中的零值
        self.input_std = np.where(self.input_std == 0, 1e-10, self.input_std)
        # 处理 NaN 值
        predata = np.nan_to_num(predata)
        self.input_mean = np.nan_to_num(self.input_mean)
        norm_data = np.divide(np.subtract(predata, self.input_mean), self.input_std)
        m, n = predata.shape
        data_ = np.empty([m, n+1])
        data_[:,0] = 1
        data_[:,1:] = norm_data
        # standard method
        return data_
    
    def _one_hot_encode(self, targets, num_classes):
        # 创建一个形状为 (len(targets), num_classes) 的全零矩阵
        _one_hot_matrix = np.zeros((len(targets), num_classes))
        
        # 在对应位置设置为1
        for idx, target in enumerate(targets):
            _one_hot_matrix[idx, target] = 1
        
        return _one_hot_matrix

    def train(self, inputs, targets, epoch_num = 500, learning_rate = 0.001, tol = None, update_type = 0, batch_size = 1): 
        # batch_size是更新一次参数所需数据量，update_type是指使用batch_update还是stochastic_update(可选0或1)，norm选择是否对数据进行normalization(可选TRUE或FALSE)，w0和w1的初始值
        # 函数功能实现
        self.loss_iter = []
        self.input_mean = np.mean(inputs, axis = 0)
        self.input_std = np.std(inputs, axis = 0)
        if tol is None:
            tol = -float('inf')
        # 数据预处理 input和target
        # input主要是在前面加1
        inputs = self._preprocess_data(inputs)
        # target主要是在讲其变成kx1的向量
        targets = self._one_hot_encode(targets, self.class_num)
        cnt = inputs.shape[0] // batch_size
        rest = inputs.shape[0] % batch_size
        # 进行batch_update的训练
        if update_type == 0:
            for epoch in range(0,epoch_num):
                for i in range(0,cnt):
                    self._batch_update(batch_size, learning_rate, inputs[i * batch_size : (i + 1)*batch_size], targets[i * batch_size : (i + 1)*batch_size])
                # 处理余项
                if rest != 0:
                    grad_rest = np.zeros(self.weight.size)
                    for j in range(0,rest):
                        grad_rest = grad_rest + self._gradient(inputs[cnt * batch_size + j], targets[cnt * batch_size + j])
                    self.weight = self.weight - learning_rate / rest * grad_rest
                # 计算loss
                outputs = self._softmax((np.dot(self.weight, inputs.T)))
                loss = self._loss(outputs, targets)
                self.loss_iter = np.append(self.loss_iter, loss)
                if loss <= tol:
                    break
        # 进行stochastic_update的训练
        elif update_type == 1:
            for epoch in range(0,epoch_num):
                for i in range(0,cnt):
                    self._stochastic_update(batch_size, learning_rate, inputs[i * batch_size : (i + 1)*batch_size], targets[i * batch_size : (i + 1)*batch_size])
                # 处理余项
                if rest != 0:
                    j = int(np.random.randint(0, rest, size=1))
                    self.weight = self.weight - learning_rate * self._gradient(inputs[cnt*batch_size + j], targets[cnt*batch_size + j])
                # 计算loss
                outputs = self._softmax((np.dot(self.weight, inputs.T)))
                loss = self._loss(outputs, targets)
                self.loss_iter = np.append(self.loss_iter, loss)
                if loss <= tol:
                    break
        
    def predict(self, inputs):
        # 直接输出标签
        data = self._preprocess_data(inputs)
        outputs =  self._softmax((np.dot(self.weight, data.T))).T
        return np.argmax(outputs, axis=1)
    
    def show(self): # 将训练结果输出
        print(f'The learned weights by LogisticRegression is W = \n {self.weight}')
    
    def plot_loss(self): # 将loss的变化画图
        plt.plot(self.loss_iter)
        plt.grid()
        plt.show()