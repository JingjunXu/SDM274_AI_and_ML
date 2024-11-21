import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self, input_feature = 1): 
    # 由于对于LogisticsRegression来说，输出的是一个值，故不需要output_size这一变量
        self.loss_iter = [] # 储存loss的值，便于后续画图
        self.weight = np.random.random(input_feature + 1) # 初始化模型参数

    def _loss(self, outputs, targets): # outputs = w^t * x + w0
        return np.sum(np.log(1.0 + np.exp(-outputs)) + np.multiply(targets, outputs)) / len(targets)
    
    def _gradient(self, input, target): 
        return np.subtract(np.multiply(target, input), np.multiply(input, np.exp(-np.dot(self.weight,input)) / (1.0 + np.exp(-np.dot(self.weight,input)))))
    
    def _batch_update(self, batch_size, learning_rate, inputs, targets):
        # 计算梯度
        grad_total = np.zeros(self.weight.size)
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
    
    def _preprocess_data(self, predata): # 将数据转化numpy中的array，并选择是否对数据进行标准化
        predata = np.array(predata)
        norm_data = np.divide(np.subtract(predata, self.input_mean), self.input_std)
        m, n = predata.shape
        data_ = np.empty([m, n+1])
        data_[:,0] = 1
        data_[:,1:] = norm_data
        # standard method
        return data_
    
    def train(self, inputs, targets, epoch_num = 500, learning_rate = 0.001, tol = None, update_type = 0, batch_size = 1): 
        # batch_size是更新一次参数所需数据量，update_type是指使用batch_update还是stochastic_update(可选0或1)，norm选择是否对数据进行normalization(可选TRUE或FALSE)，w0和w1的初始值
        # 函数功能实现
        self.loss_iter = []
        self.input_mean = np.mean(inputs, axis = 0)
        self.input_std = np.std(inputs, axis = 0)
        if tol is None:
            tol = 0
        inputs = self._preprocess_data(inputs)
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
                outputs = np.dot(inputs,self.weight)
                loss = self._loss(outputs, targets)
                self.loss_iter = np.append(self.loss_iter, loss)
                if (epoch+1) % 100 == 0:
                    print(f"Finished epoch {(epoch + 1)}/{epoch_num}: "f"loss {self.loss_iter[-1]:.6f}, "f"lr: {learning_rate:.6f}")
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
                outputs = np.dot(inputs,self.weight)
                loss = self._loss(outputs, targets)
                self.loss_iter = np.append(self.loss_iter, loss)
                if loss <= tol:
                    break
        
    def predict(self, inputs):
        data = self._preprocess_data(inputs)
        output = 1.0 / (1.0 + np.exp(-np.dot(data,self.weight)))
        res = []
        for i in output:
            if i >= 0.5:
                res.append(0)
            else:
                res.append(1)
        return res
    
    def show(self): # 将训练结果输出
        print(f'The learned weights by LogisticRegression is W = \n {self.weight}')
    
    def return_loss(self):
        return self.loss_iter

    def plot_loss(self):
        plt.plot(self.loss_iter)
        plt.grid()
        plt.show()