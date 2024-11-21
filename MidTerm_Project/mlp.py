import numpy as np
import matplotlib.pyplot as plt

class MLP:
    def __init__(self, input_feature = 1, layer_num = 1, unit_num = [1]): 
        # unit_num存储着每层hidden_layer的unit数, 最后一个为output_layer的unit数，对于二分类的问题来说，它为1
        self.loss_iter = [] # 储存loss的值，便于后续画图
        self.grad_iter = [] # 储存gradient的值，便于后续画图
        self.input_feature = input_feature
        self.layer_num = layer_num
        self.unit_num = unit_num
        self.weights = [] # weight的值，包含bias
        self.bn_param = {} # 用于batch_normalization的参数
        # 初始化模型参数, 对于n层的模型来说, 有n-1个hidden layer和1个output_layer, 对于二分类问题，output_layer只有一个unit, 因为输出的结果是一维的, 即有n组weight要训练
        # self._random_initialize()
        self._xvarier_initialize()
        
    def _random_initialize(self):
        np.random.seed(888) # 固定随机数种子使得实验能够复现
        for i in range(0, self.layer_num):
            layer = [] # 每层hidden_layer的参数
            if i == 0:
                for j in range(0, self.unit_num[0]):
                    layer.append(np.random.rand(self.input_feature + 1))
                gamma = np.ones(shape = (1, self.input_feature))
                beta = np.zeros(shape = (1, self.input_feature))
            else:
                for j in range(0, self.unit_num[i]):
                    layer.append(np.random.rand(self.unit_num[i-1] + 1))
                gamma = np.ones(shape = (1, self.unit_num[i - 1]))
                beta = np.zeros(shape = (1, self.unit_num[i - 1]))
            self.weights.append(layer) 
            self.bn_param['gamma' + str(i + 1)] = gamma
            self.bn_param['beta' + str(i + 1)] = beta

    def _xvarier_initialize(self):
        np.random.seed(2023) # 固定随机数种子使得实验能够复现
        for i in range(0, self.layer_num):
            layer = [] # 每层hidden_layer的参数
            if i == 0:
                for j in range(0, self.unit_num[0]):
                    layer.append(np.random.rand(self.input_feature + 1))
                layer = layer / np.sqrt(self.unit_num[0] + self.input_feature + 1)
                # layer = layer / np.sqrt((self.unit_num[0] + self.input_feature + 1) / 2)
                gamma = np.ones(shape = (1, self.unit_num[0]))
                beta = np.zeros(shape = (1, self.unit_num[0]))
            else:
                for j in range(0, self.unit_num[i]):
                    layer.append(np.random.rand(self.unit_num[i-1] + 1))
                #layer = layer / np.sqrt(self.unit_num[i] + self.unit_num[i-1] + 1)
                layer = layer / np.sqrt((self.unit_num[i] + self.unit_num[i-1] + 1) / 2)
                gamma = np.ones(shape = (1, self.unit_num[i]))
                beta = np.zeros(shape = (1, self.unit_num[i]))
            self.weights.append(layer) 
            self.bn_param['gamma' + str(i + 1)] = gamma
            self.bn_param['beta' + str(i + 1)] = beta

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))
    
    def _relu(self, z):
        return np.maximum(0, z)

    def _relu_derivative(self, z):
        return np.where(z > 0, 1, 0)

    def _cost_derivative(self, output, target):
        return (output - target)
        
    def _loss(self, outputs, targets): # output就是最后的输出结果是一个概率
        eps = 1e-8
        return -np.sum(targets * np.log(outputs + eps) + (1 - targets) *  np.log((1 - outputs + eps)))
    
    def _batch_norm(self, input, layer_index, mode):
        # 用于batch_normalization的batch_norm层
        eps = 1e-6
        momentum = 0.9
        D = np.array(input).shape
        global_mean = self.bn_param.get('global_mean' + str(layer_index), np.zeros(D, dtype=input.dtype))
        global_var = self.bn_param.get('global_var' + str(layer_index), np.zeros(D, dtype=input.dtype))
        cache = None
        if mode == 'train':
            sample_mean = np.mean(input, axis=0)
            sample_var = np.var(input, axis=0)
            input_hat = (input - sample_mean) / np.sqrt(sample_var + eps)
            out = self.bn_param['gamma' + str(layer_index)] * input_hat + self.bn_param['beta' + str(layer_index)]  # bn结束
            global_mean = momentum * global_mean + (1 - momentum) * sample_mean
            global_var = momentum * global_var + (1 - momentum) * sample_var
            cache = {'input': input, 'input_hat': input_hat, 'sample_mean': sample_mean, 'sample_var': sample_var}
        else:
            # 测试模式，使用全局均值和方差标准化
            input_hat = (input - global_mean) / np.sqrt(global_var + eps)
            out = self.bn_param['gamma' + str(layer_index)] * input_hat + self.bn_param['beta' + str(layer_index)]
        self.bn_param['global_mean' + str(layer_index)] = global_mean
        self.bn_param['global_var' + str(layer_index)] = global_var
        return out, cache
    
    def _forward(self, inputs):
        activations = []
        activations.append(inputs)
        zs = []
        tmp = inputs
        for i in range(0, self.layer_num):
            if i == 0:
                z = np.dot(self.weights[i], inputs)
                tmp = self._relu(z)
            elif i == self.layer_num - 1:
                z = np.dot(self.weights[i], ttmp)
                tmp = self._sigmoid(z)
            else:
                z = np.dot(self.weights[i], ttmp)
                tmp = self._relu(z)
            zs.append(z)
            # tmp = self._relu(z)
            ttmp = np.concatenate((np.ones(1), np.array(tmp)))
            activations.append(ttmp)
        return activations, zs
    
    def _forward_bn(self, inputs, bn_mode='train'):
        activations = []
        activations.append(inputs)
        zs = []
        caches = []
        caches.append(inputs)
        for i in range(0, self.layer_num):
            if i == 0:
                z = np.dot(self.weights[i], inputs)
                zs.append(z[0])
                z, cache = self._batch_norm(z, i + 1, bn_mode)  # 可以将BN理解为加在隐藏层神经元输入和输出间可训练的一层
                caches.append(cache)
                tmp = self._relu(z)
            elif i == self.layer_num - 1:
                z = np.dot(ttmp, self.weights[i].T)
                zs.append(z[0])
                z, cache = self._batch_norm(z, i + 1, bn_mode)  # 可以将BN理解为加在隐藏层神经元输入和输出间可训练的一层
                caches.append(cache)
                tmp = self._sigmoid(z)
            else:
                z = np.dot(ttmp, self.weights[i].T)
                zs.append(z[0])
                z, cache = self._batch_norm(z[0], i + 1, bn_mode)  # 可以将BN理解为加在隐藏层神经元输入和输出间可训练的一层
                caches.append(cache)
                tmp = self._relu(z)
            ttmp = np.hstack((np.ones(shape = (1,1)), np.array(tmp)))
            activations.append(ttmp[0])
        return activations, zs, caches

    def _backward(self, input, target):
        nabla_w = [np.zeros_like(w) for w in self.weights]
        activations, zs = self._forward(input)
        grad = []
        for i in range(self.layer_num, 0, -1):
            if i == self.layer_num:
                dz = [self._cost_derivative(activations[i][1], target)]
                grad.append(np.multiply(self._cost_derivative(activations[i][1], target), activations[i - 1])) 
                self.grad_iter[-1] += np.sum(grad)
            else:
                dz = np.array(dz).T @ self.weights[i][:, 1:]
                #ggrad = dz * self._sigmoid_derivative(zs[i-1])
                ggrad = dz * self._relu_derivative(zs[i-1])
                grad = ggrad[0].reshape(-1,1) @ activations[i-1].reshape(1,-1)
            nabla_w[i-1] = grad +  self.reg * self.weights[i - 1]
        return nabla_w
    
    def _backward_bn(self, input, target):
        eps = 1e-6
        nabla = [np.zeros_like(w) for w in self.weights]
        nabla_bn_params = dict()
        activations, zs, cache = self._forward_bn(input)
        grad = []
        for i in range(self.layer_num, 0, -1):
            if i == self.layer_num:
                dz = [self._cost_derivative(activations[i][1], target)]
                grad.append(np.multiply(self._cost_derivative(activations[i][1], target), activations[i - 1])) 
                self.grad_iter[-1] += np.sum(grad)
            else:
                dz = np.array(dz).T @ self.weights[i][:, 1:]
                N = cache[i]['input'].shape[0]
                nabla_bn_params['delta_gamma' + str(i)] = np.sum(dz * cache[i]['input_hat'])
                nabla_bn_params['delta_beta' + str(i)] = np.sum(dz, axis = 0)

                dx_hat = dz * self.bn_param['gamma' + str(i)]
                dsigma = -0.5 * np.sum(dx_hat * (cache[i]['input'] - cache[i]['sample_mean']), axis = 0) * np.power(cache[i]['sample_var'] + eps, -1.5)
                dmu = -np.sum(dx_hat / np.sqrt(cache[i]['sample_var'] + eps), axis = 0) - 2.0 * dsigma * np.sum(cache[i]['input'] - cache[i]['sample_mean'], axis = 0) / N
                dx = dx_hat / np.sqrt(cache[i]['sample_var'] + eps) + 2.0 * dsigma * (cache[i]['input'] - cache[i]['sample_mean']) / N + dmu / N
                # ggrad = dx * self._sigmoid_derivative(zs[i-1]) 
                ggrad = dx * self._relu_derivative(zs[i-1]) 
                self.grad_iter[-1] += np.sum(ggrad)
                grad =  ggrad[0].reshape(-1,1) @ activations[i-1].reshape(1,-1)
            nabla[i-1] = grad + self.reg * self.weights[i - 1]
        return nabla, nabla_bn_params
        
    def _batch_update(self, inputs, targets, learning_rate = 0.001):
        nabla_w = [np.zeros(np.array(w).shape) for w in self.weights]
        num = len(inputs)
        if isinstance(targets, float):
            targets = [targets]
        for i in range(0,num):
            delta_nabla_w = self._backward(inputs[i], targets[i])
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (learning_rate / num) * nw for w, nw in zip(self.weights, nabla_w)]

    def _batch_update_bn(self, inputs, targets, learning_rate = 0.001):
        nabla_w = [np.zeros(np.array(w).shape) for w in self.weights]
        num = len(inputs)
        if isinstance(targets, float):
            targets = [targets]
        for i in range(0,num):
            delta_nabla_w, delta_nabla_bn_params = self._backward_bn(inputs[i], targets[i])
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]  
            for j in range(1, self.layer_num):
                self.bn_param['gamma' + str(j)] -= learning_rate / num * delta_nabla_bn_params.get('delta_gamma' + str(j))
                self.bn_param['beta' + str(j)] -= learning_rate / num * delta_nabla_bn_params.get('delta_beta' + str(j))        
        self.weights = [w - (learning_rate / num) * nw for w, nw in zip(self.weights, nabla_w)]

    def _Adam_update_bn(self, inputs, targets, learning_rate = 0.001, beta1=0.9, beta2=0.999):
        eps = 1e-8
        first_moment = 0  # 第一动量，用于累积梯度，加速训练
        second_moment = 0  # 第二动量，用于累积梯度平方，自动调整学习率
        num = len(inputs)
        if isinstance(targets, float):
            targets = [targets]
        for i in range(1, num + 1):
            delta_nabla_w, delta_nabla_bn_params = self._backward_bn(inputs[i - 1], targets[i - 1])
            # print(delta_nabla_bn_params)
            if i == 1:
                first_moment = [beta1 * first_moment + (1 - beta1) * np.array(dw) for dw in delta_nabla_w]  # Momentum
                second_moment = [beta2 * second_moment + (1 - beta2) * np.array(dw) * np.array(dw) for dw in delta_nabla_w] # AdaGrad / RMSProp
            else:
                first_moment = [beta1 * fm + (1 - beta1) * np.array(dw) for fm, dw in zip(first_moment, delta_nabla_w)]  # Momentum
                second_moment = [beta2 * sm + (1 - beta2) * np.array(dw) * np.array(dw) for sm, dw in zip(second_moment,delta_nabla_w)] # AdaGrad / RMSProp
            first_unbias = [dw / (1 - beta1 ** i) for dw in first_moment]# 加入偏置，随次数减小，防止初始值过小
            second_unbias =  [dw / (1 - beta2 ** i) for dw in second_moment] 
            nabla_w = [dw1 / (np.sqrt(dw2) + eps) for dw1, dw2 in zip(first_unbias, second_unbias)]
            for j in range(1, self.layer_num):
                self.bn_param['gamma' + str(j)] -= learning_rate / num * delta_nabla_bn_params.get('delta_gamma' + str(j))
                self.bn_param['beta' + str(j)] -= learning_rate / num * delta_nabla_bn_params.get('delta_beta' + str(j))
            self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]

    def _Adam_update(self, inputs, targets, learning_rate = 0.001, beta1=0.9, beta2=0.999):
        eps = 1e-8
        first_moment = 0  # 第一动量，用于累积梯度，加速训练
        second_moment = 0  # 第二动量，用于累积梯度平方，自动调整学习率
        num = len(inputs)
        if isinstance(targets, float):
            targets = [targets]
        for i in range(1, num + 1):
            delta_nabla_w = self._backward(inputs[i - 1], targets[i - 1])
            if i == 1:
                first_moment = [beta1 * first_moment + (1 - beta1) * np.array(dw) for dw in delta_nabla_w]  # Momentum
                second_moment = [beta2 * second_moment + (1 - beta2) * np.array(dw) * np.array(dw) for dw in delta_nabla_w] # AdaGrad / RMSProp
            else:
                first_moment = [beta1 * fm + (1 - beta1) * np.array(dw) for fm, dw in zip(first_moment, delta_nabla_w)]  # Momentum
                second_moment = [beta2 * sm + (1 - beta2) * np.array(dw) * np.array(dw) for sm, dw in zip(second_moment,delta_nabla_w)] # AdaGrad / RMSProp
            first_unbias = [dw / (1 - beta1 ** i) for dw in first_moment]# 加入偏置，随次数减小，防止初始值过小
            second_unbias =  [dw / (1 - beta2 ** i) for dw in second_moment] 
            nabla_w = [dw1 / (np.sqrt(dw2) + eps) for dw1, dw2 in zip(first_unbias, second_unbias)]
            self.weights = [w - learning_rate * nw for w, nw in zip(self.weights, nabla_w)]

    def _preprocess_data(self, predata): # 将数据转化numpy中的array，并选择是否对数据进行标准化
        predata = np.array(predata)
        norm_data = np.divide(np.subtract(predata, self.input_mean), self.input_std)
        m, n = predata.shape
        data_ = np.empty([m, n+1])
        data_[:,0] = 1
        data_[:,1:] = norm_data
        return data_
        
    def train(self, inputs, targets, epoch_num = 500, learning_rate = 0.001, tol = None, optimize_type = 0, batch_size = 1, reg = 0, bn_mode = False): 
        # optimize_type = 0 为BGD, 为1则是Adam，bn_mode决定是否进行batch_normalization
        self.loss_iter = []
        self.grad_iter = []
        self.input_mean = np.mean(inputs, axis = 0)
        self.input_std = np.std(inputs, axis = 0)
        self.bn_mode = bn_mode
        self.reg = reg
        if tol is None:
            tol = float('-inf')
        inputs = self._preprocess_data(inputs)
        num, _ = inputs.shape
        for epoch in range(0,epoch_num):
            self.loss_iter.append(0)
            self.grad_iter.append(0)
            inputs_batches = [inputs[k:k + batch_size] for k in range(0, len(inputs), batch_size)]
            targets_batches = [targets[k:k + batch_size] for k in range(0, len(targets), batch_size)]
            if self.bn_mode:
                if optimize_type == 0:
                    for i in range(0, len(inputs_batches)):
                        self._batch_update_bn(inputs_batches[i], targets_batches[i], learning_rate)
                elif optimize_type == 1:
                    
                    for i in range(0, len(inputs_batches)):
                        self._Adam_update_bn(inputs_batches[i], targets_batches[i], learning_rate)
                    '''
                    k = int(np.random.randint(0, len(inputs_batches), size=1))
                    self._Adam_update_bn(inputs_batches[k], targets_batches[k], learning_rate)
                    '''
            else:
                if optimize_type == 0:
                    for i in range(0, len(inputs_batches)):
                        self._batch_update(inputs_batches[i], targets_batches[i], learning_rate)
                elif optimize_type == 1:
                    '''
                    for i in range(0, len(inputs_batches)):
                        self._Adam_update(inputs_batches[i], targets_batches[i], learning_rate)
                    '''
                    k = int(np.random.randint(0, len(inputs_batches), size=1))
                    self._Adam_update_bn(inputs_batches[k], targets_batches[k], learning_rate)
            output = self._predict(inputs)
            for j in range(0, num):
                self.loss_iter[-1] += self._loss(output[j], targets[j]) / num
            if (epoch+1) % 100 == 0:
                    print(f"Finished epoch {(epoch + 1)}/{epoch_num}: "f"loss {self.loss_iter[-1]:.6f}, "f"lr: {learning_rate:.6f}")
            if self.loss_iter[-1] <= tol:
                break

    def _predict(self, inputs):
        num = len(inputs)
        pred = []
        for i in range(0, num):
            if self.bn_mode:
                out, _, _ = self._forward_bn(inputs[i], 'test')
            else:
                out, _ = self._forward(inputs[i])
            pred.append(out[-1][1])
        return pred

    def predict(self, inputs):
        inputs = self._preprocess_data(inputs)
        num = len(inputs)
        pred = []
        for i in range(0, num):
            if self.bn_mode:
                out, _, _ = self._forward_bn(inputs[i], 'test')
            else:
                out, _ = self._forward(inputs[i])
            if out[-1][1] >= 0.5:
                pred.append(1)
            elif out[-1][1] < 0.5:
                pred.append(0)
        return pred
    
    def plot_loss(self):
        plt.plot(self.loss_iter)
        plt.grid()
        plt.show()

    def return_loss(self):
        return self.loss_iter

    def plot_grad(self):
        plt.plot(self.grad_iter)
        plt.grid()
        plt.show()   