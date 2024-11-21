import numpy as np
import matplotlib.pyplot as plt

class LinearAutoEncoder:
    def __init__(self, input_dim, encoding_dim):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        # 记录loss
        self.loss_iter = []
        # 初始化模型参数
        self.encoder_weights = np.random.randn(self.input_dim, self.encoding_dim)
        self.decoder_weights = np.random.randn(self.encoding_dim, self.input_dim)
        self.encoder_bias = np.zeros(self.encoding_dim)
        self.decoder_bias = np.zeros(self.input_dim)

    def _encode(self, X):
        encoded = np.dot(X, self.encoder_weights) + self.encoder_bias
        return encoded

    def _decode(self, encoded):
        decoded = np.dot(encoded, self.decoder_weights) + self.decoder_bias
        return decoded

    def fit(self, X, epochs, learning_rate, tol = None):
        if tol is None:
            tol = -float('inf')
        num = X.shape[0]
        for epoch in range(epochs):
            # Forward pass
            # Encoding
            encoded = self._encode(X)
            # Decoding
            decoded = self._decode(encoded)

            # 计算loss
            loss = np.mean((X - decoded) ** 2) / num
            self.loss_iter.append(loss)
            # 训练的时候实时观测loss的变化
            '''
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
            '''

            # 检查是否收敛
            if loss < tol:
                break
            
            # Backward Pass
            error = X - decoded
            # 计算梯度
            decoder_gradient = -np.dot(encoded.T, error) / num
            encoder_gradient = -np.dot(X.T, np.dot(error, self.decoder_weights.T)) / num
            # 更新模型参数
            self.decoder_weights -= learning_rate * decoder_gradient
            self.encoder_weights -= learning_rate * encoder_gradient
            self.decoder_bias -= learning_rate * np.mean(error, axis=0)
            self.encoder_bias -= learning_rate * np.mean(np.dot(error, self.decoder_weights.T), axis=0)

    def transform(self, X): # 将数据X降维
        return self._encode(X)
    
    def predict(self, X): # 输入的X是降维的数据
        decoded = self._decode(X)
        return decoded
    
    def plot_loss(self):
        plt.plot(self.loss_iter)
        plt.grid()
        plt.show()
