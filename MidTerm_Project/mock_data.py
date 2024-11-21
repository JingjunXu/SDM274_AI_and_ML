# 本文件实现一个类，其功能是模拟二分类数据的生成
# 生成线性分布的，二次曲线分布的，随机分布的
import numpy as np
import matplotlib.pyplot as plt

class MockData:
    def __init__(self):
        pass

    def create_linear_data(self, k = 0.5, b = 0.0, plot_show = False):
        # 以 y = k*x + b 为决策边界来创建数据集
        # 类别1在直线的上面，类别2在直线的下面
        np.random.seed(0)
        x_values = np.random.uniform(-1, 1, 50)
        y_values_class1 = (k * x_values + b) + np.random.uniform(0.1, 0.5, 50) 
        y_values_class2 = (k * x_values + b) - np.random.uniform(0.1, 0.5, 50)

        if plot_show == True:
            # 可视化数据集
            plt.figure(figsize=(8, 8))
            plt.scatter(x_values, y_values_class1, c='red', label='Class 1')
            plt.scatter(x_values, y_values_class2, c='blue', label='Class 2')
            # 创建直线并绘制
            line_y = k * x_values + b
            plt.plot(x_values, line_y, color='green', label='Separating line')
            # 绘制图表
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['top'].set_visible(False)
            plt.title('Two-Class Data with Separating Line')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 将创建的数据封装
        dataSet = []
        labels = []
        for x, y in zip(x_values, y_values_class1):
            dataSet.append([x, y])
            labels.append(0)
        for x, y in zip(x_values, y_values_class2):
            dataSet.append([x, y])
            labels.append(1) 
        
        return dataSet, labels
    
    def create_quatic_data(self, a = 1, b = 0, c = -4, plot_show = False):
        # 以 y = a*x^2 + b*x + c 为决策边界来创建数据集
        # 类别1在直线的上面，类别2在直线的下面
        np.random.seed(0)
        x_values = np.random.uniform(-1, 1, 100)
        y_values_class1 = a * x_values**2 + b * x_values + c + np.random.uniform(0.4, 1, 100)
        y_values_class2 = a * x_values**2 + b * x_values + c - np.random.uniform(0.1, 0.3, 100)

        if plot_show == True:
             # 可视化数据集
            plt.figure(figsize=(10, 6))
            plt.scatter(x_values, y_values_class1, color='red', label='Class 1')
            plt.scatter(x_values, y_values_class2, color='blue', label='Class 2')
            # 创建曲线并绘制
            x = np.linspace(-1, 1, 100)
            y = a * x**2 + b * x + c
            plt.plot(x, y, color='green', label='Quadratic Curve')
            # 绘制图表
            plt.title('Two-Class Data with Quadratic Curve')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 将创建的数据封装
        dataSet = []
        labels = []
        for x, y in zip(x_values, y_values_class1):
            dataSet.append([x, y])
            labels.append(0)
        for x, y in zip(x_values, y_values_class2):
            dataSet.append([x, y])
            labels.append(1) 

        return dataSet, labels


    def create_stochastic_data(self, num = 15, var = 1, plot_show = False): # num为每一类数据的个数，var控制数据的分散程度
        variance = var
        n_points = num

        # 随机创建两类数据
        np.random.seed(0)
        class1 = np.random.randn(n_points, 2) * variance  # Class 1
        class2 = (np.random.randn(n_points, 2) * variance) + np.array([0.3, 0.3])  # Class 2, slightly offset

        if plot_show == True:
            # 可视化数据集
            plt.figure(figsize=(6, 6))
            plt.scatter(class1[:, 0], class1[:, 1], color='red', label='Class 1')
            plt.scatter(class2[:, 0], class2[:, 1], color='blue', label='Class 2')
            plt.title('Random 2D Dataset with Two Classes')
            plt.xlabel('X-axis')
            plt.ylabel('Y-axis')
            plt.grid(True)
            plt.legend()
            plt.show()

        # 将创建的数据封装
        dataSet = []
        labels = []
        for i in range(0, len(class1)):
            dataSet.append(class1[i])
            labels.append(0)
        for i in range(0, len(class2)):
            dataSet.append(class2[i])
            labels.append(1) 

        return dataSet, labels
    