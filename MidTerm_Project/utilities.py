import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(X, y, model, title = 'Decision boundary'):
    # 绘制决策边界
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    h = 0.01

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)
    # 假设 Z 是模型对整个网格的预测类别标签
    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], colors=['pink', 'lightgreen'])
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title(title)
    plt.show()


def plot_loss(datasets):
    # 绘制多个loss的图
    """
    datasets = [
    {'x': [1, 2, 3], 'y': [2, 3, 4], 'label': 'Curve 1', 'color': 'red'},
    {'x': [1, 2, 3], 'y': [3, 4, 5], 'label': 'Curve 2', 'color': 'blue'},
    ......
    ]
    """
    plt.figure(figsize=(10, 8))
    
    for data in datasets:
        x = data['x']
        y = data['y']
        label = data.get('label', '')  
        color = data.get('color', None)  
        plt.plot(x, y, label=label, color=color)
    
    plt.title('Loss')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)
    plt.show()