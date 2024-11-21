import numpy as np

class DecisionTree:
    def __init__(self):
        pass

    def _calEnt(self, dataSet):
        # 假设dataSet是一个numpy数组，并且标签在最后一列
        n = dataSet.shape[0]  # 数据集总行数
        # 获取最后一列（标签列）
        labels = dataSet[:, -1]
        # 计算每个标签的出现次数
        _, label_counts = np.unique(labels, return_counts=True)
        # 计算每个标签的频率
        p = label_counts / n
        # 计算信息熵
        ent = -np.sum(p * np.log2(p))

        return ent
    
    def _bestSplit(self, dataSet):
        baseEnt = self._calEnt(dataSet)  # 计算原始熵
        bestGain = 0  # 初始化信息增益
        axis = -1  # 初始化最佳切分列，标签列
        n, m = dataSet.shape  # n是行数，m是列数

        for i in range(m - 1):  # 对特征的每一列进行循环
            # 提取当前列的所有值
            feature_values = dataSet[:, i]
            levels, counts = np.unique(feature_values, return_counts=True) # level储存该列中所有可能的值counts是个数
            ents = 0  # 初始化子节点的信息熵

            for level in levels:  # 对当前列的每一个取值进行循环
                # 选出当前特征值等于level的子集
                childSet = dataSet[feature_values == level] # 选择dataSet中所有在当前特征列上值等于level的行
                ent = self._calEnt(childSet)  # 计算某一个子节点的信息熵
                ents += (len(childSet) / n) * ent  # 计算当前列的信息熵

            infoGain = baseEnt - ents  # 计算当前列的信息增益
            if infoGain > bestGain:  # 选择最大信息增益
                bestGain = infoGain
                axis = i  # 最大信息增益所在列的索引

        return axis

    def _mySplit(self, dataSet, axis, value):
        # 提取满足条件的行：dataSet中axis列的值等于value
        condition = dataSet[:, axis] == value
        filtered_dataSet = dataSet[condition]
        # 删除指定的列
        redataSet = np.delete(filtered_dataSet, axis, axis=1)
        return redataSet
    
    def createTree(self, dataSet):
        dataSet = np.array(dataSet)
        # dataSet是一个numpy数组，最后一列是标签
        n_rows, n_cols = dataSet.shape
        labels = dataSet[:, -1]

        # 检查所有的行标签是否相同，或者数据集是否只有一列（标签列）
        if np.unique(labels).size == 1 or n_cols == 1:
            return np.unique(labels)[0]  # 如果是，返回类标签

        axis = self._bestSplit(dataSet)  # 确定出当前最佳切分列的索引
        myTree = {axis: {}}  # 使用列索引作为键

        # 提取最佳切分列所有属性值
        unique_values = np.unique(dataSet[:, axis])

        for value in unique_values:  # 对每一个属性值递归建树
            # 创建子数据集
            subDataSet = self._mySplit(dataSet, axis, value)
            myTree[axis][value] = self.createTree(subDataSet)

        return myTree

