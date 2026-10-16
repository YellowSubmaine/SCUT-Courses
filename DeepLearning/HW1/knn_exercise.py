import numpy as np
import os
import torch
from torchvision import datasets, transforms

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        """存储训练数据"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        预测函数
        步骤：
        1. 计算测试样本与所有训练样本的距离
        2. 找到距离最近的k个训练样本
        3. 对这k个样本的标签进行投票，选择出现次数最多的标签作为预测结果
        """
        predictions = []
        for x in X:
            # TODO: 计算欧氏距离
            # 提示：使用np.sqrt和np.sum函数
            distances = np.sqrt( np.sum( (self.X_train-x)**2 ,axis=1) )
            
            # TODO: 找到距离最近的k个样本的索引
            # 提示：使用argsort函数
            k_indices = np.argsort(distances)[:self.k]
            
            # TODO: 获取这k个样本的标签
            k_nearest_labels = self.y_train[k_indices]
            
            # TODO: 对标签进行投票，选择出现次数最多的标签
            # 提示：使用np.unique函数
            unique, counts = np.unique( k_nearest_labels , return_counts=True)
            predictions.append( unique[np.argmax(counts)])
        return np.array(predictions)
    
    def accuracy(self, X_test, y_test):
        """计算预测准确率"""
        predictions = self.predict(X_test)
        return np.sum(predictions == y_test) / len(y_test)


def download_mnist(path):
    """
    下载MNIST数据集到指定目录（使用PyTorch内置函数）
    
    参数:
        path: str, 保存MNIST数据集的目录路径
    
    返回:
        path: str, 数据集保存路径
    """
    os.makedirs(path, exist_ok=True)
    
    print(f"开始下载MNIST数据集到 {path}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().flatten())
    ])
    
    train_dataset = datasets.MNIST(
        root=path,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=path,
        train=False,
        download=True,
        transform=transform
    )
    
    print("MNIST数据集下载完成")
    return path

def load_mnist(path):
    """
    从本地文件加载MNIST数据集（使用PyTorch格式）
    
    参数:
        path: str, MNIST数据集目录路径
    
    返回:
        (X_train, y_train), (X_test, y_test): 训练和测试数据
    """
    # TODO: 定义数据转换
    transform = transform = transforms.Compose([transforms.ToTensor(),transforms.Lambda(lambda x: x.numpy().flatten())])
    
    # TODO: 加载训练集（注意设置download=False）
    train_dataset =  datasets.MNIST(root=path, train=True, download=False, transform=transform)
    
    # TODO: 加载测试集
    test_dataset = datasets.MNIST(root=path, train=False, download=False, transform=transform)
    
    # TODO: 将数据集转换为numpy数组
    # 提示：使用列表推导式和np.array
    X_train = np.array([img for img, label in train_dataset])
    y_train = np.array([label for img, label in train_dataset])
    X_test = np.array([img for img, label in test_dataset])
    y_test = np.array([label for img, label in test_dataset])

    return (X_train, y_train), (X_test, y_test)

def knn_mnist(mnist_path, k=3, test_size=1000):
    """
    MNIST数据集的KNN接口
    
    参数:
        mnist_path: str, MNIST数据集目录路径
        k: int, 最近邻的数量
        test_size: int, 使用的测试样本数量
    
    返回:
        accuracy: float, 预测准确率
    """
    # TODO: 加载数据集
    (X_train, y_train), (X_test, y_test) = load_mnist(mnist_path)
    
    # TODO: 使用子集以加快计算速度
    X_test_subset = X_test[:test_size]
    y_test_subset = y_test[:test_size]
    
    # TODO: 初始化并训练KNN
    knn  = KNN(k=k)
    knn.fit(X_train, y_train)
    
    # TODO: 计算准确率
    accuracy = knn.accuracy(X_test_subset, y_test_subset)
    print(f"KNN算法在k={k}时的准确率: {accuracy:.4f}")
    
    return accuracy

if __name__ == "__main__":
    # 1. 下载MNIST数据集
    mnist_path = "./mnist"
    download_mnist(mnist_path)
    
    # 2. 使用KNN算法
    knn_mnist(mnist_path, k=3)
