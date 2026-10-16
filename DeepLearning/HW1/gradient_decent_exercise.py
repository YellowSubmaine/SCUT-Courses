import numpy as np
import os
import torch
from torchvision import datasets, transforms

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重
        # TODO: 初始化权重W1, b1, W2, b2
        # 提示：使用np.random.randn和np.zeros
        self.W1 = np.random.randn(input_size,hidden_size) * 0.01
        self.b1 = np.zeros((1,hidden_size))
        self.W2 = np.random.randn(hidden_size,output_size) *0.01
        self.b2 = np.zeros((1,output_size))
    
    def relu(self, x):
        """ReLU激活函数"""
        # TODO: 实现ReLU激活函数
        # 提示：使用np.maximum
        return np.maximum(0,x)
    
    def relu_derivative(self, x):
        """ReLU激活函数的导数"""
        # TODO: 实现ReLU导数
        # 提示：使用np.where
        return np.where(x>0,1,0)
    
    def softmax(self, x):
        """Softmax激活函数"""
        # TODO: 实现Softmax激活函数
        # 提示：使用np.exp和np.sum，注意数值稳定性
        x_safe = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_safe)
        return exp_x / np.sum(exp_x,axis=1,keepdims=True)
    
    def forward(self, X):
        """
        前向传播
        步骤：
        1. 计算隐藏层输入z1
        2. 计算隐藏层输出a1（使用ReLU激活）
        3. 计算输出层输入z2
        4. 计算输出层输出a2（使用Softmax激活）
        """
        # TODO: 实现前向传播
        self.z1 = np.dot(X,self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1,self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y, y_pred):
        """计算交叉熵损失"""
        # TODO: 实现交叉熵损失计算
        # 提示：使用np.sum和np.log，注意数值稳定性
        m = y.shape[0]
        loss = -np.sum(y*np.log(y_pred+1e-8)) / m
        return loss
    
    def backward(self, X, y, y_pred, learning_rate):
        """
        反向传播
        步骤：
        1. 计算输出层梯度
        2. 计算隐藏层梯度
        3. 更新权重
        """
        # TODO: 实现反向传播
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = y_pred - y
        dW2 = np.dot(self.a1.T,dz2)/m
        db2 = np.sum(dz2,axis=0,keepdims=True)/m
        
        # 隐藏层梯度
        dz1 = np.dot(dz2,self.W2.T)*self.relu_derivative(self.z1)
        dW1 = np.dot(X.T,dz1)/m
        db1 = np.sum(dz1,axis=0,keepdims=True)/m
        
        # 更新权重
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs, batch_size, learning_rate):
        """训练模型"""
        # TODO: 实现训练过程
        # 提示：包含随机打乱数据、批量训练、计算损失等步骤
        m = X.shape[0]
        for epoch in range(epochs):
            # 随机打乱数据
            permutation = np.random.permutation(m) 
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            # 批量训练
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 前向传播
                y_pred = self.forward(X_batch)
                
                # 反向传播
                self.backward(X_batch,y_batch,y_pred,learning_rate)
                
            # 计算损失
            y_pred_full = self.forward(X)
            loss = self.compute_loss(y,y_pred_full)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
    
    def predict(self, X):
        """预测"""
        # TODO: 实现预测功能
        # 提示：使用前向传播和np.argmax
        y_pred = self.forward(X)
        return np.argmax(y_pred,axis=1)
    
    def accuracy(self, X, y):
        """计算准确率"""
        # TODO: 实现准确率计算
        # 提示：使用predict方法和np.argmax
        y_pred = self.predict(X)
        y_true = np.argmax(y,axis=1)
        return np.mean(y_pred == y_true)

def download_mnist(path):
    """
    下载MNIST数据集到指定目录
    
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
    从本地文件加载MNIST数据集
    
    参数:
        path: str, MNIST数据集目录路径
    
    返回:
        (X_train, y_train), (X_test, y_test): 训练和测试数据
    """
    # TODO: 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.numpy().flatten())
    ])
    
    # TODO: 加载训练集
    train_dataset = datasets.MNIST(root=path,train=True,download=False,transform=transform)
    
    # TODO: 加载测试集
    test_dataset = datasets.MNIST(root=path,train=False,download=False,transform=transform)
    
    # TODO: 转换为numpy数组
    X_train = train_dataset.data.numpy().reshape(-1,28*28)/255.0
    y_train = train_dataset.targets.numpy()
    X_test = test_dataset.data.numpy().reshape(-1,28*28)/255.0
    y_test = test_dataset.targets.numpy()
    
    # TODO: 对标签进行独热编码
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]
    
    return (X_train, y_train), (X_test, y_test)

def train_gradient_descent(mnist_path, epochs=50, batch_size=64, learning_rate=0.01):
    """
    使用梯度下降训练神经网络
    
    参数:
        mnist_path: str, MNIST数据集目录路径
        epochs: int, 训练轮数
        batch_size: int, 批次大小
        learning_rate: float, 学习率
    
    返回:
        model: NeuralNetwork, 训练好的模型
        accuracy: float, 测试集准确率
    """
    # TODO: 加载数据集
    (X_train, y_train), (X_test, y_test) = load_mnist(mnist_path)
    
    # TODO: 初始化模型
    input_size = 784  # 28x28
    hidden_size = 128
    output_size = 10
    model = NeuralNetwork(input_size,hidden_size,output_size)
    
    # TODO: 训练模型
    print("开始训练模型...")
    model.train(X_train,y_train,epochs,batch_size,learning_rate)
    
    # TODO: 计算准确率
    train_accuracy = model.accuracy(X_train, y_train)
    test_accuracy = model.accuracy(X_test, y_test)
    
    print(f"训练集准确率: {train_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")
    
    return model, test_accuracy

if __name__ == "__main__":
    # 1. 下载MNIST数据集
    mnist_path = "./mnist"
    download_mnist(mnist_path)
    
    # 2. 训练模型
    model, accuracy = train_gradient_descent(mnist_path)
