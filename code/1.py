# 处理冲突
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 导入模块
import torchvision as tv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
from PIL import Image

# 数据预处理
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 定义简单的卷积神经网络
class mycnn(nn.Module):
    def __init__(self):
        super(mycnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    # 加载数据集和测试集
    full_train_dataset = datasets.ImageFolder(root='../data/train', transform=transform)
    full_test_dataset = datasets.ImageFolder(root='../data/test', transform=transform)


    # 正确的数据集切片方式
    train_dataset = Subset(full_train_dataset, range(min(5000, len(full_train_dataset))))
    test_dataset = Subset(full_test_dataset, range(min(1000, len(full_test_dataset))))

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model = mycnn().to(device)

    # 开始训练模型
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 开始循环
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        runnning_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            runnning_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {runnning_loss / len(train_loader):.4f}")

    # 测试模型的性能
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 输出模型的准确率
    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy:.2f}%')

    # 随机显示一些测试图像以及预测结果
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # 显示图像和预测结果
    imshow(tv.utils.make_grid(images.cpu()))
    print("预测结果：", ' '.join(f'{full_test_dataset.classes[predicted[j]]}' for j in range(len(predicted))))
    print("实际标签：", ' '.join(f'{full_test_dataset.classes[labels[j]]}' for j in range(len(labels))))

    # 保存模型
    torch.save(model.state_dict(), 'mycnn_model.pth')