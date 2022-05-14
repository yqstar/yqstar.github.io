---
title: Pytorch系列自学教程(2):深度学习的“Hello World！”之Mnist图像分类
date: 2022-05-11 23:37:29
tags: Pytorch
categories: 深度学习
---
深度学习的“Hello World！”之Mnist图像分类，本章节主要目的是为了完成Mnist数据集的图像分类算法。

Mnist数据集是手写数字的MNIST数据库，包含60,000个用于训练的0~9的训练集和10,000个用于测试的测试集。这些数字已被大小归一化，并以固定尺寸的图像为中心。

对于那些想尝试学习技术和模式识别方法的人来说，这是一个很好的数据集，同时花费最少的努力来进行预处理和格式化。详细：[Mnist官网](http://yann.lecun.com/exdb/mnist/)

在介绍之前我们简单介绍下，整个模型的结构。从数据加载（见前一章）、模型构建（优化器选择、模型结构）、模型训练和评估等步骤。

## 版本环境

```python
系统：Windows 11
显卡：NVIDIA GeForce RTX 3060
python: 3.6.13
pytorch: 1.7.1
cudatoolkit: 11.0.221
torchvision: 0.8.2
```

### 导入包环境

本文使用的包主要有Pytorch的深度学习框架，torchvision用于加载Mnist数据集，matplotlib用于可视化展示数据集。

```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt

```

```python
# 保证试验结果的稳定性
seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 配置Cuda参数
is_cuda=False
if torch.cuda.is_available():
    is_cuda = True

```

## 模型数据

本章节就不详细讲解，大家可以查看前一章的讲解。

```python
# 构建Dataset
mnist_train_dataset = datasets.MNIST(root=r'./data',
                              transform=transforms.ToTensor(),
                              train=True,
                              download=True)

mnist_test_dataset = datasets.MNIST(root=r'./data',
                              transform=transforms.ToTensor(),
                              train=False,
                              download=True)

# 构建Dataloader
mnist_train_loader = DataLoader(mnist_train_dataset,batch_size=32,shuffle=True)

mnist_test_loader = DataLoader(mnist_test_dataset,batch_size=32,shuffle=True)
```

## 模型构建

```python
# 构建网络
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size= 3)
        self.conv2 = nn.Conv2d(in_channels= 32,out_channels= 64,kernel_size= 3)
        self.dropout = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x
```

## 模型训练

### 模型训练函数

如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()，在测试时添加model.eval()。其中model.train()是保证BN层用每一批数据的均值和方差，而model.eval()是保证BN用全部训练数据的均值和方差；而对于Dropout，model.train()是随机取一部分网络连接来训练更新参数，而model.eval()是利用到了所有网络连接。

```python
def fit(epoch,model,data_loader,phases="training",volatile=False):
    # 如果模型中存在BatchNormalization和Dropout,在模型训练之前需要使用model.train()和model.eval()
    if phases == "training":
        model.train()
    if phases == "validation":
        model.eval()
        volatile = True
    running_loss = 0
    running_correct = 0
    for batch_idx, (data, target) in enumerate(data_loader):
        if is_cuda:
            data, target = data.cuda(), target.cuda()
        # data, target = Variable(data, volatile=volatile), Variable(target)
        data, target = Variable(data, volatile=volatile), Variable(target)
        if phases == "training":
            optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        running_loss += F.nll_loss(output ,target ,reduction='sum').item()
        preds = output.data.max(1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phases == "training":
            loss.backward()
            optimizer.step()
    loss = running_loss/len(data_loader.dataset)
    accuracy = 100. * running_correct/len(data_loader.dataset)
    print(f"{phases} loss is {loss:{5}.{2}} and {phases} accuracy is {accuracy:{10}.{4}}")
    return loss, accuracy
```

```python
model = MnistNet()
if is_cuda:
    model.cuda()
optimizer = optim.SGD(model.parameters(),lr=0.01)

```

## 模型评估

```python
train_losses , train_accuracy = [],[]
val_losses , val_accuracy = [],[]
for epoch in range(1,20):
    epoch_loss, epoch_accuracy = fit(epoch,model,mnist_train_loader,phases='training')
    val_epoch_loss , val_epoch_accuracy = fit(epoch,model,mnist_test_loader,phases='validation')
    train_losses.append(epoch_loss)
    train_accuracy.append(epoch_accuracy)
    val_losses.append(val_epoch_loss)
    val_accuracy.append(val_epoch_accuracy)
```

```python
plt.plot(range(1,len(train_losses)+1),train_losses,'bo',label = 'training loss')
plt.plot(range(1,len(val_losses)+1),val_losses,'r',label = 'validation loss')
plt.legend()
plt.show()

```

```python
plt.plot(range(1,len(train_accuracy)+1),train_accuracy,'bo',label = 'train accuracy')
plt.plot(range(1,len(val_accuracy)+1),val_accuracy,'r',label = 'val accuracy')
plt.legend()
plt.show()

```
