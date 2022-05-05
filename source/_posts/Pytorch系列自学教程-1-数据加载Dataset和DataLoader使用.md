---
title: Pytorch系列自学教程(1):数据加载Dataset和DataLoader使用
date: 2022-05-05 23:03:19
tags:
categories:
---

区别于其他的机器学习模型，BP（Back Propagation）神经网络的训练多采用一些基于梯度下降的算法对网络的权重和偏置进行逐步地调整，是一个迭代进行的过程。因此，在训练网络之前，需要对数据集进行一些操作，将其封装成适合迭代训练的形式。这篇文章将介绍Pytorch对自定义数据集进行封装的方法。

# Dataset、Batch、Iteration和Epoch的关系

训练模型一般都是先处理数据的**输入问题**和**预处理问题**。数据加载处理在深度学习链路中起着非常重要的基础作用。在介绍如何加载数据前，简单介绍下，**Dataset**，**Batch**，**Iteration** 和 **Epoch** 的关系。


|   名词    |                                               解释                                          |
| :-------: | :-----------------------------------------------------------------------------------------: |
|  Dataset  |                                         全量待训练数据集                                     |
|   Batch   | 使用全量待训练数据集的一小部分样本对模型进行一次反向传播的参数更新，这一小部分样本称为“一个Batch” |
| Iteration |               使用一个Batch数据对模型进行一次参数更新的过程，称之为“一次Iteration”             |
|   Epoch   |               待训练数据集的全量数据对模型进行一次完整的参数更新，称之为“一个Epoch”             |

![data](Pytorch系列自学教程-1-数据加载Dataset和DataLoader使用/data_epoch.jpg)

假设DatasetSize=10，BatchSize=3，那么每个Epoch会执行4个Iteration，对应四个BatchSize大小分别为3，3，3，1个样本。

BatchSize = DatasetSize/IterationSize

# Pytoch数据处理：DataSet和DataLoader

Pytorch提供了几个有用的工具：**torch.utils.data.Dataset类**和**torch.utils.data.DataLoader类**，用于进行数据读取和预处理数据流程。
基本流程是先把原始数据转变成torch.utils.data.Dataset类，随后把得到的torch.utils.data.Dataset类当作一个参数传递给torch.utils.data.DataLoader类，得到一个数据加载器，这个数据加载器每次可以返回一个Batch的数据供模型训练使用。

## torch.utils.data.Dataset类

*torch.utils.data.Dataset*是代表这一数据的抽象类。你可以自己定义你的数据类，继承和重写这个抽象类，非常简单，只需要定义__init__,__len__和__getitem__这三个函数,其中：
* \__init__()用于初始化原始数据的路径和文件名等
* \__len__()的作用是返回数据集当中的样本总个数，
* \__getitem__()的作用是返回指定索引的样本所对应的输入变量与输出变量。

``` python
# class CustomDataset(torch.utils.data.Dataset):#需要继承data.Dataset
#     def __init__(self):
#         # TODO
#         # 1. Initialize file path or list of file names.
#         pass
#     def __getitem__(self, index):
#         # TODO
#         # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
#         # 2. Preprocess the data (e.g. torchvision.Transform).
#         # 3. Return a data pair (e.g. image and label).
#         #这里需要注意的是，第一步：read one data，是一个data
#         pass
#     def __len__(self):
#         # You should change 0 to the total size of your dataset.
#         pass
```

用原始数据造出来的 Dataset子类 其实就是一个静态的数据池，这个数据池支持我们用 索引 得到某个数据，想要让这个数据池流动起来，源源不断地输出 Batch 还需要下一个工具 DataLoader类 。所以我们把创建的 Dataset子类 当参数传入 即将构建的DataLoader类才是使用Dataset子类最终目的。

## torch.utils.data.DataLoader类
DataLoader(object)可用参数：
* dataset(Dataset): 传入的数据集
* batch_size(int, optional): 每个batch有多少个样本
* shuffle(bool, optional): 在每个epoch开始的时候，对数据进行重新排序
* sampler(Sampler, optional): 自定义从数据集中取样本的策略，如果指定这个参数，那么shuffle必须为False
* batch_sampler(Sampler, optional): 与sampler类似，但是一次只返回一个batch的indices（索引），需要注意的是，一旦指定了这个参数，那么batch_size,shuffle,sampler,drop_last就不能再制定了（互斥——Mutually exclusive）
* num_workers (int, optional): 这个参数决定了有几个进程来处理data loading。0意味着所有的数据都会被load进主进程。（默认为0）
* collate_fn (callable, optional): 将一个list的sample组成一个mini-batch的函数
* pin_memory (bool, optional)： 如果设置为True，那么data loader将会在返回它们之前，将tensors拷贝到CUDA中的固定内存（CUDA pinned memory）中.
* drop_last (bool, optional):如果设置为True：这个是对最后的未完成的batch来说的，比如你的batch_size设置为64，而一个epoch只有100个样本，那么训练的时候后面的36个就被扔掉了。 如果为False（默认），那么会继续正常执行，只是最后的batch_size会小一点。
* timeout(numeric, optional):如果是正数，表明等待从worker进程中收集一个batch等待的时间，若超出设定的时间还没有收集到，那就不收集这个内容了。这个numeric应总是大于等于0。默认为0
* worker_init_fn (callable, optional): 每个worker初始化函数 If not None, this will be called on eachworker subprocess with the worker id (an int in [0, num_workers - 1]) as input, after seeding and before data loading. (default: None) 


# 实例
## txt数据读取

使用个人创建的建议txt文件数据实例进行数据读取。

``` python
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SampleTxtDataset(Dataset):
    def __init__(self):
        # 数据文件地址
        self.txt_file_path = "./sample_easy_data.txt"

    def __getitem__(self, item):
        txt_data = np.loadtxt(self.txt_file_path, delimiter=",")
        self._x = torch.from_numpy(txt_data[:, :2])
        self._y = torch.from_numpy(txt_data[:, 2])
        return self._x[item], self._y[item]

    def __len__(self):
        txt_data = np.loadtxt(self.txt_file_path, delimiter=",")
        self._len = len(txt_data)
        return self._len

sample_txt_dataset = SampleTxtDataset()

print("Data Size:",len(sample_txt_dataset))

print("First Sample:",next(iter(sample_txt_dataset)))

print("First Sample's Type:",type(next(iter(sample_txt_dataset))[0]))

sample_dataloader = DataLoader(dataset=sample_txt_dataset, batch_size=3, shuffle=True)

num_epochs = 4

for epoch in range(num_epochs):
    for iteration, (batch_x, batch_y) in enumerate(sample_dataloader):
        print('Epoch: ', epoch, '| Iteration: ', iteration, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

```
Dataset的示例结果：

![dataset](Pytorch系列自学教程-1-数据加载Dataset和DataLoader使用/dataset_tutorial.jpg)

DataLoader的示例结果：

![dataloader](Pytorch系列自学教程-1-数据加载Dataset和DataLoader使用/dataloader_tutorial.jpg)

## csv文件读取

使用常见的离线数据csv文件进行数据加载和预处理。

``` python
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class SampleCsvDataset(Dataset):
    def __init__(self):
        self.csv_file_path = "./sample_boston.csv"


    def __getitem__(self, item):
        raw_data = pd.read_csv(self.csv_file_path)
        raw_data_shape = raw_data.shape
        self._x  = torch.from_numpy(raw_data.iloc[:,:raw_data_shape[1]-1].values)
        self._y  = torch.from_numpy(raw_data.iloc[:,raw_data_shape[1]-1].values)
        return self._x[item], self._y[item]

    def __len__(self):
        raw_data = pd.read_csv(self.csv_file_path)
        raw_data_shape = raw_data.shape
        self._len = raw_data_shape[0]
        return self._len

sample_csv_dataset = SampleCsvDataset()

print("Data Size:",len(sample_csv_dataset))

print("First Sample:",next(iter(sample_csv_dataset)))

print("First Sample's Type:",type(next(iter(sample_csv_dataset))[0]))

sample_dataloader = DataLoader(dataset=sample_csv_dataset, batch_size=3, shuffle=True)

num_epochs = 4

for epoch in range(num_epochs):
    for iteration, (batch_x, batch_y) in enumerate(sample_dataloader):
        print('Epoch: ', epoch, '| Iteration: ', iteration, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

```


## mysql数据读取

生产落地数据多为数据库，本文也针对常见Mysql数据库进行了数据加载，使用的是MYSQL8.0数据库的实例数据库表sakila.payment进行演示。

``` python
import torch
import pandas as pd
import pymysql
from torch.utils.data import Dataset, DataLoader

class SampleMysqlDataset(Dataset):
    def __init__(self):
        # 初始化MySQL数据库连接配置参数
        self.mysql_host = "localhost"
        self.mysql_port = 3307
        self.mysql_user = "utest"
        self.mysql_password = "123456xyq"
        self.mysql_db = "sakila"
        self.mysql_table = "payment"
        self.mysql_charset = "utf8"
        self.mysql_sql_data = "select payment_id, customer_id, staff_id, rental_id, amount from sakila.payment"
        self.mysql_sql_cnt = "select count(*) from sakila.payment"

    def __getitem__(self, item):
        # 创建数据库连接
        conn = pymysql.connect(host=self.mysql_host,
                        port=self.mysql_port,
                        user=self.mysql_user,
                        password=self.mysql_password,
                        db=self.mysql_db,
                        charset=self.mysql_charset)
        raw_dataframe = pd.read_sql(self.mysql_sql_data, conn)
        raw_dataframe_shape = raw_dataframe.shape
        self._x  = torch.from_numpy(raw_dataframe.iloc[:,:raw_dataframe_shape[1]-1].values)
        self._y  = torch.from_numpy(raw_dataframe.iloc[:,raw_dataframe_shape[1]-1].values)
        return self._x[item], self._y[item]

    def __len__(self):
        # 创建数据库连接
        conn = pymysql.connect(host=self.mysql_host,
                        port=self.mysql_port,
                        user=self.mysql_user,
                        password=self.mysql_password,
                        db=self.mysql_db,
                        charset=self.mysql_charset)
        raw_dataframe = pd.read_sql(self.mysql_sql_data, conn)
        raw_dataframe_shape = raw_dataframe.shape
        self._len = raw_dataframe_shape[0]
        return self._len

sample_mysql_dataset = SampleMysqlDataset()

print("Data Size:",len(sample_mysql_dataset))

print("First Sample:",next(iter(sample_mysql_dataset)))

print("First Sample's Type:",type(next(iter(sample_mysql_dataset))[0]))

sample_dataloader = DataLoader(dataset=sample_mysql_dataset, batch_size=3, shuffle=True)

num_epochs = 4

for epoch in range(num_epochs):
    for iteration, (batch_x, batch_y) in enumerate(sample_dataloader):
        print('Epoch: ', epoch, '| Iteration: ', iteration, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

```

## 使用pytorch自带数据集

为快速试验，Pytorch也集成了常见的数据集在torchaudio，torchtext和torchvision中，本代码使用torchvision读取常用的图像算法数据集MNIST，具体代码如下。
``` python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 导入训练集
sample_mnist_dataset = datasets.MNIST(root=r'./data',
                              transform=transforms.ToTensor(),
                              train=True,
                              download=True)

print("Data Size:",len(sample_mnist_dataset))

print("First Sample:",next(iter(sample_mnist_dataset)))

print("First Sample's Type:",type(next(iter(sample_mnist_dataset))[0]))

sample_dataloader = DataLoader(dataset=sample_mnist_dataset, batch_size=3, shuffle=True)

num_epochs = 4

for epoch in range(num_epochs):
    for iter, (batch_x, batch_y) in enumerate(sample_dataloader):
        print('Epoch: ', epoch, '| Iteration: ', iter, '| batch x: ', batch_x.numpy(), '| batch y: ', batch_y.numpy())

```


# 参考
More info: [pan_jinquan](https://blog.csdn.net/guyuealian/article/details/88343924)

More info: [夜和大帝](https://hexo.io/docs/generating.html)
