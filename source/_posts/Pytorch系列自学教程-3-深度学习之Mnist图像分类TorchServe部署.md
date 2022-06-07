---
title: Pytorch系列自学教程(3):深度学习之Mnist图像分类TorchServe部署
date: 2022-05-15 18:05:23
tags: Pytorch_Tutorial
categories:
---

## 部署简介

学习应该能够终端落地，这样对于个人带来的满足才是巨大的，才会是有价值的。所以单纯地跑了Demo模型，对自己的技术提升还是很大的，但是如何把技术输出也是必要的。

那么使用 *PyTorch* 训练好 Demo 模型，如何部署到生产环境提供用于提供模型服务呢？部署形式是非常多样的，本章内容主要介绍的是是大厂出品的TorchServe用于部署Pytorch的模型。

模型服务是在系统中放置经过训练的机器学习模型的过程，以便它可以接受新的输入并将推理返回给系统。

## TorchServe简介

Torchserve 是 PyTorch 的首选模型服务解决方案。它允许您为您的模型公开一个 Web API，可以直接访问或通过您的应用程序访问。

> TorchServe is a performant, flexible and easy to use tool for serving PyTorch eager mode and torschripted models.

从上面的官网介绍的内容可以看出：TorchServe的特点是性能好、灵活性好、易使用的工具，其次面向部署的模型是Pytorch的Eager模式和Script模式的模型。

TorchServe是由AWS和Facebook合作开发的PyTorch模型服务库，是 [PyTorch开源项目](https://pytorch.org/serve/index.html) 部分。

借助TorchServe，PyTorch用户可以更快地将其模型应用于生产，而无需编写自定义代码：除了提供低延迟预测API之外，TorchServe还为一些最常见的应用程序嵌入了默认处理程序，例如目标检测和文本分类。此外，TorchServe包括多模型服务、用于A/B 测试的模型版本控制、监视指标以及用于应用程序集成的RESTful端点。如你所料，TorchServe支持任何机器学习环境，包括Amazon SageMaker、容器服务和Amazon Elastic Compute Cloud。

TorchServe框架主要分为四个部分：Frontend是TorchServe的请求和响应的处理部分；Worker Process 指的是一组运行的模型实例，可以由管理API设定运行的数量；Model Store是模型存储加载的地方；Backend用于管理Worker Process。

![ts_frame](../_posts/Pytorch系列自学教程-3-深度学习之Mnist图像分类TorchServe部署/ts_frame.png)

## 环境安装

我这边使用的是Windows11，WSL2环境系统进行部署作业。

* 安装 JDK11
  
``` bash
sudo apt-get install openjdk-11-jdk
```

* 安装依赖

``` bash
pip install torch torchtext torchvision sentencepiece psutil future
pip install torchserve torch-model-archiver
```

TorchServe源安装

``` bash
git clone https://github.com/pytorch/serve.git
cd serve

./ts_scripts/setup_wsl_ubuntu
export PATH=$HOME/.local/bin:$PATH
python ./ts_scripts/install_from_src.py
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
```

## 模型打包

TorchServe 的一个关键特性是能够将所有模型工件打包到单个模型存档文件中。它是一个单独的命令行界面 (CLI)，torch-model-archiver，可以使用 state_dict 获取模型检查点或模型定义文件，并将它们打包成 .mar 文件。 然后，任何使用 TorchServe 的人都可以重新分发和提供该文件。它包含以下模型工件：在 torchscript 或模型定义文件的情况下的模型检查点文件和在急切模式的情况下的 state_dict 文件，以及服务模型可能需要的其他可选资产。 CLI 创建一个 .mar 文件，TorchServe 的服务器 CLI 使用该文件为模型提供服务。

torch-model-archiver 命令来打包模型，需要提供以下三个文件。

第 1 步：创建一个新的模型架构文件，其中包含从 torch.nn.modules 扩展的模型类。 在这个例子中，我们创建了 mnist 模型文件 mnist_model.py 文件。

``` python
import torch
from torch import nn
import torch.nn.functional as F

# 构建网络
class MnistClassificationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.max_pool2d = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout2d(p=0.25)
        # self.relu = nn.ReLU()
        self.fc1 = nn.Linear(in_features=9216, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=10)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x= self.max_pool2d(x)
        x = self.dropout(x)
        x = x.view(-1, 9216)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x= self.log_softmax(x)
        return x
```

第 2 步：使用 [mnist_sd](https://yqstar.github.io/2022/05/11/Pytorch系列自学教程-2-深度学习“Hello-World”之Mnist图像分类/) 训练 MNIST 数字识别模型并保存模型的状态字典。

``` python
torch.save(model.state_dict(), "./checkpoints/model_pth/mnist_sd.pt")
```

第 3 步：编写自定义处理程序以在您的模型上运行推理。 在此示例中，我们添加了一个 mnist_handler.py 文件，它使用上述模型对输入灰度图像进行推理并识别图像中的数字。

``` python
from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier
from torch.profiler import ProfilerActivity


class MNISTDigitClassifier(ImageClassifier):
    """
    MNISTDigitClassifier handler class. This handler extends class ImageClassifier from image_classifier.py, a
    default handler. This handler takes an image and returns the number in that image.
    Here method postprocess() has been overridden while others are reused from parent class.
    """

    image_processing = transforms.Compose([
        transforms.ToTensor()
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self):
        super(MNISTDigitClassifier, self).__init__()
        self.profiler_args = {
            "activities" : [ProfilerActivity.CPU],
            "record_shapes": True,
        }


    def postprocess(self, data):
        """The post process of MNIST converts the predicted output response to a label.
        Args:
            data (list): The predicted output from the Inference with probabilities is passed
            to the post-process function
        Returns:
            list : A list of dictionaries with predictions and explanations is returned
        """
        return data.argmax(1).tolist()
        # return data.tolist()
```

第 4 步：使用 torch-model-archiver 程序创建一个 Torch 模型存档以存档上述文件。

``` bash
$ torch-model-archiver -h
usage: torch-model-archiver [-h] --model-name MODEL_NAME  --version MODEL_VERSION_NUMBER
                      --model-file MODEL_FILE_PATH --serialized-file MODEL_SERIALIZED_PATH
                      --handler HANDLER [--runtime {python,python2,python3}]
                      [--export-path EXPORT_PATH] [-f] [--requirements-file]

Model Archiver Tool

optional arguments:
  -h, --help            show this help message and exit
  --model-name MODEL_NAME
                        Exported model name. Exported file will be named as
                        model-name.mar and saved in current working directory
                        if no --export-path is specified, else it will be
                        saved under the export path
  --serialized-file SERIALIZED_FILE
                        Path to .pt or .pth file containing state_dict in
                        case of eager mode or an executable ScriptModule
                        in case of TorchScript.
  --model-file MODEL_FILE
                        Path to python file containing model architecture.
                        This parameter is mandatory for eager mode models.
                        The model architecture file must contain only one
                        class definition extended from torch.nn.modules.
  --handler HANDLER     TorchServe's default handler name  or handler python
                        file path to handle custom TorchServe inference logic.
  --extra-files EXTRA_FILES
                        Comma separated path to extra dependency files.
  --runtime {python,python2,python3}
                        The runtime specifies which language to run your
                        inference code on. The default runtime is
                        RuntimeType.PYTHON. At the present moment we support
                        the following runtimes python, python2, python3
  --export-path EXPORT_PATH
                        Path where the exported .mar file will be saved. This
                        is an optional parameter. If --export-path is not
                        specified, the file will be saved in the current
                        working directory.
  --archive-format {tgz,default}
                        The format in which the model artifacts are archived.
                        "tgz": This creates the model-archive in <model-name>.tar.gz format.
                        If platform hosting requires model-artifacts to be in ".tar.gz"
                        use this option.
                        "no-archive": This option creates an non-archived version of model artifacts
                        at "export-path/{model-name}" location. As a result of this choice,
                        MANIFEST file will be created at "export-path/{model-name}" location
                        without archiving these model files
                        "default": This creates the model-archive in <model-name>.mar format.
                        This is the default archiving format. Models archived in this format
                        will be readily hostable on TorchServe.
  -f, --force           When the -f or --force flag is specified, an existing
                        .mar file with same name as that provided in --model-
                        name in the path specified by --export-path will
                        overwritten
  -v, --version         Model's version.
  -r, --requirements-file
                        Path to requirements.txt file containing a list of model specific python
                        packages to be installed by TorchServe for seamless model serving.
```

``` bash
torch-model-archiver --model-name mnist --version 1.0 --model-file mnist_model.py --serialized-file mnist_sd.pt --export-path ./model_store --handler mnist_handler.py -f
```

## 模型部署

``` bash
$ torchserve --help
usage: torchserve [-h] [-v | --version]
                          [--start]
                          [--stop]
                          [--ts-config TS_CONFIG]
                          [--model-store MODEL_STORE]
                          [--workflow-store WORKFLOW_STORE]
                          [--models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]]
                          [--log-config LOG_CONFIG]

torchserve

mandatory arguments:
  --model-store MODEL_STORE
                        Model store location where models can be loaded

  

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         Return TorchServe Version
  --start               Start the model-server
  --stop                Stop the model-server
  --ts-config TS_CONFIG
                        Configuration file for TorchServe

  --models MODEL_PATH1 MODEL_NAME=MODEL_PATH2... [MODEL_PATH1 MODEL_NAME=MODEL_PATH2... ...]
                        Models to be loaded using [model_name=]model_location
                        format. Location can be a HTTP URL, a model archive
                        file or directory contains model archive files in
                        MODEL_STORE.
  --log-config LOG_CONFIG
                        Log4j configuration file for TorchServe
  --ncs, --no-config-snapshots         
                        Disable snapshot feature
  --workflow-store WORKFLOW_STORE
                        Workflow store location where workflow can be loaded. Defaults to model-store
```

### 启动torchserve服务

``` bash
torchserve --start --ncs --model-store model_store --models mnist.mar
```

模型启动日志如下截图：

![ts_start](../_posts/Pytorch系列自学教程-3-深度学习之Mnist图像分类TorchServe部署/ts_start.png)

### 推理健康检查API

``` bash
curl http://localhost:8080/ping
```

如果server正常运行, 响应会如截图所示：

![ts_ping](../_posts/Pytorch系列自学教程-3-深度学习之Mnist图像分类TorchServe部署/ts_ping.png)

### 推理

``` bash
curl http://127.0.0.1:8080/predictions/mnist -T ./data/test.png
```

test.png为数字为0的图片，通过上述的调用推理，可以看出结果是能正常返回的，是可以作为下游应用调用。

![ts_infer](../_posts/Pytorch系列自学教程-3-深度学习之Mnist图像分类TorchServe部署/ts_infer.png)

### 停止torchserve服务

``` bash
torchserve --start
```

## 探索

[完整代码](https://github.com/yqstar/Awesome_Pytorch_Tutorial/tree/master/Pytorch_Lesson3)已上传Github，有需要的可以自行下载代码，如果对你有帮助，请Star，哈哈哈哈！

到此为止，已经可以使用自己数据玩耍各种Demo，快（苦）乐（逼）地进行炼丹之路。道路阻且长，行则将至，但行好事莫问前程。

* 除了使用TorchServe部署模型，还有其他的解决方案吗？
* 除了使用提供这种Web API的形式，是否可以构建一个GUI的形式提供呢？例如 PYQT5 ？这里放一张PYQT5的图，后面会填坑。

![pyqt_demo](../_posts/Pytorch系列自学教程-3-深度学习之Mnist图像分类TorchServe部署/pyqt_demo.png)

正如，人往往会对未知的事情产生恐惧，因为结局是未知的。所以当一切不再未知的时候，那么是不是就不会产生恐惧呢？

## 参考

More info: [Chenglu：如何部署PyTorch模型](https://zhuanlan.zhihu.com/p/344364948)
More info: [TorchServe](https://github.com/pytorch/serve)
More info: [TorchServe_Mnist Example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)
More info: [随便写点笔记](https://blog.csdn.net/weixin_41977938/article/details/122258595)
More info: [PyTorch Eager mode and Script mode](https://blog.csdn.net/Chris_zhangrx/article/details/117380516)
More info: [Self-host your 🤗HuggingFace Transformer NER model with Torchserve + Streamlit](https://cceyda.github.io/blog/huggingface/torchserve/streamlit/ner/2020/10/09/huggingface_streamlit_serve.html)
More info: [TorchServe搭建codeBERT分类模型服务](https://ceshiren.com/t/topic/20770)
More info: [torchserver模型本地部署和docker部署](https://blog.csdn.net/qq_15821487/article/details/122684773)
