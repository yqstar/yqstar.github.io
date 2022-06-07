---
title: Pytorch系列自学教程(3):深度学习之Mnist图像分类TorchServe部署
date: 2022-05-15 18:05:23
tags: Pytorch_Tutorial
categories:
---

## 部署简介

学习应该能够终端落地，这样对于个人带来的满足才是巨大的，才会是有价值的。所以单纯地跑了Demo模型，对自己的技术提升还是很大的，但是如何把技术输出也是必要的。

那么使用 *PyTorch* 训练好 Demo 模型，如何部署到生产环境提供用于提供服务呢？部署形式是非常多样的，本章内容主要介绍的是是大厂出品的TorchServe用于部署Pytorch的模型。

> TorchServe is a performant, flexible and easy to use tool for serving PyTorch eager mode and torschripted models.

从上面的官网介绍的内容可以看出：TorchServe的特点是性能好、灵活性好、易使用的工具，其次面向部署的模型是Pytorch的Eager模式和Script模式的模型。

## TorchServe简介

未完待续 ！！！

## 参考

More info: [Chenglu：如何部署PyTorch模型](https://zhuanlan.zhihu.com/p/344364948)
More info: [TorchServe](https://github.com/pytorch/serve)
More info: [TorchServe_Mnist Example](https://github.com/pytorch/serve/tree/master/examples/image_classifier/mnist)
More info: [随便写点笔记](https://blog.csdn.net/weixin_41977938/article/details/122258595)
More info: [PyTorch Eager mode and Script mode](https://blog.csdn.net/Chris_zhangrx/article/details/117380516)
