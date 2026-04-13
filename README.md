# MNIST-Handwritten-Digit-Recognition-CNN
Building a CNN Model from Scratch That Can Recognize Real-World Handwritten Digits



核心技术栈 & 网络结构

框架: PyTorch

预处理: torchvision.transforms / Pillow / NumPy

网络结构 (ImprovedCNN):

Conv2d (1->16, 3x3):       基础特征提取（找边缘、线条）。

ReLU:                      激活函数，引入非线性，过滤无效特征。

MaxPool2d (2x2):           核心优化。特征图降维脱水，赋予模型“平移不变性”（数字写歪了也能认）。

Flatten:                   多维张量展平为一维向量，衔接全连接层。

Linear (16*13*13 -> 128):  隐藏层，组合低级特征。

ReLU:                      再次激活。

Linear (128 -> 10):        输出层，分类 10 个数字
