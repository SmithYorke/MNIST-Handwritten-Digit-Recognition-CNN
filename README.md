
# MNIST-Handwritten-Digit-Recognition-CNN
Building a CNN Model from Scratch That Can Recognize Real-World Handwritten Digits

<img width="1200" height="1149" alt="image" src="https://github.com/user-attachments/assets/af9652a4-fd5d-438a-9993-6011077a8d1f" />


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











![b86b42ec0d1f52c8fc1cbd158137e193](https://github.com/user-attachments/assets/4b30e016-0d8a-4dd5-a8f9-d861fafe4f2c)
![e5cf371cad223b837bad10849d0cb8f6](https://github.com/user-attachments/assets/608af393-fd86-4223-ab97-9c788ff4849f)
![66730b09b0933a96768e6463beb7e08e](https://github.com/user-attachments/assets/7807a139-1110-475b-b970-2040a3237f7f)
![c1bf2be3d886c9ce4851f319dbf66210](https://github.com/user-attachments/assets/64902591-a34f-4e17-8c61-bc61169697e9)

