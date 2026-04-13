import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ==========================================
# 第一阶段：准备“教材”与“预处理工厂” (Data Pipeline)
# ==========================================

# 定义数据预处理流水线（让模型见识更多变异的数字）
transform = transforms.Compose([
    # 1. 数据增强：随机旋转（-15度 到 15度之间）
    # 目的：防止实拍时手机拿歪了，模型提前适应“歪脖子”数字
    transforms.RandomRotation(15),      
    
    # 2. 数据增强：随机缩放（80% 到 120% 之间）
    # 目的：实拍的数字忽大忽小，提前让模型适应大小变化
    transforms.RandomAffine(0, scale=(0.8, 1.2)), 
    
    # 3. 基础转换：变成 PyTorch 认识的 Tensor 张量（0~1的浮点数）
    transforms.ToTensor(),
    
    # 4. 数据标准化：减去均值，除以标准差
    # 目的：0.1307 和 0.3081 是官方统计的 MNIST 数据集所有黑白像素的平均值和标准差。
    # 这样做能把数据中心化，让 Optimizer (教练) 找规律时走得更稳、更快。
    transforms.Normalize((0.1307,), (0.3081,)) 
])

print("📦 正在下载并加载强化版教材 (MNIST数据集)...")
# 加载训练集，应用上面定义的 transform 流水线
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 雇佣搬运工，每次搬 64 张图片进显存/内存，打乱顺序（防止模型背答案）
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# ==========================================
# 第二阶段：搭建“高级数字大脑” (Model Architecture)
# ==========================================

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__() # 祖传代码，必须调用父类初始化
        
        # 1. 视网膜特征提取（卷积层）
        # 输入：1张黑白图。输出：16张特征图。用 3x3 的刷子扫。
        # 尺寸变化：28x28 -> 26x26 (因为刷子边缘不能出界)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3) 
        
        # 2. 激活函数（非线性开关）
        # 把负数（没用的特征）直接变成0，保留正数（有用的特征）
        self.relu = nn.ReLU()
        
        # 3. 【核心升级】特征脱水机（最大池化层）
        # 用 2x2 的窗口在图上滑动，每次只挑最大的那个数字保留。
        # 目的：忽略细微的位置偏移，提炼核心特征。
        # 尺寸变化：26x26 -> 13x13 (长宽各缩减一半)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) 
        
        # 4. 降维打击（拉平层）
        # 把三维的特征方块，撕碎排成一条直线
        self.flatten = nn.Flatten()
        
        # 5. 第一层大皮层（全连接层 1）
        # 接收 16张 * 13宽 * 13高 = 2704 个像素点情报。
        # 映射到 128 个高级神经元上进行逻辑组合。
        self.fc1 = nn.Linear(16 * 13 * 13, 128) 
        
        # 6. 最终决策层（全连接层 2）
        # 把 128 个神经元的讨论结果，收束到 0~9 这 10 个具体数字上
        self.fc2 = nn.Linear(128, 10)          

    def forward(self, x):
        # 这里定义了数据流过的先后顺序：
        x = self.conv1(x)  # 先用刷子扫
        x = self.relu(x)   # 过滤没用的特征
        x = self.pool(x)   # 池化脱水，缩小尺寸
        
        x = self.flatten(x) # 拍扁成一维长条
        
        x = self.fc1(x)    # 进入第一层思考
        x = self.relu(x)   # 再次过滤思考结果
        x = self.fc2(x)    # 得出最终的 10 个数字概率打分
        return x

# 实例化大脑
model = ImprovedCNN()


# ==========================================
# 第三阶段：聘请老师与教练 (Loss & Optimizer)
# ==========================================

# 判卷老师：交叉熵损失（专治多分类选择题，算错惩罚极大）
criterion = nn.CrossEntropyLoss()

# 改错教练：Adam 优化器（目前最智能的步长调节器）
# lr=0.001 是学习率，代表教练每次调旋钮的力度大小
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ==========================================
# 第四阶段：开始地狱特训 (Training Loop)
# ==========================================

print("🔥 炼丹炉点火，开始 10 轮深度特训...")
epochs = 10 # 以前只看 2 遍，现在让模型把 6 万张图看 10 遍！

for epoch in range(epochs):
    running_loss = 0.0 # 用来记录这段时间的平均错误率
    
    # 每次抽出 64 张图 (images) 和 对应的真实数字 (labels)
    for i, (images, labels) in enumerate(train_loader):
        
        # 1. 擦黑板：清空上一步残留的改错指令（梯度）
        optimizer.zero_grad()
        
        # 2. 考试：让模型看着图片猜答案
        outputs = model(images)
        
        # 3. 判卷：老师对比预测答案 outputs 和真实答案 labels，打出耻辱分 loss
        loss = criterion(outputs, labels)
        
        # 4. 追责：顺藤摸瓜算微积分，找出每个神经元该背多少锅（计算梯度）
        loss.backward()
        
        # 5. 改进：教练动手，把 10 万多个参数旋钮全部往正确的方向拨动一点点
        optimizer.step()
        
        # --- 打印进度条逻辑 ---
        running_loss += loss.item() # 累加耻辱分
        if i % 200 == 199: # 每做了 200 批题目（约12800张图），汇报一次
            avg_loss = running_loss / 200
            print(f"[第 {epoch+1}/{epochs} 轮, 批次进度: {i+1}] 当前平均误差: {avg_loss:.4f}")
            running_loss = 0.0 # 汇报完清零，重新算下一波

print("✅ 10 轮特训全部完成，你的模型已经脱胎换骨！")

# ==========================================
# 第五阶段：保存记忆 (Save Model)
# ==========================================
# 把调整到最完美的 10 万多个旋钮参数，全部保存到硬盘里
torch.save(model.state_dict(), "my_improved_model.pth")
print("💾 强力大脑已保存为 my_improved_model.pth，随时可以调用！")
