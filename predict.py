import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# ==========================================
# 1. 重建强力大脑的外壳 (必须和 train_pro.py 一字不差)
# ==========================================
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2) 
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 13 * 13, 128) 
        self.fc2 = nn.Linear(128, 10)          

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ==========================================
# 2. 唤醒记忆
# ==========================================
model = ImprovedCNN()
# 加载你刚刚花 10 轮特训出来的记忆
model.load_state_dict(torch.load("my_improved_model.pth"))
model.eval() # 开启考试模式（关闭训练时的随机性）

# ==========================================
# 3. 强化版照片洗相房 (预处理逻辑)
# ==========================================
def predict_my_photo(image_path):
    print(f"📷 正在解析照片: {image_path}...")
    
    # A. 基础处理：转灰度图，暴力缩放到 28x28
    img = Image.open(image_path).convert('L').resize((28, 28))
    img_array = np.array(img)
    
    # B. 智能二值化与黑白反转 (去阴影神器)
    # 如果背景是亮的（白纸），我们就反转颜色
    if img_array.mean() > 127: 
        img_array = 255 - img_array
        
    # 动态切掉灰色阴影，只保留最亮的核心笔迹
    threshold = img_array.mean() * 1.5 
    img_array = (img_array > threshold) * 255
    
    final_img = Image.fromarray(img_array.astype('uint8'))
    
    # C. 【核心更新】和训练时一模一样的流水线
    transform = transforms.Compose([
        transforms.ToTensor(),
        # 必须加上这行！否则模型认不出你的图
        transforms.Normalize((0.1307,), (0.3081,)) 
    ])
    
    # 包装成批次 [1, 1, 28, 28]
    img_tensor = transform(final_img).unsqueeze(0) 

    # ==========================================
    # 4. 让强力大脑出马
    # ==========================================
    with torch.no_grad(): # 考试时不需要学新东西，省点脑力
        output = model(img_tensor)
        
        # 提取最高分的那个数字
        prediction = torch.max(output, 1)[1].item()
        
        # [选看] 打印出 AI 对 0-9 每个数字的打分情况
        # 经过 Softmax 可以把乱七八糟的得分变成百分比概率
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence = probabilities[prediction].item() * 100
        
    print("-" * 30)
    print(f"🎯 AI 的结论是: 【 {prediction} 】")
    print(f"📊 它的自信心高达: {confidence:.2f}%")
    print("-" * 30)

# --- 运行这里 ---
# 确保你的照片和代码在同一个文件夹
predict_my_photo("test.jpg")