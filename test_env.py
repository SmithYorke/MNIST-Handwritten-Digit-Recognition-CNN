import torch
print("PyTorch 版本:", torch.__version__)
print("可以用的计算设备:", "GPU可用" if torch.cuda.is_available() else "仅CPU")