# -*- coding: utf-8 -*-
"""
✅ cv-yolo 环境检查脚本（最终优化版）
专为你的 RTX 5060 + cv-yolo 环境设计
"""

import sys
import os

# 关闭警告
import warnings

warnings.filterwarnings('ignore')


def print_header(title):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_item(name, status, details=""):
    icon = "✅" if status else "❌"
    if details:
        print(f"  {icon} {name}: {details}")
    else:
        print(f"  {icon} {name}")


# ================= 开始检查 =================
print_header("cv-yolo 环境检查工具")

# 1. Python
print_header("1. Python 基础环境")
print_item("Python 版本", True, sys.version.split()[0])

# 2. PyTorch & GPU
print_header("2. PyTorch & GPU 加速")
try:
    import torch

    torch_ok = True
    print_item("PyTorch", True, torch.__version__)

    cuda_ok = torch.cuda.is_available()
    print_item("CUDA 可用", cuda_ok)

    if cuda_ok:
        print_item("GPU 型号", True, torch.cuda.get_device_name(0))
        print_item("GPU 数量", True, str(torch.cuda.device_count()))
except ImportError:
    torch_ok = False
    print_item("PyTorch", False, "未安装")

# 3. 核心视觉库
print_header("3. 核心视觉库")
try:
    import cv2

    print_item("OpenCV", True, cv2.__version__)
except ImportError:
    print_item("OpenCV", False, "未安装")

try:
    import ultralytics

    print_item("YOLO (Ultralytics)", True, ultralytics.__version__)
except ImportError:
    print_item("YOLO (Ultralytics)", False, "未安装")

try:
    import numpy

    print_item("NumPy", True, numpy.__version__)
except ImportError:
    print_item("NumPy", False, "未安装")

try:
    import PIL

    print_item("Pillow", True, PIL.__version__)
except ImportError:
    print_item("Pillow", False, "未安装")

# 4. YOLO 模型文件检查（新增！）
print_header("4. YOLO 模型文件")
model_path = r"/03_computer_vision/data/models/yolov8n.pt"
if os.path.exists(model_path):
    print_item("模型文件", True, "位置正确")
    print(f"     路径: {model_path}")
else:
    print_item("模型文件", False, "未找到，请检查路径")

# 5. Jupyter
print_header("5. 开发环境")
try:
    import jupyterlab

    print_item("JupyterLab", True, "已安装")
except ImportError:
    print_item("JupyterLab", False, "未安装")

# ================= 最终总结 =================
print_header("检查完成")

if torch_ok and cuda_ok:
    print("\n🎉 恭喜！你的 cv-yolo 环境非常完美！")
    print("   GPU 已激活，可以满血跑 YOLO 了！")
else:
    print("\n⚠️  环境存在问题，请检查上面的 ❌ 项")

print("=" * 60 + "\n")