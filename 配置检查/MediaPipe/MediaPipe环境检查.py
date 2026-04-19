#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MediaPipe环境检测脚本
用于验证MediaPipe相关依赖是否正确安装
"""

import os
import sys

# 检查Python版本
print("=== 环境检测开始 ===")
print(f"Python版本: {sys.version}")
print(f"操作系统: {sys.platform}")
print()

# 检查MediaPipe依赖
try:
    import mediapipe
    print(f"✅ MediaPipe版本: {mediapipe.__version__}")
except ImportError:
    print("❌ MediaPipe未安装")

# 检查OpenCV
try:
    import cv2
    print(f"✅ OpenCV版本: {cv2.__version__}")
except ImportError:
    print("❌ OpenCV未安装")

# 检查NumPy
try:
    import numpy
    print(f"✅ NumPy版本: {numpy.__version__}")
except ImportError:
    print("❌ NumPy未安装")

# 检查Matplotlib
try:
    import matplotlib
    print(f"✅ Matplotlib版本: {matplotlib.__version__}")
except ImportError:
    print("❌ Matplotlib未安装")

# 检查Requests
try:
    import requests
    print(f"✅ Requests版本: {requests.__version__}")
except ImportError:
    print("❌ Requests未安装")

# 检查tqdm
try:
    import tqdm
    print(f"✅ tqdm版本: {tqdm.__version__}")
except ImportError:
    print("❌ tqdm未安装")

print()
print("=== 环境检测完成 ===")
print("如果所有依赖都已安装，您可以运行以下命令测试MediaPipe:")
print("  conda activate mp")
print("  python 03_MediaPipe_learning/手部姿态学习/手部检测/手部检测测试.py")