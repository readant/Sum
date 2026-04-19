## 测试模块

### 1. 人体姿态检测测试 (pose_detection)
- **功能**：检测和可视化人体姿态关键点
- **技术**：MediaPipe PoseLandmarker
- **特点**：支持全身姿态检测，实时绘制骨骼结构

### 2. 全身姿态检测测试 (full_body_detection)
- **功能**：检测和可视化全身关键点
- **技术**：MediaPipe PoseLandmarker (Full Body)
- **特点**：支持详细的全身姿态检测，包括手指关键点

## 目录结构

```
body_landmark_learning/
├── pose_detection/
│   ├── pose_detection_test.py
│   └── README.md
├── full_body_detection/
│   ├── full_body_detection_test.py
│   └── README.md
├── models/
│   └── pose_landmarker.task
└── README.md (本文件)
```

## 依赖要求

所有测试程序需要以下依赖：
- Python 3.7+
- opencv-python
- mediapipe
- numpy

## 运行方法

1. 进入对应测试模块目录
2. 运行测试程序：
   ```
   python [test_file].py
   ```
3. 按照每个模块的README.md中的说明进行测试
4. 按ESC键退出测试

## 模型文件

测试所需的模型文件：
- `pose_landmarker.task` - 姿态检测模型
- 模型文件应放置在 models 目录中

## 技术特点

- **模块化设计**：每个功能独立测试
- **实时响应**：处理速度快，适合实时应用
- **用户友好**：直观的界面和丰富的视觉反馈
- **统一标准**：与手部测试样例保持一致的结构和格式

## 测试建议

1. 从人体姿态检测开始，确保基础功能正常
2. 然后测试全身姿态检测，验证完整的人体姿态识别

## 故障排除

- **模型文件缺失**：确保 models 目录中存在相应的模型文件
- **摄像头无法打开**：确保摄像头权限正确，没有被其他程序占用
- **识别效果不佳**：确保光线充足，被检测对象在摄像头视野内

## 扩展建议

- 添加更多人体部位的检测和识别
- 实现基于姿态的动作识别
- 集成机器学习模型提高准确率
- 开发用户界面和应用程序