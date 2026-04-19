## 项目说明

本项目是专门针对面部检测、面部关键点、表情识别和面部特征分析的深入研究项目。

## 测试模块

### 1. 面部检测测试 (face_detection)
- **功能**：检测和可视化面部关键点
- **技术**：MediaPipe FaceLandmarker
- **特点**：实时面部关键点检测，绘制面部网格结构

### 2. 表情识别测试 (expression_recognition)
- **功能**：识别面部表情和情绪
- **技术**：基于面部关键点的表情分析
- **特点**：支持多种基本表情识别，可视化情绪状态

### 3. 面部特征分析测试 (facial_features)
- **功能**：分析面部特征和属性
- **技术**：基于面部关键点的特征提取
- **特点**：面部比例分析、特征点距离计算、对称性分析

## 目录结构

```
face_landmark_learning/
├── face_detection/
│   ├── face_detection_test.py
│   └── README.md
├── expression_recognition/
│   ├── expression_recognition_test.py
│   └── README.md
├── facial_features/
│   ├── facial_features_test.py
│   └── README.md
├── models/
│   └── face_landmarker.task
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
- `face_landmarker.task` - 面部检测模型
- 模型文件应放置在 models 目录中

## 技术特点

- **模块化设计**：每个功能独立测试，便于深入研究
- **实时响应**：处理速度快，适合实时应用
- **用户友好**：直观的界面和丰富的视觉反馈
- **深入分析**：不仅检测，还提供特征分析和表情识别

## 测试建议

1. 从面部检测开始，确保基础功能正常
2. 测试表情识别，了解情绪分析功能
3. 测试面部特征分析，深入研究面部属性

## 故障排除

- **模型文件缺失**：确保 models 目录中存在相应的模型文件
- **摄像头无法打开**：确保摄像头权限正确，没有被其他程序占用
- **识别效果不佳**：确保光线充足，面部朝向摄像头，距离适当

## 扩展建议

- 添加更多表情类型
- 实现面部追踪和连续表情分析
- 集成机器学习模型提高准确率
- 开发用户界面和应用程序