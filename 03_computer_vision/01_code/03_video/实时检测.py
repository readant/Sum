from ultralytics import YOLO
import cv2

# 加载训练好的轻量化模型,这个模型提前学会了80种常见的物体
model = YOLO(r"E:\Projects0\Sum\02_cv-yolo_learning\data\models\yolov8n.pt")

# 打开电脑摄像头（0 = ,默认摄像头，1=外接摄像头）
cap = cv2.VideoCapture(0) # 实时检测入口

# 设置摄像头分辨率（可选提升速度）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("实时检测开始！按q退出")
# 这是个死循环，只要摄像头开着，就一阵一阵不停地处理画面
# 视频 = 连续的图片（帧），循环处理 = 实时检测
while cap.isOpened():
    ret,frame = cap.read() # 读取每一帧画面，frame就是当前拍到的原始画面
    if not ret:
        break
    # GPU加速推理 conf = 0.5 只保留置信度50%以上的结果 imgsz模型处理尺寸 越大越准、越小越快
    results = model(frame,conf = 0.5,imgsz=640) # imgsz = 640 控制输入尺寸,影响速度

    # 绘制检测框并显示，plot()是YOLO自带的函数：自动在画面上画框、写类名、标置信度
    annotated_frame = results[0].plot()  # 自动绘制框+类别+置信度
    # 显示画面
    cv2.imshow('YOLOv8 Real-time Detection', annotated_frame)

    # 按q退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 清理资源
cap.release()
cv2.destroyAllWindows()
print("实时检测结束!")