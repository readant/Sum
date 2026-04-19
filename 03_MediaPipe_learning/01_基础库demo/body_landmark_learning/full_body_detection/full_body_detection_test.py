import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

# 禁用不必要的日志
logging.basicConfig(level=logging.ERROR)

class FullBodyDetectionTest:
    def __init__(self):
        self.model_path = r'/03_MediaPipe_learning/01_基础库demo/body_landmark_learning/models/pose_landmarker.task'
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.7,
            min_pose_presence_confidence=0.7,
            min_tracking_confidence=0.7,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)
    
    def draw_full_body_landmarks(self, frame, pose_landmarks):
        # 定义不同身体部位的颜色
        colors = {
            'head': (0, 255, 255),    # 青色
            'torso': (0, 255, 0),      # 绿色
            'arms': (255, 0, 0),       # 蓝色
            'legs': (255, 255, 0)      # 黄色
        }
        
        # 定义骨骼连接
        connections = [
            # 头部
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6),
            # 躯干
            (11, 12), (12, 24), (24, 23), (23, 11),
            # 左臂
            (11, 13), (13, 15),
            # 右臂
            (12, 14), (14, 16),
            # 左腿
            (23, 25), (25, 27), (27, 29), (29, 31),
            # 右腿
            (24, 26), (26, 28), (28, 30), (30, 32)
        ]
        
        for landmarks in pose_landmarks:
            h, w, _ = frame.shape
            
            # 绘制骨骼连接线
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    
                    start = (int(start_point.x * w), int(start_point.y * h))
                    end = (int(end_point.x * w), int(end_point.y * h))
                    
                    # 根据连接选择颜色
                    if start_idx in [0, 1, 2, 3, 4, 5, 6, 7]:
                        color = colors['head']
                    elif start_idx in [11, 12, 23, 24]:
                        color = colors['torso']
                    elif start_idx in [13, 14, 15, 16]:
                        color = colors['arms']
                    else:
                        color = colors['legs']
                    
                    cv2.line(frame, start, end, color, 2)
            
            # 绘制关键点
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                
                # 根据点的位置选择颜色
                if i in [0, 1, 2, 3, 4, 5, 6, 7]:
                    color = colors['head']
                elif i in [11, 12, 23, 24]:
                    color = colors['torso']
                elif i in [13, 14, 15, 16]:
                    color = colors['arms']
                else:
                    color = colors['legs']
                
                cv2.circle(frame, pos, 4, color, -1)
                cv2.circle(frame, pos, 6, (255, 255, 255), 1)
    
    def run_test(self):
        print("Full Body Detection Test")
        print("Press ESC to exit")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Cannot read camera frame")
                    break
                
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                
                # 绘制标题
                cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
                cv2.putText(
                    frame,
                    "Full Body Detection Test",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )
                
                # 处理帧
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)
                
                # 绘制全身关键点
                if result.pose_landmarks:
                    self.draw_full_body_landmarks(frame, result.pose_landmarks)
                    cv2.putText(
                        frame,
                        f"Poses detected: {len(result.pose_landmarks)}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        frame,
                        "No pose detected",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                
                cv2.imshow('Full Body Detection Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = FullBodyDetectionTest()
    test.run_test()