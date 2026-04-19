import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import requests
import os
import logging

# 禁用不必要的日志
logging.basicConfig(level=logging.ERROR)

class DualHandTest:
    def __init__(self):
        self.model_path = r'/03_MediaPipe_learning/01_基础库demo/hand_landmark_learning/models/hand_landmarker.task'
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def get_hand_type(self, landmarks):
        wrist = landmarks[0]
        middle_mcp = landmarks[9]
        return "Right" if middle_mcp.x < wrist.x else "Left"
    
    def draw_hand_landmarks(self, frame, hand_landmarks, hand_type):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        color = (0, 255, 0) if hand_type == "Right" else (255, 0, 0)
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            h, w, _ = frame.shape
            start = (int(start_point.x * w), int(start_point.y * h))
            end = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start, end, color, 2)
            cv2.circle(frame, start, 5, color, -1)
            cv2.circle(frame, end, 5, color, -1)
        
        wrist = hand_landmarks[0]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h) - 30)
        
        cv2.putText(
            frame,
            hand_type,
            (wrist_pos[0] - 30, wrist_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )
    
    def run_test(self):
        print("Dual Hand Detection Test")
        print("Press ESC to exit")
        print("Test both hands simultaneously")
        
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
                    "Dual Hand Detection Test",
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
                
                # 绘制双手
                if result.hand_landmarks:
                    for hand_landmarks in result.hand_landmarks:
                        hand_type = self.get_hand_type(hand_landmarks)
                        self.draw_hand_landmarks(frame, hand_landmarks, hand_type)
                    
                    cv2.putText(
                        frame,
                        f"Hands detected: {len(result.hand_landmarks)}",
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
                        "No hands detected",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                
                # 显示手型指示
                cv2.putText(
                    frame,
                    "Right hand: Green",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    "Left hand: Blue",
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA
                )
                
                cv2.imshow('Dual Hand Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = DualHandTest()
    test.run_test()
