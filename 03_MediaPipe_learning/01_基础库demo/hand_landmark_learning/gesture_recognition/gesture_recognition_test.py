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

class GestureRecognitionTest:
    def __init__(self):
        self.model_path = r'/03_MediaPipe_learning/01_基础库demo/hand_landmark_learning/models/hand_landmarker.task'
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
    
    def is_finger_extended(self, tip, pip, mcp, wrist):
        angle = self.calculate_angle(tip, pip, wrist)
        return angle > 160
    
    def is_thumb_extended(self, thumb_tip, thumb_ip, wrist, is_left_hand=False):
        if is_left_hand:
            return thumb_tip.x > thumb_ip.x
        else:
            return thumb_tip.x < thumb_ip.x
    
    def calculate_angle(self, p1, p2, p3):
        a = np.array([p1.x, p1.y])
        b = np.array([p2.x, p2.y])
        c = np.array([p3.x, p3.y])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        return np.degrees(angle)
    
    def recognize_gesture(self, hand_landmarks, is_left_hand=False):
        landmarks = hand_landmarks
        
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        index_tip = landmarks[8]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_mcp = landmarks[17]
        wrist = landmarks[0]
        
        thumb_extended = self.is_thumb_extended(thumb_tip, thumb_ip, wrist, is_left_hand)
        index_extended = self.is_finger_extended(index_tip, index_pip, index_mcp, wrist)
        middle_extended = self.is_finger_extended(middle_tip, middle_pip, middle_mcp, wrist)
        ring_extended = self.is_finger_extended(ring_tip, ring_pip, ring_mcp, wrist)
        pinky_extended = self.is_finger_extended(pinky_tip, pinky_pip, pinky_mcp, wrist)
        
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        if not thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "0"
        elif not thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "1"
        elif not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
            return "2"
        elif not thumb_extended and index_extended and middle_extended and ring_extended and not pinky_extended:
            return "3"
        elif not thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "4"
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            return "5"
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and pinky_extended:
            return "6"
        elif thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
            if thumb_index_dist > 0.15:
                return "Hello"
            else:
                return "8"
        elif thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
            return "Thank you"
        elif thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
            if thumb_index_dist < 0.1:
                return "OK"
            else:
                return "5"
        
        return "Unrecognized"
    
    def draw_hand_landmarks(self, frame, hand_landmarks):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for landmarks in hand_landmarks:
            for connection in connections:
                start_idx, end_idx = connection
                start_point = landmarks[start_idx]
                end_point = landmarks[end_idx]
                
                h, w, _ = frame.shape
                start = (int(start_point.x * w), int(start_point.y * h))
                end = (int(end_point.x * w), int(end_point.y * h))
                
                cv2.line(frame, start, end, (0, 255, 0), 2)
                cv2.circle(frame, start, 5, (0, 0, 255), -1)
                cv2.circle(frame, end, 5, (0, 0, 255), -1)
    
    def run_test(self):
        print("Gesture Recognition Test")
        print("Press ESC to exit")
        print("Test gestures: 0-6, Hello, Thank you, OK")
        
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
                    "Gesture Recognition Test",
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
                
                # 识别手势
                if result.hand_landmarks:
                    self.draw_hand_landmarks(frame, result.hand_landmarks)
                    gesture = self.recognize_gesture(result.hand_landmarks[0])
                    
                    cv2.putText(
                        frame,
                        f"Recognized: {gesture}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                else:
                    cv2.putText(
                        frame,
                        "No hand detected",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )
                
                cv2.imshow('Gesture Recognition Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = GestureRecognitionTest()
    test.run_test()
