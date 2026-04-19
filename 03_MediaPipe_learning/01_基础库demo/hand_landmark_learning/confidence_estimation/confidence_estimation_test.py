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

class ConfidenceEstimationTest:
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
    
    def calculate_gesture_confidence(self, finger_states, expected_states):
        matches = sum(1 for a, b in zip(finger_states, expected_states) if a == b)
        return matches / len(finger_states)
    
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
        
        finger_states = [thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        
        gestures = [
            ("0", [False, False, False, False, False]),
            ("1", [False, True, False, False, False]),
            ("2", [False, True, True, False, False]),
            ("3", [False, True, True, True, False]),
            ("4", [False, True, True, True, True]),
            ("5", [True, True, True, True, True]),
            ("Hello", [True, True, False, False, False]),
            ("Thank you", [True, False, False, False, False]),
            ("OK", [True, True, True, True, True])
        ]
        
        best_gesture = "Unrecognized"
        best_confidence = 0
        
        for gesture_name, expected_states in gestures:
            confidence = self.calculate_gesture_confidence(finger_states, expected_states)
            
            if gesture_name == "Hello" and confidence > 0.8:
                if thumb_index_dist > 0.15:
                    confidence = 1.0
            elif gesture_name == "OK" and confidence > 0.8:
                if thumb_index_dist < 0.1:
                    confidence = 1.0
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_gesture = gesture_name
        
        return best_gesture, best_confidence
    
    def draw_hand_landmarks(self, frame, hand_landmarks, gesture, confidence):
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20)
        ]
        
        for connection in connections:
            start_idx, end_idx = connection
            start_point = hand_landmarks[start_idx]
            end_point = hand_landmarks[end_idx]
            
            h, w, _ = frame.shape
            start = (int(start_point.x * w), int(start_point.y * h))
            end = (int(end_point.x * w), int(end_point.y * h))
            
            cv2.line(frame, start, end, (0, 255, 0), 2)
            cv2.circle(frame, start, 5, (0, 0, 255), -1)
            cv2.circle(frame, end, 5, (0, 0, 255), -1)
        
        wrist = hand_landmarks[0]
        wrist_pos = (int(wrist.x * w), int(wrist.y * h) - 30)
        
        cv2.putText(
            frame,
            f"{gesture}",
            (wrist_pos[0] - 30, wrist_pos[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )
        
        conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
        cv2.putText(
            frame,
            f"Confidence: {confidence:.0%}",
            (wrist_pos[0] - 30, wrist_pos[1] + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            conf_color,
            2,
            cv2.LINE_AA
        )
    
    def run_test(self):
        print("Confidence Estimation Test")
        print("Press ESC to exit")
        print("Test gesture confidence levels")
        
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
                    "Confidence Estimation Test",
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
                
                # 识别手势和置信度
                if result.hand_landmarks:
                    self.draw_hand_landmarks(frame, result.hand_landmarks[0], "", 0)
                    gesture, confidence = self.recognize_gesture(result.hand_landmarks[0])
                    
                    # 显示手势和置信度
                    cv2.putText(
                        frame,
                        f"Gesture: {gesture}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )
                    
                    conf_color = (0, 255, 0) if confidence > 0.8 else (0, 255, 255) if confidence > 0.6 else (0, 0, 255)
                    cv2.putText(
                        frame,
                        f"Confidence: {confidence:.0%}",
                        (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        conf_color,
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
                
                # 显示置信度说明
                cv2.putText(
                    frame,
                    "Confidence levels:",
                    (10, h - 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )
                cv2.putText(
                    frame,
                    ">80%: Green | 60-80%: Yellow | <60%: Red",
                    (10, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (200, 200, 200),
                    2,
                    cv2.LINE_AA
                )
                
                cv2.imshow('Confidence Estimation Test', frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = ConfidenceEstimationTest()
    test.run_test()
