import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging
import time

logging.basicConfig(level=logging.ERROR)

class GestureRecognitionTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'hand_landmarker.task')
        self.detector = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self._initialize_detector()

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def draw_neon_line(self, frame, start, end, color, thickness=2):
        overlay = frame.copy()
        cv2.line(overlay, start, end, color, thickness + 2)
        cv2.addWeighted(frame, 0.6, overlay, 0.4, 0, frame)
        cv2.line(frame, start, end, color, thickness)

    def draw_neon_point(self, frame, pos, color, radius=5):
        overlay = frame.copy()
        cv2.circle(overlay, pos, radius + 2, color, -1)
        cv2.addWeighted(frame, 0.5, overlay, 0.5, 0, frame)
        cv2.circle(frame, pos, radius, (255, 255, 255), -1)
        cv2.circle(frame, pos, radius - 2, color, -1)

    def recognize_gesture(self, hand_landmarks):
        landmarks = hand_landmarks
        
        # 获取关键 landmarks
        wrist = landmarks[0]
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        index_tip = landmarks[8]
        index_pip = landmarks[7]
        index_mcp = landmarks[5]
        middle_tip = landmarks[12]
        middle_pip = landmarks[11]
        middle_mcp = landmarks[9]
        ring_tip = landmarks[16]
        ring_pip = landmarks[15]
        ring_mcp = landmarks[13]
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[19]
        pinky_mcp = landmarks[17]
        
        # 计算手指是否伸展的函数
        def is_finger_extended(tip, pip, mcp):
            # 计算手指关节的角度
            # 如果指尖在PIP关节上方且PIP关节在MCP关节上方，则认为手指伸展
            return tip.y < pip.y and pip.y < mcp.y
        
        # 特殊处理拇指
        def is_thumb_extended(thumb_tip, thumb_ip, thumb_mcp, wrist):
            # 拇指的判断需要考虑水平位置
            # 对于右手，拇指应该在手掌外侧；对于左手，拇指应该在手掌内侧
            # 这里使用简单的垂直位置判断，结合水平位置
            return thumb_tip.y < thumb_ip.y and abs(thumb_tip.x - wrist.x) > 0.1
        
        # 检测各个手指的状态
        thumb_extended = is_thumb_extended(thumb_tip, thumb_ip, thumb_mcp, wrist)
        index_extended = is_finger_extended(index_tip, index_pip, index_mcp)
        middle_extended = is_finger_extended(middle_tip, middle_pip, middle_mcp)
        ring_extended = is_finger_extended(ring_tip, ring_pip, ring_mcp)
        pinky_extended = is_finger_extended(pinky_tip, pinky_pip, pinky_mcp)
        
        # 统计伸展的手指数量
        fingers_up = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])
        
        # 特殊手势识别
        if not any([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended]):
            return "Fist", (0, 0, 255)
        elif fingers_up == 1:
            if index_extended:
                return "One", (0, 255, 255)
            elif thumb_extended:
                return "Thumb Up", (255, 255, 0)
            else:
                return "One", (0, 255, 255)
        elif fingers_up == 2:
            if index_extended and middle_extended:
                return "Two", (0, 255, 0)
            elif index_extended and pinky_extended:
                return "Rock", (255, 0, 255)
            else:
                return "Two", (0, 255, 0)
        elif fingers_up == 3:
            if index_extended and middle_extended and ring_extended:
                return "Three", (255, 100, 0)
            else:
                return "Three", (255, 100, 0)
        elif fingers_up == 4:
            if not thumb_extended:
                return "Four", (255, 255, 0)
            else:
                return "Four", (255, 255, 0)
        elif fingers_up == 5:
            return "Five", (0, 255, 255)
        else:
            return "Unknown", (128, 128, 128)

    def draw_hand_landmarks(self, frame, hand_landmarks, handedness):
        palm_color = (0, 255, 255)
        finger_color = (0, 200, 255)
        
        palm_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]
        
        for idx, landmarks in enumerate(hand_landmarks):
            h, w, _ = frame.shape
            gesture, color = self.recognize_gesture(landmarks)
            hand_label = "L" if handedness[idx][0].category_name == "Left" else "R"
            
            for connection in palm_connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    start = (int(start_point.x * w), int(start_point.y * h))
                    end = (int(end_point.x * w), int(end_point.y * h))
                    self.draw_neon_line(frame, start, end, palm_color, 2)
            
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                self.draw_neon_point(frame, pos, finger_color, 4)
            
            index_tip = landmarks[8]
            tip_pos = (int(index_tip.x * w), int(index_tip.y * h))
            cv2.putText(frame, f"{hand_label}: {gesture}", (tip_pos[0] + 15, tip_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    def draw_ui(self, frame, detected, hand_count=0, gestures=None):
        h, w, _ = frame.shape
        
        header_height = 70
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header.fill(25)
        cv2.addWeighted(frame[:header_height], 0.85, header, 0.15, 0, frame[:header_height])
        
        title = "Gesture Recognition"
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        status_bar = np.zeros((50, w, 3), dtype=np.uint8)
        status_bar.fill(20)
        frame[h - 50:h, :] = status_bar
        
        if detected and gestures:
            gesture_str = " | ".join([f"{g}" for g in gestures])
            status_text = f"  {gesture_str}"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 255, 0), -1)
        else:
            status_text = "  No Hands Detected"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, "ESC to Exit", (w - 150, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    def run_test(self):
        print("Gesture Recognition Test")
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

                self.frame_count += 1
                if self.frame_count % 10 == 0:
                    elapsed = time.time() - self.start_time
                    self.fps = 10 / elapsed if elapsed > 0 else 0
                    self.start_time = time.time()

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)

                detected = False
                gestures = []
                if result.hand_landmarks:
                    detected = True
                    self.draw_hand_landmarks(frame, result.hand_landmarks, result.handedness)
                    for landmarks in result.hand_landmarks:
                        gesture, _ = self.recognize_gesture(landmarks)
                        gestures.append(gesture)

                self.draw_ui(frame, detected, len(result.hand_landmarks), gestures)

                cv2.imshow('Gesture Recognition', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = GestureRecognitionTest()
    test.run_test()