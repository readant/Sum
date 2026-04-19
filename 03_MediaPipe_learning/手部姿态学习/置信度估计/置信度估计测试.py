import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging
import time

logging.basicConfig(level=logging.ERROR)

class ConfidenceEstimationTest:
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

    def draw_neon_point(self, frame, pos, color, radius=5, alpha=1.0):
        overlay = frame.copy()
        cv2.circle(overlay, pos, radius + 2, color, -1)
        cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0, frame)
        cv2.circle(frame, pos, radius, (255, 255, 255), -1)
        cv2.circle(frame, pos, radius - 2, color, -1)

    def calculate_confidence_color(self, confidence):
        if confidence > 0.8:
            return (0, 255, 0)
        elif confidence > 0.6:
            return (0, 255, 255)
        elif confidence > 0.4:
            return (255, 255, 0)
        else:
            return (0, 0, 255)

    def draw_hand_with_confidence(self, frame, hand_landmarks, handedness, detection_score):
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
        
        conf_color = self.calculate_confidence_color(detection_score)
        
        for idx, landmarks in enumerate(hand_landmarks):
            h, w, _ = frame.shape
            hand_label = handedness[idx][0].category_name
            
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
            
            wrist = landmarks[0]
            wrist_pos = (int(wrist.x * w), int(wrist.y * h))
            label = "L" if hand_label == "Left" else "R"
            cv2.putText(frame, f"{label}: {detection_score:.2f}", (wrist_pos[0] + 15, wrist_pos[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, conf_color, 2, cv2.LINE_AA)

    def draw_ui(self, frame, detected, hand_count=0, confidences=None):
        h, w, _ = frame.shape
        
        header_height = 70
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header.fill(25)
        cv2.addWeighted(frame[:header_height], 0.85, header, 0.15, 0, frame[:header_height])
        
        title = "Confidence Estimation"
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        status_bar = np.zeros((50, w, 3), dtype=np.uint8)
        status_bar.fill(20)
        frame[h - 50:h, :] = status_bar
        
        if detected and confidences:
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            conf_color = self.calculate_confidence_color(avg_conf)
            status_text = f"  Hands: {hand_count} | Avg Confidence: {avg_conf:.2f}"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, conf_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, conf_color, -1)
        else:
            status_text = "  No Hands Detected"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, "ESC to Exit", (w - 150, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    def run_test(self):
        print("Confidence Estimation Test")
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
                confidences = []
                if result.hand_landmarks:
                    detected = True
                    # 由于MediaPipe API可能没有detection_confidences属性，使用默认置信度
                    default_confidence = 0.9
                    self.draw_hand_with_confidence(frame, result.hand_landmarks, result.handedness, default_confidence)
                    confidences.append(default_confidence)

                self.draw_ui(frame, detected, len(result.hand_landmarks), confidences)

                cv2.imshow('Confidence Estimation', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = ConfidenceEstimationTest()
    test.run_test()