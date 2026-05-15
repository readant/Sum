import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging
import time
import numpy as np

logging.basicConfig(level=logging.ERROR)

class PoseDetectionTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'pose_landmarker.task')
        self.detector = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self._initialize_detector()

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        self.detector = vision.PoseLandmarker.create_from_options(options)

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

    def draw_pose_landmarks(self, frame, pose_landmarks):
        color_map = {
            0: (0, 255, 255), 1: (0, 255, 255), 2: (0, 255, 255), 3: (0, 255, 255),
            4: (0, 255, 255), 5: (0, 255, 255), 6: (0, 255, 255), 7: (0, 255, 255),
            8: (0, 255, 255), 9: (0, 255, 255), 10: (0, 255, 255),
            11: (0, 255, 0), 12: (0, 255, 0), 13: (0, 200, 0), 14: (0, 200, 0),
            15: (0, 150, 0), 16: (0, 150, 0),
            23: (0, 255, 0), 24: (0, 255, 0),
            25: (255, 165, 0), 26: (255, 165, 0), 27: (255, 165, 0), 28: (255, 165, 0),
            29: (255, 165, 0), 30: (255, 165, 0), 31: (255, 165, 0), 32: (255, 165, 0)
        }
        
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6),
            (11, 12), (12, 24), (24, 23), (23, 11),
            (11, 13), (13, 15),
            (12, 14), (14, 16),
            (23, 25), (25, 27), (27, 29), (29, 31),
            (24, 26), (26, 28), (28, 30), (30, 32)
        ]
        
        for landmarks in pose_landmarks:
            h, w, _ = frame.shape
            
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(landmarks) and end_idx < len(landmarks):
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    start = (int(start_point.x * w), int(start_point.y * h))
                    end = (int(end_point.x * w), int(end_point.y * h))
                    color = color_map.get(start_idx, (255, 255, 0))
                    self.draw_neon_line(frame, start, end, color, 2)
            
            for i, point in enumerate(landmarks):
                pos = (int(point.x * w), int(point.y * h))
                color = color_map.get(i, (255, 255, 0))
                self.draw_neon_point(frame, pos, color, 5)

    def draw_ui(self, frame, detected):
        h, w, _ = frame.shape
        
        header_height = 70
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header.fill(25)
        cv2.addWeighted(frame[:header_height], 0.85, header, 0.15, 0, frame[:header_height])
        
        title = "Pose Detection"
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        status_bar = np.zeros((50, w, 3), dtype=np.uint8)
        status_bar.fill(20)
        frame[h - 50:h, :] = status_bar
        
        if detected:
            status_text = "  Pose Detected | 33 Landmarks"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 255, 0), -1)
        else:
            status_text = "  No Pose Detected"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, "ESC to Exit", (w - 150, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    def run_test(self):
        print("Pose Detection Test")
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
                if result.pose_landmarks:
                    detected = True
                    self.draw_pose_landmarks(frame, result.pose_landmarks)

                self.draw_ui(frame, detected)

                cv2.imshow('Pose Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = PoseDetectionTest()
    test.run_test()