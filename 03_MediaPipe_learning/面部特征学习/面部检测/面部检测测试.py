import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging
import time
import numpy as np

logging.basicConfig(level=logging.ERROR)

class FaceDetectionTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'face_landmarker.task')
        self.detector = None
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        self._initialize_detector()

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def draw_face_mesh(self, frame, face_landmarks):
        # 丰富的面部特征点集合
        all_landmarks = list(range(478))  # 使用所有478个面部特征点
        
        for landmarks in face_landmarks:
            h, w, _ = frame.shape
            
            # 只绘制点，不绘制连接线
            for i, point in enumerate(landmarks):
                if i < len(all_landmarks):
                    pos = (int(point.x * w), int(point.y * h))
                    cv2.circle(frame, pos, 2, (0, 255, 0), -1)  # 使用单一颜色

    def draw_ui(self, frame, detected):
        h, w, _ = frame.shape
        
        header_height = 70
        header = np.zeros((header_height, w, 3), dtype=np.uint8)
        header.fill(25)
        cv2.addWeighted(frame[:header_height], 0.85, header, 0.15, 0, frame[:header_height])
        
        title = "Face Detection"
        cv2.putText(frame, title, (20, 45), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255), 2, cv2.LINE_AA)
        
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 45), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        
        status_bar = np.zeros((50, w, 3), dtype=np.uint8)
        status_bar.fill(20)
        frame[h - 50:h, :] = status_bar
        
        if detected:
            status_text = "  Face Detected | Landmarks: 478"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 255, 0), -1)
        else:
            status_text = "  No Face Detected"
            cv2.putText(frame, status_text, (20, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 60, h - 25), 8, (0, 0, 255), -1)
        
        cv2.putText(frame, "ESC to Exit", (w - 150, h - 18), cv2.FONT_HERSHEY_DUPLEX, 0.7, (150, 150, 150), 1, cv2.LINE_AA)

    def run_test(self):
        print("Face Detection Test")
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
                if result.face_landmarks:
                    detected = True
                    self.draw_face_mesh(frame, result.face_landmarks)

                self.draw_ui(frame, detected)

                cv2.imshow('Face Detection', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = FaceDetectionTest()
    test.run_test()