import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

logging.basicConfig(level=logging.ERROR)

class ExpressionRecognitionTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'face_landmarker.task')
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.7,
            min_face_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def recognize_expression(self, face_landmarks):
        landmarks = face_landmarks[0]

        left_eye = landmarks[33]
        right_eye = landmarks[263]
        nose = landmarks[1]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_top = landmarks[13]
        mouth_bottom = landmarks[14]

        eye_distance = np.sqrt((left_eye.x - right_eye.x)**2 + (left_eye.y - right_eye.y)**2)
        mouth_width = np.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
        mouth_height = np.sqrt((mouth_top.x - mouth_bottom.x)**2 + (mouth_top.y - mouth_bottom.y)**2)

        mouth_ratio = mouth_height / mouth_width if mouth_width > 0 else 0

        if mouth_ratio > 0.3:
            return "Happy"
        elif mouth_ratio < 0.15:
            return "Sad"
        else:
            return "Neutral"

    def draw_face_landmarks(self, frame, face_landmarks):
        for landmarks in face_landmarks:
            h, w, _ = frame.shape
            for point in landmarks:
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 1, (0, 255, 0), -1)

    def run_test(self):
        print("Expression Recognition Test")
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

                cv2.rectangle(frame, (0, 0), (w, 60), (50, 50, 50), -1)
                cv2.putText(
                    frame,
                    "Expression Recognition Test",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)

                if result.face_landmarks:
                    self.draw_face_landmarks(frame, result.face_landmarks)
                    expression = self.recognize_expression(result.face_landmarks)

                    cv2.putText(
                        frame,
                        f"Expression: {expression}",
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
                        "No face detected",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA
                    )

                cv2.imshow('Expression Recognition Test', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = ExpressionRecognitionTest()
    test.run_test()