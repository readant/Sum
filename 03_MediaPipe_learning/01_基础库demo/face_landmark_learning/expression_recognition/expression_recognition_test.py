import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

# 禁用不必要的日志
logging.basicConfig(level=logging.ERROR)

class ExpressionRecognitionTest:
    def __init__(self):
        self.model_path = r'/03_MediaPipe_learning/01_基础库demo/face_landmark_learning/models/face_landmarker.task'
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

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

    def calculate_mouth_open_ratio(self, landmarks):
        mouth_upper = landmarks[13]
        mouth_lower = landmarks[14]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]

        vertical_dist = self.calculate_distance(mouth_upper, mouth_lower)
        horizontal_dist = self.calculate_distance(mouth_left, mouth_right)

        return vertical_dist / (horizontal_dist + 1e-6)

    def calculate_eye_open_ratio(self, landmarks):
        left_eye_upper = landmarks[159]
        left_eye_lower = landmarks[145]
        right_eye_upper = landmarks[386]
        right_eye_lower = landmarks[374]

        left_eye_dist = self.calculate_distance(left_eye_upper, left_eye_lower)
        right_eye_dist = self.calculate_distance(right_eye_upper, right_eye_lower)

        return (left_eye_dist + right_eye_dist) / 2

    def calculate_smile_ratio(self, landmarks):
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_upper = landmarks[13]
        mouth_lower = landmarks[14]

        mouth_width = self.calculate_distance(mouth_left, mouth_right)
        mouth_height = self.calculate_distance(mouth_upper, mouth_lower)

        return mouth_width / (mouth_height + 1e-6)

    def recognize_expression(self, landmarks):
        mouth_open_ratio = self.calculate_mouth_open_ratio(landmarks)
        eye_open_ratio = self.calculate_eye_open_ratio(landmarks)
        smile_ratio = self.calculate_smile_ratio(landmarks)

        if mouth_open_ratio > 0.3:
            return "Surprised", 0.9
        elif smile_ratio > 3.5:
            return "Smiling", 0.85
        elif eye_open_ratio < 0.02:
            return "Sleepy", 0.8
        else:
            return "Neutral", 0.7

    def draw_face_landmarks(self, frame, face_landmarks):
        for landmarks in face_landmarks:
            h, w, _ = frame.shape
            for point in landmarks:
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 1, (0, 255, 0), -1)

    def run_test(self):
        print("Expression Recognition Test")
        print("Press ESC to exit")
        print("Try different expressions: smile, surprise, neutral")

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
                    "Expression Recognition Test",
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

                # 绘制面部关键点和表情识别
                if result.face_landmarks:
                    self.draw_face_landmarks(frame, result.face_landmarks)

                    expression, confidence = self.recognize_expression(result.face_landmarks[0])

                    cv2.putText(
                        frame,
                        f"Expression: {expression}",
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