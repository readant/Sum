import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

logging.basicConfig(level=logging.ERROR)

class FacialFeaturesTest:
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

    def calculate_symmetry(self, landmarks):
        left_eye = landmarks[133]
        right_eye = landmarks[362]
        left_mouth = landmarks[61]
        right_mouth = landmarks[291]
        nose_tip = landmarks[1]

        eye_symmetry = abs(left_eye.x - right_eye.x)
        mouth_symmetry = abs(left_mouth.x - right_mouth.x)
        nose_offset = abs(nose_tip.x - (left_eye.x + right_eye.x) / 2)

        symmetry_score = 1 - min(nose_offset / eye_symmetry, 1)
        return symmetry_score

    def calculate_facial_ratio(self, landmarks):
        forehead = landmarks[10]
        nose_tip = landmarks[1]
        chin = landmarks[152]
        left_eye = landmarks[133]
        right_eye = landmarks[362]

        face_height = self.calculate_distance(forehead, chin)
        face_width = self.calculate_distance(left_eye, right_eye)

        eye_to_nose = self.calculate_distance(nose_tip, left_eye)
        nose_to_chin = self.calculate_distance(nose_tip, chin)

        vertical_ratio = eye_to_nose / (nose_to_chin + 1e-6)
        horizontal_ratio = face_width / (face_height + 1e-6)

        return vertical_ratio, horizontal_ratio

    def calculate_feature_distances(self, landmarks):
        left_eye = landmarks[133]
        right_eye = landmarks[362]
        nose_tip = landmarks[1]
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        forehead = landmarks[10]
        chin = landmarks[152]

        eye_distance = self.calculate_distance(left_eye, right_eye)
        nose_to_eye = self.calculate_distance(nose_tip, left_eye)
        mouth_width = self.calculate_distance(mouth_left, mouth_right)
        face_height = self.calculate_distance(forehead, chin)

        return {
            'eye_distance': eye_distance,
            'nose_to_eye': nose_to_eye,
            'mouth_width': mouth_width,
            'face_height': face_height
        }

    def draw_face_landmarks(self, frame, face_landmarks):
        for landmarks in face_landmarks:
            h, w, _ = frame.shape
            for point in landmarks:
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 1, (0, 255, 0), -1)

    def draw_feature_points(self, frame, landmarks):
        key_points = [1, 10, 61, 133, 152, 291, 362]
        h, w, _ = frame.shape

        for idx in key_points:
            point = landmarks[idx]
            pos = (int(point.x * w), int(point.y * h))
            cv2.circle(frame, pos, 5, (255, 0, 0), -1)
            cv2.putText(frame, str(idx), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1, cv2.LINE_AA)

    def run_test(self):
        print("Facial Features Analysis Test")
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
                    "Facial Features Analysis Test",
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
                    self.draw_feature_points(frame, result.face_landmarks[0])

                    symmetry = self.calculate_symmetry(result.face_landmarks[0])
                    vert_ratio, hor_ratio = self.calculate_facial_ratio(result.face_landmarks[0])
                    distances = self.calculate_feature_distances(result.face_landmarks[0])

                    cv2.putText(
                        frame,
                        f"Symmetry: {symmetry:.2%}",
                        (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        frame,
                        f"V-Ratio: {vert_ratio:.2f} H-Ratio: {hor_ratio:.2f}",
                        (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        frame,
                        f"Eye Dist: {distances['eye_distance']:.3f}",
                        (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA
                    )

                    cv2.putText(
                        frame,
                        f"Mouth Width: {distances['mouth_width']:.3f}",
                        (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
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

                cv2.imshow('Facial Features Analysis Test', frame)

                if cv2.waitKey(1) & 0xFF == 27:
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Test completed")

if __name__ == "__main__":
    test = FacialFeaturesTest()
    test.run_test()