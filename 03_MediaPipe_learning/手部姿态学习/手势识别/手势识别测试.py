import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

logging.basicConfig(level=logging.ERROR)

class GestureRecognitionTest:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'hand_landmarker.task')
        self.detector = None
        self._initialize_detector()

    def _initialize_detector(self):
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def recognize_gesture(self, hand_landmarks):
        landmarks = hand_landmarks[0]
        finger_tips = [4, 8, 12, 16, 20]
        finger_mcps = [2, 5, 9, 13, 17]

        fingers_up = []
        for tip_idx, mcp_idx in zip(finger_tips, finger_mcps):
            tip = landmarks[tip_idx]
            mcp = landmarks[mcp_idx]
            if tip.y < mcp.y:
                fingers_up.append(True)
            else:
                fingers_up.append(False)

        if fingers_up == [True, True, True, True, True]:
            return "Five"
        elif fingers_up == [False, True, True, True, True]:
            return "Four"
        elif fingers_up == [False, True, True, False, False]:
            return "Three"
        elif fingers_up == [False, True, False, False, False]:
            return "Two"
        elif fingers_up == [True, False, False, False, False]:
            return "One"
        elif fingers_up == [False, False, False, False, False]:
            return "Fist"
        else:
            return "Unknown"

    def draw_hand_landmarks(self, frame, hand_landmarks):
        for landmarks in hand_landmarks:
            h, w, _ = frame.shape
            for point in landmarks:
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 3, (0, 255, 0), -1)

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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)

                if result.hand_landmarks:
                    self.draw_hand_landmarks(frame, result.hand_landmarks)
                    gesture = self.recognize_gesture(result.hand_landmarks)
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