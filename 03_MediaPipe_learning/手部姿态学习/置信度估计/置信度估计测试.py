import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import logging

logging.basicConfig(level=logging.ERROR)

class ConfidenceEstimationTest:
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

    def draw_hand_landmarks_with_confidence(self, frame, hand_landmarks, handedness):
        for idx, (landmarks, handedness_list) in enumerate(zip(hand_landmarks, handedness)):
            h, w, _ = frame.shape
            label = handedness_list[0].category_name
            confidence = handedness_list[0].score

            color = (0, 255, 0) if label == "Left" else (255, 0, 0)

            for point in landmarks:
                pos = (int(point.x * w), int(point.y * h))
                cv2.circle(frame, pos, 3, color, -1)

            text = f"{label}: {confidence:.2f}"
            cv2.putText(
                frame,
                text,
                (10, 100 + idx * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

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

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = self.detector.detect(mp_image)

                if result.hand_landmarks:
                    self.draw_hand_landmarks_with_confidence(frame, result.hand_landmarks, result.handedness)
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