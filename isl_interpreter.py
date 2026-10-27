
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import time

from pathlib import Path
from collections import deque, Counter

#below imports are for testing and logging
# import csv
# from datetime import datetime
# LOG_FILE = Path("data/processed/live_test_log.csv")
# LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
# test_letter = None
# test_attempt = 0
# attempt_predictions = []
# attempt_confidences = []
# attempt_start = None

# Load trained model
MODEL_PATH = Path("models/isl_alphabet_rf.joblib")
bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
feature_columns = bundle["feature_columns"]

if len(feature_columns) != 126:
    raise ValueError(
        f"Expected 126 features, but model has {len(feature_columns)}"
    )

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

CONFIDENCE_THRESHOLD = 0.50
STABLE_TIME = 0.5       # Seconds required to hold a sign
RELEASE_TIME = 0.5      # Seconds without hands to re-arm capture

prediction_history = deque(maxlen=5)

# Text and automatic capture state
recognized_text = ""
stable_label = "Uncertain"
candidate_label = None
candidate_start = None
last_hands_seen = time.time()
capture_armed = True
status_message = "Show a sign"


def extract_hand_features(hand_landmarks, width, height):
    """Extract 63 normalized features from one hand."""

    coords = np.array([
        [lm.x * width, lm.y * height, lm.z * width]
        for lm in hand_landmarks.landmark
    ], dtype=np.float32)

    coords -= coords[0]
    scale = np.max(np.abs(coords[:, :2]))

    if scale <= 0:
        return np.zeros(63, dtype=np.float32)

    coords /= scale
    return coords.flatten()


# Webcam setup
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cv2.namedWindow("ISL Alphabet Interpreter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ISL Alphabet Interpreter", 1280, 720)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("Could not open webcam.")
    raise SystemExit

print("ISL Interpreter started.")
print("Hold a sign for 0.5 second to capture it.")
print("Remove hands briefly before showing the next sign.")
print("B = Backspace | SPACE = Space | X = Clear | Q = Quit")
# print("TEST MODE: Press R, S, T, or Z to select the sign being tested.")
# print("Press ENTER to finish and save that attempt.")
# print("Press ESC to cancel the current test attempt.")

try:
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read from webcam.")
                break

            # Mirrored webcam view
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            now = time.time()
            confidence = 0.0
            detected_hands = results.multi_hand_landmarks or []

            if detected_hands:
                last_hands_seen = now

                hand_data = []

                for hand_landmarks in detected_hands:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    wrist_x = hand_landmarks.landmark[0].x
                    hand_data.append((wrist_x, hand_landmarks))

                # Sort hands left-to-right
                hand_data.sort(key=lambda item: item[0])

                combined_features = []

                for _, hand_landmarks in hand_data[:2]:
                    features = extract_hand_features(
                        hand_landmarks, width, height
                    )
                    combined_features.extend(features)

                # Pad if only one hand is detected
                while len(combined_features) < 126:
                    combined_features.extend([0.0] * 63)

                combined_features = np.array(
                    combined_features, dtype=np.float32
                ).reshape(1, -1)

                input_data = pd.DataFrame(
                    combined_features,
                    columns=feature_columns
                )

                probabilities = model.predict_proba(input_data)[0]
                best_index = np.argmax(probabilities)

                predicted_label = str(model.classes_[best_index])
                confidence = float(probabilities[best_index])

                # Record predictions during a manually labelled test attempt
                # if test_letter is not None:
                #     attempt_predictions.append(predicted_label)
                #     attempt_confidences.append(confidence)

                if confidence >= CONFIDENCE_THRESHOLD:
                    prediction_history.append(predicted_label)

                    stable_label = Counter(
                        prediction_history
                    ).most_common(1)[0][0]

                    # Require a consistent prediction history
                    matching_count = sum(
                        label == predicted_label
                        for label in prediction_history
                    )

                    
                    if (
                        len(prediction_history) == prediction_history.maxlen
                        and matching_count >= 4
                        and stable_label == predicted_label
                    ):
                        if not capture_armed:
                            status_message = "Remove hands for next letter"

                        else:
                            if candidate_label != stable_label or candidate_start is None:
                                candidate_label = stable_label
                                candidate_start = now

                            elapsed = now - candidate_start

                            if elapsed >= STABLE_TIME:
                                recognized_text += stable_label
                                print("Captured:", stable_label)
                                print("Current text:", recognized_text)

                                status_message = f"Captured {stable_label}"
                                capture_armed = False
                                candidate_start = None

                            else:
                                status_message = (
                                    f"Hold still: {max(0, STABLE_TIME - elapsed):.1f}s"
                                )
                    else:
                        candidate_label = None
                        candidate_start = None
                        status_message = "Stabilizing prediction"

                else:
                    prediction_history.clear()
                    candidate_label = None
                    candidate_start = None
                    stable_label = "Uncertain"
                    status_message = "Uncertain prediction"

                cv2.putText(
                    frame,
                    f"ISL: {stable_label} ({confidence:.0%})",
                    (20, 48),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0) if stable_label != "Uncertain"
                    else (0, 200, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Hands detected: {len(detected_hands)}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1
                )

            else:
                prediction_history.clear()
                candidate_label = None
                candidate_start = None
                stable_label = "Uncertain"
                status_message = "Show a sign"

                # Re-arm only after hands are absent for a short time
                if now - last_hands_seen >= RELEASE_TIME:
                    capture_armed = True

            # Text panel
            cv2.rectangle(frame, (10, 80), (630, 180), (30, 30, 30), -1)

            cv2.putText(
                frame,
                "Text: " + recognized_text[-35:],
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                status_message[:65],
                (20, 143),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "B: Backspace | SPACE: Space | X: Clear | Q: Quit",
                (20, 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (200, 200, 200),
                1
            )

            frame = cv2.resize(frame, (1280, 720))
            cv2.imshow("ISL Alphabet Interpreter", frame)
            key = cv2.waitKey(1) & 0xFF
            #turn below code on to enable testing and logging of model predictions for R, S, T, Z letters
            # if key in (ord("r"), ord("s"), ord("t"), ord("z")):
            #     test_letter = chr(key).upper()
            #     attempt_predictions = []
            #     attempt_confidences = []
            #     attempt_start = time.time()
            #     test_attempt += 1
            #     status_message = f"Testing {test_letter}: show sign, ENTER to save"
            #     print(f"\nAttempt {test_attempt}: Actual letter = {test_letter}")

            # elif key == 13 and test_letter is not None:  # ENTER
            #     if attempt_predictions:
            #         counts = Counter(attempt_predictions)
            #         top_prediction, top_count = counts.most_common(1)[0]
            #         avg_confidence = sum(attempt_confidences) / len(attempt_confidences)

            #         row = {
            #             "timestamp": datetime.now().isoformat(timespec="seconds"),
            #             "attempt": test_attempt,
            #             "actual_letter": test_letter,
            #             "frames": len(attempt_predictions),
            #             "most_common_prediction": top_prediction,
            #             "prediction_match_percent": round(
            #                 100 * top_count / len(attempt_predictions), 2
            #             ),
            #             "average_confidence": round(avg_confidence, 4),
            #             "predictions": " ".join(attempt_predictions),
            #         }

            #         file_exists = LOG_FILE.exists()
            #         with LOG_FILE.open("a", newline="") as f:
            #             writer = csv.DictWriter(f, fieldnames=row.keys())
            #             if not file_exists or LOG_FILE.stat().st_size == 0:
            #                 writer.writeheader()
            #             writer.writerow(row)

            #         print("Saved attempt:", row)
            #         status_message = f"Saved {test_letter} attempt"

            #     else:
            #         print("No hand predictions recorded for this attempt.")

            #     test_letter = None
            #     attempt_predictions = []
            #     attempt_confidences = []
            #     attempt_start = None

            # elif key == 27 and test_letter is not None:  # ESC
            #     print("Test attempt cancelled.")
            #     test_letter = None
            #     attempt_predictions = []
            #     attempt_confidences = []
            #     attempt_start = None
            #     status_message = "Test cancelled"

            if key == ord("b"):
                recognized_text = recognized_text[:-1]

            elif key == ord(" "):
                if recognized_text and not recognized_text.endswith(" "):
                    recognized_text += " "

            elif key == ord("x"):
                recognized_text = ""
                status_message = "Text cleared"

            elif key == ord("q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
