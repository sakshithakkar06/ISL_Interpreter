import csv
import mediapipe as mp
import cv2

from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("data/dataset")
OUTPUT_FILE = Path("data/processed/landmarks_full.csv")
MAX_IMAGES_PER_CLASS = None

SPLITS = ["Training", "Validation", "Testing"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# 63 features per hand, 126 features for two hands
feature_names = [
    f"{hand}_{axis}{i}"
    for hand in ("h1", "h2")
    for i in range(21)
    for axis in ("x", "y", "z")
]

headers = ["split", "label", "image"] + feature_names
stats = defaultdict(lambda: [0, 0])

mp_hands = mp.solutions.hands


def extract_hand_features(hand_landmarks, width, height):
    points = [
        (lm.x * width, lm.y * height, lm.z * width)
        for lm in hand_landmarks.landmark
    ]

    # Normalize relative to wrist
    ox, oy, oz = points[0]

    scale = max(
        max(abs(x - ox), abs(y - oy))
        for x, y, z in points
    )

    if scale == 0:
        return None

    features = []

    for x, y, z in points:
        features.extend([
            (x - ox) / scale,
            (y - oy) / scale,
            (z - oz) / scale
        ])

    return features


with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands, open(OUTPUT_FILE, "w", newline="") as file:

    writer = csv.DictWriter(file, fieldnames=headers)
    writer.writeheader()

    for split in SPLITS:
        split_dir = DATASET_DIR / split

        if not split_dir.is_dir():
            print(f"Missing split: {split_dir}")
            continue

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            label = class_dir.name

            images = sorted(
                p for p in class_dir.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS
            )

            if MAX_IMAGES_PER_CLASS is not None:
                images = images[:MAX_IMAGES_PER_CLASS]

            key = (split, label)

            for image_path in images:
                stats[key][0] += 1

                image = cv2.imread(str(image_path))
                if image is None:
                    continue

                height, width = image.shape[:2]
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                detected = result.multi_hand_landmarks

                # Skip images where no hand was detected
                if not detected:
                    continue

                # Sort hands by wrist position, left to right in the image
                detected = sorted(
                    detected,
                    key=lambda hand: hand.landmark[0].x
                )

                all_features = []

                for hand_landmarks in detected[:2]:
                    hand_features = extract_hand_features(
                        hand_landmarks, width, height
                    )

                    if hand_features is not None:
                        all_features.append(hand_features)

                if not all_features:
                    continue

                # Missing hand gets 63 zero features
                while len(all_features) < 2:
                    all_features.append([0.0] * 63)

                features = all_features[0] + all_features[1]

                row = {
                    "split": split,
                    "label": label,
                    "image": str(image_path)
                }

                row.update(dict(zip(feature_names, features)))
                writer.writerow(row)
                stats[key][1] += 1

    file.flush()

print("\nPreprocessing summary")
print("-" * 45)

for (split, label), (total, saved) in sorted(stats.items()):
    print(f"{split:10} {label:3} detected {saved}/{total}")

total_seen = sum(v[0] for v in stats.values())
total_saved = sum(v[1] for v in stats.values())

print(f"\nImages saved: {total_saved}/{total_seen}")
print(f"Features per image: {len(feature_names)}")
print(f"Saved to: {OUTPUT_FILE}")
