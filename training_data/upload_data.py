import os
import cv2
import numpy as np
import mediapipe as mp
import math

# Mediapipe setup
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Get current directory where the script is running
dataset_dir = os.getcwd()  # This will use the current directory
print(f"Saving dataset to: {dataset_dir}")



def angle_between_three_points(pointA, pointB, pointC):
    # Calculate the angle between three points
    x1x2s = math.pow((pointA[0] - pointB[0]), 2)
    x1x3s = math.pow((pointA[0] - pointC[0]), 2)
    x2x3s = math.pow((pointB[0] - pointC[0]), 2)

    y1y2s = math.pow((pointA[1] - pointB[1]), 2)
    y1y3s = math.pow((pointA[1] - pointC[1]), 2)
    y2y3s = math.pow((pointB[1] - pointC[1]), 2)

    cosine_angle = np.arccos((x1x2s + y1y2s + x2x3s + y2y3s - x1x3s - y1y3s) /
                             (2 * math.sqrt(x1x2s + y1y2s) * math.sqrt(x2x3s + y2y3s)))

    return np.degrees(cosine_angle)


def calculate_finger_angles(landmarks):
    landmarks = np.array(landmarks)
    angles = []

    # Calculate primary angles for each finger
    finger_indices = [
        (2, 3, 4),
        (5, 6, 7),
        (9, 10, 11),
        (13, 14, 15),
        (17, 18, 19),
        (0, 2, 4),  # Thumb: wrist, MCP, tip
        (0, 5, 8),  # Index finger: wrist, MCP, tip
        (0, 9, 12),  # Middle finger: wrist, MCP, tip
        (0, 13, 16),  # Ring finger: wrist, MCP, tip
        (0, 17, 20),  # Pinky: wrist, MCP, tip
        (5,7,8)
    ]

    for wrist, p1, p3 in finger_indices:
        angles.append(angle_between_three_points(landmarks[wrist], landmarks[p1], landmarks[p3]))

    return np.array(angles)


def preprocess_landmarks(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]

    # Separate x and y distances from wrist
    x_distances = landmarks[:, 0] - wrist[0]
    y_distances = landmarks[:, 1] - wrist[1]

    # Concatenate x and y distances and angles
    finger_angles = calculate_finger_angles(landmarks)  # Calculate finger angles
    print(finger_angles)

    feature_vector = np.concatenate([x_distances, y_distances, finger_angles])  # Combine into single feature vector

    return feature_vector

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()

        image = cv2.flip(image, 1)  # Flip the image horizontally for a selfie view
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for id, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=5, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2)
                )

                # Extract (x, y) coordinates
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Wait for key press to save the data
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit
                    break
                elif 65 <= key <= 90:  # ASCII values for A-Z
                    label = chr(key)
                    # Create folder for the label if it doesn't exist
                    label_folder = os.path.join(dataset_dir, label)

                    # Debugging: Check if folder exists
                    if not os.path.exists(label_folder):
                        print(f"Creating folder: {label_folder}")
                        os.makedirs(label_folder)
                    else:
                        print(f"Folder {label_folder} already exists")

                    try:
                        # Preprocess landmarks and save data
                        feature_vector = preprocess_landmarks(landmarks)
                        file_path = os.path.join(label_folder, f"{len(os.listdir(label_folder))}.npy")
                        np.save(file_path, feature_vector)
                        print(f"Saved {label} gesture data to {file_path}")
                    except Exception as e:
                        print(f"Error while saving data for {label}: {e}")
                        continue  # Skip current frame if error occurs

        else:
            # Display message when no hands are detected
            cv2.putText(image, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image with landmarks
        cv2.imshow('MediaPipe Hands - Data Collection', image)

    cap.release()
    cv2.destroyAllWindows()
