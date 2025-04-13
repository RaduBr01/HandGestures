from time import sleep

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import math
import pyttsx3
import time

# Initialize text-to-speech engine
engine = pyttsx3.init()

# String to build the message
message = ""

model_path = "training_data/new_model.keras"
model = tf.keras.models.load_model(model_path)

# Define the label map for "A", "B", "C", "D", and "E"
label_map = {
    0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H",
    8: "I", 9: "K", 10: "L", 11: "M", 12: "N", 13: "O", 14: "P", 15: "Q",
    16: "R", 17: "S", 18: "T", 19: "U", 20: "V", 21: "W", 22: "R", 23: "W",
    24:"X", 25:"I", 26:"nustiuboss"
}


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

confidence_threshold = 0.97

cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    landmarks = np.array(landmarks)
    wrist = landmarks[0]
    normalized_landmarks = landmarks - wrist  # Subtract wrist position
    reference_distance = np.linalg.norm(normalized_landmarks[9])  # Palm to middle finger distance
    normalized_landmarks /= reference_distance if reference_distance != 0 else 1  # Normalize by reference distance
    return normalized_landmarks.flatten()

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
    feature_vector = np.concatenate([x_distances, y_distances, finger_angles])  # Combine into single feature vector

    return feature_vector

# Function to check if index finger is in the top-right region
def is_index_finger_in_top_right(landmarks, width, height):
    # Access the index finger tip's x and y coordinates directly from the tuple
    index_finger_tip = landmarks[8]  # Index finger tip position
    index_x = int(index_finger_tip[0] * width)  # x is the first element in the tuple
    index_y = int(index_finger_tip[1] * height)  # y is the second element in the tuple

    # Check if the index finger is in the top-right corner
    if index_x > width * 0.9 and index_y < height * 0.2:
        return True
    return False


def is_index_finger_in_bottom_right(landmarks, width, height):
    # Access the index finger tip's x and y coordinates directly from the tuple
    index_finger_tip = landmarks[8]  # Index finger tip position
    index_x = int(index_finger_tip[0] * width)  # x is the first element in the tuple
    index_y = int(index_finger_tip[1] * height)  # y is the second element in the tuple

    # Check if the index finger is in the top-right corner
    if index_x > width * 0.9 and index_y  > height * 0.8:
        return True
    return False

# Function to handle speech output
def speak_message(message):
    engine.say(message)
    engine.runAndWait()


prediction_counter = 0  # Initialize the counter
last_prediction = None  # To track the previous prediction
message = ""  # To store the built message

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)  # Flip the image horizontally for a selfie view
        height, width, _ = image.shape
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)



        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 0), thickness=5, circle_radius=5),
                    mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2)
                )

                # Extract the 21 (x, y) coordinates of the hand landmarks
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Preprocess the landmarks for the model (47-element feature vector)
                feature_vector = preprocess_landmarks(landmarks)

                # Reshape and predict with the model
                feature_vector = np.expand_dims(feature_vector, axis=0)
                prediction = model.predict(feature_vector)

                # Get the predicted label and its confidence
                predicted_label_idx = np.argmax(prediction)
                predicted_confidence = np.max(prediction)
                predicted_label = label_map[predicted_label_idx]
                print([predicted_label_idx])


                # If the prediction is above confidence threshold, handle it
                if predicted_confidence >= confidence_threshold:
                    # If the same prediction continues, increment the counter
                    if predicted_label == last_prediction:
                        prediction_counter += 1
                    else:
                        prediction_counter = 1  # Reset counter for a new prediction
                        last_prediction = predicted_label  # Update last prediction
                    #print(prediction_counter)
                    # Build the message if the same label has been predicted 4 times
                    if prediction_counter >= 16:
                        message += predicted_label  # Add the predicted label to the message
                        prediction_counter = 0  # Reset the counter after building the message

                # When index finger is in the top-right region, trigger text-to-speech
                if is_index_finger_in_top_right(landmarks, width, height):
                    if message:  # Only speak if there is a message to read
                        speak_message(message)
                        message = ""  # Reset the message after speaking

                if is_index_finger_in_bottom_right(landmarks, width, height):
                    if message:  # Only speak if there is a message to read
                        message=message[0:len(message)-1]
                        sleep(0.5)

                # Display the string being built on the screen
                cv2.putText(image, f"Message: {message}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Display the predicted label and confidence
                if predicted_confidence >= confidence_threshold:
                    cv2.putText(image, f"Prediction: {predicted_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)
                else:
                    cv2.putText(image, "Prediction: Uncertain", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the image with landmarks and prediction
        cv2.imshow('Hand Gesture Recognition', image)

        # Break the loop if the 'ESC' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
            break

cap.release()
cv2.destroyAllWindows()
