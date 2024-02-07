import cv2
import mediapipe as mp
import pyautogui
import webbrowser
import numpy as np
import time

# Initialize MediaPipe Hands module.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
drawing_utils = mp.solutions.drawing_utils

# Initialize the webcam.
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Variables to control the actions.
click_distance_threshold = 0.03
double_click_time = 0.4
last_click_time = 0
google_opened = False

def calculate_distance(landmark1, landmark2):
    return np.sqrt((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)

while True:
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get the hand landmark prediction.
    result = hands.process(rgb_frame)

    # If hand landmarks are detected.
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Index fingertip and thumb tip landmarks.
        index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Move the cursor based on the index finger position.
        index_finger_x = int(index_finger_tip.x * screen_width)
        index_finger_y = int(index_finger_tip.y * screen_height)
        pyautogui.moveTo(index_finger_x, index_finger_y)

        # Calculate the distance between index finger tip and thumb tip.
        distance = calculate_distance(index_finger_tip, thumb_tip)

        # Click detection.
        current_time = time.time()
        if distance < click_distance_threshold:
            if current_time - last_click_time < double_click_time:
                pyautogui.leftClick()
            else:
                pyautogui.rightClick()
            last_click_time = current_time

        # Check if the back of the hand is shown.
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        vector_index = [index_finger_tip.x - wrist.x, index_finger_tip.y - wrist.y]
        vector_middle = [middle_finger_tip.x - wrist.x, middle_finger_tip.y - wrist.y]

        if vector_index[0] > 0 and vector_middle[0] > 0 and not google_opened:
            webbrowser.open('https://www.google.com')
            google_opened = True

        if vector_index[0] <= 0 and vector_middle[0] <= 0:
            google_opened = False

    cv2.imshow('Virtual Mouse', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
