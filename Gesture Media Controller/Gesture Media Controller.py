import cv2
import mediapipe as mp
import pyautogui
import math
import time

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

last_action = 0
cooldown = 1.0


def count_fingers(lm):

    fingers = []

    # Thumb
    fingers.append(
        1 if lm[4].x < lm[3].x else 0
    )

    # Index
    fingers.append(
        1 if lm[8].y < lm[6].y else 0
    )

    # Middle
    fingers.append(
        1 if lm[12].y < lm[10].y else 0
    )

    # Ring
    fingers.append(
        1 if lm[16].y < lm[14].y else 0
    )

    # Pinky
    fingers.append(
        1 if lm[20].y < lm[18].y else 0
    )

    return fingers


while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    action_text = "Waiting..."

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            fingers = count_fingers(lm)

            current_time = time.time()

            thumb_x = int(lm[4].x * w)
            thumb_y = int(lm[4].y * h)

            index_x = int(lm[8].x * w)
            index_y = int(lm[8].y * h)

            distance = math.hypot(
                thumb_x - index_x,
                thumb_y - index_y
            )

            if current_time - last_action > cooldown:

                # Play / Pause
                if fingers == [1, 0, 0, 0, 0]:

                    pyautogui.press("playpause")
                    action_text = "Play / Pause"

                    last_action = current_time

                # Previous Track
                elif fingers == [0, 1, 0, 0, 0]:

                    pyautogui.press("prevtrack")
                    action_text = "Previous Track"

                    last_action = current_time

                # Next Track
                elif fingers == [0, 1, 1, 0, 0]:

                    pyautogui.press("nexttrack")
                    action_text = "Next Track"

                    last_action = current_time

                # Volume Up
                elif fingers == [1, 1, 1, 1, 1]:

                    pyautogui.press("volumeup")
                    action_text = "Volume Up"

                    last_action = current_time

                # Volume Down
                elif distance < 40:

                    pyautogui.press("volumedown")
                    action_text = "Volume Down"

                    last_action = current_time

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        action_text,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Gesture Media Controller",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()