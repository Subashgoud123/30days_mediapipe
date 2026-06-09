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
cooldown = 0.3

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

    volume_text = "Neutral"

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            thumb_tip = lm[4]
            index_tip = lm[8]

            x1 = int(thumb_tip.x * w)
            y1 = int(thumb_tip.y * h)

            x2 = int(index_tip.x * w)
            y2 = int(index_tip.y * h)

            cv2.circle(
                frame,
                (x1, y1),
                10,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.circle(
                frame,
                (x2, y2),
                10,
                (255, 0, 255),
                cv2.FILLED
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 0, 255),
                3
            )

            length = math.hypot(
                x2 - x1,
                y2 - y1
            )

            cv2.putText(
                frame,
                f"Distance: {int(length)}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            current_time = time.time()

            if current_time - last_action > cooldown:

                if length > 180:

                    pyautogui.press("volumeup")
                    volume_text = "Volume Up"

                    last_action = current_time

                elif length < 50:

                    pyautogui.press("volumedown")
                    volume_text = "Volume Down"

                    last_action = current_time

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.putText(
        frame,
        volume_text,
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Gesture Volume Controller",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
