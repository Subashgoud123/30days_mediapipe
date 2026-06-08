import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

canvas = np.zeros((480, 640, 3), dtype=np.uint8)

xp, yp = 0, 0

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            h, w, _ = frame.shape

            index_tip = lm[8]
            middle_tip = lm[12]

            index_x = int(index_tip.x * w)
            index_y = int(index_tip.y * h)

            middle_x = int(middle_tip.x * w)
            middle_y = int(middle_tip.y * h)

            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y

            # Selection Mode
            if index_up and middle_up:

                xp, yp = 0, 0

                cv2.rectangle(
                    frame,
                    (index_x, index_y - 25),
                    (middle_x, middle_y + 25),
                    (255, 0, 255),
                    cv2.FILLED
                )

                if index_y < 80:
                    canvas = np.zeros(
                        (480, 640, 3),
                        dtype=np.uint8
                    )

            # Drawing Mode
            elif index_up and not middle_up:

                cv2.circle(
                    frame,
                    (index_x, index_y),
                    10,
                    (255, 0, 255),
                    cv2.FILLED
                )

                if xp == 0 and yp == 0:
                    xp, yp = index_x, index_y

                cv2.line(
                    canvas,
                    (xp, yp),
                    (index_x, index_y),
                    (255, 0, 255),
                    8
                )

                xp, yp = index_x, index_y

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

    _, inv = cv2.threshold(
        gray,
        50,
        255,
        cv2.THRESH_BINARY_INV
    )

    inv = cv2.cvtColor(
        inv,
        cv2.COLOR_GRAY2BGR
    )

    frame = cv2.bitwise_and(
        frame,
        inv
    )

    frame = cv2.bitwise_or(
        frame,
        canvas
    )

    cv2.imshow(
        "Air Drawing Canvas",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
