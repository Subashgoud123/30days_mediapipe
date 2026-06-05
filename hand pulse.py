import math

import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

# MediaPipe fingertip landmark indices
FINGERTIP_IDS = [4, 8, 12, 16, 20]

# frame counter for animations (don't shadow the image variable `frame`)
frame_idx = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        # # draw all detected hands
        # for hand_landmarks in results.multi_hand_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # if at least two hands detected, connect corresponding fingertips
        if len(results.multi_hand_landmarks) >= 2:
            hand1 = results.multi_hand_landmarks[0]
            hand2 = results.multi_hand_landmarks[1]
            for idx in FINGERTIP_IDS:
                l1 = hand1.landmark[idx]
                l2 = hand2.landmark[idx]
                x1, y1 = int(l1.x * w), int(l1.y * h)
                x2, y2 = int(l2.x * w), int(l2.y * h)
                # draw small circles on the fingertip positions
                cv2.circle(frame, (x1, y1), 5, (0, 0, 255), -1)
                cv2.circle(frame, (x2, y2), 5, (0, 0, 255), -1)
                # connect the corresponding fingertips across the two hands
                # cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # p1,p2 are endpoints (x,y). use `frame_idx` for animation timing.
                for i in range(3):
                    tt = i/3.0
                    t = (frame_idx * 0.05+tt) % 1.0
                    x = int((1-t)*l1.x * w + t*l2.x * w)
                    y = int((1-t)*l1.y * h + t*l2.y * h)

                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # advance animation frame counter
    frame_idx += 1

cap.release()
cv2.destroyAllWindows()
