import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

tip_ids = [4, 8, 12, 16, 20]

while True:

    success, frame = cap.read()

    if not success:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            lm_list = hand_landmarks.landmark

            # Thumb
            if lm_list[4].x < lm_list[3].x:
                total_fingers += 1

            # Other fingers
            for tip in tip_ids[1:]:

                if lm_list[tip].y < lm_list[tip - 2].y:
                    total_fingers += 1

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.rectangle(frame, (20, 20), (180, 120), (0, 255, 0), -1)

    cv2.putText(
        frame,
        str(total_fingers),
        (70, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 255, 255),
        4
    )

    cv2.imshow("Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
