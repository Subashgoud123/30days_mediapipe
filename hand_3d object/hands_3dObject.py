import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
)

window_name = "Hand Axis Box"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
# cv2.setWindowProperty(
#     window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def to_pixel(landmark, width, height):
    return np.array([landmark.x * width, landmark.y * height], dtype=np.float32)


def normalized(vector):
    norm = np.linalg.norm(vector)
    return vector / (norm + 1e-6)


def draw_3d_box(frame, origin, x_dir, y_dir, z_dir, size):
    o = origin
    x = x_dir * size
    y = y_dir * size
    z = z_dir * size
    corners = [
        o,
        o + x,
        o + y,
        o + z,
        o + x + y,
        o + x + z,
        o + y + z,
        o + x + y + z,
    ]
    points = [tuple(map(int, c)) for c in corners]
    edges = [
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    for a, b in edges:
        cv2.line(frame, points[a], points[b], (0, 255, 255), 2)
    return points


while True:
    success, frame = cap.read()
    if not success:
        break

    height, width, _ = frame.shape
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        thumb_tip = to_pixel(hand_landmarks.landmark[4], width, height)
        index_tip = to_pixel(hand_landmarks.landmark[8], width, height)
        wrist = to_pixel(hand_landmarks.landmark[0], width, height)
        middle_mcp = to_pixel(hand_landmarks.landmark[9], width, height)

        x_dir = normalized(index_tip - thumb_tip)
        y_candidate = normalized(middle_mcp - wrist)
        y_dir = y_candidate - np.dot(y_candidate, x_dir) * x_dir
        y_dir = normalized(y_dir)
        if np.linalg.norm(y_dir) < 1e-3:
            y_dir = np.array([-x_dir[1], x_dir[0]], dtype=np.float32)
        z_dir = normalized(x_dir * 0.6 + y_dir * 0.6)

        distance = np.linalg.norm(index_tip - thumb_tip)
        box_size = float(np.clip(distance * 1.5, 40.0, 220.0))
        origin = thumb_tip

        cv2.circle(frame, tuple(map(int, thumb_tip)), 8, (0, 0, 255), -1)
        cv2.circle(frame, tuple(map(int, index_tip)), 8, (0, 255, 0), -1)
        cv2.circle(frame, tuple(map(int, wrist)), 6, (255, 0, 0), -1)
        cv2.line(frame, tuple(map(int, thumb_tip)),
                 tuple(map(int, index_tip)), (0, 255, 0), 2)

        cv2.arrowedLine(frame, tuple(map(int, origin)), tuple(
            map(int, origin + x_dir * box_size)), (0, 0, 255), 3)
        cv2.arrowedLine(frame, tuple(map(int, origin)), tuple(
            map(int, origin + y_dir * box_size)), (0, 255, 0), 3)
        cv2.arrowedLine(frame, tuple(map(int, origin)), tuple(
            map(int, origin + z_dir * box_size)), (255, 0, 0), 3)
        cv2.putText(frame, 'X', tuple(map(int, origin + x_dir * box_size +
                    np.array([5, -5]))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, 'Y', tuple(map(int, origin + y_dir * box_size +
                    np.array([5, -5]))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, 'Z', tuple(map(int, origin + z_dir * box_size +
                    np.array([5, -5]))), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        draw_3d_box(frame, origin, x_dir, y_dir, z_dir, box_size)
        cv2.putText(frame, "Pinch thumb+index to zoom box", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Scale: {int(box_size)}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "Show one hand with thumb and index finger",
                    (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
