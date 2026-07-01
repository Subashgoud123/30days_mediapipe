import cv2
import mediapipe as mp
import math

# -----------------------------
# MediaPipe Initialization
# -----------------------------
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mpDraw = mp.solutions.drawing_utils

# -----------------------------
# Calculator Buttons
# -----------------------------
buttons = [
    ['7','8','9','/'],
    ['4','5','6','*'],
    ['1','2','3','-'],
    ['0','.','=','+'],
    ['C','⌫','(',')']
]

BUTTON_W = 80
BUTTON_H = 70
START_X = 50
START_Y = 120

expression = ""

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

prev_click = False

# -----------------------------
# Draw Calculator
# -----------------------------
def draw_calculator(img):

    cv2.rectangle(img,(40,20),(400,90),(40,40,40),-1)

    cv2.putText(
        img,
        expression[-22:],
        (55,70),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255,255,255),
        2
    )

    button_positions=[]

    for r,row in enumerate(buttons):

        for c,text in enumerate(row):

            x = START_X + c*(BUTTON_W+10)
            y = START_Y + r*(BUTTON_H+10)

            cv2.rectangle(
                img,
                (x,y),
                (x+BUTTON_W,y+BUTTON_H),
                (200,200,200),
                -1
            )

            cv2.rectangle(
                img,
                (x,y),
                (x+BUTTON_W,y+BUTTON_H),
                (0,0,0),
                2
            )

            cv2.putText(
                img,
                text,
                (x+25,y+45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,0),
                2
            )

            button_positions.append((text,x,y))

    return button_positions

# -----------------------------
# Detect Finger
# -----------------------------
def get_landmarks(hand,img):

    h,w,_ = img.shape
    points=[]

    for lm in hand.landmark:

        points.append((
            int(lm.x*w),
            int(lm.y*h)
        ))

    return points

# -----------------------------
# Distance
# -----------------------------
def distance(p1,p2):

    return math.hypot(
        p1[0]-p2[0],
        p1[1]-p2[1]
    )

# -----------------------------
# Main Loop
# -----------------------------
while True:

    success,img = cap.read()

    if not success:
        break

    img = cv2.flip(img,1)

    rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    button_pos = draw_calculator(img)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mpDraw.draw_landmarks(
            img,
            hand,
            mpHands.HAND_CONNECTIONS
        )

        pts = get_landmarks(hand,img)

        index_tip = pts[8]
        thumb_tip = pts[4]

        cv2.circle(img,index_tip,10,(0,255,0),-1)

        d = distance(index_tip,thumb_tip)

        click = d < 35

        for text,x,y in button_pos:

            if x < index_tip[0] < x+BUTTON_W and y < index_tip[1] < y+BUTTON_H:

                cv2.rectangle(
                    img,
                    (x,y),
                    (x+BUTTON_W,y+BUTTON_H),
                    (0,255,0),
                    -1
                )

                cv2.putText(
                    img,
                    text,
                    (x+25,y+45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,0),
                    2
                )

                if click and not prev_click:

                    if text == "=":

                        try:
                            expression = str(eval(expression))
                        except:
                            expression = "Error"

                    elif text == "C":
                        expression = ""

                    elif text == "⌫":
                        expression = expression[:-1]

                    else:
                        expression += text

        prev_click = click

    cv2.imshow("Virtual Calculator",img)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
