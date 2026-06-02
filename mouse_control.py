import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Webcam
cap = cv2.VideoCapture(0)

# Screen size
screen_w, screen_h = pyautogui.size()

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

draw = mp.solutions.drawing_utils

last_click_time = 0
dragging = False

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    h, w, _ = frame.shape

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            landmarks = hand_landmarks.landmark

            # Fingertips
            index_tip = landmarks[8]
            thumb_tip = landmarks[4]

            # Index finger joints
            index_tip_y = landmarks[8].y
            index_pip_y = landmarks[6].y

            # Convert to screen coordinates
            x = int(index_tip.x * w)
            y = int(index_tip.y * h)

            screen_x = np.interp(x, [0, w], [0, screen_w])
            screen_y = np.interp(y, [0, h], [0, screen_h])

            # Move mouse
            pyautogui.moveTo(screen_x, screen_y)

            # Distance between thumb and index
            distance = math.hypot(
                thumb_tip.x - index_tip.x,
                thumb_tip.y - index_tip.y
            )

            # LEFT CLICK
            if distance < 0.04:
                current_time = time.time()

                # DOUBLE CLICK
                if current_time - last_click_time < 0.4:
                    pyautogui.doubleClick()
                    cv2.putText(frame, "DOUBLE CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                    time.sleep(0.3)

                else:
                    pyautogui.click()
                    cv2.putText(frame, "CLICK", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)

                last_click_time = current_time
                time.sleep(0.2)

            # DRAG / SELECT
            # If index finger bent
            if index_tip_y > index_pip_y:

                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True

                cv2.putText(frame, "DRAGGING", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

    cv2.imshow("Advanced Virtual Mouse", frame)

    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()