import cv2
import mediapipe as mp
import numpy as np
import ctypes
import time
import math

# ── Windows low-level mouse API (zero overhead, no pyautogui delay) ──────────
user32   = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

M_MOVE  = 0x0001; M_ABS   = 0x8000
M_LD    = 0x0002; M_LU    = 0x0004
M_RD    = 0x0008; M_RU    = 0x0010
M_WHEEL = 0x0800

def _me(flags, x=0, y=0, d=0):
    user32.mouse_event(flags, x, y, d, 0)

def move(sx, sy):
    _me(M_MOVE | M_ABS, int(sx * 65535 / SCREEN_W), int(sy * 65535 / SCREEN_H))

def lclick():   _me(M_LD); _me(M_LU)
def rclick():   _me(M_RD); _me(M_RU)
def ldown():    _me(M_LD)
def lup():      _me(M_LU)
def scroll(d):  _me(M_WHEEL, d=d)

# ── Config ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H  = 640, 480
ZONE          = 0.75   # only this fraction of the frame maps to full screen
SMOOTH        = 0.72   # EMA weight on prev position — higher = smoother glide
DEADZONE      = 4.0    # pixels — ignore micro-jitter below this delta
PINCH_L       = 0.065  # index-thumb distance for left click
CLICK_CD      = 0.20   # minimum seconds between any two registered clicks
DBL_WIN       = 0.40   # seconds window for double-click detection

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_h   = mp.solutions.hands
hands  = mp_h.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,               # lite model — faster, still precise enough
    min_detection_confidence=0.72,
    min_tracking_confidence=0.72,
)
draw = mp.solutions.drawing_utils
LS   = draw.DrawingSpec((0, 210, 0), 2, 2)   # landmark style
CS   = draw.DrawingSpec((180, 0, 200), 2)    # connection style

# ── Helpers ───────────────────────────────────────────────────────────────────
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def fingers_up(lm):
    """[index, middle, ring, pinky] up flags based on tip vs PIP joint."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]

# ── State ─────────────────────────────────────────────────────────────────────
sx, sy          = float(SCREEN_W) / 2, float(SCREEN_H) / 2
last_l = last_r = 0.0
dragging        = False
scroll_prev_y   = None

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # 1-frame buffer — no stale frames

prev_t = time.time()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False          # avoids an internal copy in MediaPipe
    res = hands.process(rgb)

    label       = ""
    label_color = (0, 255, 255)

    if res.multi_hand_landmarks:
        lm  = res.multi_hand_landmarks[0].landmark
        now = time.time()

        # ── Cursor: map index tip through the central control zone ────
        margin = (1.0 - ZONE) / 2.0
        tx = np.clip((lm[8].x - margin) / ZONE, 0.0, 1.0) * SCREEN_W
        ty = np.clip((lm[8].y - margin) / ZONE, 0.0, 1.0) * SCREEN_H

        nx = sx * SMOOTH + tx * (1.0 - SMOOTH)
        ny = sy * SMOOTH + ty * (1.0 - SMOOTH)

        if abs(nx - sx) > DEADZONE or abs(ny - sy) > DEADZONE:
            sx, sy = nx, ny
            move(sx, sy)

        dl = dist(lm[8], lm[4])    # index fingertip ↔ thumb tip
        fu = fingers_up(lm)        # [index, middle, ring, pinky]

        # ── Scroll: index + middle up, ring + pinky down ──────────────
        if fu[0] and fu[1] and not fu[2] and not fu[3] and dl > PINCH_L * 1.5:
            cy = lm[8].y
            if scroll_prev_y is not None:
                delta = scroll_prev_y - cy
                if abs(delta) > 0.005:
                    scroll(int(delta * 3000))
            scroll_prev_y = cy
            label = "SCROLL"
            label_color = (255, 165, 0)
            if dragging:
                lup()
                dragging = False

        # ── Right click: PINKY up, index+middle+ring all down ─────────
        elif fu[3] and not fu[0] and not fu[1] and not fu[2] \
                and (now - last_r) > CLICK_CD:
            rclick()
            last_r = now
            label = "RIGHT CLICK"
            label_color = (0, 80, 255)
            scroll_prev_y = None
            if dragging:
                lup()
                dragging = False

        # ── Left click / double-click: index-thumb pinch ─────────────
        elif dl < PINCH_L and (now - last_l) > CLICK_CD:
            scroll_prev_y = None
            if dragging:
                lup()
                dragging = False
                label = "DROP"
                label_color = (0, 220, 120)
            elif (now - last_l) < DBL_WIN:
                lclick(); lclick()
                label = "DOUBLE CLICK"
                label_color = (0, 255, 0)
            else:
                lclick()
                label = "LEFT CLICK"
                label_color = (0, 255, 0)
            last_l = now

        # ── Drag: closed fist (all 4 fingers down) ────────────────────
        elif not any(fu):
            if not dragging:
                ldown()
                dragging = True
            label = "DRAG"
            label_color = (0, 0, 255)
            scroll_prev_y = None

        # ── Idle: release any held state ──────────────────────────────
        else:
            if dragging:
                lup()
                dragging = False
            scroll_prev_y = None

        # Draw skeleton
        draw.draw_landmarks(
            frame, res.multi_hand_landmarks[0], mp_h.HAND_CONNECTIONS, LS, CS
        )
        # Highlight index fingertip
        ix, iy = int(lm[8].x * CAM_W), int(lm[8].y * CAM_H)
        cv2.circle(frame, (ix, iy), 11, (0, 255, 0), -1)
        cv2.circle(frame, (ix, iy), 11, (0, 160, 0),  2)

        # Live pinch distance bar (bottom-left) — fill goes green when close enough
        bar_max = 0.15
        bar_w   = 160
        fill    = int(np.clip(1.0 - dl / bar_max, 0, 1) * bar_w)
        bar_col = (0, 255, 0) if dl < PINCH_L else (100, 100, 255)
        cv2.rectangle(frame, (10, CAM_H - 35), (10 + bar_w, CAM_H - 20), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, CAM_H - 35), (10 + fill,  CAM_H - 20), bar_col, -1)
        cv2.putText(frame, f"pinch {dl:.3f}", (10, CAM_H - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

    else:
        if dragging:
            lup()
            dragging = False
        scroll_prev_y = None

    # ── HUD ───────────────────────────────────────────────────────────────────
    # Control zone box
    mw = int(CAM_W * (1 - ZONE) / 2)
    mh = int(CAM_H * (1 - ZONE) / 2)
    cv2.rectangle(frame, (mw, mh), (CAM_W - mw, CAM_H - mh), (70, 70, 70), 1)

    # Gesture label (outlined for readability)
    if label:
        cv2.putText(frame, label, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
        cv2.putText(frame, label, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, label_color, 2)

    # FPS counter
    t = time.time()
    fps = 1.0 / max(t - prev_t, 1e-9)
    prev_t = t
    cv2.putText(frame, f"FPS {fps:.0f}", (CAM_W - 95, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Drag indicator
    if dragging:
        cv2.circle(frame, (CAM_W - 20, CAM_H - 20), 8, (0, 0, 255), -1)

    cv2.putText(frame, "ESC = quit", (10, CAM_H - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (130, 130, 130), 1)

    cv2.imshow("Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
