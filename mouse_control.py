import cv2
import mediapipe as mp
import numpy as np
import ctypes
import time

# ── Windows API ───────────────────────────────────────────────────────────────
user32 = ctypes.windll.user32

# Multi-monitor: virtual desktop spans all screens
VIRT_X = user32.GetSystemMetrics(76)   # SM_XVIRTUALSCREEN
VIRT_Y = user32.GetSystemMetrics(77)   # SM_YVIRTUALSCREEN
VIRT_W = user32.GetSystemMetrics(78)   # SM_CXVIRTUALSCREEN
VIRT_H = user32.GetSystemMetrics(79)   # SM_CYVIRTUALSCREEN

M_MOVE  = 0x0001; M_ABS  = 0x8000; M_VIRT = 0x4000
M_LD    = 0x0002; M_LU   = 0x0004
M_RD    = 0x0008; M_RU   = 0x0010
M_MD    = 0x0020; M_MU   = 0x0040
M_WHEEL = 0x0800
KEYUP   = 0x0002

def _me(flags, x=0, y=0, d=0):
    user32.mouse_event(flags, x, y, d, 0)

def move(sx, sy):
    nx = int((sx - VIRT_X) * 65535 / VIRT_W)
    ny = int((sy - VIRT_Y) * 65535 / VIRT_H)
    _me(M_MOVE | M_ABS | M_VIRT, nx, ny)

def lclick():  _me(M_LD); _me(M_LU)
def rclick():  _me(M_RD); _me(M_RU)
def mclick():  _me(M_MD); _me(M_MU)
def ldown():   _me(M_LD)
def lup():     _me(M_LU)
def scroll(d): _me(M_WHEEL, d=d)

def key_tap(*vks):
    for k in vks:          user32.keybd_event(k, 0, 0,     0)
    for k in reversed(vks): user32.keybd_event(k, 0, KEYUP, 0)

VK_ALT  = 0x12; VK_LEFT = 0x25; VK_RIGHT = 0x27
VK_TAB  = 0x09; VK_WIN  = 0x5B; VK_D     = 0x44
VK_VOLU = 0xAF; VK_VOLD = 0xAE; VK_SNAP  = 0x2C

# ── Config ────────────────────────────────────────────────────────────────────
CAM_W, CAM_H = 640, 480
ZONE         = 0.75
DEADZONE     = 5.0
CLICK_CD     = 0.25
DBL_WIN      = 0.45
VOL_CD       = 0.08     # seconds between volume key taps
VOL_STEP     = 0.012    # palm-Y delta to trigger one volume step
SWIPE_VEL    = 2.0      # normalized units/sec for swipe
SWIPE_CD     = 1.0      # cooldown between swipes
SPREAD_HOLD  = 1.2      # seconds to hold open palm for screenshot

# ── Kalman 2-D cursor smoother ────────────────────────────────────────────────
class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix   = np.array([[1,0,0,0],[0,1,0,0]], np.float32)
        self.kf.transitionMatrix    = np.array([[1,0,1,0],[0,1,0,1],
                                                [0,0,1,0],[0,0,0,1]], np.float32)
        self.kf.processNoiseCov     = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kf.errorCovPost        = np.eye(4, dtype=np.float32)
        self._init = False

    def update(self, x, y):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self._init:
            self.kf.statePre  = np.array([[x],[y],[0],[0]], np.float32)
            self.kf.statePost = np.array([[x],[y],[0],[0]], np.float32)
            self._init = True
        self.kf.predict()
        s = self.kf.correct(meas)
        return float(s[0]), float(s[1])

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_h  = mp.solutions.hands
hands = mp_h.Hands(
    static_image_mode=False, max_num_hands=1, model_complexity=0,
    min_detection_confidence=0.75, min_tracking_confidence=0.75,
)
draw = mp.solutions.drawing_utils
LS   = draw.DrawingSpec((0, 210, 0), 2, 2)
CS   = draw.DrawingSpec((180, 0, 200), 2)

# ── Helpers ───────────────────────────────────────────────────────────────────
def fingers_up(lm):
    tips = [8, 12, 16, 20]; pips = [6, 10, 14, 18]
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]

def palm_center(lm):
    """Average of wrist + 4 knuckle bases — much more stable than fingertip."""
    pts = [lm[i] for i in [0, 5, 9, 13, 17]]
    return sum(p.x for p in pts)/5, sum(p.y for p in pts)/5

def hand_scale(lm):
    """Wrist-to-middle-MCP distance — used to normalise pinch threshold."""
    return np.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y)

# ── State ─────────────────────────────────────────────────────────────────────
kf            = Kalman2D()
sx, sy        = VIRT_X + VIRT_W/2.0, VIRT_Y + VIRT_H/2.0
last_l        = 0.0
last_r        = 0.0
last_vol      = 0.0
last_swipe    = 0.0
dragging      = False
scroll_prev_y = None
vol_prev_y    = None
pinched       = False
pinky_up      = False
ring_up       = False
spread_since  = None
palm_hist     = []        # [(norm_x, norm_y, timestamp)]
show_help     = False

# ── Camera ────────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

prev_t = time.time()

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    res   = hands.process(rgb)

    label = ""; label_color = (0, 255, 255)

    if res.multi_hand_landmarks:
        lm  = res.multi_hand_landmarks[0].landmark
        now = time.time()
        fu  = fingers_up(lm)
        px, py = palm_center(lm)
        scale  = hand_scale(lm)

        # ── Adaptive pinch threshold ──────────────────────────────────
        dyn_pinch = float(np.clip(scale * 0.40, 0.03, 0.10))

        # ── Palm velocity (for swipe detection) ───────────────────────
        palm_hist.append((px, py, now))
        palm_hist[:] = [(x, y, t) for x, y, t in palm_hist if now - t < 0.12]
        vel_x = vel_y = 0.0
        if len(palm_hist) >= 2:
            dt = palm_hist[-1][2] - palm_hist[0][2]
            if dt > 0:
                vel_x = (palm_hist[-1][0] - palm_hist[0][0]) / dt
                vel_y = (palm_hist[-1][1] - palm_hist[0][1]) / dt

        # ── Gesture flags — computed BEFORE cursor update ─────────────
        pinch_d    = np.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
        is_pinched = pinch_d < dyn_pinch
        is_pinky   = fu[3] and not fu[0] and not fu[1] and not fu[2]
        is_ring    = fu[2] and not fu[0] and not fu[1] and not fu[3]
        is_volume  = fu[2] and fu[3] and not fu[0] and not fu[1]
        is_spread  = all(fu)

        lock_cursor = is_pinched or is_pinky or is_ring or is_volume

        # ── Cursor update: palm center + Kalman + pointer ballistics ──
        if not lock_cursor:
            margin = (1.0 - ZONE) / 2.0
            tx = np.clip((px - margin) / ZONE, 0.0, 1.0) * VIRT_W + VIRT_X
            ty = np.clip((py - margin) / ZONE, 0.0, 1.0) * VIRT_H + VIRT_Y

            # pointer ballistics: amplify fast movements
            dist  = np.hypot(tx - sx, ty - sy)
            accel = min(1.0 + dist / 300.0, 2.5)
            tx    = sx + (tx - sx) * accel
            ty    = sy + (ty - sy) * accel

            nx, ny = kf.update(tx, ty)
            if abs(nx - sx) > DEADZONE or abs(ny - sy) > DEADZONE:
                sx, sy = nx, ny
                move(sx, sy)

        # ── LEFT CLICK: pinch onset ───────────────────────────────────
        if is_pinched and not pinched and (now - last_l) > CLICK_CD:
            if (now - last_l) < DBL_WIN:
                lclick(); lclick()
                label = "DOUBLE CLICK"; label_color = (0, 255, 0)
            else:
                lclick()
                label = "LEFT CLICK";  label_color = (0, 255, 0)
            last_l = now
            scroll_prev_y = None
            if dragging: lup(); dragging = False

        # ── RIGHT CLICK: pinky tap (up → down) ───────────────────────
        elif pinky_up and not is_pinky and (now - last_r) > CLICK_CD:
            rclick(); last_r = now
            label = "RIGHT CLICK"; label_color = (0, 80, 255)
            scroll_prev_y = None
            if dragging: lup(); dragging = False

        # ── MIDDLE CLICK: ring tap (up → down) ───────────────────────
        elif ring_up and not is_ring and (now - last_r) > CLICK_CD:
            mclick(); last_r = now
            label = "MIDDLE CLICK"; label_color = (255, 0, 200)
            scroll_prev_y = None

        # ── All other gestures ────────────────────────────────────────
        elif not is_pinched and not is_pinky and not is_ring:

            # VOLUME: ring+pinky up, move hand up/down
            if is_volume:
                if vol_prev_y is not None and (now - last_vol) > VOL_CD:
                    delta = vol_prev_y - py   # positive = hand moved up
                    if delta > VOL_STEP:
                        key_tap(VK_VOLU); last_vol = now
                        label = "VOL +"; label_color = (0, 220, 255)
                    elif delta < -VOL_STEP:
                        key_tap(VK_VOLD); last_vol = now
                        label = "VOL -"; label_color = (0, 220, 255)
                vol_prev_y    = py
                scroll_prev_y = None
                if dragging: lup(); dragging = False

            # SCROLL: index + middle up (V sign)
            elif fu[0] and fu[1] and not fu[2] and not fu[3]:
                cy = lm[8].y
                if scroll_prev_y is not None:
                    delta = scroll_prev_y - cy
                    if abs(delta) > 0.005:
                        scroll(int(delta * 3000))
                scroll_prev_y = cy
                vol_prev_y    = None
                label = "SCROLL"; label_color = (255, 165, 0)
                if dragging: lup(); dragging = False

            # SWIPE: open palm + fast movement
            elif is_spread and (now - last_swipe) > SWIPE_CD:
                scroll_prev_y = None; vol_prev_y = None
                if dragging: lup(); dragging = False
                fired = False
                if abs(vel_x) > SWIPE_VEL and abs(vel_x) > abs(vel_y):
                    if vel_x > 0:
                        key_tap(VK_ALT, VK_RIGHT); label = ">> FORWARD"
                    else:
                        key_tap(VK_ALT, VK_LEFT);  label = "<< BACK"
                    fired = True
                elif abs(vel_y) > SWIPE_VEL and abs(vel_y) > abs(vel_x):
                    if vel_y < 0:
                        key_tap(VK_ALT, VK_TAB); label = "ALT + TAB"
                    else:
                        key_tap(VK_WIN, VK_D);   label = "SHOW DESKTOP"
                    fired = True
                if fired:
                    label_color  = (255, 255, 0)
                    last_swipe   = now
                    spread_since = None   # cancel screenshot on swipe

            # DRAG: closed fist
            elif not any(fu):
                if not dragging: ldown(); dragging = True
                label = "DRAG"; label_color = (0, 0, 255)
                scroll_prev_y = None; vol_prev_y = None

            # IDLE / MOVE
            else:
                if dragging: lup(); dragging = False
                scroll_prev_y = None; vol_prev_y = None

        # ── SCREENSHOT: hold open palm for SPREAD_HOLD seconds ───────
        if is_spread and not (is_pinched or is_pinky or is_ring):
            if spread_since is None:
                spread_since = now
            held = now - spread_since
            if held >= SPREAD_HOLD:
                key_tap(VK_WIN, VK_SNAP)
                label = "SCREENSHOT!"; label_color = (0, 255, 255)
                spread_since = None
            elif not label:
                pct = int(held / SPREAD_HOLD * 100)
                label = f"SCREENSHOT {pct}%"; label_color = (0, 200, 200)
        elif not is_spread:
            spread_since = None

        # pending-gesture hints (show while gesture is held but not yet fired)
        if is_pinky and not label:
            label = "RIGHT CLICK...";  label_color = (0, 80, 255)
        if is_ring  and not label:
            label = "MIDDLE CLICK..."; label_color = (255, 0, 200)
        if is_volume and not label:
            label = "VOLUME — move up/down"; label_color = (0, 220, 255)

        # update state
        pinched  = is_pinched
        pinky_up = is_pinky
        ring_up  = is_ring

        # draw
        draw.draw_landmarks(frame, res.multi_hand_landmarks[0],
                            mp_h.HAND_CONNECTIONS, LS, CS)
        pcx, pcy = int(px * CAM_W), int(py * CAM_H)
        cv2.circle(frame, (pcx, pcy), 9, (0, 200, 255), -1)   # palm centre
        ix, iy   = int(lm[8].x * CAM_W), int(lm[8].y * CAM_H)
        cv2.circle(frame, (ix, iy), 7, (0, 255, 0), -1)        # index tip

    else:
        if dragging: lup(); dragging = False
        scroll_prev_y = vol_prev_y = None
        pinched = pinky_up = ring_up = False
        spread_since = None
        palm_hist.clear()

    # ── HUD ───────────────────────────────────────────────────────────────────
    mw = int(CAM_W * (1 - ZONE) / 2)
    mh = int(CAM_H * (1 - ZONE) / 2)
    cv2.rectangle(frame, (mw, mh), (CAM_W-mw, CAM_H-mh), (70, 70, 70), 1)

    if label:
        cv2.putText(frame, label, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 5)
        cv2.putText(frame, label, (18, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.0, label_color, 2)

    t   = time.time()
    fps = 1.0 / max(t - prev_t, 1e-9); prev_t = t
    cv2.putText(frame, f"FPS {fps:.0f}", (CAM_W-95, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

    if dragging:
        cv2.circle(frame, (CAM_W-20, CAM_H-20), 8, (0,0,255), -1)

    cv2.putText(frame, "H = help    ESC = quit", (10, CAM_H-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (130,130,130), 1)

    if show_help:
        lines = [
            "MOVE HAND          Move cursor (palm centre)",
            "PINCH 1x           Left click",
            "PINCH 2x fast      Double click",
            "PINKY  up -> down  Right click",
            "RING   up -> down  Middle click",
            "V-SIGN + move      Scroll",
            "RING+PINKY up      Volume  (move hand up/down)",
            "FIST               Drag",
            "OPEN PALM swipe    Forward / Back / Alt+Tab / Desktop",
            "OPEN PALM hold     Screenshot  (1.2 sec)",
        ]
        ov = frame.copy()
        cv2.rectangle(ov, (8, 65), (390, 65 + len(lines)*24 + 12), (0,0,0), -1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        for i, ln in enumerate(lines):
            cv2.putText(frame, ln, (16, 85+i*24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220,220,220), 1)

    cv2.imshow("Virtual Mouse", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27: break
    if key in (ord('h'), ord('H')): show_help = not show_help

cap.release()
cv2.destroyAllWindows()
