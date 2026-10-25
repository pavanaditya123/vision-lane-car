"""
Vision-Guided Lane-Following Car  (fully simulated autonomous system)

A car drives around an oval road. It does NOT know the map: its only input
is a small camera image of the road in front of it. Every step it:
  OBSERVE  - grabs a camera frame
  DECIDE   - finds the white lane lines with computer vision and works out
             where the middle of the lane is
  ACT      - steers toward the middle of the lane and moves forward
  FEEDBACK - the next camera frame shows whether the steering worked

Run:   python lane_car.py
Keys:  SPACE pause   R restart   Q quit
"""
import math
import numpy as np
import cv2

# =====================================================================
# 1. WORLD SETTINGS
# =====================================================================
W, H = 900, 700                          # top-down world image (pixels)
CENTER, AXES = (450, 350), (330, 230)    # oval track: centre and half-axes of its middle line
ROAD_HALF = 48                           # road is 96 px wide
LINE_OFFSET = 38                         # white edge lines sit 38 px each side of the middle
CAM_W, CAM_H, CAM_NEAR = 160, 120, 10    # camera image size; it starts 10 px ahead of the car
BAND = slice(40, 100)                    # camera rows the car analyses (30-90 px ahead)
SPEED = 4.0                              # pixels per step at full speed
KP = 0.006                               # steering gain: radians of turn per pixel of error
LAPS_GOAL, MAX_STEPS = 2, 1500
GRASS, ASPHALT, PAINT = (40, 90, 40), (60, 60, 60), (255, 255, 255)   # BGR colours


def build_world():
    """Draw the road once. Returns the colour world, a road mask, and the
    distance of every pixel from the lane middle (used only to SCORE the car,
    never to drive it)."""
    world = np.full((H, W, 3), GRASS, np.uint8)
    cv2.ellipse(world, CENTER, AXES, 0, 0, 360, ASPHALT, 2 * ROAD_HALF)
    for d in (-LINE_OFFSET, LINE_OFFSET):
        cv2.ellipse(world, CENTER, (AXES[0] + d, AXES[1] + d), 0, 0, 360, PAINT, 5)
    road = np.zeros((H, W), np.uint8)
    cv2.ellipse(road, CENTER, AXES, 0, 0, 360, 255, 2 * ROAD_HALF)
    middle = np.full((H, W), 255, np.uint8)
    cv2.ellipse(middle, CENTER, AXES, 0, 0, 360, 0, 1)
    return world, road, cv2.distanceTransform(middle, cv2.DIST_L2, 5)


WORLD, ROAD, DIST_MIDDLE = build_world()


# =====================================================================
# 2. THE CAMERA  (the car's only sensor)
# =====================================================================
def camera(scene, x, y, th):
    """Cut out a CAM_W x CAM_H patch of the scene in front of the car,
    rotated so that 'forward' points up. Column CAM_W/2 is straight ahead."""
    c, s = math.cos(th), math.sin(th)
    ahead = CAM_H + CAM_NEAR
    # Maps camera pixel (u, v) -> world pixel: car + forward*(ahead - v) + right*(u - CAM_W/2)
    M = np.float32([[-s, -c, x + c * ahead + s * CAM_W / 2],
                    [c, -s, y + s * ahead - c * CAM_W / 2]])
    return cv2.warpAffine(scene, M, (CAM_W, CAM_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=GRASS)


# =====================================================================
# 3. COMPUTER VISION  (find the lane lines)
# =====================================================================
def line_mask(gray):
    """Which pixels in the look-ahead band are lane paint?
    Basic rule used by many beginner line followers: 'brighter than 180'."""
    band = cv2.GaussianBlur(gray, (5, 5), 0)[BAND]
    return band > 180


def lane_centre(mask):
    """From the line pixels, estimate the lane middle (camera column) and a
    confidence score = share of analysed rows in which any line was seen."""
    xs = np.nonzero(mask)[1]
    left, right = xs[xs < CAM_W // 2], xs[xs >= CAM_W // 2]
    confidence = float(mask.any(axis=1).mean())
    if len(left) and len(right):
        return (left.mean() + right.mean()) / 2, confidence
    if len(left):                                # only one line visible:
        return left.mean() + LINE_OFFSET, confidence     # lane middle is 38 px to its right
    if len(right):
        return right.mean() - LINE_OFFSET, confidence
    return None, confidence


# =====================================================================
# 4. THE AUTONOMOUS LOOP  (one step = observe -> decide -> act -> feedback)
# =====================================================================
class Car:
    def __init__(self):
        self.x, self.y, self.th = CENTER[0], CENTER[1] - AXES[1], 0.0   # top of the oval, facing right
        self.steps, self.laps_angle, self.last_angle = 0, 0.0, self._angle()
        self.speed, self.status, self.why = SPEED, 'driving', 'Starting.'
        self.errors = []                         # distance from lane middle each step (for scoring)
        self.cam = self.mask = None

    def _angle(self):
        return math.atan2((self.y - CENTER[1]) / AXES[1], (self.x - CENTER[0]) / AXES[0])

    @property
    def laps(self):
        return self.laps_angle / (2 * math.pi)

    def step(self, scene):
        if self.status != 'driving':
            return
        self.steps += 1

        # ---- OBSERVE: take a camera picture, convert to grayscale ----
        self.cam = camera(scene, self.x, self.y, self.th)
        gray = cv2.cvtColor(self.cam, cv2.COLOR_BGR2GRAY)

        # ---- DECIDE: where is the lane middle, how much should I turn? ----
        self.mask = line_mask(gray)
        centre, self.confidence = lane_centre(self.mask)
        if centre is None:
            error = 0.0
            self.why = 'No lane lines found -> drive straight and hope.'
        else:
            error = centre - CAM_W / 2           # + means the lane middle is to my right
            side = 'right' if error > 0 else 'left'
            self.why = f'Lane middle is {abs(error):.0f}px to the {side} -> steer {side}.'

        # ---- ACT: steer, then move forward ----
        self.th += KP * error
        self.x += self.speed * math.cos(self.th)
        self.y += self.speed * math.sin(self.th)

        # ---- FEEDBACK: score the result (the car itself never sees these numbers) ----
        a = self._angle()
        self.laps_angle += (a - self.last_angle + math.pi) % (2 * math.pi) - math.pi
        self.last_angle = a
        xi, yi = int(self.x), int(self.y)
        on_road = 0 <= xi < W and 0 <= yi < H and ROAD[yi, xi] > 0
        self.errors.append(DIST_MIDDLE[yi, xi] if on_road else ROAD_HALF)
        if not on_road:
            self.status, self.why = 'OFF ROAD', 'Left the road! (crash)'
        elif self.laps >= LAPS_GOAL:
            self.status, self.why = 'FINISHED', f'Completed {LAPS_GOAL} laps.'
        elif self.steps >= MAX_STEPS:
            self.status = 'TIMEOUT'


# =====================================================================
# 5. DRAWING
# =====================================================================
def text(img, s, y, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(scene, car):
    view = scene.copy()
    c, s = math.cos(car.th), math.sin(car.th)
    # Camera footprint: the area the car can see (yellow box)
    corners = []
    for u, v in ((0, 0), (CAM_W, 0), (CAM_W, CAM_H), (0, CAM_H)):
        ahead, side = CAM_H + CAM_NEAR - v, u - CAM_W / 2
        corners.append((car.x + c * ahead - s * side, car.y + s * ahead + c * side))
    cv2.polylines(view, [np.int32(corners)], True, (0, 220, 255), 1)
    # The car: a red rotated rectangle
    box = cv2.boxPoints(((car.x, car.y), (22, 12), math.degrees(car.th)))
    cv2.fillPoly(view, [np.int32(box)], (40, 40, 230))
    view = cv2.resize(view, (720, 560))

    # Right-hand panel: camera image, what the vision found, and the explanation
    panel = np.zeros((560, 330, 3), np.uint8)
    if car.cam is not None:
        panel[25:205, 5:245] = cv2.resize(car.cam, (240, 180))
        seen = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        seen[BAND][car.mask] = (255, 255, 255)
        seen[BAND.start, :] = seen[BAND.stop - 1, :] = (0, 220, 255)   # analysed band
        panel[230:410, 5:245] = cv2.resize(seen, (240, 180), interpolation=cv2.INTER_NEAREST)
    text(panel, 'Camera (what the car sees)', 18)
    text(panel, 'Vision: detected lane paint', 223)
    text(panel, f'Status: {car.status}   step {car.steps}', 430)
    text(panel, f'Laps: {car.laps:.2f}   speed: {car.speed:.1f}', 452)
    text(panel, f'Vision confidence: {getattr(car, "confidence", 0) * 100:.0f}%', 474)
    words, line, y = car.why.split(), '', 500       # wrap the explanation
    for w in words:
        if len(line + w) > 38:
            text(panel, line, y, (120, 255, 255)); line, y = '', y + 20
        line += w + ' '
    text(panel, line, y, (120, 255, 255))
    return np.hstack([view, panel])


def main():
    car, paused = Car(), False
    while True:
        if not paused:
            car.step(WORLD)
        cv2.imshow('Vision-Guided Lane-Following Car', render(WORLD, car))
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('r'):
            car = Car()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
