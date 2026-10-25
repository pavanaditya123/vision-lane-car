"""
Vision-Guided Lane-Following Car  (fully simulated autonomous system)

A car drives around an oval road. It does NOT know the map: its only input
is a small camera image of the road in front of it. Every step it:
  OBSERVE  - grabs a camera frame
  DECIDE   - finds the white lane lines with computer vision and works out
             where the middle of the lane is
  ACT      - steers toward the middle of the lane and moves forward
  FEEDBACK - the next camera frame shows whether the steering worked

Run:   python lane_car.py             (live demo, starts in scenario 2)
       python lane_car.py 3           (start in scenario 3)
       python lane_car.py --compare   (measure Basic vs Smart, no window)
Keys:  1 / 2 / 3 load scenario   SPACE pause   R restart   Q quit
       T toggle dark tunnel   N toggle storm (world gets dark)
       B switch Basic/Smart brain   H auto-help (simulated human clicks for you)
Mouse: when the car asks for help, click a white lane line in the
       "vision" picture on the right.
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
NOISE = 3.0                              # camera sensor noise (std dev in gray levels)
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


class Lighting:
    """How bright the world is: 1.0 = daylight, 0.0 = pitch black.
    The environment can change while the car drives."""
    def __init__(self, tunnel=None, storm=None):
        self.tunnel = tunnel        # (x_from, brightness): everything right of x_from is dark
        self.storm = storm          # (step, brightness): at that step the whole world dims
        self.storm_on = False
        self.news = 'A dark tunnel covers the right bend.' if tunnel else 'Daylight.'
        self._scene = None

    def update(self, step):
        """Scheduled change: the storm rolls in at its step."""
        if self.storm and step == self.storm[0] and not self.storm_on:
            self.toggle_storm()

    def toggle_tunnel(self):
        self.tunnel = None if self.tunnel else (680, 0.2)
        self.news = 'A dark tunnel now covers the right bend!' if self.tunnel else 'Tunnel removed.'
        self._scene = None

    def toggle_storm(self):
        self.storm_on = not self.storm_on
        if self.storm is None:
            self.storm = (-1, 0.3)
        self.news = f'WORLD CHANGED: storm! Light dropped to {self.storm[1]:.0%}.' if self.storm_on else 'Storm passed.'
        self._scene = None

    def scene(self):
        """The world as it looks right now (recomputed only when the light changes)."""
        if self._scene is None:
            light = np.ones((H, W), np.float32)
            if self.tunnel:
                light[:, self.tunnel[0]:] = self.tunnel[1]
            if self.storm_on:
                light *= self.storm[1]
            self._scene = (WORLD * light[..., None]).astype(np.uint8)
        return self._scene


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
FACTORY_THRESHOLD = 180                  # "paint is brighter than 180" - tuned for daylight


def line_mask(band, thresholds):
    """Which pixels in the look-ahead band are lane paint?
    A pixel is paint if it is brighter than the threshold for its row."""
    return band > thresholds[:, None]


def row_light(band):
    """How bright the road is in each row (the median pixel is almost always road)."""
    return np.median(band, axis=1)


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
ASK_BELOW = 0.25    # smart car stops and asks for help when confidence drops below 25%


class Car:
    """brain = 'basic': fixed threshold, keeps driving when blind (common approach).
       brain = 'smart': stops when blind, asks a human to click ONE lane line,
                        learns a new threshold from that click and remembers it
                        for that light level (our novel feature)."""
    def __init__(self, brain='smart'):
        self.brain = brain
        # Memory: road brightness -> paint threshold. Starts with the daylight setting only.
        self.memory = {60.0: float(FACTORY_THRESHOLD)}
        self.clicks = []                         # lap number of every help request answered
        self.x, self.y, self.th = CENTER[0], CENTER[1] - AXES[1], 0.0   # top of the oval, facing right
        self.steps, self.laps_angle, self.last_angle = 0, 0.0, self._angle()
        self.speed, self.status, self.why = SPEED, 'driving', 'Starting.'
        self.errors = []                         # distance from lane middle each step (for scoring)
        self.cam = self.mask = None
        self.rng = np.random.default_rng(0)      # camera noise (same every run)

    def _angle(self):
        return math.atan2((self.y - CENTER[1]) / AXES[1], (self.x - CENTER[0]) / AXES[0])

    @property
    def laps(self):
        return self.laps_angle / (2 * math.pi)

    def step(self, scene):
        if self.status != 'driving':             # finished, crashed, or waiting for a click
            return
        self.steps += 1

        # ---- OBSERVE: take a camera picture, convert to grayscale ----
        self.cam = camera(scene, self.x, self.y, self.th)
        gray = cv2.cvtColor(self.cam, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray = np.clip(gray + self.rng.normal(0, NOISE, gray.shape), 0, 255).astype(np.uint8)

        # ---- DECIDE: where is the lane middle, how much should I turn? ----
        self.band = cv2.GaussianBlur(gray, (5, 5), 0)[BAND].astype(np.float32)
        self.mask = line_mask(self.band, self.thresholds(self.band))
        centre, self.confidence = lane_centre(self.mask)
        if self.brain == 'smart' and self.confidence < ASK_BELOW:
            # Novel feature, part 1: don't drive blind - stop and ask for help.
            self.status = 'ASKING'
            self.why = (f'I cannot see the lane (confidence {self.confidence:.0%}). Stopped safely. '
                        'Please click a white lane line in my camera view.')
            return
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

    def thresholds(self, band):
        """Paint threshold for every row of the band.
        Basic: always the factory value. Smart: for each row, use the remembered
        threshold whose light level is closest to that row's road brightness -
        but only thresholds that are still brighter than that road (sanity check)."""
        if self.brain == 'basic':
            return np.full(band.shape[0], FACTORY_THRESHOLD, np.float32)
        levels = np.array(list(self.memory))
        values = np.array(list(self.memory.values()), np.float32)
        road = row_light(band)[:, None]
        distance = np.abs(road - levels[None, :])
        distance[values[None, :] <= road + 1] = np.inf      # would mark the whole road as paint
        return values[distance.argmin(axis=1)]

    def teach(self, u, v):
        """Novel feature, part 2: learn from ONE click at camera pixel (u, v)."""
        if self.status != 'ASKING' or not (BAND.start <= v < BAND.stop):
            self.why = 'Please click on a white line between the two yellow lines of the vision view.'
            return False
        row = self.band[v - BAND.start]
        paint = float(row[max(0, u - 3):u + 4].max())     # brightest pixel near the click
        road = float(np.median(row))                       # typical road brightness in that row
        if paint - road < 8:
            self.why = f"That spot ({paint:.0f}) is not brighter than the road ({road:.0f}). Please click a white line."
            return False
        threshold = (paint + road) / 2                     # halfway between road and paint
        self.memory[round(road, 1)] = threshold            # part 3: remember it for this light level
        self.clicks.append(int(self.laps))
        self.status = 'driving'
        self.why = (f'Thanks! In this light paint looks like {paint:.0f} and road like {road:.0f}, '
                    f'so my new threshold is {threshold:.0f}. I will remember it.')
        return True


def simulated_human(car):
    """Stands in for a person during automatic tests: clicks a real lane-line
    pixel in the darkest analysed row (where the car is struggling)."""
    truth = camera(WORLD, car.x, car.y, car.th)[BAND]       # same view, but in daylight
    paint = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY) > 200
    rows = [r for r in np.argsort(row_light(car.band)) if paint[r].any()]
    if not rows:
        return False
    r = rows[0]
    return car.teach(int(np.nonzero(paint[r])[0][0]) + 1, r + BAND.start)


# =====================================================================
# 5. SCENARIOS AND MEASUREMENT
# =====================================================================
SCENARIOS = {
    '1': ('Sunny day', lambda: Lighting()),
    '2': ('Dark tunnel on the right bend', lambda: Lighting(tunnel=(680, 0.2))),
    '3': ('Storm rolls in at step 300 (world changes)', lambda: Lighting(storm=(300, 0.3))),
}


def run_to_end(brain, lighting):
    """Drive until finished/crashed; a simulated human answers help requests."""
    car = Car(brain)
    while car.status in ('driving', 'ASKING'):
        if car.status == 'ASKING' and not simulated_human(car):
            car.status = 'SAFE STOP'                 # too dark even for a human-taught threshold
            car.why = 'Too dark to see the paint even with help. Staying stopped (safe).'
            break
        lighting.update(car.steps)
        car.step(lighting.scene())
    return car


def random_lighting(rng):
    """A random world: a tunnel of random size/darkness, and maybe a storm."""
    tunnel = (int(rng.integers(560, 741)), float(rng.uniform(0.1, 0.4)))
    storm = (int(rng.integers(100, 701)), float(rng.uniform(0.15, 0.5))) if rng.random() < 0.7 else None
    return Lighting(tunnel=tunnel, storm=storm)


def compare(trials=30):
    """Run Basic and Smart on the 3 scenarios and on random worlds; print a table."""
    def row(name, car):
        lap1 = sum(1 for lap in car.clicks if lap == 0)
        return (f'  {name:<44} {car.brain:<6} {car.status:<9} laps {car.laps:4.2f}  '
                f'clicks {len(car.clicks)} (lap1 {lap1}, lap2 {len(car.clicks) - lap1})')
    print('\nNamed scenarios')
    for name, make in SCENARIOS.values():
        for brain in ('basic', 'smart'):
            print(row(name, run_to_end(brain, make())))
    print(f'\n{trials} random worlds (random tunnel + 70% chance of a storm)')
    for brain in ('basic', 'smart'):
        rng = np.random.default_rng(7)               # same worlds for both brains
        cars = [run_to_end(brain, random_lighting(rng)) for _ in range(trials)]
        crashes = sum(c.status == 'OFF ROAD' for c in cars)
        done = sum(c.status == 'FINISHED' for c in cars)
        safe = sum(c.status == 'SAFE STOP' for c in cars)
        lap1 = sum(sum(1 for lap in c.clicks if lap == 0) for c in cars)
        lap2 = sum(len(c.clicks) for c in cars) - lap1
        err = np.mean([np.mean(c.errors) for c in cars])
        print(f'  {brain:<6} crashed {crashes}/{trials}   finished 2 laps {done}/{trials}   '
              f'safe stops {safe}/{trials}   help clicks: lap1 {lap1}, lap2 {lap2}   '
              f'avg distance from lane middle {err:.1f}px')


# =====================================================================
# 6. DRAWING
# =====================================================================
def text(img, s, y, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render(lighting, car):
    view = lighting.scene().copy()
    if lighting.tunnel:                                    # outline the tunnel so the class can see it
        cv2.rectangle(view, (lighting.tunnel[0], 0), (W - 1, H - 1), (0, 140, 255), 2)
        cv2.putText(view, 'TUNNEL', (lighting.tunnel[0] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2)
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
    panel = np.zeros((560, 380, 3), np.uint8)
    if car.cam is not None:
        panel[25:205, 5:245] = cv2.resize(car.cam, (240, 180))
        seen = np.zeros((CAM_H, CAM_W, 3), np.uint8)
        seen[BAND][car.mask] = (255, 255, 255)
        seen[BAND.start, :] = seen[BAND.stop - 1, :] = (0, 220, 255)   # analysed band
        panel[230:410, 5:245] = cv2.resize(seen, (240, 180), interpolation=cv2.INTER_NEAREST)
    if car.status == 'ASKING':                             # red frame: "click here!"
        cv2.rectangle(panel, (3, 228), (247, 412), (0, 0, 255), 3)
        text(panel, 'CLICK A WHITE LINE BELOW', 223, (80, 80, 255))
    else:
        text(panel, 'Vision: detected lane paint', 223)
    text(panel, 'Camera (what the car sees)', 18)
    mem = '  '.join(f'{k:.0f}->{v:.0f}' for k, v in sorted(car.memory.items(), reverse=True))
    text(panel, f'{car.brain.upper()} brain    help clicks: {len(car.clicks)}', 428, (255, 200, 120))
    text(panel, f'Memory (road light -> threshold): {mem}', 446, (255, 200, 120), 0.4)
    text(panel, lighting.news, 464, (0, 160, 255), 0.45)
    text(panel, f'{car.status}  step {car.steps}  laps {car.laps:.2f}  '
                f'confidence {getattr(car, "confidence", 0):.0%}', 482, scale=0.45)
    words, line, y = car.why.split(), '', 504       # wrap the explanation
    for w in words:
        if len(line + w) > 44:
            text(panel, line, y, (120, 255, 255), 0.45); line, y = '', y + 18
        line += w + ' '
    text(panel, line, y, (120, 255, 255), 0.45)
    return np.hstack([view, panel])


WINDOW = 'Vision-Guided Lane-Following Car'


def main(scenario='2'):
    brain, auto_help, paused = 'smart', False, False
    car, lighting = Car(brain), SCENARIOS[scenario][1]()

    def on_mouse(event, mx, my, flags, param):
        """A click on the 'vision' picture (right panel) answers the car's help request."""
        if event == cv2.EVENT_LBUTTONDOWN and 725 <= mx < 965 and 230 <= my < 410:
            car.teach(int((mx - 725) * CAM_W / 240), int((my - 230) * CAM_H / 180))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    while True:
        if not paused:
            if car.status == 'ASKING' and auto_help:
                simulated_human(car)
            lighting.update(car.steps)
            car.step(lighting.scene())
        cv2.imshow(WINDOW, render(lighting, car))
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord(' '):
            paused = not paused
        if chr(key) in SCENARIOS:                # 1/2/3: load a scenario
            scenario = chr(key)
        if key == ord('b'):                      # switch brain
            brain = 'basic' if brain == 'smart' else 'smart'
        if key in (ord('r'), ord('b')) or chr(key) in SCENARIOS:
            car, lighting = Car(brain), SCENARIOS[scenario][1]()
            lighting.news = f'Scenario {scenario}: {SCENARIOS[scenario][0]}'
        if key == ord('h'):                      # let a simulated human answer help requests
            auto_help = not auto_help
            lighting.news = f'Auto-help (simulated human) {"ON" if auto_help else "OFF"}.'
        if key == ord('t'):                      # change the environment live
            lighting.toggle_tunnel()
        if key == ord('n'):
            lighting.toggle_storm()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    if '--compare' in sys.argv:
        compare()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SCENARIOS else '2')
