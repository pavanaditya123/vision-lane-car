"""
Vision-Guided Lane-Following Car  (fully simulated autonomous system)

A car drives around a road (an oval, or a random shape with several
tunnels). It does NOT know the map: its only input
is a small camera image of the road in front of it. Every step it:
  OBSERVE  - grabs a camera frame
  DECIDE   - finds the white lane lines with computer vision and works out
             where the middle of the lane is
  ACT      - steers toward the middle of the lane and moves forward
  FEEDBACK - the next camera frame shows whether the steering worked

Run:   python lane_car.py             (live demo, starts in scenario 2)
       python lane_car.py 4           (random road: 3-5 tunnels of mixed light + surprise dark patches)
       python lane_car.py 3           (start in scenario 3)
       python lane_car.py --compare   (measure Basic vs Smart, no window)
Keys:  1 / 2 / 3 / 4 load scenario   G new random road   SPACE pause   R restart   Q quit
       T toggle dark tunnel   N toggle storm (world gets dark)   P drop a dark patch ahead
       - make the P patch darker   + make it lighter (5% to 60% light, changes live)
       B switch Basic/Smart brain   H auto-help (simulated human clicks for you)
Mouse: when the car asks for help, click NEAR a grey lane line in either
       picture on the right (it snaps to the line within 20 px).
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


def track_angle(x, y):
    """Where around the track a point is (radians). One full turn = one lap."""
    return np.arctan2((y - CENTER[1]) / AXES[1], (x - CENTER[0]) / AXES[0])


def random_road_points(seed):
    """A random curvy closed road. The radius wobbles around the oval with
    3 random waves; shapes with bends too sharp for the car are re-drawn."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-np.pi / 2, 1.5 * np.pi, 720, endpoint=False)    # start at the top
    while True:
        r = np.ones_like(t)
        for k in (2, 3, 4):
            r += rng.uniform(0, 0.14) * np.cos(k * t + rng.uniform(0, 2 * np.pi))
        x, y = CENTER[0] + AXES[0] * r * np.cos(t), CENTER[1] + AXES[1] * r * np.sin(t)
        dx, dy = (np.roll(x, -1) - np.roll(x, 1)) / 2, (np.roll(y, -1) - np.roll(y, 1)) / 2
        ddx, ddy = np.roll(x, -1) - 2 * x + np.roll(x, 1), np.roll(y, -1) - 2 * y + np.roll(y, 1)
        bend = np.abs(dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5    # 1 / bend radius
        if r.max() <= 1.15 and r.min() >= 0.6 and 1 / bend.max() >= MIN_BEND_RADIUS:
            return np.stack([x, y], axis=1)


MIN_BEND_RADIUS = 130       # sharpest bend allowed on a random road (the oval's is 160)


class Track:
    """The road. seed=None: the classic oval. A number: a random road shape
    (the same number always gives the same road, so tests are repeatable)."""
    def __init__(self, seed=None):
        self.seed = seed
        self.world = np.full((H, W, 3), GRASS, np.uint8)
        self.road = np.zeros((H, W), np.uint8)
        middle = np.full((H, W), 255, np.uint8)
        if seed is None:
            cv2.ellipse(self.world, CENTER, AXES, 0, 0, 360, ASPHALT, 2 * ROAD_HALF)
            for d in (-LINE_OFFSET, LINE_OFFSET):
                cv2.ellipse(self.world, CENTER, (AXES[0] + d, AXES[1] + d), 0, 0, 360, PAINT, 5)
            cv2.ellipse(self.road, CENTER, AXES, 0, 0, 360, 255, 2 * ROAD_HALF)
            cv2.ellipse(middle, CENTER, AXES, 0, 0, 360, 0, 1)
            self.start = (CENTER[0], CENTER[1] - AXES[1], 0.0)        # top of the oval, facing right
            t = np.linspace(-np.pi / 2, 1.5 * np.pi, 720, endpoint=False)
            self.points = np.stack([CENTER[0] + AXES[0] * np.cos(t), CENTER[1] + AXES[1] * np.sin(t)], axis=1)
        else:
            p = self.points = random_road_points(seed)                    # the lane middle line
            pts = [np.int32(np.round(p))]
            cv2.polylines(self.world, pts, True, ASPHALT, 2 * ROAD_HALF)
            cv2.polylines(self.world, pts, True, PAINT, 2 * LINE_OFFSET + 5)   # wide white band...
            cv2.polylines(self.world, pts, True, ASPHALT, 2 * LINE_OFFSET - 5)  # ...minus its middle
            cv2.polylines(self.road, pts, True, 255, 2 * ROAD_HALF)           # = two white lines
            cv2.polylines(middle, pts, True, 0, 1)
            self.start = (p[0, 0], p[0, 1], math.atan2(p[1, 1] - p[0, 1], p[1, 0] - p[0, 0]))
        # Distance of every pixel from the lane middle: used only to SCORE the car, never to drive it.
        self.dist_middle = cv2.distanceTransform(middle, cv2.DIST_L2, 5)


OVAL = Track()
ZONE_FADE = 10              # light zones fade in/out over this many degrees (mixed, not on/off)
ANGLE_MAP = np.degrees(track_angle(*np.meshgrid(np.arange(W), np.arange(H))))   # for light zones


class Lighting:
    """How bright the world is: 1.0 = daylight, 0.0 = pitch black.
    The environment can change while the car drives."""
    def __init__(self, track=OVAL, tunnel=None, storm=None, zones=(), dapple=None, surprises=()):
        self.track = track
        self.tunnel = tunnel        # (x_from, brightness): everything right of x_from is dark
        self.storm = storm          # (step, brightness): at that step the whole world dims
        self.zones = list(zones)    # [(centre_deg, width_deg, brightness)]: patches of different light
        self.dapple = dapple        # (seed, strength): smooth random light pattern, like light through trees
        self.surprises = list(surprises)   # [(from_step, to_step, x, y, radius, brightness)]: dark patches
        self.patches = []                  # the surprise patches showing right now
        self.manual = None                 # the patch dropped with the P key (only one at a time)
        self.patch_level = 0.3             # its light level, changed live with keys - and +
        self.storm_on = False
        self.news = ('A dark tunnel covers the right bend.' if tunnel else
                     f'{len(self.zones)} tunnels: ' + ', '.join(f'{b:.0%}' for _, _, b in self.zones) + ' light'
                     if self.zones else 'Daylight.')
        self._scene = None

    def update(self, step):
        """Scheduled changes: the storm rolls in at its step, and surprise dark
        patches appear and disappear while the car is driving."""
        if self.storm and step == self.storm[0] and not self.storm_on:
            self.toggle_storm()
        showing = [s[2:] for s in self.surprises if s[0] <= step < s[1]]
        if showing != self.patches:
            new = [p for p in showing if p not in self.patches]
            self.news = (f'SURPRISE: dark patch ({new[0][3]:.0%} light)!' if new
                         else 'A dark patch went away.')
            self.patches, self._scene = showing, None

    def drop_patch(self, x, y, step):
        """Live demo (key P): a dark patch appears right now, 140 px in front of the car,
        at the chosen darkness (keys - and +). Only one P-patch at a time:
        pressing P again moves it, instead of stacking."""
        self.surprises = [s for s in self.surprises if s is not self.manual]
        self.manual = (step, step + 300, x, y, 70.0, self.patch_level)
        self.surprises.append(self.manual)
        self.update(step)

    def change_patch_level(self, change, step):
        """Live demo (keys - and +): make the P-patch darker or lighter, 5% at a time
        (5% = almost black, 60% = light shade). A patch already on the road changes at once."""
        self.patch_level = round(min(0.6, max(0.05, self.patch_level + change)), 2)
        if self.manual in self.surprises:                  # update the patch that is showing
            i = self.surprises.index(self.manual)
            self.manual = self.manual[:5] + (self.patch_level,)
            self.surprises[i] = self.manual
            self.update(step)
        self.news = f'P-patch darkness: {self.patch_level:.0%} light   (- darker, + lighter)'

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
            for centre, width, brightness in self.zones:   # wedge-shaped patches around the track
                off = np.abs((ANGLE_MAP - centre + 180) % 360 - 180)     # degrees from the zone centre
                mix = np.clip((width / 2 + ZONE_FADE - off) / (2 * ZONE_FADE), 0, 1)  # soft edges
                light = light * (1 - mix) + brightness * mix             # overlapping zones blend
            if self.dapple:
                seed, strength = self.dapple
                coarse = np.random.default_rng(seed).normal(0, 1, (H // 50 + 1, W // 50 + 1))
                field = cv2.resize(coarse.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)
                light *= 1 + strength * np.clip(field, -2, 2) / 2       # brightness varies by +-strength
            if self.patches:                                # round dark patches with soft edges
                ys, xs = np.mgrid[0:H, 0:W]
                for x, y, radius, brightness in self.patches:
                    f = np.clip(1.3 - np.hypot(xs - x, ys - y) / radius, 0, 1)   # 1 inside, fades at edge
                    light *= 1 - (1 - brightness) * f
            if self.storm_on:
                light *= self.storm[1]
            lit = self.track.world * light[..., None]
            self._scene = np.clip(lit, 0, 255).astype(np.uint8)   # cap at 255: brighter than white is white
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


LIGHT_WINDOW = 31       # size (px) of the area used to judge the light around each pixel


def line_mask(band, thresholds):
    """Which pixels in the look-ahead band are lane paint?
    A pixel is paint if it is brighter than its own threshold."""
    return band > thresholds


def local_light(gray):
    """How bright the road is AROUND each pixel: the median of the 31 x 31 area
    around it. Lane lines are only 5 px wide, so that median is road (or grass),
    even where a tunnel edge or a dark patch cuts across the picture."""
    return cv2.medianBlur(gray, LIGHT_WINDOW)


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
SNAP = 20           # a help click within 20 px of a lane line is good enough


class Car:
    """brain = 'basic': fixed threshold, keeps driving when blind (common approach).
       brain = 'smart': stops when blind, asks a human to click ONE lane line,
                        learns a new threshold from that click and remembers it
                        for that light level (our novel feature)."""
    def __init__(self, brain='smart', track=OVAL):
        self.brain, self.track = brain, track
        # Memory: road brightness -> paint threshold. Starts with the daylight setting only.
        self.memory = {60.0: float(FACTORY_THRESHOLD)}
        self.clicks = []                         # lap number of every help request answered
        self.just_taught = False                 # True right after a click, until vision is checked
        self.snapped = None                      # (u, v, step) of the last line pixel learned from
        self.x, self.y, self.th = track.start    # start of the road, facing along it
        self.steps, self.laps_angle, self.last_angle = 0, 0.0, self._angle()
        self.speed, self.status, self.why = SPEED, 'driving', 'Starting.'
        self.errors = []                         # distance from lane middle each step (for scoring)
        self.cam = self.mask = None
        self.rng = np.random.default_rng(0)      # camera noise (same every run)

    def _angle(self):
        return float(track_angle(self.x, self.y))

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
        smooth = cv2.GaussianBlur(gray, (5, 5), 0)
        self.band = smooth[BAND].astype(np.float32)               # the rows we analyse
        self.light = local_light(smooth)[BAND].astype(np.float32)  # light around each of their pixels
        self.mask = line_mask(self.band, self.thresholds(self.light))
        centre, self.confidence = lane_centre(self.mask)
        if self.brain == 'smart' and self.confidence < ASK_BELOW:
            # Novel feature, part 1: don't drive blind - stop and ask for help.
            if self.just_taught:                 # a click did not bring the lines back:
                self.status = 'SAFE STOP'        # don't ask forever, stay stopped (fault tolerance)
                self.why = (f'Even after learning I see too little (confidence {self.confidence:.0%}). '
                            'Staying stopped (safe). Click near another lane line to try again, or press R.')
                return
            self.status = 'ASKING'
            self.why = (f'I cannot see the lane (confidence {self.confidence:.0%}). Stopped safely. '
                        'Click near a grey lane line in either picture (or press H).')
            return
        self.just_taught = False
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
        on_road = 0 <= xi < W and 0 <= yi < H and self.track.road[yi, xi] > 0
        self.errors.append(self.track.dist_middle[yi, xi] if on_road else ROAD_HALF)
        if not on_road:
            self.status, self.why = 'OFF ROAD', 'Left the road! (crash)'
        elif self.laps >= LAPS_GOAL:
            self.status, self.why = 'FINISHED', f'Completed {LAPS_GOAL} laps.'
        elif self.steps >= MAX_STEPS:
            self.status = 'TIMEOUT'

    def thresholds(self, light):
        """Paint threshold for every pixel, given the light around it.
        Basic: always the factory value. Smart: use the remembered threshold whose
        light level is closest to the light around that pixel - but only thresholds
        still brighter than that road (sanity check)."""
        if self.brain == 'basic':
            return np.full_like(light, FACTORY_THRESHOLD)
        levels = np.array(list(self.memory), np.float32)
        values = np.array(list(self.memory.values()), np.float32)
        road = light[..., None]                             # rows x columns x 1
        distance = np.abs(road - levels)
        distance[values <= road + 1] = np.inf               # would mark the whole road as paint
        return values[distance.argmin(axis=2)]

    def teach(self, u, v):
        """Novel feature, part 2: learn from ONE click near a lane line at camera
        pixel (u, v). The click does not have to be exact: the car snaps to the
        clearest line pixel within SNAP px of it. Works while asking for help,
        and also to rescue the car from a safe stop."""
        if self.status not in ('ASKING', 'SAFE STOP'):
            return False                                   # driving fine: ignore clicks
        r0 = min(max(v - BAND.start, 0), self.band.shape[0] - 1)   # nearest analysed row
        rows = slice(max(0, r0 - SNAP), r0 + SNAP + 1)
        cols = slice(max(0, u - SNAP), u + SNAP + 1)
        stand_out = (self.band - self.light)[rows, cols]   # how much brighter than the road around it
        r, c = np.unravel_index(int(np.argmax(stand_out)), stand_out.shape)
        r, c = r + rows.start, c + cols.start              # the snapped line pixel
        paint = float(self.band[r, c])                     # paint brightness there
        road = float(self.light[r, c])                     # road brightness around it
        if paint - road < 8:
            self.why = 'No lane line near your click. Click on (or right next to) a grey line, or press H.'
            return False
        self.snapped = (c, r + BAND.start, self.steps)     # shown as a green circle for a moment
        threshold = (paint + road) / 2                     # halfway between road and paint
        self.memory[round(road, 1)] = threshold            # part 3: remember it for this light level
        self.clicks.append(int(self.laps))
        self.status, self.just_taught = 'driving', True
        self.why = (f'Thanks! In this light paint looks like {paint:.0f} and road like {road:.0f}, '
                    f'so my new threshold is {threshold:.0f}. I will remember it.')
        return True


def simulated_human(car):
    """Stands in for a person during automatic tests: clicks a real lane-line
    pixel in the darkest analysed row (where the car is struggling)."""
    truth = camera(car.track.world, car.x, car.y, car.th)[BAND]   # same view, but in daylight
    paint = cv2.cvtColor(truth, cv2.COLOR_BGR2GRAY) > 200
    ys, xs = np.nonzero(paint)
    if not len(ys):
        return False
    i = int(np.argmin(car.light[ys, xs]))                   # the paint pixel in the darkest spot
    return car.teach(int(xs[i]), int(ys[i]) + BAND.start)


# =====================================================================
# 5. SCENARIOS AND MEASUREMENT
# =====================================================================
SCENARIOS = {
    '1': ('Sunny day', lambda: Lighting()),
    '2': ('Dark tunnel on the right bend', lambda: Lighting(tunnel=(680, 0.2))),
    '3': ('Storm rolls in at step 300 (world changes)', lambda: Lighting(storm=(300, 0.3))),
    '4': ('Random road, 3-5 tunnels of mixed light (G = new road)', lambda: random_road_world(1)),
}


def random_road_world(seed, dapple=0.15, surprises=3):
    """A random road with 3-5 tunnels: open road -> tunnel -> open road -> tunnel...
    Every tunnel has its own darkness (12% to 70% light) and length, entrances
    fade in and out, and the whole world gets a patchy +-15% light pattern,
    so light levels are mixed, never clean steps. On top of that, 3 surprise
    dark patches (8% to 50% light) appear on the road WHILE the car is driving
    and go away again a few hundred steps later."""
    rng = np.random.default_rng(seed + 1000)
    n = int(rng.integers(3, 6))
    slot = 360 / n                                          # each tunnel gets its own part of the track
    first = rng.uniform(0, 360)
    zones = [((first + i * slot + rng.uniform(-10, 10)) % 360 - 180,
              float(rng.uniform(25, slot - 30)),            # leaves open road between tunnels
              float(rng.uniform(0.12, 0.7))) for i in range(n)]
    track = Track(seed)
    surprise_rng = np.random.default_rng(seed + 2000)
    plan = []
    for _ in range(surprises):
        start = int(surprise_rng.integers(50, 800))
        x, y = track.points[int(surprise_rng.integers(len(track.points)))]   # somewhere on the road
        plan.append((start, start + int(surprise_rng.integers(150, 400)), float(x), float(y),
                     float(surprise_rng.uniform(50, 110)), float(surprise_rng.uniform(0.08, 0.5))))
    return Lighting(track=track, zones=zones, dapple=(seed, dapple) if dapple else None, surprises=plan)


def run_to_end(brain, lighting):
    """Drive until finished/crashed; a simulated human answers help requests."""
    car = Car(brain, lighting.track)
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
    print(f'\n{trials} random worlds (oval, random tunnel + 70% chance of a storm)')
    for brain in ('basic', 'smart'):
        rng = np.random.default_rng(7)               # same worlds for both brains
        summarise(brain, [run_to_end(brain, random_lighting(rng)) for _ in range(trials)])
    print(f'\n{trials} random roads (3-5 tunnels of mixed light each)')
    for brain in ('basic', 'smart'):                 # seeds 1..trials: same roads for both brains
        summarise(brain, [run_to_end(brain, random_road_world(seed)) for seed in range(1, trials + 1)])


def summarise(brain, cars):
    """Print one result line for a list of finished runs."""
    n = len(cars)
    crashes = sum(c.status == 'OFF ROAD' for c in cars)
    done = sum(c.status == 'FINISHED' for c in cars)
    safe = sum(c.status == 'SAFE STOP' for c in cars)
    lap1 = sum(sum(1 for lap in c.clicks if lap == 0) for c in cars)
    lap2 = sum(len(c.clicks) for c in cars) - lap1
    err = np.mean([np.mean(c.errors) for c in cars])
    print(f'  {brain:<6} crashed {crashes}/{n}   finished 2 laps {done}/{n}   '
          f'safe stops {safe}/{n}   help clicks: lap1 {lap1}, lap2 {lap2}   '
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
    for x, y, radius, brightness in lighting.patches:             # outline the surprise dark patches
        cv2.circle(view, (int(x), int(y)), int(radius), (255, 120, 255), 2, cv2.LINE_AA)
        cv2.putText(view, f'{brightness:.0%}', (int(x) - 30, int(y - radius) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 120, 255), 2, cv2.LINE_AA)
    for i, (centre, _, brightness) in enumerate(lighting.zones):   # label each tunnel inside the track
        a = math.radians(centre)
        lx, ly = CENTER[0] + 0.55 * AXES[0] * math.cos(a), CENTER[1] + 0.55 * AXES[1] * math.sin(a)
        cv2.putText(view, f'T{i + 1} {brightness:.0%}', (int(lx) - 45, int(ly) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 140, 255), 2, cv2.LINE_AA)
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
    needs_click = car.status in ('ASKING', 'SAFE STOP')
    if car.cam is not None:
        cam = car.cam
        if needs_click:                                    # "night view": brighten dark areas so
            gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)    # the human can see the lines to click
            cam = cv2.cvtColor(cv2.createCLAHE(4.0, (4, 4)).apply(gray), cv2.COLOR_GRAY2BGR)
        panel[25:205, 5:245] = cv2.resize(cam, (240, 180))
        seen = cam.copy() if needs_click else np.zeros((CAM_H, CAM_W, 3), np.uint8)
        seen[BAND][car.mask] = (255, 255, 255)
        seen[BAND.start, :] = seen[BAND.stop - 1, :] = (0, 220, 255)   # analysed band
        panel[230:410, 5:245] = cv2.resize(seen, (240, 180), interpolation=cv2.INTER_NEAREST)
        if car.snapped and car.steps - car.snapped[2] < 25:   # where the last click snapped to
            for top in (25, 230):
                cv2.circle(panel, (5 + int(car.snapped[0] * 1.5), top + int(car.snapped[1] * 1.5)),
                           10, (0, 255, 0), 2, cv2.LINE_AA)
    if needs_click:                                        # red frames: "click here!"
        for top in (25, 230):
            cv2.rectangle(panel, (3, top - 2), (247, top + 182), (0, 0, 255), 3)
        text(panel, 'NIGHT VIEW: click near a grey line', 18, (80, 80, 255))
        text(panel, 'either picture works  (or press H)', 223, (80, 80, 255))
    else:
        text(panel, 'Camera (what the car sees)', 18)
        text(panel, 'Vision: detected lane paint', 223)
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


def panel_to_camera(mx, my):
    """Window pixel -> camera pixel, for a click on the camera picture (top)
    or the vision picture (bottom) of the right panel; None elsewhere.
    Both pictures are the camera view shown 1.5x bigger (240 x 180)."""
    if not 725 <= mx < 965:
        return None
    for top in (25, 230):
        if top <= my < top + 180:
            return int((mx - 725) * CAM_W / 240), int((my - top) * CAM_H / 180)
    return None


def main(scenario='2'):
    brain, auto_help, paused, road_seed = 'smart', False, False, 1

    def make_world():
        return random_road_world(road_seed) if scenario == '4' else SCENARIOS[scenario][1]()

    lighting = make_world()
    car = Car(brain, lighting.track)

    def on_mouse(event, mx, my, flags, param):
        """A click on either picture in the right panel answers the car's help request."""
        spot = panel_to_camera(mx, my)
        if event == cv2.EVENT_LBUTTONDOWN and spot:
            car.teach(*spot)

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
        if chr(key) in SCENARIOS:                # 1/2/3/4: load a scenario
            scenario = chr(key)
        if key == ord('b'):                      # switch brain
            brain = 'basic' if brain == 'smart' else 'smart'
        if key == ord('g'):                      # generate a new random road with new tunnels
            scenario, road_seed = '4', road_seed + 1
        if key in (ord('r'), ord('b'), ord('g')) or chr(key) in SCENARIOS:
            lighting = make_world()
            car = Car(brain, lighting.track)
            if scenario != '4':
                lighting.news = f'Scenario {scenario}: {SCENARIOS[scenario][0]}'
            else:
                lighting.news = f'Road #{road_seed}: ' + lighting.news
        if key == ord('h'):                      # let a simulated human answer help requests
            auto_help = not auto_help
            lighting.news = f'Auto-help (simulated human) {"ON" if auto_help else "OFF"}.'
        if key == ord('t'):                      # change the environment live
            lighting.toggle_tunnel()
        if key == ord('n'):
            lighting.toggle_storm()
        if key == ord('p'):                      # drop a surprise dark patch in front of the car
            lighting.drop_patch(car.x + 140 * math.cos(car.th), car.y + 140 * math.sin(car.th), car.steps)
        if key in (ord('-'), ord('_')):          # make the P-patch darker (live)
            lighting.change_patch_level(-0.05, car.steps)
        if key in (ord('+'), ord('=')):          # make the P-patch lighter (live)
            lighting.change_patch_level(+0.05, car.steps)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    import sys
    if '--compare' in sys.argv:
        compare()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in SCENARIOS else '2')
