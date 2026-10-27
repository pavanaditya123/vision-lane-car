# How the Project Is Implemented (step by step)

This guide walks through `lane_car.py` **in the order it was built**. Each step
says what we needed, the code that does it, and how to check it works. All
code below is copied from the real file.

The whole program is **one Python file (about 400 lines)** using only two libraries:

| Library | Used for |
|---|---|
| **OpenCV** (`cv2`) | drawing the road, cutting out the camera picture, blurring, the window, mouse clicks |
| **NumPy** (`np`) | treating images as grids of numbers and doing fast maths on them |

> **Key idea:** to a computer, an image is just a grid of numbers. A grayscale
> picture 160 wide and 120 tall is a 120 × 160 table of brightness values from 0 to 255.

---

## The build order at a glance

| Step | What we build | Function / class | Git commit |
|---|---|---|---|
| 0 | Setup: Python, venv, libraries | `requirements.txt`, `run.bat` | "Add virtual environment setup" |
| 1 | Draw the world (road, lines, grass) | `build_world()` | "Add simulated road world…" |
| 2 | The camera (the car's only sensor) | `camera()` | same commit |
| 3 | Computer vision: find the lane lines | `line_mask()`, `lane_centre()` | same commit |
| 4 | The control loop: steer and move | `Car.step()` | same commit |
| 5 | Changing light: tunnel and storm | `Lighting` | "Add changing lighting…" |
| 6 | **Novel feature:** ask, learn, remember | `Car.step()`, `teach()`, `thresholds()` | "Add novel feature…" |
| 7 | Scenarios and measurement | `run_to_end()`, `compare()` | "Add three scenarios…" |
| 8 | Window, keys and mouse | `render()`, `main()` | across commits |

You can see each step's exact changes on GitHub: **Commits** → click a commit.

---

## Step 0: Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt     # opencv-python, numpy
python lane_car.py
```

Or just double-click `run.bat`, which does all of this for you.

---

## Step 1: Draw the world

**Goal:** a top-down picture of an oval road with two white lane lines on green grass.

```python
W, H = 900, 700                          # world image size (pixels)
CENTER, AXES = (450, 350), (330, 230)    # oval: centre and half-width / half-height
ROAD_HALF = 48                           # road is 96 px wide
LINE_OFFSET = 38                         # white lines 38 px each side of the middle

def build_world():
    world = np.full((H, W, 3), GRASS, np.uint8)                        # all grass
    cv2.ellipse(world, CENTER, AXES, 0, 0, 360, ASPHALT, 2 * ROAD_HALF)  # thick grey oval = road
    for d in (-LINE_OFFSET, LINE_OFFSET):                               # two thin white ovals = lines
        cv2.ellipse(world, CENTER, (AXES[0] + d, AXES[1] + d), 0, 0, 360, PAINT, 5)
    road = np.zeros((H, W), np.uint8)                                   # road mask (for scoring)
    cv2.ellipse(road, CENTER, AXES, 0, 0, 360, 255, 2 * ROAD_HALF)
    middle = np.full((H, W), 255, np.uint8)
    cv2.ellipse(middle, CENTER, AXES, 0, 0, 360, 0, 1)
    return world, road, cv2.distanceTransform(middle, cv2.DIST_L2, 5)
```

**In simple words:**
- `cv2.ellipse` draws an oval. A **thick** grey oval is the road; two **thin** white ovals are the lane lines.
- We also make two **hidden helper images** that are used **only to score** the car, never to drive it:
  - `road`: which pixels are road (to detect "off road = crash");
  - `distanceTransform`: how far every pixel is from the lane middle (to measure driving accuracy).

**Colours:** grass `(40, 90, 40)`, road `(60, 60, 60)`, paint `(255, 255, 255)`.
OpenCV writes colours as **B, G, R**, not R, G, B.

---

## Step 2: The camera

**Goal:** the car must NOT see the whole map. It only gets a small **160 × 120 picture
of the road in front of it**, rotated so "forward" is always "up".

```python
CAM_W, CAM_H, CAM_NEAR = 160, 120, 10    # picture size; starts 10 px ahead of the car

def camera(scene, x, y, th):
    c, s = math.cos(th), math.sin(th)
    ahead = CAM_H + CAM_NEAR
    # camera pixel (u, v) -> world pixel: car + forward*(ahead - v) + right*(u - CAM_W/2)
    M = np.float32([[-s, -c, x + c * ahead + s * CAM_W / 2],
                    [c, -s, y + s * ahead - c * CAM_W / 2]])
    return cv2.warpAffine(scene, M, (CAM_W, CAM_H), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=GRASS)
```

**In simple words:**
- The car has a position `(x, y)` and a heading angle `th` (where it points).
- `cv2.warpAffine` **cuts out a rotated rectangle** from the world image. `M` says, for
  every camera pixel, which world pixel to copy.
- **Forward** direction = `(cos th, sin th)`. **Right** direction = `(−sin th, cos th)`.
- Camera column 80 (the middle) is straight ahead. Columns below 80 are to the left; above 80 to the right.
- Outside the map the camera sees grass (`borderValue=GRASS`).

The yellow box drawn on the map in the window is exactly the area this function cuts out.

---

## Step 3: Computer vision (find the lane lines)

**Goal:** turn the camera picture into "where is the lane middle?" plus "how sure am I?".

### 3a. Prepare the picture (inside `Car.step`)

```python
gray = cv2.cvtColor(self.cam, cv2.COLOR_BGR2GRAY).astype(np.float32)      # colour -> brightness
gray = np.clip(gray + self.rng.normal(0, NOISE, gray.shape), 0, 255).astype(np.uint8)  # sensor noise
self.band = cv2.GaussianBlur(gray, (5, 5), 0)[BAND].astype(np.float32)     # blur, keep rows 40-99
```

| Operation | Why |
|---|---|
| Grayscale | We only need brightness (0 to 255), not colour |
| Add noise (`NOISE = 3`) | Real cameras are a little noisy; makes the test honest |
| Gaussian blur 5 × 5 | Smooths the noise so single noisy pixels don't look like paint |
| `BAND = slice(40, 100)` | Only look at **60 rows, 30 to 90 px ahead**, like a driver looking a little ahead |

### 3b. Mark the paint pixels: the threshold

```python
FACTORY_THRESHOLD = 180

def line_mask(band, thresholds):
    return band > thresholds[:, None]          # True where a pixel is brighter than its row's threshold
```

The result is a grid of **True/False**: True = "this pixel is lane paint".
(Each row can have its own threshold. That matters later for the memory feature.)

### 3c. From paint pixels to the lane middle and a confidence score

```python
def lane_centre(mask):
    xs = np.nonzero(mask)[1]                               # column of every paint pixel
    left, right = xs[xs < CAM_W // 2], xs[xs >= CAM_W // 2]  # split into left / right half
    confidence = float(mask.any(axis=1).mean())            # share of rows that contain ANY paint
    if len(left) and len(right):
        return (left.mean() + right.mean()) / 2, confidence  # middle between both lines
    if len(left):
        return left.mean() + LINE_OFFSET, confidence         # only left line: middle is 38 px right of it
    if len(right):
        return right.mean() - LINE_OFFSET, confidence
    return None, confidence                                  # nothing found
```

**Example:** left pixels average 40, right pixels average 140 → lane middle **90**.
If paint is found in 54 of the 60 rows → confidence **0.9 (90%)**.

---

## Step 4: The control loop (observe → decide → act → feedback)

**Goal:** steer toward the lane middle and move forward, every step.

```python
error = centre - CAM_W / 2             # + means the lane middle is to my right
self.th += KP * error                  # KP = 0.006: turn a little per pixel of error
self.x += self.speed * math.cos(self.th)   # move 4 px forward in the new direction
self.y += self.speed * math.sin(self.th)
```

**In simple words:**
- `error` = how many pixels the lane middle is away from the centre of the picture (80).
- The car turns by `0.006 × error` radians. A bigger error gives a sharper turn. This is a
  **proportional controller** (the "P" in PID).
- Then it moves 4 pixels forward. The **next** picture shows if the turn was right: that's **feedback**.

**Scoring (the car never sees this):**

```python
on_road = 0 <= xi < W and 0 <= yi < H and ROAD[yi, xi] > 0
self.errors.append(DIST_MIDDLE[yi, xi] if on_road else ROAD_HALF)
if not on_road:            self.status = 'OFF ROAD'      # crash
elif self.laps >= LAPS_GOAL: self.status = 'FINISHED'    # 2 laps done
```

Laps are counted by tracking the car's **angle around the oval centre** (`_angle()`):
one full turn of 2π = 1 lap.

✅ **Check:** at this stage, in daylight, the car drives 2 laps on its own, about 5 px from the lane middle on average.

---

## Step 5: Changing light (tunnel and storm)

**Goal:** make the environment change, so the simple vision rule breaks.

```python
class Lighting:
    def __init__(self, tunnel=None, storm=None):
        self.tunnel = tunnel   # (x_from, brightness): everything right of x_from is dark
        self.storm = storm     # (step, brightness): at that step the whole world dims

    def scene(self):
        light = np.ones((H, W), np.float32)            # 1.0 = full daylight everywhere
        if self.tunnel:
            light[:, self.tunnel[0]:] = self.tunnel[1]  # e.g. 0.2 = 20% light in the tunnel
        if self.storm_on:
            light *= self.storm[1]                      # e.g. 0.3 = whole world at 30%
        return (WORLD * light[..., None]).astype(np.uint8)   # darken the picture
```

**In simple words:** we multiply every pixel's brightness by a "light level".
Paint 255 × 0.2 = **51**. Road 60 × 0.2 = **12**. Now nothing is brighter than 180,
so the basic car finds no lines, drives straight and **crashes**.

- `update(step)` switches the storm on at its scheduled step: **the world changes while the car drives.**
- `toggle_tunnel()` / `toggle_storm()` are wired to keys **T** and **N** for live demos.
- The picture is only recalculated when the light changes (`_scene` cache), so it stays fast.

---

## Step 6: The novel feature (ask → learn → remember)

There are two brains: `Car('basic')` and `Car('smart')`. Both use the same camera,
vision and steering. Only the smart car has the three parts below.

### 6a. Ask: stop when blind

```python
ASK_BELOW = 0.25
if self.brain == 'smart' and self.confidence < ASK_BELOW:
    self.status = 'ASKING'          # stop moving; step() does nothing until a click
    self.why = 'I cannot see the lane ... Please click a white lane line in my camera view.'
    return
```

The basic car has no such check: with no lines found it sets `error = 0` and drives straight.

### 6b. Learn: one click → a new threshold

```python
def teach(self, u, v):                                # (u, v) = clicked camera pixel
    row = self.band[v - BAND.start]                   # that row of the picture
    paint = float(row[max(0, u - 3):u + 4].max())     # brightest pixel near the click (≈ 51)
    road = float(np.median(row))                      # typical pixel in the row = road (≈ 12)
    if paint - road < 8:
        return False                                  # not brighter than the road: reject click
    threshold = (paint + road) / 2                    # halfway: (51 + 12) / 2 ≈ 32
    self.memory[round(road, 1)] = threshold           # 6c: remember it for this light level
    self.status = 'driving'
    return True
```

**Why the median is the road:** in one row, most pixels are road and only a few are
paint, so the middle value (median) is almost always a road pixel.

**Why halfway:** the threshold must sit **between** road and paint. Halfway gives the
most room for noise on both sides.

After the click the car drives on, and **the very next picture re-checks confidence**.
If it's still below 25%, it asks again.

### 6c. Remember: memory by light level

```python
self.memory = {60.0: 180.0}       # start: daylight road 60 -> threshold 180 (factory)
# after one click in the tunnel:  {60.0: 180.0, 12.0: 31.5}

def thresholds(self, band):
    levels = np.array(list(self.memory))                    # [60, 12]
    values = np.array(list(self.memory.values()))           # [180, 31.5]
    road = row_light(band)[:, None]                         # road brightness of EACH row
    distance = np.abs(road - levels[None, :])               # how close each row is to each memory
    distance[values[None, :] <= road + 1] = np.inf          # sanity check: never pick a threshold
                                                            # that isn't brighter than the road
    return values[distance.argmin(axis=1)]                  # closest memory, row by row
```

**In simple words:** for every row of the picture, the car measures how bright the road
is and picks the remembered threshold whose light level is **closest**.
- Daylight row (road ≈ 60) → uses **180**.
- Tunnel row (road ≈ 12) → uses **32**, so **no click is needed on lap 2**.
- Storm row (road ≈ 18) → closest is 12 → uses **32**. Learning **transfers** to new places.

Choosing **per row** means a picture that is half daylight and half tunnel (at the
tunnel entrance) still works.

**Sanity check (a real bug we fixed):** a remembered threshold darker than the current
road would make the whole road look like paint. We skip those. This removed the last crash.

---

## Step 7: Scenarios and measurement

**Goal:** prove the feature helps, with numbers, fairly.

```python
SCENARIOS = {
    '1': ('Sunny day',                    lambda: Lighting()),
    '2': ('Dark tunnel on the right bend', lambda: Lighting(tunnel=(680, 0.2))),
    '3': ('Storm rolls in at step 300',    lambda: Lighting(storm=(300, 0.3))),
}
```

- **`simulated_human(car)`** stands in for a person during automatic tests. It looks at
  the same view in daylight to find a **real** paint pixel, then calls `car.teach()` on it.
- **`run_to_end(brain, lighting)`** drives until finished, crashed, or a **safe stop**
  (too dark even with help: paint 6 vs road 2).
- **`random_lighting(rng)`** makes a random tunnel (position and darkness) and a 70% chance of a storm.
- **`compare()`** runs both brains on the 3 scenarios and **the same 30 random worlds**
  (fixed seed `7`, so the results are the same every run) and prints the table.

```bash
python lane_car.py --compare      # about 40 seconds
```

| 30 random worlds | Basic | Smart |
|---|---|---|
| Crashed | 30 / 30 | **0 / 30** |
| Finished 2 laps | 0 / 30 | **27 / 30** |
| Safe stops | 0 | 3 / 30 |

---

## Step 8: The window, keys and mouse

- **`render(lighting, car)`** builds one image: the map on the left (scaled to 720 × 560)
  with the car (red box) and its camera area (yellow box), and on the right the camera
  picture, the detected paint, the memory, and the car's explanation ("why").
- **`main()`** is the live loop: every 30 ms it calls `lighting.update()`, `car.step()`,
  shows the image, and reads a key.
- **Mouse:** `cv2.setMouseCallback` gives the click position in the window. We convert it
  to camera pixels and call `car.teach(u, v)`:

```python
if 725 <= mx < 965 and 230 <= my < 410:                    # click inside the "vision" picture
    car.teach(int((mx - 725) * CAM_W / 240), int((my - 230) * CAM_H / 180))
```

(The vision picture is shown 1.5× bigger, 240 × 180, so we scale back to 160 × 120.)

---

## Settings you can change (try them!)

| Setting | Where | Default | What happens if you change it |
|---|---|---|---|
| `KP` | top of file | 0.006 | Higher = sharper steering (can wobble); lower = lazy (can drift off on curves) |
| `ASK_BELOW` | section 4 | 0.25 | Higher = asks for help sooner; lower = drives on with less evidence |
| `NOISE` | top of file | 3.0 | More camera noise makes vision harder |
| `SPEED` | top of file | 4.0 | Faster car, less time to react |
| Tunnel darkness | `SCENARIOS['2']` | 0.2 | Darker tunnel; below about 0.05 even a click can't help |
| `FACTORY_THRESHOLD` | section 3 | 180 | The basic rule; lower helps in the dark but can mistake road for paint |

After changing something, run `python lane_car.py --compare` to see the effect in numbers.

---

## Summary: the whole program in 6 lines

1. **Draw** an oval road (`build_world`).
2. **Cut out** a camera picture in front of the car (`camera`).
3. **Find** paint pixels with a threshold, then the lane middle and a confidence score (`line_mask`, `lane_centre`).
4. **Steer** by `0.006 × error` and move forward (`Car.step`).
5. **Darken** the world with a tunnel or storm (`Lighting`).
6. **When blind:** stop, learn a threshold from one click, and remember it by light level (`teach`, `thresholds`).
