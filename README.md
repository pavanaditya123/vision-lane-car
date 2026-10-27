# Vision-Guided Lane-Following Car that Asks for Help and Remembers

**Topic:** Autonomous systems using computer vision. Fully simulated in Python + OpenCV, with no hardware.

📖 **New here? Read [EXPLANATION.md](EXPLANATION.md) first**: the whole project step by step, in simple English, with diagrams.
🛠️ How the code is built, step by step: [IMPLEMENTATION.md](IMPLEMENTATION.md).
🎤 Presenting as a team? See [TEAM_PRESENTATION.md](TEAM_PRESENTATION.md).

## 1. The problem:-

A self-driving car must stay in its lane. Our car has **only one sensor: a camera**.
It looks at the white lane lines and steers to stay between them.

The usual simple method decides "a pixel is a lane line if it is brighter
than 180". That works in daylight. But when the light changes (a dark
tunnel, a storm), the white paint looks grey, nothing is brighter than 180,
and the car becomes **blind**. Most simple lane followers then just keep
driving, and they crash.

## 2. Our novel feature

> When the camera cannot see the lane, our car **stops safely and asks a
> human to click once on a lane line**. From that **single click** it learns
> what paint and road look like in this light and computes a new vision
> threshold. It **remembers it for that light level**, so the same lighting
> never needs help again, even on a different part of the track or in a storm
> with similar light.

### Is it really new?

We searched for this exact feature. The single pieces exist, but **we did not
find any project, paper or patent that combines them like this**. We cannot
prove it is new worldwide; we can only say that our search found no prior work
doing it.

| Piece | Existing work we found | How ours is different |
|---|---|---|
| Robot asks for help when unsure | [Learning When to Ask for Help (arXiv 2305.16502)](https://alphaxiv.org/abs/2305.16502) | They ask an expert to take over. We **learn a vision setting from the answer**. |
| Human helps through a picture | [Autonomous robot with on-demand teleoperation (patent)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10824142) | The human picks an object. We use one click to **recalibrate lane vision**. |
| Remember where perception fails | [Gurău et al., "Learn from Experience", IJRR 2017](https://ori.ox.ac.uk/media/5703/2017ijrr_gurau.pdf) | They remember **places**. We remember **light levels**, so learning carries over to new places and new weather. |
| Slow or stop when perception is unsure | [arXiv 2103.00345](https://arxiv.org/pdf/2103.00345), [arXiv 2510.22680](https://arxiv.org/html/2510.22680v1) | They brake or slow down. We brake **and then learn from the human so it doesn't happen again**. |
| Fixed thresholds fail in shadows | [OpenCV forum: shadows in lane lines](https://answers.opencv.org/question/184167/shadow-removal-in-traffic-lane-lines/), [PyImageSearch: adaptive thresholding](https://pyimagesearch.com/2021/05/12/adaptive-thresholding-with-opencv-cv2-adaptivethreshold/), [ROS line follower limitation](https://github.com/chrisdalke/ros-line-follower-robot/blob/main/README.md) | This is the known problem we start from. |

## 3. How to run

You need Python 3.9 or newer. Nothing else.

**Easiest (Windows):** double-click **`run.bat`**.
The first time, it creates a virtual environment (`.venv`) and installs
`requirements.txt` (about 1 minute). After that it starts straight away.

**Manual (any OS):**

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
python lane_car.py
```

| Command | What it does |
|---|---|
| `python lane_car.py` (or `run.bat`) | Live demo window, starting in scenario 2 (tunnel) |
| `python lane_car.py 3` (or `run.bat 3`) | Start in scenario 3 (storm) |
| `python lane_car.py 4` (or `run.bat 4`) | Random road: 3 to 5 tunnels of mixed light plus surprise dark patches |
| `python lane_car.py --compare` (or `run.bat --compare`) | Measure Basic vs Smart (no window, about 3 minutes) |

Tested with OpenCV 4.13 and 5.0; both give identical results.

| Key | Action |
|---|---|
| `1` `2` `3` `4` | Load scenario |
| `G` | Generate a new random road with new tunnels and surprises |
| `B` | Switch Basic / Smart brain |
| `T` | Add or remove the dark tunnel (change the world live) |
| `N` | Start or stop a storm (change the world live) |
| `P` | Drop a dark patch right in front of the car (change the world live) |
| `H` | Auto-help: a simulated human clicks for you |
| `SPACE` | Pause |
| `R` | Restart |
| `Q` | Quit |
| **Mouse** | When the car asks (red frames, brightened **night view**), click **near** a grey lane line in **either** picture. It snaps to the line within 20 px, and a green circle shows where. Also works to rescue the car from a SAFE STOP. |

## 4. How it works

Every step the car runs one loop:

| Stage | What happens | Code |
|---|---|---|
| **Observe** | Take a 160×120 camera picture of the road 10 to 130 px ahead. Convert it to grayscale and add a little sensor noise. | `camera()`, start of `Car.step()` |
| **Decide (vision)** | Blur the picture. In the band 30 to 90 px ahead, mark pixels brighter than the threshold as **paint**. The left paint and right paint give the lane middle. **Confidence** = share of rows where paint was seen. | `line_mask()`, `lane_centre()` |
| **Decide (safety, novel)** | If confidence < 25% → **stop and ask for help**. | `Car.step()` |
| **Learn (novel)** | The human clicks a line. Paint brightness = brightest pixel near the click. Road brightness = the light around the click (median of the 31×31 area). New threshold = halfway between them. Store `road light → threshold` in memory. | `Car.teach()` |
| **Remember (novel)** | For **every pixel** of every new picture, measure the light around it (31×31 median; lines are thin, so this is the road) and use the remembered threshold for the closest light level. Skip any threshold that is not brighter than the road (sanity check). This works even when a tunnel edge or dark patch cuts diagonally across the picture. | `local_light()`, `Car.thresholds()` |
| **Fault tolerance** | If one click still doesn't bring the lines back, the car **stops safely** instead of asking forever. | `Car.step()` |
| **Act** | Steer by `0.006 × error` (error = lane middle minus picture middle), then move 4 px forward. | end of `Car.step()` |
| **Feedback** | The next picture shows whether the steering worked. For scoring only (the car never sees it): distance from the lane middle, laps, off-road. | end of `Car.step()` |

**Basic brain (for comparison):** same camera and steering, but always uses threshold 180 and keeps driving straight when it sees nothing.

## 5. Demonstration scenarios

| # | Scenario | Basic brain | Smart brain (ours) |
|---|---|---|---|
| 1 | Sunny day | ✅ 2 laps | ✅ 2 laps, 0 clicks (same as basic) |
| 2 | Dark tunnel on the right bend | ❌ off road at step 98 | ✅ asks **once** on lap 1, **0 clicks on lap 2**, finishes |
| 3 | Storm rolls in at step 300 (**the world changes**) | ❌ off road at step 339 | ✅ stops, asks once, learns, finishes |
| 4 | **Random road** with 3 to 5 tunnels of mixed light and surprise dark patches (press `G` for a new one) | ❌ off road | ✅ road #1: 2 clicks on lap 1, 0 on lap 2, finishes |

## 6. Measured result (`python lane_car.py --compare`)

Both brains drive exactly the same worlds (fixed random seeds), so the comparison is fair.

**Test A: 30 random oval worlds.** Each has a random tunnel (position and darkness) and a 70% chance of a storm at a random time.

| Metric | Basic | Smart (ours) |
|---|---|---|
| Crashed (left the road) | **30 / 30** | **0 / 30** |
| Finished 2 laps | 0 / 30 | **29 / 30** |
| Safe stop (too dark even with help) | 0 | 1 / 30 |
| Help clicks on lap 1 / lap 2 | – | 32 / 14 |
| Avg distance from lane middle | 8.6 px | 5.2 px |

**Test B: 30 random roads.** Each is a different curvy road with 3 to 5 tunnels (12% to 70% light, fading entrances), patchy ±15% light everywhere, and 3 surprise dark patches (8% to 50% light) that appear and disappear **while the car drives**.

| Metric | Basic | Smart (ours) |
|---|---|---|
| Crashed (left the road) | **30 / 30** | **0 / 30** |
| Finished 2 laps | 0 / 30 | **30 / 30** |
| Help clicks on lap 1 / lap 2 | – | 52 / 3 |
| Avg distance from lane middle | 9.8 px | 5.2 px |

- In Test B, **124 tunnels needed only 52 clicks**: similar tunnels share one memory.
- Lap-2 clicks happen only when something new appears in lap 2 (a storm, or a surprise patch with unseen light). In the fixed tunnel scenario lap 2 needs **0** clicks.

## 7. Limitations

- It is a simulation: a flat road (an oval or a random smooth loop), perfect white paint, a top-down camera, and simple noise.
- Light changes (tunnels, storm, patchy light, surprise dark patches) only make things darker or brighter. Real glare, rain, reflections and worn paint are harder.
- Random roads are smooth closed loops with no junctions, and bends no sharper than a 130 px radius.
- In automatic tests a **simulated human** clicks a correct line. Real people may click wrongly. The car rejects clicks that are not brighter than the road, but nothing more.
- The car needs a human nearby. A real car could ask a remote operator instead.
- 1 of 30 random oval worlds was so dark (tunnel plus storm) that the paint was nearly invisible. The car stops safely, but it cannot finish.
- Our novelty claim is "not found in our search", not "proven new".
