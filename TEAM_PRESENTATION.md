# Team Presentation Guide (3 people)

**Project:** Vision-Guided Lane-Following Car that Asks for Help and Remembers
**Total time:** about 10 to 12 minutes, plus questions

## The whole story in 5 lines (everyone must know this)

1. Our car drives using **only a camera**: it finds the white lane lines and steers between them.
2. **Problem:** when the light changes (tunnel, storm), the simple vision rule "brighter than 180 = paint" stops working, the car goes blind and crashes.
3. **Our novel feature:** when blind, the car **stops safely, asks a human to click one lane line, learns a new vision setting from that one click, and remembers it for that light level**.
4. **Result:** in 30 random dark worlds, the basic car crashed **30 of 30** times; our car crashed **0 of 30** times.
5. **Honest claim:** the pieces exist separately, but we found **no prior work that combines them**. We say "not found in our search", not "first in the world".

## Who does what

| Person | Part | Slides | Time |
|---|---|---|---|
| **M. Pavan Aditya** | Title, the problem, existing work, our novelty | 1 to 5 | ~3.5 min |
| **Hartik** | How the system works (camera → vision → decision → action → learning) and the harder tests | 6 to 10 | ~4 min |
| **Zafar Mohammad** | Live demo, results, limitations, conclusion | 11 to 14 | ~4 min |

**Slide deck:** the 14-slide deck has every line of this script in its speaker notes. The "Slide 1–9" headings below are the shorter outline; in the deck, M. Pavan Aditya presents slides 1–5, Hartik 6–10 (slide 10: random roads and harder tests), and Zafar Mohammad 11–14.

**Laptop setup before you start (Zafar Mohammad does this):**
open a terminal in the project folder, run `python lane_car.py`, then press **SPACE** to pause on the first frame. Keep the README open in a second window in case the demo fails.

## M. PAVAN ADITYA (Person A): The problem and our novelty

### Slide 1: Title
**Show:** project title, the three names, one screenshot of the window.

**Say:**
> "Good morning. Our project is a self-driving car that uses computer vision. The car has only one sensor, a camera, just like a person driving with only their eyes. Our car has one special skill: when it can't see, it stops, asks for help, learns from one click, and remembers what it learned. I'll explain the problem, Hartik will explain how it works, and Zafar will show you a live demo and the results."

**Notes for A:**
- Speak slowly. This slide sets up the whole talk.
- "Autonomous" means it makes its own decisions in a loop: **observe → decide → act → feedback**.

### Slide 2: The problem
**Show:** two camera pictures side by side, one in daylight (bright white lines) and one in the tunnel (grey lines, almost invisible).

**Say:**
> "To find the lane lines, a simple program uses one rule: if a pixel is brighter than 180, it is white paint. In daylight, paint is about 255 bright and the road is about 60, so the rule works. But in a dark tunnel everything gets darker. The paint is now only about 51, so nothing is brighter than 180. The car sees zero lane lines. The car is blind. A simple car just keeps driving straight and crashes. This is a known problem; people on the OpenCV forum and in tutorials report that fixed thresholds fail with shadows and changing light."

**Notes for A:**
- The numbers: paint **255**, road **60** in daylight; paint **51**, road **12** in the tunnel.
- The known-problem sources are the OpenCV forum and PyImageSearch (both in the README).

### Slide 3: What exists, and what is new in ours
**Show:** the comparison table from README section 2.

**Say:**
> "We searched for existing work. Some robots ask for help when they are unsure, but a human then takes control; the robot doesn't learn anything. Some research remembers *places* where the camera failed. Some cars slow down when unsure. Our idea combines these in a new way. Our car asks for help, but it **learns a vision setting from a single click**, and it remembers that setting by **light level**, not by place. That means what it learns in the tunnel also helps later in a storm with similar light. We did not find any project, paper or patent that does this exact combination. We are careful here: we say 'not found in our search', not 'first in the world'."

**Handover line (A → B):**
> "So how does a camera picture become a steering decision, and how does one click teach the car? Hartik will explain."

**Notes for A:**
- If someone asks "Is it really new?": "The single parts exist; the combination of one-click learning and memory by light level was not in anything we found. We list the closest works in our README."

## HARTIK (Person B): How the system works

### Slide 4: The autonomous loop
**Show:** a simple loop diagram: **Camera → Vision → Decision → Steering → (car moves) → Camera …**

**Say:**
> "Our car repeats one loop about 30 times per second. Step 1, **observe**: it takes a small camera picture, 160 by 120 pixels, of the road in front of it. Step 2, **decide**: computer vision finds the white lines. Left line plus right line gives the middle of the lane. Step 3, **act**: if the lane middle is to the right, steer right; if it's to the left, steer left. Then move forward a little. Step 4, **feedback**: the next picture shows if the steering worked, and the loop starts again. Nobody presses any buttons; the car does this on its own."

**Notes for B:**
- Vision steps: grayscale → blur (removes noise) → threshold (bright = paint) → find left and right line → lane middle.
- Steering: `turn = 0.006 × error`. Error = how many pixels the lane middle is away from the picture's centre.
- It looks only at a band **30 to 90 pixels ahead**, like a driver looking a little ahead, not at their feet.

### Slide 5: Confidence: the car knows when it is blind
**Show:** the vision picture with good lines (100% confidence) next to one with no lines (0 to 20%).

**Say:**
> "We added a confidence score: the percentage of rows in the picture where the car found any lane line. In daylight it's about 100%. When it drives into the tunnel, the score drops. Our rule: if confidence falls below 25%, the car does not guess. It **stops safely** and shows a red box: 'Please click a white lane line.' The basic car has no such rule; it drives straight and hopes."

**Notes for B:**
- Threshold for stopping: **25%** (`ASK_BELOW = 0.25`).
- "Stopping is the safe choice when you can't see. A human driver also slows down or stops in thick fog."

### Slide 6: Learn from one click, then remember (the novel part)
**Show:** the red-box screenshot, then the "Thanks! … new threshold is 32" message and the memory line `60->180 12->32`.

**Say:**
> "When the human clicks on a line, the car looks at that spot. The brightest pixel near the click is the paint, about 51 in the tunnel. The typical pixel in that row is the road, about 12. The car picks a new threshold halfway between them: 32. Now it can see the lines again. Then the important part: it **saves this in memory as 'road brightness 12 → threshold 32'**. From then on, for every row of every picture, it checks how bright the road is and uses the remembered threshold for the closest light level. So on lap 2, when it reaches the tunnel again, it doesn't ask; it already knows. We also added a safety check: it never uses a remembered threshold that is darker than the road, because then the whole road would look like paint."

**Handover line (B → C):**
> "That's the theory. Now Zafar will show you the car doing it live, and the numbers that prove it works."

**Notes for B:**
- Formula: **threshold = (paint + road) / 2**.
- Memory starts with one entry, the factory daylight setting: **60 → 180**.
- If the human clicks the wrong place (not brighter than the road), the car says so and asks again.
- Code functions to point at if asked: `Car.step()`, `Car.teach()`, `Car.thresholds()`.

## ZAFAR MOHAMMAD (Person C): Demo, results, limits, conclusion

### Slide 7: Live demo
**Do the demo in this order:**

| Step | Press | What the class sees | What C says |
|---|---|---|---|
| 1 | `B` (switch to BASIC), then SPACE | Car drives into the tunnel, confidence drops to 0%, car crashes | "This is the basic car with the fixed rule. In the tunnel it goes blind… and crashes." |
| 2 | `B` (back to SMART) | Car drives to the tunnel and **stops**, red box appears | "Same tunnel, our car. It sees the confidence drop and stops safely. It's asking me for help." |
| 3 | **Click** a grey line inside the red box | Message: "paint 51, road 12, new threshold 32" | "One click. It learned the new threshold and continues through the tunnel." |
| 4 | Let it drive to lap 2 | Passes the tunnel **without stopping**; help clicks stay at 1 | "Lap 2: no question this time. It remembered." |
| 5 | Press `N` (storm) | The whole world gets darker | "Now the environment changes: a storm. Watch: it adapts, or asks once for this new light, then continues." |
| 6 | Press `4`, then `H` (auto-help) | A **random curvy road** with 3 to 5 labelled tunnels (T1 25%, T2 60% …) | "It's not memorising one oval. Every road is new, with tunnels of different darkness. Similar tunnels share one memory, so it needs fewer clicks than tunnels." |
| 7 | Press `P` | A pink circle: a **dark patch drops right in front of the car** | "A surprise while driving. It handles it, or asks once, and never drives blind." Press `G` for another new road if time allows. |

**Notes for C:**
- If the click misses, the car says "Please click a white line"; just click again on the grey line.
- If the demo breaks: press `R` to restart, or `H` so a simulated human answers automatically.
- Backup: show the screenshots/numbers from the README.

### Slide 8: Results
**Show:** run `python lane_car.py --compare` before class and put this table on the slide.

| Metric (30 random dark worlds) | Basic | Ours |
|---|---|---|
| Crashed | **30 / 30** | **0 / 30** |
| Finished 2 laps | 0 / 30 | **28 / 30** |
| Safe stop (too dark even with help) | 0 | 2 / 30 |
| Help clicks lap 1 / lap 2 | – | 32 / 13 |

**Say:**
> "To be fair we tested both cars on the same 30 random worlds, each with a random tunnel and often a storm at a random time. The basic car crashed every time. Our car never crashed. It finished 28 worlds. In 2 worlds it was so dark that even a human couldn't see the paint, so it stayed stopped, which is the safe choice. We also tested 30 random curvy roads, each with 3 to 5 tunnels of different darkness and surprise dark patches: the basic car crashed on all 30, ours finished all 30. Those roads had 124 tunnels but needed only 51 clicks, because similar tunnels share one memory. And it needed fewer clicks on lap 2 than on lap 1, because it remembered. The clicks on lap 2 happen only when a storm came during lap 2, bringing new lighting."

**Notes for C:**
- The whole test takes about 40 seconds. Run it before class, not live.
- "Same 30 worlds for both" = fair comparison (fixed random seed).

### Slide 9: Limitations and conclusion
**Say:**
> "Our project has limits. It's a simulation: a flat oval road with clean white paint. Real roads have glare, rain and faded paint. It needs a human nearby to click; a real car could ask a remote operator. The automatic tests use a simulated human who always clicks correctly. And our novelty is 'not found in our search', not proven. To conclude: a car that knows when it can't see, stops instead of crashing, learns from one click, and remembers. Simple idea, measurable result: 30 crashes became 0. Thank you, and we're happy to take questions."

## Questions: who answers

| Question | Who | Simple answer |
|---|---|---|
| Why is this "autonomous" if a human helps? | **M. Pavan Aditya** | It drives by itself all the time. It asks only when it is blind, rarely, and learns so it asks less later. Real self-driving cars also use remote human help. |
| Why not just use adaptive thresholding (OpenCV)? | **Hartik** | That's a good alternative for normal light. But when it's very dark, adaptive methods also guess, and noise can look like lines. Our car stops and gets one reliable answer, and it knows it is unsure instead of guessing. |
| What if the human clicks the wrong place? | **Hartik** | The car checks that the click is brighter than the road. If not, it says "please click a white line" and waits. |
| Why remember by light level instead of by place? | **Hartik** | Light level transfers: what it learned in the tunnel also works in a storm with similar light, even in a new place. |
| Why 25% confidence? | **Hartik** | Below 25% it sees lines in fewer than 1 in 4 rows, which is too little to steer safely. It's a design choice; we could tune it. |
| Is it really novel? | **M. Pavan Aditya** | The pieces exist (asking for help, remembering failures, slowing down when unsure). We found no work combining one-click learning with memory by light level. We show the closest works in our README. |
| Why 2 safe stops? | **Zafar Mohammad** | The tunnel plus storm made the paint almost as dark as the road (brightness 6 vs 2). Nobody could see it, so stopping is correct. |
| Does it work on a real car? | **Zafar Mohammad** | Not tested. It's a simulation. Next steps: real camera images, curved and faded lines, a remote operator instead of a mouse click. |

## How to prepare tonight (about 30 minutes each)

| Time | Everyone does |
|---|---|
| 0 to 5 min | Run `python lane_car.py`, watch scenario 2, press `B` to compare both brains. |
| 5 to 10 min | Learn "The whole story in 5 lines" at the top of this file. |
| 10 to 25 min | Practise **your own part** out loud twice, including your handover line. |
| 25 to 30 min | Read the Questions table; each person learns their own answers. |

**Zafar Mohammad, extra:** practise the demo steps twice and run `--compare` once to see the table.
**Hartik, extra:** open `lane_car.py` and find `Car.step()`, `Car.teach()` and `Car.thresholds()`, so you can show them if asked.
