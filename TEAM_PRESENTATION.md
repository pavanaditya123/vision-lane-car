# Team Presentation Guide (4 people)

**Project:** Vision-Guided Lane-Following Car that Asks for Help and Remembers
**Total time:** about 12 to 14 minutes, plus questions
**Slide deck:** 14 slides. Every line below is also in each slide's speaker notes.

## The whole story in 5 lines (everyone must know this)

1. Our car drives using **only a camera**: it finds the white lane lines and steers between them.
2. **Problem:** when the light changes (tunnel, storm), the simple vision rule "brighter than 180 = paint" stops working, the car goes blind and crashes.
3. **Our novel feature:** when blind, the car **stops safely, asks a human to click near one lane line, learns a new vision setting from that click, and remembers it for that light level**.
4. **Result:** in 60 random dark worlds, the basic car crashed **60 of 60** times; our car crashed **0 of 60** times and finished 59.
5. **Honest claim:** the pieces exist separately, but we found **no prior work that combines them**. We say "not found in our search", not "first in the world".

## Who does what

| Person | Slides | Part | Time |
|---|---|---|---|
| **M. Pavan Aditya** | 1–5 | Title, team, the problem, why the tunnel breaks vision, our novelty | ~3.5 min |
| **Hartik** | 6–8 | How it works: the loop, the confidence score, learning from one click | ~3 min |
| **Varada Reddy** | 9–11 | Memory by light level, harder random-road tests, measured results | ~3 min |
| **Zafar Mohammad** | 12–14 | Live demo, limits, conclusion and questions | ~4 min |

**Laptop setup before you start (Zafar Mohammad does this):** double-click `run.bat`, click once on the car window, check Caps Lock is OFF, then press **SPACE** to pause. Keep the README open in a second window in case the demo fails.

---

## M. PAVAN ADITYA: slides 1–5 (the problem and our novelty)

### Slide 1: Title
> "Good morning. Our project is a self-driving car that uses computer vision. The car has only one sensor, a camera, just like a person driving with only their eyes. Our car has one special skill: when it can't see, it stops, asks for help, learns from one click, and remembers. I'll explain the problem, Hartik will explain how it works, Varada will explain how it remembers and show the results, and Zafar will show you a live demo."

### Slide 2: Team
> "We split the talk into four parts. I'll start with the problem and our novelty. Hartik explains how the car sees, decides and learns from one click. Varada explains how it remembers, how we tested it on harder roads, and the results. Zafar shows the live demo and wraps up."

### Slide 3: One sensor, a camera
> "Our car must stay between two white lines. Each step it asks: where is the left line, where is the right line, and which way should I steer? The camera picture is 160 pixels wide, so its centre is 80. If the left line is at 40 and the right at 140, the lane middle is 90, which is 10 pixels to the right, so it steers right. Then it moves a little, takes a new picture and repeats. This observe, decide, act loop is what makes it autonomous."

### Slide 4: Same rule, darker world
> "A simple program uses one rule: a pixel brighter than 180 is paint. In daylight, paint is about 255 and the road about 60, so it works. In a dark tunnel, paint is only 51 and road 12. Nothing is brighter than 180, so the car sees zero lines. It is blind. The lane is still there; only the rule failed. A simple car keeps driving straight and, in our test, crashes at step 98."

### Slide 5: What is new
> "Some robots ask for help when unsure, but then a human takes over and the robot learns nothing. Some research remembers *places* where the camera failed. Some cars slow down when unsure. Our car combines these: it **learns a vision setting from one click** and **remembers it by light level, not by place**. We found no project, paper or patent doing this exact combination. We say 'not found in our search', not 'first in the world'."

**Handover (Pavan → Hartik):** "So how does a camera picture become a steering decision, and how does one click teach the car? Hartik will explain."

**Notes for Pavan:** numbers to remember are daylight paint **255** / road **60**, tunnel paint **51** / road **12**, rule **180**. "Autonomous" means it decides by itself in a loop.

---

## HARTIK: slides 6–8 (how it works)

### Slide 6: One loop, repeated every step
> "The car repeats one loop about 30 times a second. One, observe: take a 160 by 120 camera picture of the road ahead. Two, see: pixels brighter than the threshold are paint. Three, decide: the left and right lines give the lane middle; minus 80 gives the error. Four, act: turn by 0.006 times the error and move forward 4 pixels. Then the next picture shows if it worked. Nobody presses a button."

### Slide 7: A confidence score
> "We added a confidence score: the share of the 60 rows the car checks where it found any lane line. In daylight it's about 100%. In the tunnel it drops. Our rule: below 25%, the car does not guess. It stops safely and asks for help. The basic car has no such rule; it drives straight and hopes."

### Slide 8: One click teaches a new threshold
> "When the car stops, both pictures turn into a brightened night view, so the human can see the lines. The human clicks **near** a grey line; it doesn't need to be exact, because the car snaps to the clearest line within 20 pixels and shows a green circle there. The paint there is about 51, the road around it about 12, so the new threshold is halfway: 51 plus 12, divided by 2, is about 32. Now the road is below 32 and the paint above it, so the car can see again. That's learning in a simple sense: one number updated by human feedback, no neural network."

**Handover (Hartik → Varada):** "But the clever part is what happens next time. Varada will explain how the car remembers."

**Notes for Hartik:** formula **threshold = (paint + road) / 2**; stop rule **25%**; steering **0.006 × error**. If asked for code: `Car.step()`, `Car.teach()`.

---

## VARADA REDDY: slides 9–11 (memory, harder tests, results)

### Slide 9: It remembers by light level, not place
> "Thanks, Hartik. Now the important part: the car saves what it learned in memory, 'road light 12 means threshold 32'. For every pixel of every picture it checks how bright the road is around that pixel and uses the remembered threshold for the closest light level. So on lap 2 it doesn't ask; it already knows: 1 click on lap 1, 0 on lap 2. Because it remembers light, not place, a storm or another tunnel with similar light reuses it too. Safety check: it never uses a threshold darker than the road. Honest note: memory resets when the program closes. Next: does this only work on one oval?"

### Slide 10: Not one oval: random roads, mixed light
> "So we made the test much harder. Every road is a new random curvy loop with 3 to 5 tunnels, each with its own darkness, from 12% to 70% light. Entrances fade in gradually, the light is patchy, and surprise dark patches appear while the car drives; those are the pink circles. On 30 random roads the car finished all 30 with zero crashes. Those roads had 124 tunnels but needed only 52 clicks, because similar tunnels share one memory. These tests also found real weaknesses, which we fixed: the car now judges light around each pixel, and if one click doesn't help, it stops safely instead of asking forever. Now let's look at the measured numbers."

### Slide 11: Crashes went from 60 to 0
> "We tested both cars on the same 60 random worlds: 30 ovals with a random tunnel and often a storm, and 30 random roads with tunnels and surprise patches. The basic car crashed every time, 60 out of 60. Our car never crashed and finished 59. In 1 world it was so dark that even a human couldn't see the paint, so it stayed stopped, which is the safe choice. Finished laps matter too: a car could avoid every crash by never moving."

**Handover (Varada → Zafar):** "Numbers are one thing; now Zafar will show you the car doing all of this live."

**Notes for Varada:** memory starts as **60 → 180**; after the tunnel it's **60 → 180, 12 → 32**; key numbers are **0/60 crashes, 59/60 finished, 1 safe stop, 124 tunnels, 52 clicks**. Run `python lane_car.py --compare` once before class (about 3 minutes) to see it yourself.

---

## ZAFAR MOHAMMAD: slides 12–14 (demo, limits, close)

### Slide 12: Live demo
Start with: "Thanks, Varada. Now let's watch it live."

| Step | Press | What the class sees | What Zafar says |
|---|---|---|---|
| 1 | `B` (Basic), then SPACE | In the tunnel confidence drops to 0% and the car crashes | "The basic car with the fixed rule goes blind… and crashes." |
| 2 | `B` (Smart) | The car stops at the tunnel; red frames and a night view appear | "Same tunnel, our car. It stops safely and asks me for help." |
| 3 | **Click near** a grey line in either picture | A green circle; "new threshold ≈ 32"; it drives on | "One click, not even exact. It learned and continues." |
| 4 | Wait for lap 2 | Passes the tunnel without stopping; help clicks stay at 1 | "Lap 2: no question this time. It remembered." |
| 5 | `N` (storm) | The whole world gets darker | "The environment changes. It asks once for this new light, then continues." |
| 6 | `4`, then `H` | A random curvy road with labelled tunnels | "Every road is new. Similar tunnels share one memory." |
| 7 | `P`, then `-` a few times | A pink dark patch ahead, getting darker live | "A surprise while driving. It handles it, or asks once, and never drives blind." Tip: drop it just before a bend. `G` gives a new road. |

**If something goes wrong:** "No lane line near your click" means click closer to a grey line. SAFE STOP means a click near a line still rescues it. `R` restarts. `H` lets the computer click for you.

### Slide 13: Limits and next steps
> "Our project has limits. It's a simulation: smooth roads with clean white paint, even though we test random shapes, many tunnels and surprise dark patches. Real roads have glare, rain and faded paint. It needs a human nearby to click; a real car could ask a remote operator. Our tests use a simulated human who always clicks correctly, and memory resets when the program closes. Next we'd try real camera footage, a remote operator, saving memory to a file, and comparing against a version with no memory."

### Slide 14: Thank you
> "To conclude: a car that knows when it can't see, stops instead of crashing, learns from one click, and remembers. Simple idea, measurable result: across 60 random worlds, crashes went from 60 to 0. Thank you, and we're happy to take questions."

---

## Questions: who answers

| Question | Who | Simple answer |
|---|---|---|
| Why is this "autonomous" if a human helps? | **M. Pavan Aditya** | It drives by itself all the time. It asks only when it's blind, and learns so it asks less later. Real self-driving cars also use remote human help. |
| Is it really novel? | **M. Pavan Aditya** | The pieces exist (asking for help, remembering failures, slowing when unsure). We found no work combining one-click learning with memory by light level. |
| Why not just use adaptive thresholding (OpenCV)? | **Hartik** | Good in normal light, but in very dark scenes it also guesses and noise can look like lines. Our car knows it's unsure, stops, and gets one reliable answer. |
| What if the human clicks the wrong place? | **Hartik** | It snaps to the clearest line within 20 px. If there's no line nearby, it says "No lane line near your click" and waits. |
| Why 25% confidence? | **Hartik** | Below 25% it sees lines in fewer than 1 in 4 rows, too little to steer safely. It's a design choice we could tune. |
| Why remember by light level instead of by place? | **Varada Reddy** | Light carries over: what it learned in the tunnel also works in a storm or another tunnel with similar light, even in a new place. |
| Does it only work on one oval? | **Varada Reddy** | No: 30 random roads with 3–5 tunnels and surprise dark patches, all 30 finished, 0 crashes. |
| Why 1 safe stop? | **Varada Reddy** | Tunnel plus storm made the paint almost as dark as the road (brightness 6 vs 2). Nobody could see it, so stopping is correct. |
| Does it work on a real car? | **Zafar Mohammad** | Not tested; it's a simulation. Next: real camera images, faded lines, a remote operator instead of a mouse click. |
| Which keys did you press? | **Zafar Mohammad** | B switches the car, N storm, 4 random road, P dark patch, - darker, H auto-help, R restart. |

## How to prepare tonight (about 30 minutes each)

| Time | Everyone does |
|---|---|
| 0 to 5 min | Double-click `run.bat`, watch scenario 2, press `B` to compare both cars. |
| 5 to 10 min | Learn "The whole story in 5 lines" at the top of this file. |
| 10 to 25 min | Practise **your own slides** out loud twice, including your handover line. |
| 25 to 30 min | Read the Questions table; learn your own answers. |

**Zafar Mohammad, extra:** practise the 7 demo steps twice with the real mouse and keys.
**Varada Reddy, extra:** run `python lane_car.py --compare` once and match its numbers to slide 11.
**Hartik, extra:** open `lane_car.py` and find `Car.step()` and `Car.teach()`, so you can show them if asked.
