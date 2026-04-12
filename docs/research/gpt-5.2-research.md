<!-- Synthesized in docs/RESEARCH_SUMMARY.md. Raw source â€” 2026-04-12. -->
## Summary
I looked for an **already-built, reliable, end-to-end open-source system** that takes **excavator-mounted camera video** and outputs **“# of trailers/trucks filled per day”** (i.e., counts completed loading cycles). **I did not find a widely used “turnkey” GitHub/Hugging Face repo that does exactly that**.

What I *did* find is the next-best thing:
- **Public datasets + benchmarks specifically for excavator ? dump-truck loading cycles** (very close to your use case).   
- **Construction equipment detection / multi-object tracking datasets and code** you can adapt.   
- **Vehicle counting (detection + tracking + line-cross counting) implementations** that solve the “count each truck once” problem (but usually assume a more fixed camera).   
- Evidence that **industry and academia have counted truck/trailer leaving events and bucket-load events with CV**, but those implementations are often **not open-sourced**.   

Below is a platform-by-platform dump of the most relevant things I found, plus what’s missing and how people typically assemble a reliable solution from these parts.

---

## Hugging Face (datasets/models/spaces): closest public building blocks

### 1) EXC (Excavator Truck-Loading) video dataset (very relevant)
**What it is:** A dataset explicitly curated around **“excavator loading a truck”** with **complete loading trajectories**. The dataset description says each clip is a “complete loading process” ending when the **truck bed is filled** and the **truck leaves**—which is essentially the event you want to count.   

**Why it matters for you:** If your excavator camera view is similar (arm/bucket visible + truck in frame), you can train an **event detector** (“loading cycle complete”) or a **temporal model** that detects “truck leaving after loading,” and count those per day.

Code link (HF page) (URL in code block as requested):
```text
https://huggingface.co/datasets/fuxi-robot/excavator-video
```
  

---

### 2) Excavator / dump-truck detection datasets
If you need to (re)train detection for your Indian sites (lighting, dust, local vehicle shapes), these help as starting points:

- **keremberke/excavator-detector** (labels include *excavators*, *dump truck*, *wheel loader*).   
```text
https://huggingface.co/datasets/keremberke/excavator-detector
```

- **aswin00000/ConstructionSite** (construction imagery; includes excavator + dump truck scenes).   
```text
https://huggingface.co/datasets/aswin00000/ConstructionSite
```

These don’t directly output “trailers filled,” but they reduce your cold-start time for detection.

---

### 3) Pretrained “construction site” weights (detection)
I found a Hugging Face repo publishing “SiteSense” weights that include classes like **Excavator** and **Dump Truck** (useful as an initialization point).   
```text
https://huggingface.co/Zaafan/sitesense-weights
```

---

### 4) “Vehicle counting” Spaces (general counting patterns)
There are multiple Spaces demonstrating YOLO-based counting and tracking patterns (again: generally not excavator-specific, but the counting logic is relevant). Example:   
```text
https://huggingface.co/spaces/noochila/Vehicle-Classification-Counting-YOLO-v8
```

---

## GitHub: relevant datasets + partial implementations (not end-to-end “trailer filled/day”)

### 1) CMOT-Dataset (Construction Multi-Object Tracking) — includes dump-truck & excavator
**What it is:** A dataset repo for **tracking** construction objects; the table of sequences explicitly includes **dump-truck** and **excavator**, across conditions like **night**, and even **moving/shaky** camera categories in the dataset listing.   

This is useful because “counting trailers filled” is really **(detection + tracking + event counting)**, and tracking is usually where “reliability” lives or dies.

```text
https://github.com/XZ-YAN/CMOT-Dataset
```
  

---

### 2) OpenConstruction (construction object detection framework)
**What it is:** “Official implementation” of a construction object detection framework (MMDetection-based) trained across multiple construction datasets and categories including dump truck and excavator classes (via referenced datasets).   

```text
https://github.com/YUZ128pitt/OpenConstruction
```
  

This is a solid “CV plumbing” base, but it’s not a finished “count loaded trailers per day” solution.

---

### 3) Generic YOLOv8 vehicle detection + tracking + counting logic (adaptable)
A representative example repo that does the classic pattern:
1) detect vehicles (YOLO),
2) track IDs across frames,
3) count when crossing entry/exit lines.

Example: **Subhadip7/yolov8-multiple-vehicle-detection**   
```text
https://github.com/Subhadip7/yolov8-multiple-vehicle-detection
```

This kind of repo solves the *count-once* logic, but you’d still need to define **what “filled” means** (more on that below).

---

### 4) “Object counting” guidance (Ultralytics-style patterns)
I also found an object-counting guide document mirrored on HF that describes YOLO-based counting/tracking patterns.   
This is not excavator-specific, but it’s the standard recipe most GitHub counting projects implement.

---

## Reddit (and forums): people discuss the exact “count once” problem, not excavator-specific completion logic
The most relevant Reddit signal I found is that when people ask “count dump trucks,” the standard advice is:

- YOLO gives per-frame detections, **not** “count once per vehicle”
- You need a **tracker** to maintain IDs across frames and count each truck once

Example thread: **“Count dump trucks going through our town”**   

This is directly applicable to your “trailers filled/day” metric because your count should be per *unique loading completion event*, not per-frame.

---

## “Reliable execution” evidence (but not open-source): industry + academic work shows it’s doable

### 1) Industry-style pilot: counting truck+trailer leaving and bucket-unload counts
A Building 4.0 CRC report (“Lighthouse Project #28 – The APPL”) explicitly lists validation metrics including:
- **Truck + Trailer Leaving** counts (AI vs manual)
- **Count of Buckets Unloaded to Trucks** (AI vs manual)
…and describes labeling/verification with CVAT.   

This is very close to your goal (daily count of loaded units leaving), but I did not find an associated public GitHub repo in the material I could access.

---

### 2) Academic approach: “# loaded dump trucks × capacity = productivity”
A 2025 paper describing a multi-vision productivity monitoring platform states that total earthwork volume can be calculated by multiplying the **number of loaded dump trucks** by capacity.   

That implies they solve (at least) “count loaded trucks,” even if their code isn’t openly posted.

---

### 3) “Full vs Empty” classification for trucks (directly related to “filled”)
A 2019 Applied Sciences (MDPI) paper is explicitly about **vision-based classification of full/empty-load trucks** in earthmoving operations (FEQS scenario).   

This supports a very practical architecture: count trucks at the exit/haul road camera and classify **full** vs **empty**.

---

## X (Twitter / x.com)
I attempted multiple searches targeting “excavator + dump truck + computer vision + counting / YOLO” and did not get usable results for this specific task in the search tool (either irrelevant hits or no indexed results). So: **nothing credible to report from X** for “excavator-camera ? count filled trailers” beyond generic CV chatter.

---

# What’s missing (the honest answer)
Across GitHub + Hugging Face + Reddit, I did **not** find a popular, well-documented repo where someone says (and demonstrates) something like:

> “Mount a camera on an excavator, run this repo, and it reliably counts filled trailers/trucks per day in production.”

Instead, the ecosystem is:
- **Open data & benchmarks** (EXC) to *build* such a system   
- **Tracking datasets** (CMOT) and detection frameworks (OpenConstruction) to assemble components   
- **Generic vehicle counting code** that needs your domain logic for “filled”   
- **Closed / not-open implementations** in pilots and papers   

---

## How people typically make this reliable anyway (practical recipe, using the found components)
Even though you asked for “has this already been done,” the fastest path in practice is to combine two concepts that *are* proven:

### Option A (most reliable in real sites): Count “FULL trucks leaving” at a chokepoint camera
1) Put a camera at a gate / haul road exit (fixed view).
2) Detect + track trucks.
3) Classify **full vs empty** (this is exactly what the MDPI paper category is about).   
4) Count only **full** leaving events per day.

This avoids the biggest excavator-camera problems: constant viewpoint motion, occlusion by boom, dust bursts right at dumping, etc.

### Option B (what you described): From excavator camera, count “completed loading cycles”
Typical robust logic looks like:
1) Detect **truck/trailer** + (optionally) bucket/arm.
2) Track the truck through the loading window.
3) Detect “loading interaction” / “bucket dump events” / “truck departure after loading”
4) Increment count when the tracked truck **exits** and loading events were observed.

This aligns well with EXC’s framing of complete loading trajectories that end when the truck leaves.   

If you go this route, CMOT-style tracking resources are relevant, especially if your excavator camera is shaky/moving.   

---

## Links (quick copy/paste)
```text
HF dataset (complete excavator?truck loading trajectories):
https://huggingface.co/datasets/fuxi-robot/excavator-video

GitHub tracking dataset (dump-truck + excavator):
https://github.com/XZ-YAN/CMOT-Dataset

GitHub detection framework (construction equipment):
https://github.com/YUZ128pitt/OpenConstruction

Example GitHub vehicle detect+track+count pattern:
https://github.com/Subhadip7/yolov8-multiple-vehicle-detection

HF detection dataset (excavator + dump truck):
https://huggingface.co/datasets/keremberke/excavator-detector
```

---

If your real goal is **operational reporting you can trust for billing/production**, the strongest pattern I’ve seen in the literature and industry pilots is: **count at a chokepoint (leaving events) and/or fuse with non-vision signals** (weighbridge, RFID, telematics), and treat excavator video as supporting evidence—because “excavator-mounted camera” is usually the hardest viewpoint to make 99% reliable in dust/night.