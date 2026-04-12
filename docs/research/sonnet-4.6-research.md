<!-- Synthesized in docs/RESEARCH_SUMMARY.md. Raw source â€” 2026-04-12. -->
Great findings! Here is a comprehensive research summary across **GitHub**, **Hugging Face**, **academic sources**, and related communities on whether people have executed excavator-camera-based trailer/truck counting:

---

## ?? Research Summary: Excavator Camera ? Trailer/Truck Count Per Day

### ? Verdict: **No exact end-to-end solution exists publicly**, but ALL the building blocks are well-established and some are very close to your use case. Here's a platform-by-platform breakdown:

---

## ?? GitHub

### 1. **Smart Construction Site** *(Closest match to your use case)*
**Repo:** `arpitmaur07/Smart-Construction-Site`

This is a smart construction site tool built in Python that utilizes object detection and counting techniques to detect and count the number of workers and vehicles on the construction site in real-time. Object detection is done via YOLO, and counting via OpenCV.

The tool captures the live video stream from a camera placed on the construction site and detects and counts the number of workers and vehicles on the site.

?? **Gap:** Counts vehicles in frame but does **not** specifically track "trailer filled and driven away" events.

---

### 2. **YOLOv5 + DeepSort Vehicle Counting**
**Repo:** `charnkanit/Yolov5-Vehicle-Counting`

This is a computer vision and AI project to detect and count vehicles, using YOLOv5 and the Deep Sort Algorithm to perform object recognition and tracking in real time.

?? **Gap:** Designed for traffic counting, not construction loading cycles. But can be retrained on dump truck/trailer classes.

---

### 3. **YOLO Vehicle Counter (YOLOv3)**
**Repo:** `guptavasu1213/Yolo-Vehicle-Counter`

This project aims to count every vehicle detected in the input video using YOLOv3. When vehicles in the frame are detected, they are counted. Since YOLO misses a few detections for consecutive frames, the project resolves this by saving detections for the last 10 frames and comparing them to the current frame.

?? **Gap:** Count logic is per-frame entry, not per "loading completed" event.

---

### 4. **Construction Hazard Detection (YOLO + HuggingFace)**
**Repo:** `yihong1120/Construction-Hazard-Detection`

Enhances construction site safety using YOLO for object detection, identifying hazards like workers without helmets or safety vests, and proximity to machinery or vehicles. HDBSCAN clusters safety cone coordinates to create monitored zones, and post-processing algorithms improve detection accuracy.

The system is built for real-time deployment, offering instant analysis and alerts for identified hazards. It also integrates AI recognition results in real-time via a web interface.

?? **Gap:** Safety-focused, not loading-count focused. But the architecture (multi-stream YOLO + zone monitoring) is directly applicable.

---

### 5. **Construction Machine Dataset (YOLOv4)**
**Repo:** `miniexcav/Construction-Machines-Images-Dataset`

The dataset consists of 223 images with YOLO format label files. Despite the low image count, it achieves good performance when trained with YOLOv4.

Classes include excavators, trucks, bulldozers, wheel loaders, cranes, backhoe loaders, concrete mixture trucks, graders, compactors, and workers. — This covers **backhoe loaders and trucks**, exactly your use case!

---

### 6. **Pictor-YOLO (Construction Object Detection)**
**Repo:** `ciber-lab/pictor-yolo`

The Pictor-v2 dataset contains 1,105 crowd-sourced and 1,402 web-mined images of buildings, equipment, and workers. The best performing model is YOLOv3 trained on the combined dataset.

---

### 7. **Construction Progress Tracking (Material Counting)**
Listed under GitHub topic `construction-site`:

There is official code for *"Deep-Learning Based Construction Progress Tracking and Material Counting Using Existing Site Surveillance Camera"* — a project using deep learning, object detection for indoor construction site material tracking and counting construction progress.

?? This is the **most conceptually aligned** project with your idea — it explicitly does **material counting from site cameras**.

---

## ?? Hugging Face

### 1. **fuxi-robot/excavator-video** *(Extremely relevant dataset!)*

The dataset requires scenes that include both an excavator and a truck to be loaded, with the hydraulic arm and bucket clearly visible. It contains 10 different trucks, with each truck having 100 complete loading video trajectories, for a total of 1,000 video trajectories.

It covers 200+ distinct operational scenarios across 5 key industrial domains including mining, urban construction, and quarrying, with advanced multi-modal sensing: High-density LiDAR, 360° RGB-D cameras, boom/arm/bucket IMUs, hydraulic pressure and engine telemetry. It also covers varying material types (soil, gravel, rock), dynamic truck positioning, confined workspaces, and adverse weather & lighting.

?? **This is the closest dataset to your exact problem** — complete loading cycles, excavator+truck together!

---

### 2. **fuxi-robot/excavator-motion**

Excavator-motion is a large-scale dataset of excavator motion trajectories, encompassing three distinct excavator models. Each file records a complete motion trajectory from the initiation of digging to the full loading of a truck.

---

### 3. **FlywheelAI/excavator-dataset**

This dataset captures real excavator operations through synchronized multi-camera video recordings and corresponding joystick control inputs. — Useful for training loading-cycle detectors.

---

## ?? Academic Papers

### 1. **Bucket Fill Estimation (Faster R-CNN + Depth Maps)**

Excavators are crucial in the construction industry, and accurate estimation of the volume of excavator bucket fill is key for monitoring and evaluating system automation performance. This paper presents the use of 2D depth maps as input to a Faster R-CNN model for bucket volume estimation.

The proposed approach achieves **96.91% accuracy** in fill factor estimation and predicts in real time at about 10 fps.

---

### 2. **Video-Based Hydraulic Excavator + Dump Truck Detection**

Intelligent monitoring systems that aim at detecting hydraulic excavators and dump trucks on state-owned land have not been thoroughly explored. A paper presents an automatic, video-based algorithm for detecting hydraulic excavators and dump trucks, using foreground detection based on an improved frame difference algorithm.

Methods were proposed based on inverse valley feature of mechanical arm and spatial-temporal reasoning for hydraulic excavator detection, along with dump truck detection strategies that combine structured component projection with spatial relationships.

---

## ?? What's Missing for Your Exact Use Case

No one has publicly released a **complete, reliable pipeline** that does exactly this:
> *"Detect when excavator/backhoe has finished loading a trailer ? trailer departs ? count += 1 ? daily report"*

Here's what needs to be assembled:

| Component | Status |
|---|---|
| Detect excavator/backhoe in video | ? Well solved (YOLO) |
| Detect trucks/trailers in video | ? Well solved (YOLO) |
| Detect loading action (bucket dump) | ? Partially solved (fuxi-robot dataset, Faster R-CNN paper) |
| Track truck entry + exit from zone | ? Solvable (DeepSort + virtual line crossing) |
| Trigger "filled & departed" event | ?? Not publicly done end-to-end |
| Daily count report | ?? Not done in any public repo |

---

## ??? Recommended Stack to Build This

1. **Model:** YOLOv8 (Ultralytics) fine-tuned on construction datasets
2. **Tracking:** DeepSort or ByteTrack for truck ID persistence
3. **Loading Detection:** Use bucket-above-truck zone + arm motion heuristic OR fine-tune on `fuxi-robot/excavator-video` dataset
4. **Count Logic:** Virtual line crossing — when a truck crosses the exit line after being in the loading zone, `count += 1`
5. **Dataset:** `fuxi-robot/excavator-video` on Hugging Face + `miniexcav/Construction-Machines-Images-Dataset` on GitHub

Would you like me to help design the full architecture or code skeleton for this?