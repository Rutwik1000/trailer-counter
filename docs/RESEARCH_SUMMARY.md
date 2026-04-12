# Research Summary

> Synthesized from two parallel research runs. Raw sources: `docs/research/gpt-5.2-research.md`, `docs/research/sonnet-4.6-research.md` (2026-04-12).

## Verdict

- **No public end-to-end solution exists** for "excavator cab-camera → count filled trailers per day."
- **All required building blocks are available** individually and have been validated in research or industry pilots.

## Relevant Public Resources

| Resource | Type | Relevance |
|---|---|---|
| `fuxi-robot/excavator-video` (HuggingFace) | Dataset | 1,000 complete loading cycles, 10 trucks, from excavator-mounted camera, day/night/adverse |
| `Zaafan/sitesense-weights` (HuggingFace) | Model weights | RF-DETR + DINOv3 Re-ID head domain-adapted to aerial construction sites, mAP@50=0.834 |
| `keremberke/excavator-detector` (HuggingFace) | Dataset | 2,656 overhead-view images, COCO format, excavator + dump truck classes |
| `CMOT-Dataset` (GitHub: XZ-YAN) | Dataset | Multi-object tracking with dump-truck + excavator sequences |
| Building 4.0 CRC APPL Report | Industry pilot | Validated truck-leaving + bucket-unload counts vs manual — closed source |
| MDPI 2019: Full/Empty Truck Classification | Paper | Vision-based full vs empty truck state — relevant to loading confirmation |

## Gap Analysis

| Component | Publicly Solved? | Our Approach |
|---|---|---|
| Detect trucks in video | Yes (YOLO, RF-DETR) | SiteSense RF-DETR weights |
| Track trucks across frames | Yes (BotSort, ByteTrack) | BoxMOT BotSort |
| Define loading zone | No standard | Operator-drawn polygon (one-time calibration) |
| Count departures after loading | Not end-to-end | State machine in `event_counter.py` |
| Re-identify same vehicle | Partial (pedestrian Re-ID) | DINOv3 ViT-B/16 + SiteSense projection head |
| Daily count report | None found | JSON output + Streamlit dashboard |

## Why No Off-the-Shelf Solution

- Industry implementations (CRC APPL) are closed-source
- Open-source vehicle counters target fixed traffic cameras, not rotating cab-mounted cameras
- Loading cycle detection requires domain-specific zone logic no generic library provides

## Dataset Decision

`fuxi-robot/excavator-video` was chosen as the primary dataset: the only public dataset with complete loading trajectories, confirmed from an excavator-mounted camera perspective, 1,000 videos covering adverse conditions. Total size: 85.5 GB — drives the Kaggle platform requirement (107 GB disk).
