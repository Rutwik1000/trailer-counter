# Architecture Decision Records

> Append-only. Each ADR answers "why was this chosen instead of X?" Do not edit completed records. Add new ADRs at the bottom.

---

## ADR-001: Departure-Based Fill Counting

**Status:** Accepted

**Decision:** Any vehicle that enters the loading zone and then departs = 1 fill. No minimum dwell time. No partial/full distinction.

**Why:** Simplest reliable trigger. Alternative (minimum dwell time) requires per-site tuning and varies by trailer size; alternative (bucket-dump detection) requires a second model. Both add complexity without improving accuracy for the MVP.

**Consequence:** A vehicle that enters the zone briefly without being loaded would be counted. Acceptable on an active site — vehicles approach the excavator only for loading.

---

## ADR-002: Static Loading Zone Polygon

**Status:** Accepted

**Decision:** Operator draws a polygon on the first frame at installation. Saved to `config/loading_zone.json`. Used for all subsequent videos from that camera.

**Why:** The excavator cam rotates with the cab — the loading position is always in the same image region relative to the camera. One-time human calibration is cheaper and more reliable than dynamic arm-tracking.

**Consequence:** Re-calibration required if the camera is repositioned. The calibration tool is a notebook cell (interactive mouse-click, or manual coordinate entry for headless Kaggle environments).

---

## ADR-003: BoxMOT BotSort for Tracking

**Status:** Accepted

**Decision:** `pip install boxmot`, use `BotSort` with OSNet Re-ID weights.

**Why:** Appearance-based (not pure IoU), so robust to camera rotation. BoxMOT is a unified library supporting 8 trackers — swapping is one constructor argument change. Actively maintained (v17.0.0, March 2026).

**Alternatives rejected:**
- ByteTrack: pure IoU; fails when the excavator cab rotates because predicted positions become unreliable
- DeepSort: older, less maintained than BotSort

**License note:** BoxMOT is AGPL-3.0. Internal/research use is fine; commercial deployment requires licensing consideration.

---

## ADR-004: RF-DETR as Primary Detector Candidate

**Status:** Pending — to be evaluated in Phase 2

**Decision:** Evaluate RF-DETR (SiteSense weights, `pip install rfdetr`) and YOLOv8n (COCO pretrained) on 10 fuxi-robot frames. Manually score both. Commit to the higher-performing one.

**Why deferred:** Cannot select a detector without seeing both perform on actual cab-view footage. The `Detector` wrapper class accepts `model_type="yolo"` or `"rfdetr"` — switching costs zero code change after evaluation.

**Contingency C2A — if `Zaafan/sitesense-weights` RF-DETR weights are unavailable:**
- Use `ultralytics/yolov8n.pt` (COCO pretrained) as the primary detector candidate
- Evaluate YOLOv8n-COCO vs YOLOv8n fine-tuned on `keremberke/excavator-detector` in Phase 2
- Update this ADR status to: "YOLOv8 only — SiteSense RF-DETR weights unavailable"
- No code changes needed: the `Detector` wrapper already supports `model_type="yolo"`

---

## ADR-005: Day-Scoped Re-ID Gallery

**Status:** Accepted

**Decision:** The Re-ID gallery dict resets at the start of each day's processing run. No cross-day persistence.

**Why:** Cross-day persistence requires robust embedding stability across large lighting/weather/dust variation over multiple days. Without real-footage validation this would be a reliability liability. MVP requirement is daily counts only.

**Consequence:** No cross-day vehicle identity tracking. If tracking the same vehicle across days is needed, that is post-MVP work.

---

## ADR-006: DINOv3 + SiteSense Projection Head for Re-ID

**Status:** Accepted — model ID and dimensions to be confirmed in Phase 0B

**Decision:** Use `facebook/dinov3-vitb16-pretrain-lvd1689m` (86M params, CLS token → 1536-d) + SiteSense head (`dinov3_reid_head.pth`: Linear 1536→256 → ReLU → Linear 256→128 → L2 normalize → 128-d).

**Why:** SiteSense head is already domain-adapted to construction equipment from overhead/aerial views (96.8% accuracy on 12k construction contrastive pairs). No fine-tuning required for MVP.

**Alternatives rejected:**
- torchreid OSNet: trained on pedestrian data — wrong domain, features may not transfer well
- Custom trained Re-ID: no labeled Re-ID data available for this specific camera setup

**⚠ Phase 0B verification required:** The model ID `facebook/dinov3-vitb16-pretrain-lvd1689m` must be confirmed accessible on HuggingFace before Phase 5. Update the BACKBONE_ID and BACKBONE_DIM values below after running Phase 0B Task 0B.1 Step 2.

```
BACKBONE_ID  = [CONFIRM IN PHASE 0B]   # e.g. facebook/dinov3-vitb16-... or facebook/dinov2-base
BACKBONE_DIM = [CONFIRM IN PHASE 0B]   # e.g. 1536 (DINOv3) or 768 (DINOv2-base)
```

**Contingency C1A — if backbone ID is DINOv2:**
- Replace all `facebook/dinov3-vitb16-pretrain-lvd1689m` references with `facebook/dinov2-base`
- CLS token dimension: 768-d (not 1536-d)
- Projection head becomes: `Linear(768→256) → ReLU → Linear(256→128) → L2 norm`
- Update `ARCHITECTURE.md` Re-ID Pipeline section with new dimensions

**Contingency C2B — if SiteSense Re-ID head weights are unavailable:**
- Use DINOv2 backbone CLS token directly as embedding (no projection head)
- Dimension: 768-d L2-normalized vector stored in gallery
- Cosine similarity on raw backbone features (accuracy may be lower — validate in Phase 5)
- Update this ADR status: "SiteSense head unavailable — raw DINOv2 backbone embeddings used"

---

## ADR-007: Centroid-Based Zone Containment Test

**Status:** Accepted

**Decision:** Check if the centroid `((x1+x2)/2, (y1+y2)/2)` of the bounding box is inside the polygon using `cv2.pointPolygonTest`.

**Why:** Simple, fast, deterministic. Correct when the vehicle is partially outside the zone boundary.

**Alternatives rejected:**
- IoU with polygon: complex to compute for irregular polygon shapes
- Overlap percentage threshold: requires tuning; adds a parameter with no clear default

---

## ADR-008: fuxi-robot Dataset for Development

**Status:** Accepted

**Decision:** Use `fuxi-robot/excavator-video` (HuggingFace) with `streaming=True` on Colab; full download on Kaggle.

**Why:** Only public dataset with complete loading trajectories confirmed from an excavator-mounted camera. 1,000 videos, 10 trucks, day/night, adverse weather and lighting — closely matches the operating environment.

**Constraint:** 85.5 GB total size. This drives the Kaggle platform selection (107 GB disk).

---

## ADR-009: Kaggle Notebook as Primary Platform

**Status:** Accepted

**Decision:** Kaggle Notebook primary; Google Colab + Drive mount as fallback.

**Why:** The fuxi-robot dataset is 85.5 GB. Kaggle provides 107 GB persistent disk and T4/P100 GPU at no cost. Colab's disk is ~80 GB ephemeral — insufficient to hold the full dataset without Drive mounting overhead.
