"""
Offline VFM-derived bounding-box proposal generation, used to condition
``NormalMlpPreheated``-style slot initialization (see smoothsa.py) on
something better than a shared learned Gaussian, for datasets (eg LIBERO)
that have no ground-truth segmentation/bbox to condition on instead.

Uses SAM (via the HF `transformers` "mask-generation" pipeline) to propose
class-agnostic object masks, reduced to bboxes in the exact same format
MOVi's ground-truth bboxes use: ltrb, both-side normalized to [0,1],
missing/padded slots filled with -1 (see dataset_movi.py, NormalMlpPreheated
in smoothsa.py).
"""

import numpy as np


def build_sam_mask_generator(model_name="facebook/sam-vit-base", device=None):
    from transformers import pipeline
    import torch as pt

    if device is None:
        device = 0 if pt.cuda.is_available() else -1
    return pipeline("mask-generation", model=model_name, device=device, points_per_batch=64)


def _box_iou(a, b):
    l = max(a[0], b[0])
    t = max(a[1], b[1])
    r = min(a[2], b[2])
    bo = min(a[3], b[3])
    if r <= l or bo <= t:
        return 0.0
    inter = (r - l) * (bo - t)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(boxes, iou_thresh):
    """boxes: list of (area, l,t,r,b), sorted by area descending."""
    kept = []
    for box in boxes:
        ltrb = box[1:]
        if all(_box_iou(ltrb, k[1:]) < iou_thresh for k in kept):
            kept.append(box)
    return kept


def generate_sam_bboxes(
    image: np.ndarray,
    mask_generator,
    max_boxes=8,
    points_per_side=32,
    max_area_frac=0.10,
    min_area_frac=0.002,
    border_margin=3,
    nms_iou=0.5,
) -> np.ndarray:
    """
    - image: (h,w,3) uint8, RGB
    - returns: (max_boxes,4) float32, ltrb both-side normalized to [0,1],
        sorted by mask area descending, padded with -1 if fewer than
        max_boxes proposals survive filtering.

    Filtering rationale (calibrated on real LIBERO frames):
    - max_area_frac: SAM's single largest mask is consistently the
        table/background surface, not an object; excluded by an upper
        area-fraction bound.
    - border_margin: masks touching the frame border are almost always
        walls/counters/the robot mount rather than tabletop objects.
    - nms_iou: SAM's grid-of-points prompting frequently produces multiple
        near-duplicate masks for the same object from different prompt
        points; deduplicated by bbox IoU, keeping the larger one.
    """
    from PIL import Image

    h, w = image.shape[:2]
    out = mask_generator(Image.fromarray(image), points_per_side=points_per_side)
    masks = out["masks"]

    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if len(xs) == 0:
            continue
        area = len(xs)
        frac = area / (w * h)
        if frac > max_area_frac or frac < min_area_frac:
            continue
        l, r, t, b = xs.min(), xs.max(), ys.min(), ys.max()
        if l <= border_margin or t <= border_margin or r >= w - 1 - border_margin or b >= h - 1 - border_margin:
            continue
        boxes.append((area, float(l), float(t), float(r), float(b)))

    boxes.sort(key=lambda _: -_[0])
    boxes = _nms(boxes, nms_iou)
    top = boxes[:max_boxes]

    bbox = np.full((max_boxes, 4), -1, dtype="float32")
    for i, (area, l, t, r, b) in enumerate(top):
        bbox[i] = [l / w, t / h, r / w, b / h]
    return bbox
