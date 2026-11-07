"""
Side-by-side streaming results of several checkpoints on whole LIBERO trajectories, eg
before vs after fine-tuning, optionally next to the SAM 3 masks.

Each model runs as at deployment (``SmoothSAVideoStream``, frame-0 bboxes once, one frame
at a time). Its slots are Hungarian-matched to the SAM segments over the whole trajectory
and drawn in their matched segment's color, so one object has one color in every panel;
unmatched slots are white, background (SAM's and its matched slot) dark navy. Writes
one mp4 per trajectory and an overview png (first / middle / last frame of a few trajectories).

    python visualize_compare.py --cfg_file config-smoothsa/smoothsav_c_masksupmatch_curric_ft-libero.py \
        --ckpt "before (MOVi-C)=archive-smoothsa/smoothsav_c-movi_c/best.pth" \
        --ckpt "after (fine-tuned)=save/smoothsav_c_masksupmatch_curric_ft-libero/42/0999.pth" \
        --data_name libero10 --indices 0-9,50-59 --sam_panel --out_dir compare/libero10
"""

from argparse import ArgumentParser
from pathlib import Path
import pickle as pkl

import cv2
import imageio
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import tqdm

from eval_libero_stream import parse_indices, read_metas
from object_centric_bench.datum.dataset import lmdb_open_read
from object_centric_bench.learn.callback_video import SaveValVideo
from object_centric_bench.learn.metric_masksup import match_slot_segment, pool_segment
from object_centric_bench.model import ModelWrap, SmoothSAVideoStream
from object_centric_bench.util import Config, build_from_config


@pt.inference_mode()
def stream_attent(model, sample, attent_key, seed):
    """Whole trajectory, one frame at a time. Returns (t,n,hh,ww) attention, (1,n) smask."""
    pt.manual_seed(seed)  # slot init samples noise
    video = sample["video"][None].cuda()
    bbox0 = sample["bbox"][None, :1].cuda()
    model.m.reset()
    attents = []
    for i in range(video.size(1)):
        with pt.autocast("cuda", enabled=True):
            output = model(batch=dict(video=video[:, i : i + 1], bbox=bbox0))
        attents.append(output[attent_key].float()[0])
    return pt.concat(attents, 0), output.get("smask", None)


def slot_masks(attent, segment, smask):
    """Slots as (t,h,w,s+u) bool, channel k = slot matched to segment k, then unmatched
    slots; plus matched-pair mean IoU (hard masks, whole trajectory)."""
    t, h, w, s = segment.shape
    n = attent.size(1)
    target = pool_segment(segment[None].cuda(), attent.shape[-2:])
    assign, _ = match_slot_segment(attent[None], target, smask)
    assign = assign[0].cpu()
    slot_of_seg = {int(k): int(j) for j, k in zip(*assign.nonzero(as_tuple=True))}
    unmatched = [j for j in range(n) if j not in slot_of_seg.values()]
    argmax = ptnf.interpolate(attent, size=[h, w], mode="bilinear").argmax(1).cpu()
    onehot = ptnf.one_hot(argmax, n).bool()  # (t,h,w,n)
    empty = pt.zeros(t, h, w, 1, dtype=pt.bool)
    order = [slot_of_seg.get(k, -1) for k in range(s)] + unmatched
    masks = pt.concat([onehot[..., j : j + 1] if j >= 0 else empty for j in order], -1)
    ious = []
    for k, j in slot_of_seg.items():
        p, g = onehot[..., j], segment[..., k]
        ious.append(((p & g).sum() / (p | g).sum().clamp_min(1)).item())
    return masks, len(unmatched), float(np.mean(ious)) if ious else 0.0


def read_names(dataset, indices):
    """object names per trajectory, in segment-channel order (from the LMDB)"""
    env = lmdb_open_read(dataset.data_file)
    names = []
    with env.begin(write=False) as txn:
        for i in indices:
            d = pkl.loads(txn.get(dataset.idxs[i]))["names"]  # {j+1: name}
            names.append([d[k] for k in sorted(d)])
    env.close()
    return names


def box_panel(rgb0, bbox0, names, color):
    """Frame 0 with the bboxes the slots are initialized from. Slot 0 is the
    background slot (padding-value bbox, nothing to draw); slot j >= 1 is foreground
    object j, drawn in its SAM segment's color.
    - rgb0: (h,w,3) uint8; bbox0: (n,4) ltrb, normalized; color: (>=n,3)"""
    img = rgb0.copy()
    h, w, _ = img.shape
    for j in range(1, bbox0.size(0)):
        l, t, r, b = (bbox0[j] * pt.tensor([w, h, w, h])).round().int().tolist()
        if r <= l and b <= t:
            continue  # not visible in frame 0: no box
        c = tuple(int(_) for _ in color[j])
        cv2.rectangle(img, (l, t), (r, b), c, 2)
        name = names[j - 1] if names and len(names) == bbox0.size(0) - 1 else f"slot {j}"
        name = name.replace("_1", "").replace("_", " ")[:18]
        y = t - 3 if t > 12 else b + 11
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            cv2.putText(img, name, (l + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 0, 0), 1)
        cv2.putText(img, name, (l, y), cv2.FONT_HERSHEY_SIMPLEX, 0.33, c, 1)
    SaveValVideo.label(img, "frame-0 boxes (slot init)")
    SaveValVideo.label(img, f"+ background slot (box -1): {bbox0.size(0)} slots", y=28)
    return img


def split_task_name(task_name):
    """'KITCHEN_SCENE4_put_the_bowl' -> ('KITCHEN_SCENE4', 'put the bowl')"""
    words = task_name.split("_")
    k = 0
    while k < len(words) and (words[k].isupper() or words[k].isdigit()):
        k += 1
    scene = "_".join(words[:k]) if k else ""
    return scene, " ".join(words[k:])


def header_band(width, lines, line_h=18):
    """dark strip with white text lines, above the panels"""
    band = np.full((line_h * len(lines) + 6, width, 3), 25, "uint8")
    for i, text in enumerate(lines):
        cv2.putText(band, text, (6, line_h * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return band


def parse_seen(specs):
    """['title=suite:0-7', ...] -> {title: {(suite, task), ...}}: models trained on those tasks"""
    seen = {}
    for spec in specs or []:
        title, _, rest = spec.partition("=")
        suite, _, tasks = rest.partition(":")
        seen.setdefault(title, set()).update((suite, t) for t in parse_indices(tasks))
    return seen


def main(args):
    cfg = Config.fromfile(args.cfg_file)
    cfg.dataset_v.base_dir = args.data_dir
    cfg.dataset_v.data_file = f"{args.data_name}/val.lmdb"
    if "transform0" in cfg.dataset_v:
        cfg.dataset_v.pop("transform0")  # whole trajectory
    dataset = build_from_config(cfg.dataset_v)
    indices = parse_indices(args.indices) if args.indices else list(range(len(dataset)))
    metas = read_metas(dataset, indices)
    all_names = read_names(dataset, indices)
    seen = parse_seen(args.seen)

    cfg.model.type = SmoothSAVideoStream
    models = []
    for spec in args.ckpt:
        name, ckpt = spec.split("=", 1)
        m = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
        m.load(ckpt, None, verbose=False)
        models.append((name, m.cuda().eval()))

    mean = np.array(cfg.IMAGENET_MEAN, "float32").reshape(3, 1, 1)
    std = np.array(cfg.IMAGENET_STD, "float32").reshape(3, 1, 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = min(args.overview_rows, len(indices))
    overview_idx = {indices[_] for _ in np.linspace(0, len(indices) - 1, rows).astype(int)}
    overview = []
    for idx, meta, names in zip(tqdm.tqdm(indices), metas, all_names):
        sample = dataset[idx]
        segment = sample["segment"]  # (t,h,w,s)
        t, h, w, s = segment.shape
        rgb = (sample["video"].numpy() * std + mean).clip(0, 255).astype("uint8")
        rgb = rgb.transpose(0, 2, 3, 1)

        panels = []  # (title, masks (t,h,w,c), colors)
        if args.sam_panel:
            panels.append(("SAM 3 masks", segment, SaveValVideo.palette(s, 0)[:s]))
        for name, model in models:
            attent, smask = stream_attent(model, sample, args.attent, args.seed)
            masks, nu, miou = slot_masks(attent, segment, smask)
            title = f"{name}  mIoU {miou:.2f}"
            if (meta.get("suite"), meta["task_idx"]) in seen.get(name, ()):
                title += "\n(trained on this task)"
            panels.append((title, masks, SaveValVideo.palette(s, nu)))

        name = f"{idx:03d}-{(meta['suite'] + '-' if meta.get('suite') else '')}task{meta['task_idx']}-demo{meta['demo_idx']:02d}"
        boxes = None
        if args.box_panel:  # static: what the slots start from
            boxes = box_panel(rgb[0], sample["bbox"][0], names, SaveValVideo.palette(s, 0))
        frames = []
        for f in range(t):
            row = [] if boxes is None else [boxes.copy()]
            for title, masks, color in panels:
                img = SaveValVideo.draw(rgb[f], masks[f].numpy(), args.alpha, color)
                for i, line in enumerate(title.split("\n")):
                    SaveValVideo.label(img, line, y=14 + 14 * i)
                row.append(img)
            frame = np.concatenate(row, 1)
            if args.header is None:
                SaveValVideo.label(row[0], f"{name}  frame {f}", y=h - 6)
            else:
                scene, instruction = split_task_name(meta.get("task_name", ""))
                num_obj = len(names)
                num_slot = sample["bbox"].size(1)  # background + objects visible in frame 0
                frame = np.concatenate([header_band(frame.shape[1], [
                    f"{meta.get('suite', args.data_name)}  |  task {meta['task_idx']}"
                    + (f"  |  {scene}" if scene else ""),
                    f"instruction: {instruction}",
                    f"trajectory: demo {meta['demo_idx']}  |  {args.header}  |  {num_obj} objects "
                    f"(incl. robot arm) + background = {num_slot} slots  |  frame {f + 1}/{t}",
                ]), frame], 0)
            frames.append(frame)
        imageio.mimwrite(args.out_dir / f"{name}.mp4", frames, fps=args.fps, macro_block_size=1)
        if idx in overview_idx:  # first / middle / last frame
            overview.append(np.concatenate([frames[p] for p in [0, t // 2, t - 1]], 1))
    if overview:
        sep = lambda img: np.concatenate([img, np.full((4, img.shape[1], 3), 255, "uint8")], 0)
        imageio.imwrite(args.out_dir / "overview.png", np.concatenate([sep(_) for _ in overview], 0))
    print(f"saved {len(indices)} videos and overview.png to {args.out_dir}")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg_file", type=Path, required=True)
    parser.add_argument(
        "--ckpt", action="append", required=True, help="'panel title=checkpoint path', repeatable"
    )
    parser.add_argument("--data_dir", type=Path, default=Path("."))
    parser.add_argument("--data_name", default="libero10")
    parser.add_argument("--indices", default="", help="eg '0-9,50-59'; '' = all")
    parser.add_argument("--sam_panel", action="store_true", help="add a SAM 3 masks panel")
    parser.add_argument(
        "--header", default=None,
        help="add a header (suite, task, instruction, demo, object count) with this test-set "
        "label, eg 'held-out task (new task)'",
    )
    parser.add_argument(
        "--seen", action="append",
        help="'panel title=suite:0-7': that model trained on these tasks; marked in its panel",
    )
    parser.add_argument(
        "--box_panel", action="store_true", help="add a first panel: frame 0 with the init bboxes"
    )
    parser.add_argument("--attent", default="attenta", choices=["attenta", "attentd"])
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--overview_rows", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
