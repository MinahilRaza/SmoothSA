"""
Streaming evaluation of a bbox-conditioned SmoothSAVideo on whole LIBERO trajectories,
as in deployment: slots are initialized once from frame-0 bboxes, then
``SmoothSAVideoStream`` carries them frame by frame to the end of the episode.

Metrics are computed per trajectory over its whole (t h w) volume, so a slot switching
objects mid-trajectory is penalized, then averaged over trajectories. Reported for both
the decoder attention (``attentd``, what training val scores) and the encoder slot
attention (``attenta``).
"""

from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
import json
import pickle as pkl
import random
import time

import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import imageio
import tqdm
from torch.utils.data import Subset

from object_centric_bench.datum import DataLoader
from object_centric_bench.datum.dataset import lmdb_open_read
from object_centric_bench.learn import MetricWrap
from object_centric_bench.model import ModelWrap, SmoothSAVideo, SmoothSAVideoStream
from object_centric_bench.util import Config, build_from_config
from object_centric_bench.util_datum import draw_segmentation_np
from object_centric_bench.util_model import interpolat_argmax_attent


@pt.inference_mode()
def eval_trajectory(model, acc_fn, batch, chunk, resolut0):
    """
    - batch.video: (b=1,t,c,h,w); batch.segment: (1,t,h,w,s); batch.bbox: (1,t,n,c=4)
    Returns {"<attent>.<metric>": float} and {"<attent>": (t,h,w,n) bool pred, on cpu}.
    """
    video = batch["video"].cuda()
    segment = batch["segment"].cuda()
    bbox0 = batch["bbox"][:, :1].cuda()  # frame-0 bboxes only: the single initialization
    t = video.size(1)

    model.m.reset()
    attents = defaultdict(list)
    for i in range(0, t, chunk):
        with pt.autocast("cuda", enabled=True):
            output = model(batch=dict(video=video[:, i : i + chunk], bbox=bbox0))
        for key in ["attenta", "attentd"]:
            attents[key].append(output[key].float())  # (1,c,n,h,w)

    accs, preds = {}, {}
    for key, attent in attents.items():
        attent = pt.concat(attent, 1)  # (1,t,n,h,w)
        n = attent.size(2)
        pred = ptnf.one_hot(interpolat_argmax_attent(attent, size=resolut0).long(), n)
        acc = acc_fn(output=dict(segment=pred.bool()), batch=dict(segment=segment))
        preds[key] = pred[0].bool().cpu()
        for name, (value, valid) in acc.items():
            if valid.all():
                accs[f"{key}.{name}"] = value.mean().item()
    return accs, preds


def save_videos(video_dir, video, segment, mean, std, fps, alpha=0.6):
    """
    Final slots over the input video, as in ``rough/eval_libero_dataset.py``:
    ``all_slots.mp4`` with each slot in its own color, plus ``slotNN.mp4`` with that
    slot alone (green). Slot 0 is background, slot j+1 is frame-0 bbox j.
    - video: (t,c,h,w) normalized; segment: (t,h,w,n) bool, argmax slot per pixel
    """
    video_dir.mkdir(parents=True, exist_ok=True)
    rgb = (video * std + mean).clip(0, 255).byte().permute(0, 2, 3, 1).numpy()
    segment = segment.numpy()
    n = segment.shape[-1]
    green = np.array([[0, 255, 0]])
    imageio.mimwrite(
        video_dir / "all_slots.mp4",
        [draw_segmentation_np(f, s, alpha) for f, s in zip(rgb, segment)],
        fps=fps, macro_block_size=1,
    )
    for j in range(n):
        imageio.mimwrite(
            video_dir / f"slot{j:02d}.mp4",
            [draw_segmentation_np(f, s[:, :, j : j + 1], alpha, green) for f, s in zip(rgb, segment)],
            fps=fps, macro_block_size=1,
        )


def main(args):
    pt.manual_seed(args.seed)  # slot init samples noise even in eval mode
    random.seed(args.seed)
    np.random.seed(args.seed)

    cfg = Config.fromfile(args.cfg_file)
    assert issubclass(cfg.model.type, SmoothSAVideo)

    cfg.dataset_v.base_dir = args.data_dir
    # val transforms (no flip) either way; train split only to inspect fitting
    cfg.dataset_v.data_file = f"{args.data_name}/{args.split}.lmdb"
    if "transform0" in cfg.dataset_v:
        cfg.dataset_v.pop("transform0")  # whole trajectory, no clip
    dataset = build_from_config(cfg.dataset_v)
    indices = parse_indices(args.indices) if args.indices else list(range(len(dataset)))
    dataset = Subset(dataset, indices)
    dataload = DataLoader(
        dataset,
        1,  # trajectories differ in length
        shuffle=False,
        num_workers=cfg.num_work,
        collate_fn=build_from_config(cfg.collate_fn_v),
        pin_memory=True,
    )

    cfg.model.type = SmoothSAVideoStream
    cfg.model.history_stride = args.history_stride
    model = ModelWrap(build_from_config(cfg.model), cfg.model_imap, cfg.model_omap)
    model.load(args.ckpt_file, None, verbose=False)
    model = model.cuda().eval()

    # segmentation metrics only (ARI, mBO, mIoU), computed per attention below;
    # training monitors that read other model outputs are skipped
    acc_cfg = {k: v for k, v in cfg.acc_fn_v.items() if v.map.input == "output.segment"}
    acc_fn = MetricWrap(detach=True, **build_from_config(acc_cfg))

    per_traj = []
    t0 = time.time()
    num_frame = 0
    metas = read_metas(dataset.dataset, indices)
    if args.video_dir:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        mean = pt.tensor(cfg.IMAGENET_MEAN)
        std = pt.tensor(cfg.IMAGENET_STD)
    for i, batch in enumerate(tqdm.tqdm(dataload)):
        accs, preds = eval_trajectory(model, acc_fn, batch, args.chunk, cfg.resolut0)
        t = batch["video"].size(1)
        num_frame += t
        meta = metas[i]
        per_traj.append(dict(index=indices[i], **meta, num_frame=t, **accs))
        if args.video_dir:
            name = f"{indices[i]:03d}-{(meta['suite'] + '-' if meta.get('suite') else '')}task{meta['task_idx']}-demo{meta['demo_idx']:02d}"
            save_videos(
                args.video_dir / name, batch["video"][0], preds[args.video_attent],
                mean, std, args.fps,
            )

    keys = sorted({k for _ in per_traj for k in _ if "." in k})
    mean = {k: float(np.mean([_[k] for _ in per_traj if k in _])) for k in keys}
    elapsed = time.time() - t0
    for k, v in mean.items():
        print(f"{k}: {v:.4f}")
    print(f"{len(per_traj)} trajectories, {num_frame} frames, {num_frame / elapsed:.1f} frame/s")

    out_file = args.out_file or Path(args.ckpt_file).with_suffix(".stream.json")
    result = dict(
        cfg_file=str(args.cfg_file),
        ckpt_file=str(args.ckpt_file),
        chunk=args.chunk,
        history_stride=args.history_stride,
        seed=args.seed,
        data_name=args.data_name,
        split=args.split,
        indices=args.indices,
        mean=mean,
        per_traj=per_traj,
    )
    Path(out_file).write_text(json.dumps(result, indent=1))
    print(f"saved to {out_file}")


def read_metas(dataset, indices):
    """task/demo ids of the LIBERO trajectories at ``indices``, straight from the LMDB"""
    env = lmdb_open_read(dataset.data_file)
    metas = []
    with env.begin(write=False) as txn:
        for i in indices:
            sample = pkl.loads(txn.get(dataset.idxs[i]))
            keys = ["suite", "task_idx", "demo_idx", "task_name"]  # suite: multi-suite LMDBs
            metas.append({k: sample[k] for k in keys if k in sample})
            metas[-1]["suite"] = sample.get("suite", "")  # multi-suite LMDBs
    env.close()
    return metas


def parse_indices(spec):
    """'0-9,50-59' -> [0..9, 50..59]: dataset indices (LMDB order) of the trajectories"""
    indices = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        indices.extend(range(int(lo), int(hi or lo) + 1))
    return indices


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--cfg_file", type=Path, required=True)
    parser.add_argument("--ckpt_file", type=Path, required=True)
    parser.add_argument("--data_dir", type=Path, default=Path("."))
    parser.add_argument(
        "--data_name", default="libero10", help="LMDB dir under data_dir, eg libero_spatial"
    )
    parser.add_argument("--split", default="val", choices=["val", "train"])
    parser.add_argument(
        "--chunk", type=int, default=1, help="frames per forward call; 1 = online"
    )
    parser.add_argument("--history_stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--indices", type=str, default="", help="eg '0-9,50-59'; '' = all trajectories"
    )
    parser.add_argument("--out_file", type=Path, default=None)
    parser.add_argument(
        "--video_dir", type=Path, default=None, help="save slot overlay mp4s per trajectory"
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video_attent", default="attentd", choices=["attentd", "attenta"],
        help="whose argmax gives the slots drawn: decoder (default, as in eval) or encoder",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
