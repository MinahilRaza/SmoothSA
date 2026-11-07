"""
LIBERO dataset for loading and processing trajectories.
"""

from pathlib import Path
import pickle as pkl

from einops import rearrange
from tqdm import tqdm
import cv2
import numpy as np
import torch as pt
import torch.nn.functional as ptnf
import torch.utils.data as ptud

from .dataset import lmdb_open_read, lmdb_open_write
from ..util_datum import mask_segment_to_bbox_np, draw_segmentation_np


class LIBERO(ptud.Dataset):
    """
    Example
    ```
    dataset = LIBERO(
        data_file="libero10/train.lmdb",
        extra_keys=["segment", "bbox"],
        transform0=dict(type=Slice1, keys=["video", "segment"], dim=0, start=0, end=6),
        transform=dict(type=Compose, transforms=[...]),
        base_dir=Path(...),
    )
    for sample in dataset:
        dataset.visualiz(
            video=sample["video"].permute(0, 2, 3, 1).numpy(),
            segment=sample["segment"].numpy(),
            bbox=sample["bbox"].numpy(),
        )
    ```
    """

    def __init__(
        self,
        data_file,
        extra_keys=["segment", "bbox"],
        transform0=lambda **_: _, 
        transform=lambda **_: _,
        base_dir: Path = None,
        bbox_bg=None,
    ):
        """
        - bbox_bg: bbox prepended for a background slot, eg ``[-1,-1,-1,-1]`` (padding
            value, MOVi convention) or ``[0,0,1,1]`` (full frame); then slot 0 is
            background and slot j+1 is foreground object j. None: no background slot
        """
        if base_dir:
            data_file = base_dir / data_file
        self.data_file = data_file

        env = lmdb_open_read(data_file)
        with env.begin(write=False) as txn:
            self.idxs = pkl.loads(txn.get(b"__keys__"))
        env.close()

        self.extra_keys = extra_keys
        self.transform0 = transform0
        self.transform = transform
        self.bbox_bg = bbox_bg

    def __getitem__(self, index):
        """
        - video: (t,c,h,w), uint8, rgb
        - segment: (t,h,w,s), uint8 -> bool
        - bbox: (t,s,c=4), float32. both side normalized ltrb, only foreground
        """
        if not hasattr(self, "env"):  # torch>2.6
            self.env = lmdb_open_read(self.data_file)

        with self.env.begin(write=False) as txn:
            sample0 = pkl.loads(txn.get(self.idxs[index]))
        sample0 = self.transform0(**sample0)  # slice video's frame window in advance, for efficiency
        sample1 = {}

        video0 = np.array(  # rgb; imencode/imdecode round-trip preserves channel order as-is
            [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in sample0["video"]]
        )
        video = pt.from_numpy(video0).permute(0, 3, 1, 2)
        sample1["video"] = video  # (t,c,h,w) uint8

        if "segment" in self.extra_keys:
            segment0 = np.array(
                [cv2.imdecode(_, cv2.IMREAD_UNCHANGED) for _ in sample0["segment"]]
            )
            segment = pt.from_numpy(segment0)
            sample1["segment"] = segment  # (t,h,w) uint8
            s0 = 1 + sample0["s"]  # bg+fg

        sample2 = self.transform(**sample1)

        if "segment" in self.extra_keys:
            segment2 = sample2["segment"]  # (t,h,w); index format
            # (t,h,w,s); mask format
            segment2_ = ptnf.one_hot(segment2.long(), s0).bool()

            t, h, w, _ = segment2_.shape

            # ``RandomCrop`` and ``CenterCrop`` can diminish segments
            cond = segment2_.any([0, 1, 2])  # (s,)
            segment3 = segment2_[:, :, :, cond]
            sample2["segment"] = segment3  # (t,h,w,s) bool

            if "bbox" in self.extra_keys:
                segment3_ = rearrange(  # skip bg
                    segment3[:, :, :, 1 if cond[0] else 0 :], "t h w s -> h w (t s)"
                )
                bbox2_ = pt.from_numpy(  # (t*s,c=4)
                    mask_segment_to_bbox_np(segment3_.numpy())
                ).float()
                bbox2 = rearrange(bbox2_, "(t s) c -> t s c", t=t)
                bbox2[:, :, 0::2] /= w  # normalize
                bbox2[:, :, 1::2] /= h
                if self.bbox_bg is not None:
                    bbox_bg = pt.tensor(self.bbox_bg, dtype=bbox2.dtype)
                    bbox2 = pt.concat([bbox_bg.expand(t, 1, 4), bbox2], 1)
                sample2["bbox"] = bbox2  # (t,s,c=4) float32

        return sample2

    def __len__(self):
        return len(self.idxs)

    @staticmethod
    def convert_dataset(
        masks_dir,
        dst_dir: Path,
        val_tasks=(),  # tasks to hold out entirely, eg [8, 9] or [("libero_goal", 3)]
        val_traj_ids=(),  # (task_idx,demo_idx) or (suite,task_idx,demo_idx) to hold out
    ):
        """
        Packs the SAM-3-masked trajectories written by
        `ocl-vla/data-prep/generate_sam_masks.py` into `dst_dir/{train,val}.lmdb`.

        `masks_dir` is one directory or a list of them, eg several LIBERO suites. Each is
        `.../masked/<suite>`; the suite name (its directory name) is stored per sample,
        since task indices repeat across suites. `val_tasks` / `val_traj_ids` entries
        without a suite match any suite.

        Reads every `<masks_dir>/<task_idx:02d>_<demo_idx:03d>.npz` (each has
        `video (t,h,w,3)` uint8 and `masks (t,k,h,w)` uint8/bool -- see that
        script), splits them train/val by `val_tasks` / `val_traj_ids`
        (everything else -> train), composites each trajectory's per-object
        masks into a single per-frame index map (largest object painted
        first, so smaller ones win overlaps), and writes the LMDBs in the
        MOVi convention: `video` + `segment` as lists of per-frame JPEG/PNG
        bytes, plus `s` (foreground instance count).
        """
        masks_dirs = [masks_dir] if isinstance(masks_dir, (str, Path)) else masks_dir
        masks_dirs = [Path(_) for _ in masks_dirs]
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        val_task_set = set(tuple(_) if isinstance(_, (list, tuple)) else _ for _ in val_tasks)
        val_traj_set = set(tuple(_) for _ in val_traj_ids)

        npz_files = []  # (suite, path)
        for d in masks_dirs:
            files = sorted(d.glob("*.npz"))
            if not files:
                raise FileNotFoundError(
                    f"no .npz trajectories under {d} -- run "
                    "ocl-vla/data-prep/render_libero_trajectories.py then "
                    "ocl-vla/data-prep/generate_sam_masks.py first"
                )
            npz_files += [(d.name, p) for p in files]
        npz_files.sort(key=lambda _: (_[0], _[1].name))
        ids = [(suite, p.stem) for suite, p in npz_files]
        assert len(ids) == len(set(ids)), "same trajectory in several masks_dirs"

        train_files, val_files = [], []
        for suite, p in npz_files:
            task_idx, demo_idx = (int(_) for _ in p.stem.split("_"))
            is_val = (
                task_idx in val_task_set
                or (suite, task_idx) in val_task_set
                or (task_idx, demo_idx) in val_traj_set
                or (suite, task_idx, demo_idx) in val_traj_set
            )
            (val_files if is_val else train_files).append((suite, p))
        print(f"train: {len(train_files)} trajectories, val: {len(val_files)} trajectories")

        jpeg_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        png_param = [
            cv2.IMWRITE_PNG_COMPRESSION, 9,
            cv2.IMWRITE_PNG_STRATEGY, cv2.IMWRITE_PNG_STRATEGY_FILTERED,
        ]

        for split, files in [("train", train_files), ("val", val_files)]:
            if not files:
                print(f"{split}: no trajectories, skipping")
                continue

            env = lmdb_open_write(dst_dir / f"{split}.lmdb")
            keys = []
            txn = env.begin(write=True)

            for i, (suite, p) in enumerate(tqdm(files, desc=f"writing {split}")):
                d = np.load(p, allow_pickle=False)
                video = d["video"]  # (t,h,w,3) uint8 rgb
                masks = d["masks"]  # (t,k,h,w) uint8/bool
                names = [str(_) for _ in d["names"]]
                k = masks.shape[1]
                if k == 0:
                    print(f"  skip {p.name}: no tracked objects")
                    continue
                segment = __class__._masks_to_segment(masks)

                key = f"{i:06d}".encode("ascii")
                sample = dict(
                    suite=suite,
                    task_idx=int(d["task_idx"]),
                    demo_idx=int(d["demo_idx"]),
                    task_name=str(d["task_name"]),
                    names={j + 1: n for j, n in enumerate(names)},
                    video=[cv2.imencode(".jpg", f, jpeg_param)[1] for f in video],
                    segment=[cv2.imencode(".png", s, png_param)[1] for s in segment],
                    s=k,  # foreground instance count, see __getitem__
                )
                txn.put(key, pkl.dumps(sample))
                keys.append(key)

                if (i + 1) % 64 == 0:
                    txn.commit()
                    txn = env.begin(write=True)

            txn.put(b"__keys__", pkl.dumps(keys))
            txn.commit()
            env.close()
        print("Done.")

    @staticmethod
    def _masks_to_segment(masks):
        """(t,k,h,w) uint8/bool per-object masks -> (t,h,w) uint8 index map
        (0=bg, j+1 for object j). Objects are painted largest-first (by
        frame-0 mask area) so smaller ones, painted after, win any overlap."""
        t, k, h, w = masks.shape
        seg = np.zeros((t, h, w), dtype=np.uint8)
        order = np.argsort(-masks[0].reshape(k, -1).astype(np.int64).sum(1))
        for ti in range(t):
            for j in order:
                m = masks[ti, j]
                if m.any():
                    seg[ti][m.astype(bool)] = int(j) + 1
        return seg

    @staticmethod
    def visualiz(video, segment=None, bbox=None, wait=0):
        """
        - video: (t,h,w,c=3) uint8, rgb format
        - segment: (t,h,w,s) bool, mask format
        - bbox: (t,s,c=4) float32, ltrb format, dual normalized
        """
        t, h, w, cv_ = video.shape
        assert cv_ == 3 and video.dtype == np.uint8

        if bbox is not None:
            bbox = (bbox.copy() * [w, h, w, h]).round().astype(int)

        c1 = (255, 255, 255)
        imgs, segs = [], []

        for ti, img in enumerate(video):
            cv2.imshow("v", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            imgs.append(img)

            if segment is not None:
                seg = draw_segmentation_np(img, segment[ti, :, :, :], alpha=0.75)
                if bbox is not None and bbox.shape[0]:
                    for box in bbox[ti, :, :]:
                        seg = cv2.rectangle(seg, box[:2], box[2:], color=c1)
                cv2.imshow("s", cv2.cvtColor(seg, cv2.COLOR_RGB2BGR))
                segs.append(seg)

            cv2.waitKey(wait)

        return imgs, segs
