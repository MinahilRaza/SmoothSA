"""
Periodic videos of whole validation trajectories during training.
"""

from copy import copy
from pathlib import Path
import pickle as pkl

import cv2
import imageio
import numpy as np
import torch as pt
import torch.nn.functional as ptnf

from ..datum.dataset import lmdb_open_read
from ..util_datum import draw_segmentation_np
from .callback import Callback
from .metric_masksup import match_slot_segment, pool_segment


class SaveValVideo(Callback):
    """
    After every ``every``-th validation epoch, run the model over whole validation
    trajectories, slots initialized once from frame-0 bboxes as at deployment, and save
    one mp4 per trajectory with two panels: SAM masks | predicted slots.

    Slots are Hungarian-matched to SAM segments over the whole trajectory, as in
    ``SlotMaskLoss``, and drawn in their matched segment's color, so a slot keeping its
    object keeps its color; unmatched slots are white; background (segment 0, and the
    slot matched to it) is dark navy. Running the full trajectory at once equals
    streaming it frame by frame with ``SmoothSAVideoStream``.
    """

    def __init__(
        self,
        indices=(0,),
        every=1,
        attent_key="attenta",
        mean=None,
        std=None,
        fps=10,
        alpha=0.6,
        save_dir=None,
        dataset_key="dataset_v",
        model_key="model",
        step_count_key="step_count",
    ):
        """
        - indices: validation dataset indices (LMDB order) of the trajectories
        - attent_key: model output whose argmax gives the drawn slots
        - mean, std: input normalization, to recover RGB, shape=(c,1,1)
        """
        super().__init__()
        self.indices = list(indices)
        self.every = every
        self.attent_key = attent_key
        self.mean = np.array(mean, "float32").reshape(3, 1, 1)
        self.std = np.array(std, "float32").reshape(3, 1, 1)
        self.fps = fps
        self.alpha = alpha
        self.save_dir = save_dir
        self.dataset_key = dataset_key
        self.model_key = model_key
        self.step_count_key = step_count_key
        self.count = 0

    @pt.inference_mode()
    def after_epoch(self, **pack):
        self.count += 1
        if (self.count - 1) % self.every != 0:
            return pack
        model = pack[self.model_key]
        dataset = copy(pack[self.dataset_key].dataset)
        dataset.transform0 = lambda **_: _  # whole trajectory, no clip
        step = pack[self.step_count_key]
        out_dir = Path(self.save_dir) / "videos" / f"step{step:06d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        metas = __class__.read_metas(dataset, self.indices)
        # same slot-init noise in every video; training's random state untouched
        with pt.random.fork_rng(devices=[pt.cuda.current_device()]):
            pt.manual_seed(0)
            for i, meta in zip(self.indices, metas):
                name = f"{i:03d}-{(meta['suite'] + '-' if meta.get('suite') else '')}task{meta['task_idx']}-demo{meta['demo_idx']:02d}"
                self.save_one(model, dataset[i], out_dir / f"{name}.mp4", step)
        print(f"[{__class__.__name__}] saved {len(self.indices)} videos to {out_dir}")
        return pack

    def save_one(self, model, sample, video_file, step):
        video = sample["video"][None].cuda()  # (1,t,c,h,w)
        segment = sample["segment"]  # (t,h,w,s) bool
        t, h, w, s = segment.shape
        with pt.autocast("cuda", enabled=True):
            output = model(batch=dict(video=video, bbox=sample["bbox"][None].cuda()))
        attent = output[self.attent_key].float()  # (1,t,n,hh,ww)
        n = attent.size(2)
        smask = output.get("smask", None)

        target = pool_segment(segment[None].cuda(), attent.shape[-2:])
        assign, _ = match_slot_segment(attent, target, smask)  # (1,n,s)
        assign = assign[0].cpu()
        # slot channels reordered to their matched segment; unmatched slots after
        slot_of_seg = {int(k): int(j) for j, k in zip(*assign.nonzero(as_tuple=True))}
        unmatched = [j for j in range(n) if j not in slot_of_seg.values()]
        order = [slot_of_seg.get(k, -1) for k in range(s)] + unmatched

        argmax = ptnf.interpolate(attent[0], size=[h, w], mode="bilinear").argmax(1).cpu()
        slots = ptnf.one_hot(argmax, n).bool()  # (t,h,w,n)
        empty = pt.zeros(t, h, w, 1, dtype=pt.bool)
        slots = pt.concat([slots[..., j : j + 1] if j >= 0 else empty for j in order], -1)

        color = __class__.palette(s, len(unmatched))
        rgb = (sample["video"].numpy() * self.std + self.mean).clip(0, 255)
        rgb = rgb.astype("uint8").transpose(0, 2, 3, 1)  # (t,h,w,c)
        frames = []
        for f in range(t):
            left = __class__.draw(rgb[f], segment[f].numpy(), self.alpha, color[:s])
            right = __class__.draw(rgb[f], slots[f].numpy(), self.alpha, color)
            __class__.label(left, f"SAM 3 masks  frame {f}")
            __class__.label(right, f"slots ({self.attent_key})  step {step}")
            frames.append(np.concatenate([left, right], 1))
        imageio.mimwrite(video_file, frames, fps=self.fps, macro_block_size=1)

    BG_COLOR = (20, 40, 140)  # dark navy

    @staticmethod
    def palette(s, num_unmatched):
        """segment k -> color k, kept away from the background navy; segment 0
        (background) navy; unmatched slots white"""
        rng = np.random.RandomState(0)
        color = []
        while len(color) < s - 1:
            c = rng.randint(60, 256, 3)
            if np.abs(c - __class__.BG_COLOR).sum() > 200:
                color.append(c)
        color = [__class__.BG_COLOR] + color
        return np.concatenate([color, np.full((num_unmatched, 3), 255)]).astype("uint8")

    @staticmethod
    def draw(image, masks, alpha, color):
        """- masks: (h,w,c) bool, channel 0 = background"""
        return draw_segmentation_np(image, masks, alpha, color)

    @staticmethod
    def label(image, text, y=14):
        # outline as offset thin copies: thick text is spaced wider and would not align
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 1), (1, 0), (-1, 0), (0, -1)]:
            cv2.putText(image, text, (4 + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(image, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    @staticmethod
    def read_metas(dataset, indices):
        env = lmdb_open_read(dataset.data_file)
        metas = []
        with env.begin(write=False) as txn:
            for i in indices:
                sample = pkl.loads(txn.get(dataset.idxs[i]))
                metas.append({k: sample[k] for k in ["task_idx", "demo_idx"]})
                metas[-1]["suite"] = sample.get("suite", "")
        env.close()
        return metas
