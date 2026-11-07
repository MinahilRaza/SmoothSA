"""
Mask supervision of slot attention with (pseudo) instance segments, eg SAM 3.
"""

from einops import rearrange
import torch as pt
import torch.nn.functional as ptnf

from .metric import Metric
from ..util_learn import hungarian_matching


def pool_segment(segment, size):
    """
    Area fraction of each segment inside each attention patch.
    - segment: shape=(b,t,hh,ww,s), bool
    - return: shape=(b,t,s,h,w), float
    """
    b = segment.size(0)
    segment = rearrange(segment.float(), "b t hh ww s -> (b t) s hh ww")
    segment = ptnf.adaptive_avg_pool2d(segment, size)
    return rearrange(segment, "(b t) s h w -> b t s h w", b=b)


@pt.no_grad()
def match_slot_segment(attent, target, smask=None, eps=1e-8):
    """
    Hungarian matching of slots to segments, once per clip, maximizing soft IoU
    accumulated over all frames and patches.
    - attent: shape=(b,t,n,h,w), softmax over n
    - target: shape=(b,t,s,h,w), area fractions
    - smask: shape=(b,n), dtype=bool. True means there is a valid slot.
    - return: assignment, shape=(b,n,s), float 0/1, slot j <- segment k;
        iou, shape=(b,n,s), soft IoU
    """
    b, t, n, h, w = attent.shape
    a = rearrange(attent.float(), "b t n h w -> b n (t h w)")
    q = rearrange(target, "b t s h w -> b s (t h w)")
    inter = pt.einsum("bnx,bsx->bns", a, q)
    union = a.sum(2)[:, :, None] + q.sum(2)[:, None, :] - inter
    iou = inter / union.clamp_min(eps)

    valid_slot = pt.ones(b, n, dtype=pt.bool, device=a.device) if smask is None else smask
    valid_seg = q.sum(2) > 0  # empty or padded segments
    valid = valid_slot[:, :, None] & valid_seg[:, None, :]  # (b,n,s)
    _, rcidx = hungarian_matching(iou.where(valid, -1e6), maximize=True)  # (b,m,2)
    rcidx = rcidx.clone()  # out of inference mode

    bidx = pt.arange(b, device=a.device)[:, None].expand_as(rcidx[:, :, 0])
    row, col = rcidx[:, :, 0], rcidx[:, :, 1]
    assign = pt.zeros(b, n, q.size(1), device=a.device)
    assign[bidx, row, col] = valid[bidx, row, col].float()  # drop forced invalid pairs
    return assign, iou


class SlotMaskLoss(Metric):
    """
    Matched mask supervision of slot attention with (pseudo) instance segments.

    Slots and segments are Hungarian-matched once per clip on soft IoU accumulated over
    all frames, so each slot is supervised with the same segment in every frame: a slot
    switching objects mid-clip, eg across an occlusion, is penalized in the frames after
    the switch. Background is a segment like any other. Each matched slot is then
    supervised with its segment's area fraction per patch by:
    - ``dice``: soft Dice loss per matched pair over the whole clip, averaged over pairs.
        Uniform attention scores badly, so it does not reward hedging; consistent with
        the soft IoU used for matching.
    - ``ce``: cross entropy over patches. Rewards hedging while assignments are poor:
        uniform attention (log n) beats sharp but partly wrong attention, which can
        collapse slot attention to uniform.

    Masked slots and empty segments are never matched. Unmatched slots get no target
    (with ``ce``, their attention is still pushed down where matched slots have
    targets); unmatched segments, eg objects without a frame-0 bbox, are not supervised.
    """

    def __init__(self, dice=1.0, ce=0.0, eps=1e-8, mean=()):
        super().__init__(mean)
        self.dice = dice
        self.ce = ce
        self.eps = eps

    def compute(self, input, target, smask=None):
        """
        - input: slot attention, shape=(b,t,n,h,w), softmax over n
        - target: segment, shape=(b,t,hh,ww,s), bool
        - smask: slots' mask, shape=(b,n), dtype=bool. True means there is a valid slot.
        """
        b, t, n, h, w = input.shape
        target = pool_segment(target, [h, w])  # (b,t,s,h,w)
        assign, _ = match_slot_segment(input, target, smask, self.eps)  # (b,n,s)
        target = pt.einsum("bns,btshw->btnhw", assign, target)  # matched, per slot
        attent = input.float()
        loss = attent.new_zeros([])

        matched = assign.sum(2) > 0  # (b,n)
        if self.dice and matched.any():
            a = rearrange(attent, "b t n h w -> b n (t h w)")
            q = rearrange(target, "b t n h w -> b n (t h w)")
            dice = 2 * (a * q).sum(2) / (a.sum(2) + q.sum(2)).clamp_min(self.eps)
            loss = loss + self.dice * (1 - dice[matched]).mean()

        if self.ce:
            logp = attent.clamp_min(self.eps).log()
            ce = -(target * logp).sum() / target.sum().clamp_min(self.eps)
            loss = loss + self.ce * ce

        return loss[None], None  # (b=1,) (b=1,)


class SlotAttentStat(Metric):
    """
    Monitors against slot-attention collapse, per clip:
    - ``entropy``: mean per-patch entropy over valid slots / log(#valid slots);
        ~0 sharp, 1 uniform (collapsed)
    - ``miou``: hard-mask IoU of argmax slots vs argmax segments, after the same
        per-clip Hungarian matching as ``SlotMaskLoss``; averaged over segments
    - ``top_share``: fraction of patches taken by the largest slot
    """

    def __init__(self, stat, eps=1e-8, mean=()):
        super().__init__(mean)
        assert stat in ["entropy", "miou", "top_share"]
        self.stat = stat
        self.eps = eps

    def compute(self, input, target, smask=None):
        """
        - input: slot attention, shape=(b,t,n,h,w), softmax over n
        - target: segment, shape=(b,t,hh,ww,s), bool
        - smask: shape=(b,n), dtype=bool
        """
        b, t, n, h, w = input.shape
        attent = input.float()
        if smask is None:
            smask = pt.ones(b, n, dtype=pt.bool, device=attent.device)
        attent = attent * smask[:, None, :, None, None]

        if self.stat == "entropy":
            a = attent.clamp_min(self.eps)
            ent = -(a * a.log() * smask[:, None, :, None, None]).sum(2)  # (b,t,h,w)
            nv = smask.sum(1).clamp_min(2).float()  # (b,)
            return ent.flatten(1).mean(1) / nv.log(), None

        pred = ptnf.one_hot(attent.argmax(2), n).float()  # (b,t,h,w,n)
        pred = rearrange(pred, "b t h w n -> b t n h w")
        if self.stat == "top_share":
            return pred.mean([1, 3, 4]).max(1).values, None

        target = pool_segment(target, [h, w])  # (b,t,s,h,w)
        s = target.size(2)
        gt = ptnf.one_hot(target.argmax(2), s).float()  # (b,t,h,w,s)
        gt = rearrange(gt, "b t h w s -> b t s h w") * (target.sum(2, keepdim=True) > 0)
        assign, _ = match_slot_segment(pred, gt, smask, self.eps)  # (b,n,s)
        p = rearrange(pred, "b t n h w -> b n (t h w)")
        g = rearrange(gt, "b t s h w -> b s (t h w)")
        inter = pt.einsum("bnx,bsx->bns", p, g)
        union = p.sum(2)[:, :, None] + g.sum(2)[:, None, :] - inter
        iou = (inter / union.clamp_min(self.eps) * assign).sum(1)  # (b,s)
        nseg = (g.sum(2) > 0).sum(1).clamp_min(1)  # (b,)
        return iou.sum(1) / nseg, None
