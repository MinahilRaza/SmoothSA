"""
Mask-conditioned SmoothSA video model: slot queries come from externally
supplied, temporally consistent segment masks (e.g. a tracker) instead of
frame-0 bounding boxes.

Differences from ``SmoothSAVideo``:

- Slot index ``j`` is tied to segment ``j`` of ``segment`` for the whole clip
  (identity comes from the tracker IDs, not from slot competition).
- Validity is per frame, ``svalid: (b,t,n)``, instead of per clip. Invalid
  slots are masked out of slot attention, ``transit`` and the decoder, and are
  zeroed in the outputs.
- A slot is (re)bootstrapped by ``initializ`` from its mask-pooled features the
  first time it is valid in the clip. Afterwards its query comes from
  ``transit`` over the slot's own last *valid* state (carried through
  occlusion), plus an optional mask-pooled correction term.
- ``initializ`` gets mask-pooled features (b,n,kv_dim) as its condition, in
  place of bboxes; ``NormalMlpPreheated`` is reused with ``in_dim=kv_dim``.
- Optional deterministic evaluation: no sampling noise in eval mode.

Strictly causal: slots at frame ``t`` depend only on frames ``<= t``. The
decoder reconstructs each frame from that frame's slots only.
"""

from einops import rearrange
import torch as pt
import torch.nn as nn

from .basic import MLP
from .smoothsa import NormalMlpPreheated, SmoothSAVideo


def mask_pool(encode, segment, eps=1e-6):
    """Masked average pooling of features over each segment.

    - encode: (b,h*w,c)
    - segment: (b,n,h*w), non-negative soft masks (eg patch area fractions)
    - return: (b,n,c); all-zero masks give all-zero features
    """
    weight = segment / (segment.sum(2, keepdim=True) + eps)
    return pt.einsum("bnm,bmc->bnc", weight, encode)


class MaskPoolPreheated(NormalMlpPreheated):
    """``NormalMlpPreheated`` conditioned on mask-pooled features.

    - eval_noise: if False, no sampling noise in eval mode, so the frozen
        encoder is deterministic (needed for caching and checkpoint checks).
    """

    def __init__(self, in_dim, dims, kv_dim, mlpln="post", eval_noise=False):
        super().__init__(in_dim, dims, kv_dim, mlpln, pad_value=None)
        self.eval_noise = eval_noise

    def forward(self, encode, condit, smask=None):
        """
        - encode: shape=(b,h*w,c)
        - condit: mask-pooled features, shape=(b,n,in_dim)
        - smask: shape=(b,n), dtype=bool. True means there is a valid slot.
        """
        noisy = self.training or self.eval_noise
        mean = self.mlp(condit)
        smpl = mean + pt.randn_like(mean) * self.logstd.exp() if noisy else mean

        if self.detach_flag:
            encode = encode.detach()
        qinit = self.qinit(
            smpl,
            self.qproj_kv(encode),
            tgt_key_padding_mask=None if smask is None else ~smask,
        )
        if noisy:
            query = qinit.detach() + pt.randn_like(qinit) * self.logstd2.exp()
        else:
            query = qinit.detach()
        return qinit, query


class SmoothSAVideoMaskCond(SmoothSAVideo):
    """SmoothSA video model with tracker-mask-conditioned, identity-fixed slots.

    - mask_query: add ``MLP(mask_pool(segment_t))`` to the transit query at
        every frame, so the current tracked mask steers its slot. The last
        layer is zero-initialized: at init the model behaves like SmoothSA.
    """

    def __init__(
        self,
        encode_backbone,
        encode_posit_embed,
        encode_project,
        initializ,
        aggregat,
        transit,
        decode,
        mask_query=True,
        kv_dim=384,
        emb_dim=256,
    ):
        super().__init__(
            encode_backbone,
            encode_posit_embed,
            encode_project,
            initializ,
            aggregat,
            transit,
            decode,
            slot_mask=False,
            bg_slot=False,
        )
        self.mask_query = mask_query
        if mask_query:
            self.mask_query_proj = MLP(kv_dim, [emb_dim, emb_dim], "pre", 0)
            nn.init.zeros_(self.mask_query_proj[-1].weight)
            nn.init.zeros_(self.mask_query_proj[-1].bias)

    def encode(self, input):
        """
        - input: video, shape=(b,t,c,h,w)
        - return: feature (b,t,c,h,w), encode (b,t,h*w,c)
        """
        b = input.size(0)
        feature = self.encode_backbone(input.flatten(0, 1)).detach()
        encode = feature.permute(0, 2, 3, 1)
        encode = self.encode_posit_embed(encode)
        encode = encode.flatten(1, 2)
        encode = self.encode_project(encode)
        feature = rearrange(feature, "(b t) c h w -> b t c h w", b=b)
        encode = rearrange(encode, "(b t) hw c -> b t hw c", b=b)
        return feature, encode

    def forward(self, input, segment, svalid, state=None):
        """
        - input: video, shape=(b,t,c,h,w)
        - segment: tracked masks on the feature grid, shape=(b,t,n,h',w'),
            soft (eg patch area fractions); slot j <-> segment j
        - svalid: shape=(b,t,n), dtype=bool. True means segment j is observed
        - state: carried state from a previous call (streaming), or None to
            bootstrap. See ``init_state``.
        - return: feature (b,t,c,h',w'), qinit (b,t,n,c), slotz (b,t,n,c),
            attenta (b,t,n,h',w'), recon (b,t,c,h',w'), attentd (b,t,n,h',w'),
            svalid (b,t,n), state (dict)
        """
        b, t = input.shape[:2]
        n = segment.size(2)
        feature, encode = self.encode(input)
        h, w = feature.shape[-2:]
        assert segment.shape[-2:] == (h, w), f"segment grid {tuple(segment.shape[-2:])} != feature grid {(h, w)}"
        assert svalid.shape == (b, t, n), f"svalid {tuple(svalid.shape)} != {(b, t, n)}"
        svalid = svalid.bool()
        segment = segment.flatten(3) * svalid[..., None]  # (b,t,n,h*w)

        if state is None:
            state = self.init_state(b, n, encode.device, encode.dtype)
        slot_hist = state["slot_hist"]  # (b,i,n,c): carried last-valid states
        encode_hist = state["encode_hist"]  # (b,i,h*w,c) or None
        seen = state["seen"]  # (b,n): slot has been valid before

        qinits, slotzs, attentas = [], [], []
        for i in range(t):
            enc_i = encode[:, i]
            valid_i = svalid[:, i]
            pooled_i = mask_pool(enc_i, segment[:, i])  # (b,n,c)
            encode_hist = enc_i[:, None] if encode_hist is None else pt.concat([encode_hist, enc_i[:, None]], 1)

            # bootstrap from mask-pooled features; also gives the preheat target
            qinit_i, query_init = self.initializ(enc_i, pooled_i, smask=valid_i)
            if slot_hist.size(1) > 0:
                query_tran = self.transit(slot_hist, encode_hist, smask=seen | valid_i)
                if self.mask_query:
                    query_tran = query_tran + self.mask_query_proj(pooled_i)
                query_i = pt.where(seen[..., None], query_tran, query_init)
            else:
                query_i = query_init
            # first frame (or a lone new slot): full refinement, else 1 iteration
            niter = None if slot_hist.size(1) == 0 else 1
            slotz_i, attenta_i = self.aggregat(enc_i, query_i, smask=valid_i, num_iter=niter)

            # carry the last valid state through occlusion
            last = slot_hist[:, -1] if slot_hist.size(1) > 0 else pt.zeros_like(slotz_i)
            carried = pt.where(valid_i[..., None], slotz_i, last)
            slot_hist = pt.concat([slot_hist, carried[:, None]], 1)
            seen = seen | valid_i

            qinits.append(qinit_i)
            slotzs.append(slotz_i)
            attentas.append(attenta_i)

        maxlen = self.transit.dt
        state = dict(
            slot_hist=slot_hist[:, -(maxlen - 1) :].detach(),
            encode_hist=encode_hist[:, -(maxlen - 1) :].detach(),
            seen=seen,
        )

        qinit = pt.stack(qinits, 1)  # (b,t,n,c)
        slotz = pt.stack(slotzs, 1)
        attenta = pt.stack(attentas, 1)  # (b,t,n,h*w)
        attenta = rearrange(attenta, "b t n (h w) -> b t n h w", h=h)

        clue = rearrange(feature, "b t c h w -> (b t) (h w) c")
        recon, attentd = self.decode(clue, slotz.flatten(0, 1), svalid.flatten(0, 1))
        recon = rearrange(recon, "(b t) (h w) c -> b t c h w", b=b, h=h)
        attentd = rearrange(attentd, "(b t) n (h w) -> b t n h w", b=b, h=h)

        vmask = svalid[..., None]
        qinit = qinit * vmask
        slotz = slotz * vmask
        return feature, qinit, slotz, attenta, recon, attentd, svalid, state

    def init_state(self, b, n, device, dtype):
        """Empty streaming state: no slot history, no slot seen yet."""
        c = self.aggregat.proj_q.in_features
        return dict(
            slot_hist=pt.zeros(b, 0, n, c, device=device, dtype=dtype),
            encode_hist=None,
            seen=pt.zeros(b, n, dtype=pt.bool, device=device),
        )
