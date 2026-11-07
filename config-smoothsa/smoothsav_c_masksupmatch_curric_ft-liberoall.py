from einops import rearrange
import torch.nn.functional as ptnf

from object_centric_bench.datum import (
    RandomSliceToSequence,
    RandomCrop,
    Resize,
    RandomFlip,
    Normalize,
    CenterCrop,
    Lambda,
    LIBERO,
    ClPadToMax1,
    ClPadTo1,
    DefaultCollate,
)
from object_centric_bench.learn import (
    Adam,
    GradScaler,
    ClipGradNorm,
    MSELoss,
    SlotContrastLoss,
    SlotMaskLoss,
    SlotAttentStat,
    mBO,
    ARI,
    mIoU,
    CbLinearCosine,
    CbSquarewave,
    Callback,
    AverageLog,
    SaveModel,
    SaveValVideo,
)
from object_centric_bench.model import (
    SmoothSAVideo,
    Sequential,
    Interpolate,
    DINO2ViT,
    Identity,
    MLP,
    NormalMlpPreheated,
    SlotAttention,
    RSFQTransit,
    ARRandTransformerDecoder,
    LearntPositionalEmbedding,
    Linear,
    LayerNorm,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from object_centric_bench.util import Compose, ComposeNoStar
from object_centric_bench.util_model import interpolat_argmax_attent

### global

resolut0 = [256, 256]
resolut1 = [16, 16]
emb_dim = 256
vfm_dim = 384

total_step = 50000  # 100000 better
val_interval = total_step // 40
batch_size_t = 32 // 4  # 64 better
batch_size_v = batch_size_t
num_work = 4
lr = 2e-4 / 4  # scale with batch_size

# curriculum on training window length: slots are bbox-initialized at the window's
# first frame then carried through it, so longer windows train longer tracking
window_points = [0, total_step // 10, total_step * 3 // 10, total_step * 6 // 10, total_step]
window_sizes = [6, 12, 24, 48]

### datum

IMAGENET_MEAN = [[[123.675]], [[116.28]], [[103.53]]]
IMAGENET_STD = [[[58.395]], [[57.12]], [[57.375]]]
transform_t = [
    # the following 2 == RandomResizedCrop: better than max sized random crop
    dict(type=RandomCrop, keys=["video", "segment"], size=None, scale=[0.75, 1]),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=RandomFlip, keys=["video", "segment"], dims=[-1], p=0.5),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
transform_v = [
    dict(type=CenterCrop, keys=["video", "segment"], size=None),
    dict(type=Resize, keys=["video"], size=resolut0, interp="bilinear"),
    dict(type=Resize, keys=["segment"], size=resolut0, interp="nearest-exact", c=0),
    dict(type=Normalize, keys=["video"], mean=[IMAGENET_MEAN], std=[IMAGENET_STD]),
]
dataset_t = dict(
    type=LIBERO,
    data_file="libero_all/train.lmdb",
    # all LIBERO suites (spatial, object, goal, 10, 90), 10 random demos per task;
    # held out: one task per suite (see data-prep/DataPrep.md). LIBERO doesn't ship
    # bbox -- it derives them from the SAM-3 masks (see LIBERO.__getitem__).
    extra_keys=["segment", "bbox"],
    # any start frame, not only multiples of size: long windows would repeat otherwise
    transform0=dict(
        type=RandomSliceToSequence, keys=["video", "segment"], size=window_sizes[0]
    ),
    transform=dict(type=Compose, transforms=transform_t),
    base_dir=...,
    bbox_bg=[-1, -1, -1, -1],  # slot 0: background, padding-value bbox as in MOVi
)
dataset_v = dict(
    type=LIBERO,
    data_file="libero_all/val.lmdb",
    extra_keys=["segment", "bbox"],
    # trajectories differ in length (~250-400 frames): fixed episode-start clip
    transform0=dict(type=Lambda, ikeys=[["video", "segment"]], func=lambda _: _[:24]),
    transform=dict(type=Compose, transforms=transform_v),
    base_dir=...,
    bbox_bg=[-1, -1, -1, -1],  # slot 0: background, padding-value bbox as in MOVi
)
collate_fn_t = dict(
    type=ComposeNoStar,
    transforms=[
        dict(type=ClPadToMax1, keys=["segment"], dims=[3]),
        dict(type=ClPadTo1, keys=["bbox"], dims=[1], num=None, value=-1),
        dict(type=DefaultCollate),
    ],
)
collate_fn_v = collate_fn_t

### model

model = dict(
    type=SmoothSAVideo,
    encode_backbone=dict(
        type=Sequential,
        modules=[
            dict(type=Interpolate, scale_factor=0.875, interp="bicubic"),
            dict(
                type=DINO2ViT,
                model_name="vit_small_patch14_reg4_dinov2.lvd142m",
                in_size=int(resolut0[0] * 0.875),
                rearrange=True,
                norm_out=False,
            ),
        ],
    ),
    encode_posit_embed=dict(type=Identity),
    encode_project=dict(
        type=MLP, in_dim=vfm_dim, dims=[vfm_dim, vfm_dim], ln="pre", dropout=0.0
    ),
    initializ=dict(
        type=NormalMlpPreheated,
        in_dim=4,
        dims=[emb_dim, emb_dim],
        kv_dim=vfm_dim,
        pad_value=-1,
    ),
    aggregat=dict(
        type=SlotAttention,
        num_iter=3,
        embed_dim=emb_dim,
        ffn_dim=emb_dim * 4,
        dropout=0,
        kv_dim=vfm_dim,
        trunc_bp="bi-level",
    ),
    transit=dict(
        type=RSFQTransit,
        dt=6,  # XXX XXX XXX
        ci=vfm_dim,
        c=emb_dim,
        nhead=4,
        expanz=4,
        pdo=0.5,  # XXX XXX XXX
        norm_first=False,
        bias=False,
    ),
    decode=dict(
        type=ARRandTransformerDecoder,
        emb_dim=vfm_dim,
        posit_embed=dict(
            type=LearntPositionalEmbedding,
            resolut=[resolut1[0] * resolut1[1]],
            embed_dim=vfm_dim,
        ),
        project1=dict(  # fc>fc+ln
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=vfm_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        project2=dict(  # fc+ln>fc
            type=Sequential,
            modules=[
                dict(
                    type=Linear, in_features=emb_dim, out_features=vfm_dim, bias=False
                ),
                dict(type=LayerNorm, normalized_shape=vfm_dim),
            ],
        ),
        backbone=dict(
            type=TransformerDecoder,
            decoder_layer=dict(
                type=TransformerDecoderLayer,
                d_model=vfm_dim,
                nhead=4,  # 4@384, 6@768
                dim_feedforward=vfm_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
                bias=False,
            ),
            num_layers=4,
        ),
        readout=dict(type=Identity),
    ),
    # exactly 1 + (#frame-0 bboxes) slots per sample, as at deployment
    slot_mask=True,
    bg_slot=True,  # slot 0 keeps its -1 bbox unmasked
)
model_imap = dict(input="batch.video", condit="batch.bbox")
model_omap = ["feature", "qinit", "slotz", "attenta", "recon", "attentd", "smask"]
# fine-tune from smoothsav_c-movi_c: ``--ckpt_file archive-smoothsa/smoothsav_c-movi_c/best.pth``
ckpt_map = None  # None: load all weights; [] would load nothing
freez = [r"^m\.encode_backbone\..*"]

### learn

param_groups = None
optimiz = dict(type=Adam, params=param_groups, lr=lr)
gscale = dict(type=GradScaler)
gclip = dict(type=ClipGradNorm, max_norm=0.05)

loss_fn_t = loss_fn_v = dict(
    recon=dict(
        metric=dict(type=MSELoss),
        map=dict(input="output.recon", target="output.feature"),
        transform=dict(type=Lambda, ikeys=[["target"]], func=lambda _: _.detach()),
    ),
    ssc=dict(
        metric=dict(type=SlotContrastLoss),
        map=dict(input="output.slotz", smask="output.smask"),
        weight=0.5,
    ),
    preheat=dict(
        metric=dict(type=MSELoss),  # (b,s,c) (b,t,s,c)
        map=dict(input="output.qinit", target="output.slotz"),
        transform=dict(
            type=Lambda, ikeys=[["target"]], func=lambda _: _[:, 0, :, :].detach()
        ),
        weight=100,
    ),
    # supervise encoder slot attention with SAM 3 pseudo masks: slots and segments
    # (incl background) Hungarian-matched once per clip, so identities stay fixed
    masksup=dict(
        metric=dict(type=SlotMaskLoss, dice=1.0, ce=0.0),  # ce rewards hedging
        map=dict(input="output.attenta", target="batch.segment", smask="output.smask"),
        weight=1,  # TODO sweep
    ),
)
_acc_dict_ = dict(
    # metric=...,
    map=dict(input="output.segment", target="batch.segment"),
    transform=dict(
        type=Lambda,
        ikeys=[["input", "target"]],
        func=lambda _: rearrange(_, "b t h w s -> b (t h w) s"),
    ),
)
# collapse monitors on encoder slot attention
_stat_dict_ = dict(
    map=dict(input="output.attenta", target="batch.segment", smask="output.smask")
)
_stats_ = dict(
    attenta_entropy=dict(metric=dict(type=SlotAttentStat, stat="entropy"), **_stat_dict_),
    attenta_miou=dict(metric=dict(type=SlotAttentStat, stat="miou"), **_stat_dict_),
    attenta_top_share=dict(
        metric=dict(type=SlotAttentStat, stat="top_share"), **_stat_dict_
    ),
)
acc_fn_t = dict(
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    **_stats_,
)
acc_fn_v = dict(
    ari=dict(metric=dict(type=ARI, skip=[]), **_acc_dict_),
    ari_fg=dict(metric=dict(type=ARI, skip=[0]), **_acc_dict_),
    mbo=dict(metric=dict(type=mBO, skip=[]), **_acc_dict_),
    miou=dict(metric=dict(type=mIoU, skip=[]), **_acc_dict_),
    **_stats_,
)

before_step = [
    dict(
        type=Lambda,
        ikeys=[["batch.video", "batch.segment", "batch.bbox"]],
        func=lambda _: _.cuda(),
    ),
    dict(
        type=CbLinearCosine,
        assigns=["optimiz.param_groups[0]['lr']=value"],
        nlin=total_step // 20,
        ntotal=total_step,
        vstart=0,
        vbase=lr,
        vfinal=lr / 1e3,
    ),
    dict(
        type=CbSquarewave,
        assigns=["model.m.initializ.detach_flag.data[...]=value"],
        points=[0, total_step // 10, total_step],
        values=[1, 0],
    ),
    # dataloader workers are re-created each epoch, so this applies from the next one
    dict(
        type=CbSquarewave,
        assigns=["dataset_t.dataset.transform0.size=value"],
        points=window_points,
        values=window_sizes,
    ),
]
after_forward = [
    dict(
        type=Lambda,
        ikeys=[["output.attentd"]],  # (b,t,s,h,w) -> (b,t,h,w,s)
        func=lambda _: ptnf.one_hot(
            interpolat_argmax_attent(_.detach(), size=resolut0).long()
        ).bool(),
        okeys=[["output.segment"]],
    ),
]
callback_t = [
    dict(type=Callback, before_step=before_step, after_forward=after_forward),
    dict(type=AverageLog, log_file=...),
]
callback_v = [
    dict(type=Callback, before_step=before_step[:1], after_forward=after_forward),
    callback_t[1],
    dict(type=SaveModel, save_dir=..., since_step=total_step * 0.5),
    # whole val trajectories, one per held-out task/suite: SAM masks | slots
    dict(
        type=SaveValVideo,
        indices=[0, 10, 20, 30, 40],
        every=2,  # every 2nd val epoch: 2 * val_interval steps
        attent_key="attenta",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        save_dir=...,
    ),
]
