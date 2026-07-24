"""
Modified VFIMamba flow module.

CORE CHANGE vs. the original file:
------------------------------------------------------------------
Original:  the Head/IFBlock stack directly regresses the BILATERAL,
           timestep-conditioned flow (F_{t->0}, F_{t->1}) at every
           refinement stage. `timestep` is concatenated as an input
           channel throughout, so flow must be recomputed from
           scratch for every t you want to interpolate at.

This file: the Head/IFBlock stack (renamed BiHead/BiIFBlock) instead
           regresses a single, t-INDEPENDENT bidirectional flow pair
           (F_{0->1}, F_{1->0}) between the two input frames, plus a
           t-independent occlusion/visibility field. `timestep` is
           removed from every conv input. This "backbone" work
           happens exactly once per (img0, img1) pair.

           A cheap, non-learned step then derives F_{t->0}/F_{t->1}
           (and a t-dependent blend mask) from that single flow pair
           for as many t values as you like, using the classical
           "linear combination of bidirectional flows" formula
           (Jiang et al., Super SloMo). This is the paradigm change
           discussed: constant-velocity motion between frame0 and
           frame1 is now an explicit assumption, not something the
           network has to re-learn/re-infer per timestep.

           The synthesis stage (feature warping + UNet residual) is
           unchanged in spirit but now consumes the derived F_t
           instead of a directly-regressed one, and is called once
           per requested t.

Practical payoff: for Nx / arbitrary-timestep interpolation, the
feature backbone and BiHead/BiIFBlock stack run ONCE per frame pair;
only the cheap scaling step + synthesis UNet reruns per t. Compare to
the original, where the whole timestep-conditioned Head/IFBlock stack
reruns per t.

Trade-off (see conversation): constant-velocity scaling has no way to
represent acceleration/curved motion within the (0,1) interval. The
synthesis UNet is now the only place left that can partially correct
for that, since it still sees the actual warped images.

NOTE: in_else channel counts below are recomputed for the new (no
timestep) concatenation pattern. Verify against your actual
`backbone`/`refine.py` embed_dims before training — they depend on
exactly what you concatenate, and are only as correct as the
docstring's channel bookkeeping.

PACKAGING NOTE: this file must live inside the SAME package as
`warplayer.py` that train.py imports as `model_oldRepo.warplayer` and
clears every epoch via `warplayer.backwarp_tenGrid.clear()` (needed
because training uses native/mixed resolutions per the curriculum).
`warp()` caches a resolution-keyed grid in a module-level dict on
first use. If this file's `from .warplayer import warp` resolves to a
different module object than the one train.py clears (e.g. this file
placed in a different package, or imported under a different path),
you'll get two independent caches and train.py's per-epoch clear()
will silently do nothing for the grids this file actually uses --
place this file at `model_oldRepo/flow_estimation.py` (same directory
as the original), not elsewhere.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from .warplayer import warp
from .refine import *


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.PReLU(out_planes)
    )


class BiHead(nn.Module):
    """Coarse-to-fine block that predicts (F_{0->1}, F_{1->0}, visibility).

    CHANGED from the original `Head`:
      - No `timestep` argument / channel anywhere.
      - Output is still 5 channels, but their meaning changed:
          out[:, :2] = F_{0->1}
          out[:, 2:4] = F_{1->0}
          out[:, 4:5] = visibility logit (t-independent occlusion cue)
        (previously: out[:,:4] was F_{t->0},F_{t->1} and out[:,4:5]
         was the t-dependent blend mask.)
    """
    def __init__(self, in_planes, scale, c, in_else=6):
        super(BiHead, self).__init__()
        self.upsample = nn.Sequential(nn.PixelShuffle(2), nn.PixelShuffle(2))
        self.scale = scale
        self.conv = nn.Sequential(
            conv(in_planes * 2 // (4 * 4) + in_else, c),
            conv(c, c),
            conv(c, 5),
        )

    def forward(self, motion_feature, x, bi_flow):
        # `bi_flow` replaces the old `flow` arg: still a 4-channel field,
        # now holding (F_{0->1}, F_{1->0}) instead of (F_{t->0}, F_{t->1}).
        motion_feature = self.upsample(motion_feature)
        if self.scale != 4:
            x = F.interpolate(x, scale_factor=4. / self.scale, mode="bilinear", align_corners=False)
        if bi_flow is not None:
            if self.scale != 4:
                bi_flow = F.interpolate(bi_flow, scale_factor=4. / self.scale,
                                         mode="bilinear", align_corners=False) * 4. / self.scale
            x = torch.cat((x, bi_flow), 1)
        x = self.conv(torch.cat([motion_feature, x], 1))
        if self.scale != 4:
            x = F.interpolate(x, scale_factor=self.scale // 4, mode="bilinear", align_corners=False)
            bi_flow = x[:, :4] * (self.scale // 4)
        else:
            bi_flow = x[:, :4]
        visibility = x[:, 4:5]
        return bi_flow, visibility


class BiIFBlock(nn.Module):
    """Local refinement block for the bidirectional (t-independent) flow.
    CHANGED from `IFBlock`: no timestep channel; operates purely on
    img0/img1/warped0/warped1/visibility.
    """
    def __init__(self, in_planes, c, scale):
        super(BiIFBlock, self).__init__()
        self.conv0 = nn.Sequential(
            conv(in_planes, c // 2, 3, 2, 1),
            conv(c // 2, c, 3, 2, 1),
        )
        self.convblock = nn.Sequential(
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
            conv(c, c), conv(c, c), conv(c, c), conv(c, c),
        )
        self.lastconv = nn.ConvTranspose2d(c, 5, 4, 2, 1)
        self.scale = scale

    def forward(self, x, bi_flow):
        scale = self.scale
        if scale != 1:
            x = F.interpolate(x, scale_factor=1. / scale, mode="bilinear", align_corners=False)
            bi_flow = F.interpolate(bi_flow, scale_factor=1. / scale, mode="bilinear", align_corners=False) * 1. / scale
        x = torch.cat((x, bi_flow), 1)
        x = self.conv0(x)
        x = self.convblock(x) + x
        tmp = self.lastconv(x)
        tmp = F.interpolate(tmp, scale_factor=scale * 2, mode="bilinear", align_corners=False)
        bi_flow = tmp[:, :4] * scale * 2
        visibility = tmp[:, 4:5]
        return bi_flow, visibility


class MultiScaleFlow(nn.Module):
    def __init__(self, backbone, **kargs):
        super(MultiScaleFlow, self).__init__()
        self.flow_num_stage = len(kargs['hidden_dims'])
        self.local_num = kargs['local_num']
        self.feature_bone = backbone

        # CHANGED: in_else no longer includes a timestep channel.
        #   stage 0 (coarsest): x = cat(img0, img1)                       -> 6 channels
        #   stage i>0:          x = cat(img0, img1, warp0, warp1, vis)    -> 13 channels
        self.block = nn.ModuleList([
            BiHead(kargs['embed_dims'][-1 - i],
                   kargs['scales'][-1 - i],
                   kargs['hidden_dims'][-1 - i],
                   in_else=6 if i == 0 else 13)
            for i in range(self.flow_num_stage)
        ])
        # local refinement input: cat(img0,img1,warp0,warp1,vis) = 13ch, +4ch bi_flow = 17
        self.local_block = nn.ModuleList([
            BiIFBlock(17, c=kargs['local_hidden_dims'], scale=2 - i)
            for i in range(self.local_num)
        ])
        self.unet = Unet(kargs['c'] * 2, kargs['M'])

    # ------------------------------------------------------------------
    # t-independent stage: run ONCE per (img0, img1) pair.
    # ------------------------------------------------------------------
    def estimate_bi_flow(self, img0, img1, local=False, af=None):
        """Coarse-to-fine estimation of F_{0->1}, F_{1->0} and an
        occlusion/visibility field. Contains no timestep dependence
        at all -- this is the piece that used to be recomputed per-t
        in the original architecture and now only runs once.
        """
        B = img0.size(0)
        if af is None:
            af = self.feature_bone(img0, img1)

        bi_flow, visibility = None, None
        for i in range(self.flow_num_stage):
            if bi_flow is not None:
                warped_img0 = warp(img0, bi_flow[:, :2])   # img0 -> img1 space
                warped_img1 = warp(img1, bi_flow[:, 2:4])  # img1 -> img0 space
                flow_d, vis_d = self.block[i](
                    torch.cat([af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1, warped_img0, warped_img1, visibility), 1),
                    bi_flow,
                )
                bi_flow = bi_flow + flow_d
                visibility = visibility + vis_d
            else:
                bi_flow, visibility = self.block[i](
                    torch.cat([af[-1 - i][:B], af[-1 - i][B:]], 1),
                    torch.cat((img0, img1), 1),
                    None,
                )

        if local:
            for i in range(self.local_num):
                warped_img0 = warp(img0, bi_flow[:, :2])
                warped_img1 = warp(img1, bi_flow[:, 2:4])
                flow_d, vis_d = self.local_block[i](
                    torch.cat((img0, img1, warped_img0, warped_img1, visibility), 1),
                    bi_flow,
                )
                bi_flow = bi_flow + flow_d
                visibility = visibility + vis_d

        return bi_flow, visibility, af

    # ------------------------------------------------------------------
    # t-dependent stage: cheap, no learned parameters. Run per requested t.
    # ------------------------------------------------------------------
    @staticmethod
    def flow_from_bi(bi_flow, t):
        """Classic linear-combination-of-bidirectional-flow formula
        (Super SloMo / Jiang et al.). Assumes constant velocity between
        frame0 and frame1 -- this is exactly the assumption that breaks
        down under acceleration, discussed earlier.

        t: python float, or a tensor broadcastable against bi_flow's
           (B,1,H,W) spatial dims -- e.g. shape (B,1,1,1) if different
           samples in the batch want different timestamps.
        """
        F01 = bi_flow[:, 0:2]
        F10 = bi_flow[:, 2:4]
        Ft0 = -(1 - t) * t * F01 + (t * t) * F10
        Ft1 = (1 - t) * (1 - t) * F01 - t * (1 - t) * F10
        return torch.cat([Ft0, Ft1], dim=1)

    @staticmethod
    def mask_from_visibility(visibility, t, eps=1e-6):
        """Derives a t-dependent blend weight (probability that a pixel
        should be taken from warped_img0) from the single t-independent
        visibility/occlusion logit, using the same
        occlusion-aware-blend idea as Super SloMo/DAIN:

            mask_t = (1-t)*V0 / ((1-t)*V0 + t*(1-V0))

        where V0 = sigmoid(visibility). Returned directly as a
        probability in (0,1) -- no further sigmoid needed downstream.
        """
        V0 = torch.sigmoid(visibility)
        num = (1 - t) * V0
        den = (1 - t) * V0 + t * (1 - V0) + eps
        return num / den

    # ------------------------------------------------------------------
    # synthesis: same spirit as the original coraseWarp_and_Refine,
    # but takes an already-derived (flow_t, mask_t) instead of the
    # network's direct bilateral prediction.
    # ------------------------------------------------------------------
    def synthesize(self, img0, img1, af, flow_t, mask_t):
        warped_img0 = warp(img0, flow_t[:, :2])
        warped_img1 = warp(img1, flow_t[:, 2:4])
        c0, c1 = self.warp_features(af, flow_t)
        tmp = self.unet(img0, img1, warped_img0, warped_img1, mask_t, flow_t, c0, c1)
        res = tmp[:, :3] * 2 - 1
        merged = warped_img0 * mask_t + warped_img1 * (1 - mask_t)
        pred = torch.clamp(merged + res, 0, 1)
        return pred, merged, warped_img0, warped_img1

    def warp_features(self, xs, flow):
        y0 = []
        y1 = []
        B = xs[0].size(0) // 2
        for x in xs:
            y0.append(warp(x[:B], flow[:, 0:2]))
            y1.append(warp(x[B:], flow[:, 2:4]))
            flow = F.interpolate(flow, scale_factor=0.5, mode="bilinear", align_corners=False,
                                  recompute_scale_factor=False) * 0.5
        return y0, y1

    # ------------------------------------------------------------------
    # Backward-compat API: matches the ORIGINAL VFIMamba method names/
    # signatures used by Trainer.py's hr_inference (calculate_flow +
    # coraseWarp_and_Refine, called as two separate steps so a caller
    # can run flow estimation at a downscaled resolution and synthesis
    # at full resolution, exactly as hr_inference already does). Keeping
    # these means Trainer.py does not need to change at all.
    # ------------------------------------------------------------------
    def estimate_flow_and_mask(self, img0, img1, timestep, local=False, af=None):
        """Model-agnostic entry point: returns (flow_t, mask_t, af) for
        the requested timestep. AccelFlow overrides this to also fold
        in its acceleration term; callers (Trainer.py) don't need to
        know which subclass they're holding.
        """
        bi_flow, visibility, af = self.estimate_bi_flow(img0, img1, local=local, af=af)
        t = timestep if torch.is_tensor(timestep) else \
            (img0[:, :1].clone() * 0 + 1) * float(timestep)
        flow_t = self.flow_from_bi(bi_flow, t)
        mask_t = self.mask_from_visibility(visibility, t)
        return flow_t, mask_t, af

    def calculate_flow(self, imgs, timestep, local=False, af=None):
        """Original signature: imgs is the channel-concatenated
        (img0, img1) tensor. Returns (flow, mask) for `timestep`.
        `flow` is now derived from the once-computed bidirectional
        pair rather than being directly regressed, and `mask` is
        returned as a probability in (0,1) rather than a pre-sigmoid
        logit (see `synthesize`, which expects it that way already --
        no double-sigmoid bug when these two are used together).
        """
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        flow_t, mask_t, _ = self.estimate_flow_and_mask(img0, img1, timestep, local=local, af=af)
        return flow_t, mask_t

    def coraseWarp_and_Refine(self, imgs, af, flow, mask):
        """Original signature/return type (a single predicted frame),
        thin wrapper around `synthesize`.
        """
        img0, img1 = imgs[:, :3], imgs[:, 3:6]
        pred, merged, warped_img0, warped_img1 = self.synthesize(img0, img1, af, flow, mask)
        return pred

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, x, local=False, timestep=0.5, scale=0):
        """Single-timestep interpolation. Signature-compatible with the
        original `forward`, but internally the expensive bi_flow
        estimation is decoupled from `timestep`.
        """
        if scale > 0:
            x_o = x
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

        img0, img1 = x[:, :3], x[:, 3:6]

        # ---- t-independent, runs once ----
        bi_flow, visibility, af = self.estimate_bi_flow(img0, img1, local=local)

        if scale > 0:
            img0, img1 = x_o[:, :3], x_o[:, 3:6]
            af = self.feature_bone(img0, img1)
            up = img0.shape[3] / bi_flow.shape[3]
            bi_flow = F.interpolate(bi_flow, scale_factor=up, mode="bilinear", align_corners=False) * up
            visibility = F.interpolate(visibility, scale_factor=up, mode="bilinear", align_corners=False)

        t = timestep if torch.is_tensor(timestep) else \
            (img0[:, :1].clone() * 0 + 1) * float(timestep)

        # ---- t-dependent, cheap, runs per requested t ----
        flow_t = self.flow_from_bi(bi_flow, t)
        mask_t = self.mask_from_visibility(visibility, t)
        pred, merged, warped_img0, warped_img1 = self.synthesize(img0, img1, af, flow_t, mask_t)

        # NOTE: flow_list holds flow_t (the actual per-t flow used for
        # warping), not the raw bidirectional pair -- matches what the
        # original architecture's flow_list contained, in case any
        # demo/visualization code inspects it.
        return [flow_t], [mask_t], [merged], pred

    def forward_multi_t(self, x, timesteps, local=False, scale=0):
        """Nx / arbitrary-timestamp interpolation. This is where the
        architectural change pays off: `estimate_bi_flow` (the
        Mamba backbone + BiHead/BiIFBlock stack) runs exactly ONCE,
        no matter how many timesteps you request. Only the cheap
        `flow_from_bi` + `mask_from_visibility` + `synthesize` steps
        repeat per t.

        timesteps: EITHER
        - a (B, T_max) tensor (train.py's batched path, one
            timestep column per interior-frame slot, potentially
            different per sample b) -- this is what
            dataset.ragged_collate / train.py's training loop produce.
        - a plain iterable of python floats, one t shared across the
            WHOLE batch (Trainer.py's inference_multi_t calling
            convention, unaffected by the batching revision) -- each
            entry is broadcast to every sample the same way `forward()`
            already does.

        Returns: list of predicted frames. Length T_max (batched path)
        or len(timesteps) (legacy path); each entry is (B, C, H, W).
        """
        if scale > 0:
            x_o = x
            x = F.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=False)

        img0, img1 = x[:, :3], x[:, 3:6]
        bi_flow, visibility, af = self.estimate_bi_flow(img0, img1, local=local)

        if scale > 0:
            img0, img1 = x_o[:, :3], x_o[:, 3:6]
            af = self.feature_bone(img0, img1)
            up = img0.shape[3] / bi_flow.shape[3]
            bi_flow = F.interpolate(bi_flow, scale_factor=up, mode="bilinear", align_corners=False) * up
            visibility = F.interpolate(visibility, scale_factor=up, mode="bilinear", align_corners=False)

        preds = []
        if torch.is_tensor(timesteps) and timesteps.dim() == 2:
            # Batched path: timesteps is (B, T_max), one column per interior
            # slot, potentially a different t per sample. Loop over T_max
            # (time), not B (batch) -- each iteration handles the WHOLE
            # batch's slot i in one call.
            B, T_max = timesteps.shape
            for i in range(T_max):
                t = timesteps[:, i].view(B, 1, 1, 1).to(img0.dtype)
                flow_t = self.flow_from_bi(bi_flow, t)
                mask_t = self.mask_from_visibility(visibility, t)
                pred, _, _, _ = self.synthesize(img0, img1, af, flow_t, mask_t)
                preds.append(pred)
        else:
            # Legacy path: plain iterable of python floats, one t shared by
            # the whole batch per call -- unchanged from before this revision.
            for ts in timesteps:
                t = (img0[:, :1].clone() * 0 + 1) * float(ts)
                flow_t = self.flow_from_bi(bi_flow, t)
                mask_t = self.mask_from_visibility(visibility, t)
                pred, _, _, _ = self.synthesize(img0, img1, af, flow_t, mask_t)
                preds.append(pred)
        return preds