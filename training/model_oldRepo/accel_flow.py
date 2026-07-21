"""
Acceleration extension for the bi-flow VFI model (IFNet_bi_flow.py).

WHY THIS EXISTS
------------------------------------------------------------------
We established earlier that acceleration cannot be recovered from
pure geometric consistency of (F_{0->1}, F_{1->0}) -- there just
isn't enough information in two frames. Any two-frame model can only
ever produce a *learned prior* standing in for real acceleration.

This file makes that prior EXPLICIT and DIRECTLY SUPERVISED, instead
of leaving it as an implicit, unverifiable side-effect of a black-box
photometric loss. The trick: at *train* time we do have a third,
real frame (the ground-truth middle frame in any VFI triplet/dataset).
We use it to derive a closed-form target for what the acceleration
"should have been", and train a blind predictor (which only sees
img0/img1, same as at inference) to match that target. This is
exactly a guess-vs-actual distillation loop, made mathematically
concrete rather than left implicit.

MOTION MODEL
------------------------------------------------------------------
    D  = F_{0->1}                          (already estimated, unprivileged)
    a  = learned acceleration/curvature field (NEW, unprivileged predictor)

    p(t) = t*D + a*t*(t-1)      <- vanishes at t=0 and t=1, so it can
                                    only bend the path *between* the
                                    two observed frames, never disagree
                                    with the directly observed D.
    F_{t->0} = -p(t)
    F_{t->1} =  p(t) - D

    a=0 reproduces the pure linear-scaling baseline from before.

    `a` and `D` do not depend on t at all -- t only enters through the
    cheap p(t) evaluation. This is what lets a SINGLE guess of `a` be
    reused for ANY intermediate timestep(s) requested, including many
    at once (see forward_multi_t below): the expensive backbone +
    accel_head work happens once per (img0, img1) pair, and only the
    cheap p(t)/mask_t/synthesize steps repeat per t.

TRAIN-TIME TARGET FOR a (closed form, no separate teacher network)
------------------------------------------------------------------
    Given a real triplet (img0, img_mid_gt, img1) at known t_gt (any
    value strictly between 0 and 1 -- t_gt is NOT assumed to be 0.5;
    variable-t datasets like X4K1000FPS work here unchanged):
      1. Run the SAME bi-flow estimator on (img0, img_mid_gt) to get
         a real (not guessed) flow  p_true = F_{0->mid}.
      2. Since p_true = t_gt*D + a_true*t_gt*(t_gt-1), solve:
             a_true = (p_true - t_gt*D) / (t_gt*(t_gt-1))
      3. L_accel = || a_student(img0,img1) - a_true.detach() ||_1

This a_true pass uses privileged information (img_mid_gt) and is
detached -- it's a target, not something optimized jointly. At
inference, a_student never sees img_mid_gt; it has to guess from
img0/img1 content alone, same as before, but now it was explicitly
trained to guess well rather than only implicitly rewarded for it.
"""

# PACKAGING NOTE: same requirement as flow_estimation.py -- this file
# must sit in the same package as `warplayer.py` that train.py clears
# each epoch via `warplayer.backwarp_tenGrid.clear()`, or the cache
# this file's warp() calls use won't be the one train.py is clearing.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .warplayer import warp
from .flow_estimation import MultiScaleFlow, BiIFBlock, conv


class AccelHead(nn.Module):
    """Small extra head, run once after estimate_bi_flow converges.
    Predicts a single acceleration/curvature field 'a' from img0, img1,
    and the already-estimated D=F_{0->1} (warped context included).
    Bounded via tanh so it can only ever be a *correction* to the
    linear path, not dominate it -- keeps early training stable.
    """
    def __init__(self, c=48, accel_scale=1.0):
        super().__init__()
        self.accel_scale = accel_scale
        # img0(3) + img1(3) + warped_img0_via_D(3) + D(2) = 11 channels in
        self.net = nn.Sequential(
            conv(11, c),
            conv(c, c),
            conv(c, c),
            nn.Conv2d(c, 2, 3, 1, 1),
        )

    def forward(self, img0, img1, D):
        warped_img0 = warp(img0, D)  # img0 pushed forward by the linear estimate
        x = torch.cat([img0, img1, warped_img0, D], dim=1)
        raw = self.net(x)
        return torch.tanh(raw) * self.accel_scale


class AccelFlow(MultiScaleFlow):
    def __init__(self, backbone, accel_scale=1.0, **kargs):
        super().__init__(backbone, **kargs)
        self.accel_head = AccelHead(c=kargs.get('accel_hidden', 48), accel_scale=accel_scale)

    # ------------------------------------------------------------------
    @staticmethod
    def flow_from_bi_accel(D, a, t):
        """Replaces MultiScaleFlow.flow_from_bi. D=F_{0->1} only (F_{1->0}
        is kept solely as an occlusion-consistency cue elsewhere, not
        used in the trajectory itself anymore -- 'a' carries the
        nonlinearity instead).

        t: python float, or a tensor broadcastable against D's (B,1,H,W)
           spatial dims -- e.g. shape (B,1,1,1)/(B,1,H,W) if different
           samples (or different requested timestamps) need different t.
           No restriction to t=0.5 anywhere in this formula.
        """
        p = t * D + a * t * (t - 1)
        Ft0 = -p
        Ft1 = p - D
        return torch.cat([Ft0, Ft1], dim=1)

    @staticmethod
    def solve_accel_target(D, p_true, t_gt, eps=1e-4):
        """Closed-form 'actual acceleration' derived from the real
        middle frame. t_gt must not be 0 or 1 (no interior information
        there), but is otherwise UNRESTRICTED -- works for any
        intermediate frame position, not just t_gt=0.5. D and p_true:
        (B,2,H,W). t_gt: python float or tensor broadcastable to
        (B,1,H,W).
        """
        denom = t_gt * (t_gt - 1)
        denom = torch.where(denom.abs() < eps, torch.full_like(denom, eps), denom) \
            if torch.is_tensor(denom) else (denom if abs(denom) > eps else eps)
        return (p_true - t_gt * D) / denom

    # ------------------------------------------------------------------
    # Overrides MultiScaleFlow.estimate_flow_and_mask so the
    # calculate_flow/coraseWarp_and_Refine compat shims inherited from
    # the base class (used by Trainer.py's hr_inference) automatically
    # pick up the acceleration term too, with zero Trainer.py changes.
    # ------------------------------------------------------------------
    def estimate_flow_and_mask(self, img0, img1, timestep, local=False, af=None):
        bi_flow, visibility, af = self.estimate_bi_flow(img0, img1, local=local, af=af)
        D = bi_flow[:, 0:2]
        a = self.accel_head(img0, img1, D)
        t = timestep if torch.is_tensor(timestep) else \
            (img0[:, :1].clone() * 0 + 1) * float(timestep)
        flow_t = self.flow_from_bi_accel(D, a, t)
        mask_t = self.mask_from_visibility(visibility, t)
        return flow_t, mask_t, af

    def forward(self, x, timestep=0.5, local=False, scale=0):
        """Kept SIGNATURE- AND ARITY-COMPATIBLE with MultiScaleFlow.forward
        (4-tuple return), so `_, _, _, pred = self.net(...)` in
        Trainer.py/train.py keeps working unchanged regardless of which
        of the two model classes is loaded. `a` is intentionally not
        returned here -- training against it goes through
        `training_step`/`accel_distillation_loss` instead, which
        recompute what they need directly.

        `timestep` accepts ANY value in (0, 1) (or a per-sample tensor
        of such values) -- there is no assumption of 0.5 anywhere in
        this path.

        FIXED: previously this ignored `scale` entirely, unlike the
        base class's forward(), which downsamples for the expensive
        estimate_bi_flow pass and re-runs feature_bone/accel_head at
        full resolution afterward. That silently disabled the scale
        speed-up for AccelFlow. Now mirrors the base class's handling.
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

        D = bi_flow[:, 0:2]
        a = self.accel_head(img0, img1, D)  # now computed at full res when scale>0, matching base class

        t = timestep if torch.is_tensor(timestep) else \
            (img0[:, :1].clone() * 0 + 1) * float(timestep)

        flow_t = self.flow_from_bi_accel(D, a, t)
        mask_t = self.mask_from_visibility(visibility, t)
        pred, merged, warped_img0, warped_img1 = self.synthesize(img0, img1, af, flow_t, mask_t)
        return [flow_t], [mask_t], [merged], pred

    # ------------------------------------------------------------------
    # NEW: Nx / arbitrary-timestamp interpolation for AccelFlow.
    #
    # MultiScaleFlow.forward_multi_t (inherited by default) computes
    # bi_flow/visibility ONCE and then calls self.flow_from_bi(bi_flow, t)
    # per timestep -- but flow_from_bi is the LINEAR-ONLY formula. Left
    # un-overridden, calling forward_multi_t on an AccelFlow instance
    # would silently ignore accel_head and the learned acceleration term
    # for every timestep, defeating the entire point of this file.
    #
    # This override keeps the same efficiency payoff (backbone +
    # accel_head run exactly once, regardless of how many timesteps are
    # requested) while actually using flow_from_bi_accel per t, so any
    # intermediate frame -- or a whole batch of them for slow-motion /
    # Nx interpolation -- gets the acceleration-aware trajectory.
    # ------------------------------------------------------------------
    def forward_multi_t(self, x, timesteps, local=False, scale=0):
        """
        timesteps: iterable of floats in (0, 1) -- any intermediate
        frame position(s), not restricted to 0.5 or any fixed grid.
        Returns: list of predicted frames, one per timestep.
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

        D = bi_flow[:, 0:2]
        a = self.accel_head(img0, img1, D)  # computed ONCE, reused for every requested t below

        preds = []
        for ts in timesteps:
            t = (img0[:, :1].clone() * 0 + 1) * float(ts)
            flow_t = self.flow_from_bi_accel(D, a, t)
            mask_t = self.mask_from_visibility(visibility, t)
            pred, _, _, _ = self.synthesize(img0, img1, af, flow_t, mask_t)
            preds.append(pred)
        return preds

    # ------------------------------------------------------------------
    def training_step(self, img0, img_mid_gt, img1, t_gt, w_photo=1.0, w_accel=1.0, local=False):
        """One train step implementing the guess -> compare-to-actual
        -> correct loop. t_gt: python float (e.g. 0.5) or per-sample
        tensor if your dataset has variable timestamps -- ANY value in
        (0, 1) is supported, so datasets whose triplets sit at varied
        interior positions (not just a fixed t=0.5 middle frame) train
        correctly with no changes needed here.

        local: should match whatever `local` setting you use for this
        model elsewhere (e.g. at inference) -- accel_head is a shared
        module, so the student's D here should come from the same
        local-refinement setting it will see in deployment. The
        teacher pass always uses local=True regardless (see
        accel_distillation_loss's docstring for why).

        Returns (loss, logs_dict). Wire this into your existing
        optimizer loop; nothing else about your training harness needs
        to change.
        """
        B = img0.size(0)
        t = (img0[:, :1].clone() * 0 + 1) * (t_gt if not torch.is_tensor(t_gt) else t_gt)

        # ---- student: blind guess, same inputs available at inference ----
        bi_flow, visibility, af = self.estimate_bi_flow(img0, img1, local=local)
        D = bi_flow[:, 0:2]
        a_student = self.accel_head(img0, img1, D)

        flow_t = self.flow_from_bi_accel(D, a_student, t)
        mask_t = self.mask_from_visibility(visibility, t)
        pred, merged, _, _ = self.synthesize(img0, img1, af, flow_t, mask_t)

        loss_photo = F.l1_loss(pred, img_mid_gt)

        # ---- privileged target: real flow to the real middle frame ----
        with torch.no_grad():
            bi_flow_0mid, _, _ = self.estimate_bi_flow(img0, img_mid_gt, local=True)
            p_true = bi_flow_0mid[:, 0:2]                     # real F_{0->mid}
            a_true = self.solve_accel_target(D.detach(), p_true, t)

        loss_accel = F.l1_loss(a_student, a_true)

        loss = w_photo * loss_photo + w_accel * loss_accel
        logs = {"loss_photo": loss_photo.item(), "loss_accel": loss_accel.item()}
        return loss, logs


def accel_distillation_loss(net, img0, img_mid_gt, img1, t_gt, local=False):
    """Standalone version of the acceleration loss inside
    `AccelFlow.training_step`, usable when a training script (e.g.
    train.py) already computes `pred` itself via `model.net(...)` and
    has its own loss-accumulation/scaling logic that training_step
    doesn't know about. Call this once per step and add its result to
    whatever loss you already have, before dividing by the
    accumulation-window size.

    t_gt: any value in (0, 1) -- per-item variable timesteps (e.g. a
    dataset that samples a different interior frame/position each
    epoch, as described in train.py's per-epoch seeded timestep
    selection) are supported with no changes here.

    local: MUST match the `local` flag the main forward pass used for
    this same step (e.g. `model.local` in train.py). `accel_head` is a
    single shared module that also gets called inside `AccelFlow.forward`
    on a D that WAS run through local refinement whenever `local` is
    truthy there. If this function always recomputed D with local=False
    regardless, accel_head would be trained against a coarser D
    distribution than the one it actually sees at inference -- a real
    train/inference mismatch, not just a cosmetic difference. Passing
    the same `local` here keeps the two consistent.

    The TEACHER pass (img0, img_mid_gt) always uses local=True
    (full refinement), independent of the `local` argument above: it
    runs under `torch.no_grad()` so it costs nothing at inference time,
    and a better-refined target is strictly better supervision for
    a_true regardless of what the student's D looked like.

    Recomputes estimate_bi_flow on (img0, img1) rather than reusing
    anything from an earlier forward() call, since that earlier call
    may have run at a downsampled `scale` for speed -- this needs
    full-resolution img0/img1 to line up with img_mid_gt.
    """
    t = (img0[:, :1].clone() * 0 + 1) * float(t_gt) if not torch.is_tensor(t_gt) else t_gt
    bi_flow, visibility, af = net.estimate_bi_flow(img0, img1, local=local)
    D = bi_flow[:, 0:2]
    a_student = net.accel_head(img0, img1, D)
    with torch.no_grad():
        bi_flow_0mid, _, _ = net.estimate_bi_flow(img0, img_mid_gt, local=True)
        p_true = bi_flow_0mid[:, 0:2]
        a_true = net.solve_accel_target(D.detach(), p_true, t)
    return F.l1_loss(a_student, a_true)