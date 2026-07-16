"""
Same Laplacian-pyramid loss as the base repo, with one change: the entire
pyramid computation is forced to run in float32, even when this loss is
called from inside a torch.cuda.amp.autocast(enabled=True) block.

WHY THIS IS NEEDED
-------------------
conv2d is on autocast's "always run in fp16" whitelist. That means if you
only do `input = input.float()` at the top of forward() and then call
conv_gauss() -> F.conv2d(...), autocast will silently cast the tensor back
to float16 before running the convolution anyway -- the .float() call gets
overridden for any op autocast controls. So laplacian_pyramid() ends up
producing float16 tensors regardless of what dtype you handed it.

Then upsample() does:
    torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
which defaults to float32 (no dtype/device inheritance from x), and
torch.cat() refuses to concatenate a float16 tensor with a float32 one.
That's the actual crash.

THE FIX
-------
Wrap the whole forward() body in `torch.cuda.amp.autocast(enabled=False)`.
This locally disables autocast for every op inside the block -- including
conv2d -- so the fp32 casts on `input`/`target` actually stick all the way
through the pyramid, and upsample()'s float32 torch.zeros(...) now matches
everything else. This is different from (and strictly stronger than) just
calling .float() on the inputs, which does NOT survive autocast's op
whitelist on its own.

Net effect: this loss always computes in float32, whether or not the
surrounding training step is under autocast(enabled=True). That costs a
small amount of speed relative to a "true" fp16 loss, but Laplacian
pyramid losses are numerically sensitive (repeated small differences
across pyramid levels), so full precision here is the safer default.
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gauss_kernel(channels=3):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def upsample(x):
    # NOTE: these torch.zeros(...) calls default to float32. That's fine
    # as long as x is also float32 by this point -- which, with the
    # autocast(enabled=False) wrapper in LapLoss.forward(), it now always
    # is. If you ever call upsample() from somewhere NOT protected by that
    # context manager, you're exposed to the same dtype-mismatch bug again.
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return conv_gauss(x_up, 4*gauss_kernel(channels=x.shape[1]))

def conv_gauss(img, kernel):
    # conv2d is autocast-whitelisted -- under a live autocast(enabled=True)
    # context this would normally run in fp16 no matter what dtype `img`
    # is, which is exactly the behavior LapLoss.forward() now disables.
    img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out

def laplacian_pyramid(img, kernel, max_levels=3):
    current = img
    pyr = []
    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down)
        diff = current - up
        pyr.append(diff)
        current = down
    return pyr

class LapLoss(torch.nn.Module):
    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = gauss_kernel(channels=channels)

    def forward(self, input, target):
        # THE FIX: disable autocast for this whole block, not just cast
        # the inputs. Without this context manager, conv_gauss()'s
        # F.conv2d call would get silently re-cast to fp16 by the
        # *outer* autocast context (if the caller is inside one), and
        # you'd hit the exact float16/float32 torch.cat crash this
        # docstring describes, even with `.float()` calls left in place.
        with torch.cuda.amp.autocast(enabled=False):
            input = input.float()
            target = target.float()
            pyr_input = laplacian_pyramid(img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
            pyr_target = laplacian_pyramid(img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
            return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))