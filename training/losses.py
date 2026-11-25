import torch
from torch import nn
from torch.nn import functional as F


def ycbcr(rgb):
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    # Full-range BT.601, values in [0,1], no per-image min/max normalization.
    y = .299*r + .587*g + .114*b
    cb = -.168736*r - .331264*g + .5*b + .5
    cr = .5*r - .418688*g - .081312*b + .5
    return y, cb, cr


class FusionLoss(nn.Module):
    def __init__(self, intensity=1., gradient=10., color=1.):
        super().__init__()
        self.weights = intensity, gradient, color
        kernel = torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]]) / 8
        self.register_buffer('sobel', torch.stack([kernel, kernel.T])[:,None])

    def edges(self, image):
        xy = F.conv2d(F.pad(image, (1,1,1,1), mode='replicate'), self.sobel)
        return xy.abs().sum(1, keepdim=True)

    def forward(self, output, visible, infrared):
        yf, cbf, crf = ycbcr(output)
        yv, cbv, crv = ycbcr(visible)
        intensity = F.l1_loss(yf, torch.maximum(yv, infrared))
        gradient = F.l1_loss(self.edges(yf), torch.maximum(self.edges(yv), self.edges(infrared)))
        color = F.l1_loss(cbf, cbv) + F.l1_loss(crf, crv)
        total = sum(w*x for w,x in zip(self.weights, (intensity,gradient,color)))
        return total, dict(intensity=intensity, gradient=gradient, color=color)


def restoration_loss(output, batch):
    return 0.5 * ((output["vis"] - batch["vh"]).abs().mean()
                  + (output["ir"] - batch["ih"]).abs().mean())


def routed_backward(model, recovery, fusion):
    """Use full recovery gradients and a quarter of fusion gradients on VIS.

    The 0.5 recovery coefficient is applied here, in addition to the mean over
    the two restoration modalities. Both terms are already microbatch-weighted.
    """
    prefixes = ("ev.", "av.", "rv.", "hv.")
    vis = [p for name, p in model.named_parameters() if name.startswith(prefixes)]
    extra = torch.autograd.grad(fusion, vis, allow_unused=True, retain_graph=True)
    (0.5 * recovery + fusion).backward()
    with torch.no_grad():
        for parameter, gradient in zip(vis, extra):
            if gradient is not None:
                parameter.grad.sub_(gradient, alpha=0.75)
