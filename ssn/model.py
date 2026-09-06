"""SSN - Spiking Survivor Network.

A temporal verifier for indoor survivor detection. The single-frame proposer
(YOLOv8n) is tuned for recall and therefore over-proposes: debris, mannequin
limbs, motion-blurred clutter and reflections all come back as candidate people.
This network votes on a short sequence of ego-motion-aligned crops of one tracked
candidate. LIF membranes integrate evidence across frames, so a survivor that is
weakly visible in every frame accumulates past threshold, while a one-frame
artefact leaks away before it can fire.

Design notes for the NIDAR AirMouse (indoor, GPS-denied) mission:
  - crops are 64x64, so the network is cheap enough to run 5 candidates per frame
    on a Raspberry Pi 5 CPU alongside the proposer and the SLAM front-end;
  - T is one timestep per real captured frame, so the network's time axis is
    wall-clock time, not an internal simulation axis;
  - channel 0 is luma, channel 1 is the ego-motion-compensated frame delta.
    Warp before differencing - an unwarped delta on a moving drone is dominated
    by camera translation and the survivor signal is lost in it.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

CROP = 64  # crop side, px - shared by training and runtime
T_STEPS = 6  # frames per verification window


class ATanSpike(torch.autograd.Function):
    """Heaviside forward, arctan surrogate gradient backward."""

    alpha = 2.0

    @staticmethod
    def forward(ctx, v):
        ctx.save_for_backward(v)
        return (v >= 0.0).to(v.dtype)

    @staticmethod
    def backward(ctx, grad):
        (v,) = ctx.saved_tensors
        a = ATanSpike.alpha
        return grad * (a / 2.0) / (1.0 + (math.pi / 2.0 * a * v).pow(2))


def spike(v):
    return ATanSpike.apply(v)


class LIF(nn.Module):
    """Leaky integrate-and-fire neuron with a per-channel learnable time constant.

        v[t] = v[t-1] * decay + x[t]
        s[t] = 1 if v[t] >= v_th else 0
        v[t] <- v[t] - s[t] * v_th        (hard reset by subtraction)

    decay is parameterised as sigmoid(w) so it stays in (0, 1) with no optimizer
    constraint. w = 0 gives decay 0.5, i.e. tau = 2 frames.
    """

    def __init__(self, channels: int, tau: float = 2.0, v_th: float = 1.0):
        super().__init__()
        d = 1.0 - 1.0 / tau
        self.w = nn.Parameter(torch.full((channels,), math.log(d / (1.0 - d))))
        self.register_buffer("v_th", torch.tensor(float(v_th)))

    def forward(self, x, v):
        shape = (1, -1, 1, 1) if x.dim() == 4 else (1, -1)
        decay = torch.sigmoid(self.w).view(shape)
        v = v * decay + x
        s = spike(v - self.v_th)
        return s, v - s * self.v_th


class SpikingBlock(nn.Module):
    """conv -> BN -> LIF, evaluated over the whole time axis.

    Convolution and normalisation are folded into one batched call over T*B; only
    the neuron state is looped. That fusion is the difference between usable and
    unusable inference speed on ARM.
    """

    def __init__(self, cin: int, cout: int, stride: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.lif = LIF(cout)

    def forward(self, x):  # x: (T, B, C, H, W) -> (T, B, C', H', W')
        t, b = x.shape[0], x.shape[1]
        h = self.bn(self.conv(x.flatten(0, 1)))
        h = h.view(t, b, *h.shape[1:])
        v = torch.zeros_like(h[0])
        out = []
        for step in range(t):
            s, v = self.lif(h[step], v)
            out.append(s)
        return torch.stack(out)


class SSN(nn.Module):
    """Binary survivor / not-survivor verifier over a crop sequence.

    Input  : (B, T, C, 64, 64) float in [-1, 1]
    Output : logit (B,) - readout is the mean over time of a linear head on the
             pooled spike rate of the last block ("voting" readout). The temporal
             integration lives in the hidden LIF membranes; averaging the readout
             keeps training stable and still reads only spikes.
    """

    def __init__(self, in_ch: int = 2, widths=(16, 32, 64)):
        super().__init__()
        chans = (in_ch,) + tuple(widths)
        self.blocks = nn.ModuleList(
            [SpikingBlock(chans[i], chans[i + 1]) for i in range(len(widths))]
        )
        self.head = nn.Linear(widths[-1], 1)
        self.last_spike_rate = 0.0  # diagnostic, updated in forward

    def forward(self, x):
        x = x.transpose(0, 1)  # (T, B, C, H, W)
        for block in self.blocks:
            x = block(x)
        self.last_spike_rate = float(x.detach().mean())
        pooled = x.mean(dim=(-2, -1))  # (T, B, C) - spike rate per channel
        return self.head(pooled).mean(dim=0).squeeze(-1)  # (B,)

    @torch.no_grad()
    def confidence(self, x):
        return torch.sigmoid(self(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def load(cls, path, map_location="cpu"):
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt.get("kwargs", {}))
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save(self, path, **kwargs):
        torch.save({"state_dict": self.state_dict(), "kwargs": kwargs}, path)


def _demo():
    torch.manual_seed(0)
    m = SSN()
    x = torch.randn(2, T_STEPS, 2, CROP, CROP)
    y = m(x)
    assert y.shape == (2,), y.shape
    assert m.confidence(x).shape == (2,)
    assert 0.0 <= m.last_spike_rate <= 1.0

    # the point of the LIF: weak-but-persistent evidence fires, an equally weak
    # one-off leaks away. Checked on the neuron, not the untrained net, whose
    # random weights say nothing.
    def fires(drive):  # drive: list of per-step inputs to one neuron
        lif, v, total = LIF(1), torch.zeros(1, 1, 1, 1), 0.0
        for value in drive:
            s, v = lif(torch.full((1, 1, 1, 1), value), v)
            total += float(s.detach().sum())
        return total

    assert fires([0.6] * T_STEPS) >= 1.0
    assert fires([0.6] + [0.0] * (T_STEPS - 1)) == 0.0

    y.sum().backward()  # surrogate gradient reaches the first conv
    assert m.blocks[0].conv.weight.grad.abs().sum() > 0
    print(f"ok: logits {tuple(y.shape)}, params {m.num_params()}, "
          f"spike rate {m.last_spike_rate:.3f}")


if __name__ == "__main__":
    _demo()
