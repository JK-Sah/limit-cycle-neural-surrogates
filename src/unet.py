"""
Convolutional U-Net on the periodic Navier-Stokes field.

Why a second field architecture. The segment-length result was measured on a
Fourier neural operator only, so it is open to the reading that it belongs to
spectral convolutions rather than to the objective. A U-Net is the other family
deployed autoregressively on PDE fields, and it shares nothing with the FNO
below the training loop: local convolutions instead of a truncated global
spectrum, a multiscale encoder-decoder instead of a fixed-resolution stack.

The interface matches FNO2d exactly -- [B, ch, H, W] in, same shape out, with
the same residual switch -- so `train` and `measure_delta` in fno.py run it
unchanged and only the model differs between the two arms.

Padding is circular because the Navier-Stokes domain is a periodic box. The
forcing is fixed in space, which a translation-equivariant network cannot see,
so position is supplied as sin/cos coordinate channels; the FNO gets the same
information from its appended linear grid.
"""
import math
import torch
import torch.nn as nn


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular"), nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1, padding_mode="circular"), nn.GELU())


class UNet2d(nn.Module):
    def __init__(self, ch=1, base=32, levels=3, residual=True, zero_out=False):
        super().__init__()
        self.residual = residual
        widths = [base * 2 ** i for i in range(levels + 1)]   # 32, 64, 128, 256
        self.enc = nn.ModuleList()
        cin = ch + 4
        for w in widths[:-1]:
            self.enc.append(_block(cin, w))
            cin = w
        self.pool = nn.AvgPool2d(2)
        self.mid = _block(widths[-2], widths[-1])
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for w_hi, w_lo in zip(widths[:0:-1], widths[-2::-1]):
            self.up.append(nn.ConvTranspose2d(w_hi, w_lo, 2, stride=2))
            self.dec.append(_block(2 * w_lo, w_lo))
        self.out = nn.Conv2d(widths[0], ch, 1)
        if zero_out:
            # start the residual network at the identity map. Without this the
            # untrained U-Net's update is O(1) and a 25- or 40-step unroll can
            # sit on a loss plateau near the field variance (~3e-2) for all of
            # 3000 iterations: 5 of 10 default-initialised seeds did. It did not
            # cure it -- 5 of 10 zero-initialised seeds stayed there too.
            nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    @staticmethod
    def _coords(x):
        B, _, H, W = x.shape
        gx = torch.arange(H, device=x.device, dtype=x.dtype) * (2 * math.pi / H)
        gy = torch.arange(W, device=x.device, dtype=x.dtype) * (2 * math.pi / W)
        gx = gx.view(1, 1, H, 1).expand(B, 1, H, W)
        gy = gy.view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, torch.sin(gx), torch.cos(gx),
                          torch.sin(gy), torch.cos(gy)], 1)

    def forward(self, x):
        h = self._coords(x)
        skips = []
        for blk in self.enc:
            h = blk(h)
            skips.append(h)
            h = self.pool(h)
        h = self.mid(h)
        for up, dec, s in zip(self.up, self.dec, reversed(skips)):
            h = dec(torch.cat([up(h), s], 1))
        y = self.out(h)
        return x + y if self.residual else y
