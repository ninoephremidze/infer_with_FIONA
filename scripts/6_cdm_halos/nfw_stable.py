"""NFW lens potential with a numerically stable small-u branch (scratch; FIONA repo untouched).

psi(r) = (psi0/2) h(u), u = r/xs, with FIONA's NFW (lenses.NFW):
  h = ln^2(u/2) + (u^2 - 1) F(u)^2.
For u < 1, (1 - u^2) F^2 = atanh^2(s), s = sqrt(1 - u^2), and atanh(s) = ln((1 + s)/u) exactly, so with
L = ln(2/u), A = ln((1 + s)/u):  h = L^2 - A^2 = (L - A)(L + A),  L - A = log1p(u^2 / (1 + s)^2).
This avoids the cancellation of two ~ln^2(2/u) terms that FIONA's form suffers for u << 1 (needed when
lengths are in units of the Fresnel radius, u ~ 1e-7 - 1).
"""
import numpy as np
from infer_with_fiona.fiona.lenses import AxisymmetricLens


class NFWStable(AxisymmetricLens):
    def __init__(self, psi0=1.0, xs=1.0):
        self.psi0 = float(psi0)
        self.xs = float(xs)

    def psi_r(self, r):
        u = np.asarray(r, float) / self.xs
        h = np.empty_like(u)
        lo = u < 1.0
        if np.any(lo):
            ul = np.maximum(u[lo], 1e-300)
            s = np.sqrt((1.0 - ul) * (1.0 + ul))
            h[lo] = np.log1p(ul * ul / (1.0 + s) ** 2) * (np.log(2.0 / ul) + np.log((1.0 + s) / ul))
        hi = ~lo
        if np.any(hi):
            uh = u[hi]
            t = np.sqrt(np.maximum(uh * uh - 1.0, 0.0))
            F = np.where(t > 0, np.arctan(t) / np.where(t > 0, t, 1.0), 1.0)
            h[hi] = np.log(uh / 2.0) ** 2 + t * t * F * F
        return 0.5 * self.psi0 * h
