"""Item statistics on a binary response matrix X (models x items)."""
import numpy as np


def ability(X):
    """Standardised logit of each model's accuracy."""
    s = X.mean(1)
    n = X.shape[1]
    s = (s * n + 0.5) / (n + 1.0)
    z = np.log(s / (1 - s))
    sd = z.std()
    return (z - z.mean()) / (sd if sd > 0 else 1.0)


def logistic_fit(X, z, iters=25, ridge=1e-3):
    """Per-item logistic regression of X[:, j] on (1, z). Returns (b0, b1, residuals, P)."""
    Xf = X.astype(np.float64)
    J = X.shape[1]
    p = np.clip(Xf.mean(0), 1e-3, 1 - 1e-3)
    b0 = np.log(p / (1 - p)); b1 = np.zeros(J)
    zc = z[:, None]
    for _ in range(iters):
        P = 1 / (1 + np.exp(-np.clip(b0[None, :] + zc * b1[None, :], -30, 30)))
        W = P * (1 - P)
        g0 = (Xf - P).sum(0) - ridge * b0
        g1 = (zc * (Xf - P)).sum(0) - ridge * b1
        h00 = W.sum(0) + ridge; h01 = (zc * W).sum(0); h11 = (zc ** 2 * W).sum(0) + ridge
        det = h00 * h11 - h01 ** 2
        d0 = (h11 * g0 - h01 * g1) / det; d1 = (-h01 * g0 + h00 * g1) / det
        step = np.maximum(np.abs(d0), np.abs(d1))
        scale = np.minimum(1.0, 1.0 / np.maximum(step, 1e-12))
        b0 += scale * d0; b1 += scale * d1
        if step.max() < 1e-7:
            break
    P = 1 / (1 + np.exp(-np.clip(b0[None, :] + zc * b1[None, :], -30, 30)))
    return b0, b1, Xf - P, P


def residuals(X, z=None):
    """Ability-adjusted residuals of every response."""
    z = ability(X) if z is None else z
    return logistic_fit(X, z)[2]


def residual_similarity(X, z=None):
    """Model-by-model correlation of ability-adjusted residuals."""
    E = residuals(X, z)
    E = E - E.mean(1, keepdims=True)
    nrm = np.sqrt((E ** 2).sum(1)); nrm[nrm == 0] = 1.0
    E = E / nrm[:, None]
    return E @ E.T


def item_rest(X):
    """Corrected item-rest correlation of every item (NaN where undefined)."""
    Xf = X.astype(np.float64)
    rest = (Xf.sum(1)[:, None] - Xf) / (X.shape[1] - 1)
    xc = Xf - Xf.mean(0); rc = rest - rest.mean(0)
    den = np.sqrt((xc ** 2).sum(0) * (rc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (xc * rc).sum(0) / den
    r[den == 0] = np.nan
    return r
