"""Shared loaders and estimators for the cultural-field project.

Conventions: unit = distrito electoral (CODELEC); spatial set = domestic,
non-institutional, non-offshore, with CRTM05 coordinates; election tags are strings.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "figures"
TAGS = ["2010", "2014", "2018", "2022r1", "2022r2", "2026"]
# log-spaced distance bins (km)
R_EDGES = np.array([1, 2, 3, 4.5, 7, 10, 15, 20, 30, 45, 65, 90, 130, 180, 250, 350], float)


# ----------------------------------------------------------------------------- data
def load_master(spatial: bool = True) -> pd.DataFrame:
    df = pd.read_csv(PROC / "de_master.csv", dtype={"codelec": str})
    if spatial:
        df = df[(df.institutional == 0) & (df.offshore == 0) & df.E_crtm05.notna() & df.N_crtm05.notna()].copy()
    return df.reset_index(drop=True)


def winner_votes(tag: str) -> pd.DataFrame:
    """codelec, W (winner votes), validos for one result set. Winner = national plurality."""
    df = pd.read_csv(PROC / f"results_{tag}_de.csv", dtype={"codelec": str})
    df = df[df.codelec.notna() & (df.codelec != "")]
    meta = {"codelec", "n_jrv", "n_jrv_zero_elec", "electores_padron", "electores_results", "recibidos",
            "validos", "nulos", "blancos", "year", "prov", "cnum", "canton", "dnum", "dist_admin",
            "de_name", "n_codes", "institutional", "juntas", "electorado"}
    parties = [c for c in df.columns if c not in meta]
    win = df[parties].sum().idxmax()
    out = df[["codelec", "validos", win]].rename(columns={win: "W"})
    out.attrs["winner"] = win.replace("P:", "")
    return out.groupby("codelec", as_index=False).sum() if out.codelec.duplicated().any() else out.reset_index(drop=True)


def pair_distance_km(E: np.ndarray, N: np.ndarray) -> np.ndarray:
    return np.hypot(E[:, None] - E[None, :], N[:, None] - N[None, :]) / 1e3


# ----------------------------------------------------------------------- observables
def size_bin_residual(x: np.ndarray, N: np.ndarray, bin_size: int = 100, sigma_cut: float = 5.0):
    """Rank units by N, subtract the size-bin mean (Borghesi–Bouchaud Eq. 1, m_alpha), drop
    outliers beyond sigma_cut within-bin sigmas. Returns residual, keep mask, bin id."""
    order = np.argsort(N, kind="stable")
    nb = max(1, len(x) // bin_size)
    bins = np.empty(len(x), int)
    res = np.empty(len(x), float)
    keep = np.ones(len(x), bool)
    for b in range(nb):
        idx = order[b * bin_size:(b + 1) * bin_size if b < nb - 1 else None]
        bins[idx] = b
        m, s = x[idx].mean(), x[idx].std()
        res[idx] = x[idx] - m
        keep[idx] = np.abs(res[idx]) <= sigma_cut * max(s, 1e-12)
    # re-centre after outlier removal
    for b in range(nb):
        sel = (bins == b) & keep
        if sel.any():
            res[sel] -= res[sel].mean()
    return res, keep, bins


def logit(p): return np.log(p / (1 - p))


# ------------------------------------------------------------- correlation function
def pair_stats(x: np.ndarray, D: np.ndarray, edges: np.ndarray = R_EDGES, n_boot: int = 200, seed: int = 0):
    """Flat pair-average correlation C(r) = <x_a x_b>_bin / <x^2> and variogram
    V(r) = <(x_a - x_b)^2>_bin, with unit-level bootstrap SE (resampled unit multiplicities
    weight each pair by w_a w_b, self-pairs excluded)."""
    n = len(x)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    bid = np.digitize(d, edges) - 1
    nbin = len(edges) - 1
    prod = (x[:, None] * x[None, :])[iu]
    sq = ((x[:, None] - x[None, :]) ** 2)[iu]
    var = np.mean(x ** 2)
    rows = []
    C0 = np.full(nbin, np.nan); V0 = np.full(nbin, np.nan); npairs = np.zeros(nbin, int)
    for b in range(nbin):
        m = bid == b
        npairs[b] = m.sum()
        if m.sum() >= 30:
            C0[b] = prod[m].mean() / var
            V0[b] = sq[m].mean()
    rng = np.random.default_rng(seed)
    Cb = np.full((n_boot, nbin), np.nan); Vb = np.full((n_boot, nbin), np.nan)
    for r in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        ww = (w[:, None] * w[None, :])[iu]
        vb = np.sum(w * x ** 2) / np.sum(w)
        for b in range(nbin):
            m = bid == b
            sw = ww[m].sum()
            if npairs[b] >= 30 and sw > 0:
                Cb[r, b] = np.sum(ww[m] * prod[m]) / sw / vb
                Vb[r, b] = np.sum(ww[m] * sq[m]) / sw
    out = pd.DataFrame(dict(r_lo=edges[:-1], r_hi=edges[1:], r_mid=np.sqrt(edges[:-1] * edges[1:]),
                            n_pairs=npairs, C=C0, C_se=np.nanstd(Cb, 0), V=V0, V_se=np.nanstd(Vb, 0)))
    return out, var, Cb


def fit_log(r: np.ndarray, C: np.ndarray, se: np.ndarray, rmin=8.0, rmax=300.0):
    """C = -lambda^2 ln(r/L) = a - lambda^2 ln r, weighted LS on the window. Returns lambda^2, L."""
    m = (r >= rmin) & (r <= rmax) & np.isfinite(C) & np.isfinite(se) & (se > 0)
    if m.sum() < 3:
        return np.nan, np.nan
    X = np.c_[np.ones(m.sum()), -np.log(r[m])]
    w = 1 / se[m] ** 2
    beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * C[m]))
    a, lam2 = beta
    L = np.exp(a / lam2) if lam2 > 0 else np.nan
    return lam2, L


def save_json(obj, path):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(type(o))
    Path(path).write_text(json.dumps(obj, indent=1, default=conv))


# ------------------------------------------------------------------ kernels, Laplacians, GFF likelihood
from scipy.linalg import solve_triangular
from scipy.optimize import minimize


def geo_kernel(D: np.ndarray, lc: float) -> np.ndarray:
    G = np.exp(-D / lc); np.fill_diagonal(G, 0.0); return G


def knn_graph(M: np.ndarray, k: int) -> np.ndarray:
    """symmetrised k-nearest-neighbour adjacency in the metric M (zero diagonal)."""
    n = len(M); A = np.zeros((n, n))
    nn = np.argsort(M + np.eye(n) * 1e9, 1)[:, :k]
    A[np.arange(n)[:, None], nn] = 1.0
    return np.maximum(A, A.T)


def match_k(G: np.ndarray, M: np.ndarray) -> int:
    """k such that the symmetrised kNN graph in M has mean degree closest to G's mean weighted degree."""
    target = G.sum(1).mean(); best = (np.inf, 1)
    for k in range(1, 40):
        d = knn_graph(M, k).sum(1).mean()
        if abs(d - target) < best[0]: best = (abs(d - target), k)
    return best[1]


def mix_kernels(G: np.ndarray, A: np.ndarray, w: float) -> np.ndarray:
    """w*Gamma_geo + (1-w)*Gamma_sur after scaling both to equal total weight."""
    return w * (G * A.sum() / G.sum()) + (1 - w) * A


def laplacian(Gw: np.ndarray, variant: str = "comb_rw") -> np.ndarray:
    n = len(Gw); d = Gw.sum(1)
    if variant == "comb_rw":      # adopted 2026-09-21 (Korniss beta = -1)
        Gr = Gw / np.outer(d, d); L = np.diag(Gr.sum(1)) - Gr; return L * (n / np.trace(L))
    if variant == "comb":
        L = np.diag(d) - Gw; return L * (n / np.trace(L))
    if variant == "sym":
        Dm = 1 / np.sqrt(d); return np.eye(n) - Dm[:, None] * Gw * Dm[None, :]
    raise ValueError(variant)


def eig_pinv(L: np.ndarray, tol: float = 1e-9):
    """eigendecomposition and pseudo-inverse (zero modes dropped). Returns P, lam, U."""
    lam, U = np.linalg.eigh(L); keep = lam > tol * lam.max()
    return (U[:, keep] / lam[keep]) @ U[:, keep].T, lam, U


def gff_nll(theta, P, taus, binom, het=None):
    """-log L of taus ~ N(0, s2 P + diag(smu * het + binom)); theta = log(s2, smu); het defaults to 1."""
    s2, smu = np.exp(theta)
    diag = smu * (het if het is not None else 1.0) + binom
    S = s2 * P + np.diag(diag)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30
    Y = solve_triangular(Lc, np.asarray(taus).T, lower=True, check_finite=False)
    return 0.5 * (len(taus) * 2 * np.log(np.diag(Lc)).sum() + (Y * Y).sum())


def profile_nll(taus, P, binom, x0, het=None, maxfev=60):
    r = minimize(gff_nll, x0, args=(P, taus, binom, het), method="Nelder-Mead",
                 options=dict(xatol=0.03, fatol=0.3, maxfev=maxfev, initial_simplex=np.array([x0, x0 + [0.3, 0], x0 + [0, 0.3]])))
    return r.fun, r.x
