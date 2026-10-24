"""Gate G3a — synthetic recovery of the kernel mixing weight w on the REAL district geometry and
REAL Lasker matrix (2026 roll, pooled surnames, N >= 200 electors).

Field model: tau' ~ N(0, sigma^2 L_w^+ + sigma_mu^2 I + diag(h / [N p (1-p)])),
Gamma_w = w Gamma_geo + (1-w) Gamma_sur, Gamma_geo = exp(-r/l_c) (l_c = 4.5 km), Gamma_sur = kNN in Lasker
space with k = round(mean weighted degree of Gamma_geo); both adjacencies scaled to equal total weight.
Laplacian variants (Korniss 2007 digest: the normalised L^+ is not a diffusive-field covariance):
  comb_rw : combinatorial Laplacian of Gamma_ij / (k_i k_j) (Korniss beta = -1) -- ADOPTED 2026-09-21 (only variant passing G3a)
  sym     : I - D^-1/2 Gamma D^-1/2 (original plan choice; fails the gate at the surname end) -- robustness only
  comb    : plain combinatorial Laplacian (Borghesi-Bouchaud / kill panel; marginal) -- robustness only
For each variant: simulate 5-election sets at w_true in {0, 0.5, 1} (3 sets), refit w on an 11-point grid
by profile likelihood over (sigma^2, sigma_mu^2); report w_hat, dNLL at +-0.2, and the model C(r) shape.
Gate: joint-5 w_hat within 0.15 of w_true for every set and dNLL(+-0.2) >= 3.
Outputs: data/processed/stage3a_recovery.csv, stage3a_summary.json, figures/stage3a_*.png
"""
import sys, time, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

ap = argparse.ArgumentParser(); ap.add_argument("--nmin", type=int, default=200); ap.add_argument("--lc", type=float, default=4.5)
ap.add_argument("--sets", type=int, default=3); ap.add_argument("--nel", type=int, default=5); ap.add_argument("--variants", default="comb_rw,sym,comb")
ap.add_argument("--sigma_phi2", type=float, default=0.08); ap.add_argument("--sigma_mu2", type=float, default=0.10)
ap.add_argument("--k", type=int, default=0, help="override the matched k of the surname kNN graph (0 = match the geo mean degree)"); ap.add_argument("--suffix", default="")
ap.add_argument("--lc_profile", action="store_true", help="generate at --lc and profile l_c (w profiled at each) over the real-data grid; skips the panel and the gate")
ap.add_argument("--landscape", action="store_true", help="with --lc_profile: generate from the fitted landscape model (K_r at 20 km + K_BC + unit term, no Laplacian) instead of Eq. (1)")
args = ap.parse_args()
T0 = time.time()

# ------------------------------------------------------------------ real geometry + Lasker
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True)
res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
s1 = __import__("json").load(open(cf.PROC / "stage1_summary.json"))
h = float(s1["variance_decomposition"]["2022r1-2026"]["h"]); h = min(max(h, 1.0), 3.0)
codes = las["codelec"].astype(str); Nel = las["N"] / 2.0
sel = (Nel >= args.nmin) & np.isin(codes, res.index)
codes = codes[sel]; L_las = las["L"][np.ix_(sel, sel)].astype(float); n = len(codes)
E = res.loc[codes, "E_crtm05"].to_numpy(float); Nc = res.loc[codes, "N_crtm05"].to_numpy(float)
D = cf.pair_distance_km(E, Nc)
N = res.loc[codes, "N_2026"].to_numpy(float); p = res.loc[codes, "p_2026"].to_numpy(float); p = np.where(np.isfinite(p), p, np.nanmean(p))
binom = h / (N * p * (1 - p))
tau_real = res.loc[codes, "tau_2026"].to_numpy(float)
print(f"units n={n}; h={h:.2f}; mean binomial var {binom.mean():.4f}; sigma_phi2={args.sigma_phi2} sigma_mu2={args.sigma_mu2}")

# ------------------------------------------------------------------ kernels
def kernels(lc):
    G = np.exp(-D / lc); np.fill_diagonal(G, 0)
    k = args.k if args.k > 0 else int(round(G.sum(1).mean()))
    A = np.zeros((n, n)); nn = np.argsort(L_las + np.eye(n) * 1e9, 1)[:, :k]
    A[np.arange(n)[:, None], nn] = 1.0; A = np.maximum(A, A.T)
    G *= A.sum() / G.sum()          # equal total weight (equal trace of the combinatorial Laplacians)
    return G, A, k
G, A, kdeg = kernels(args.lc)
print(f"l_c={args.lc} km: mean weighted degree of geo kernel -> k={kdeg}; surname kNN mean degree {A.sum(1).mean():.1f}")

def laplacian(Gw, variant):
    d = Gw.sum(1)
    if variant == "sym":
        Dm = 1 / np.sqrt(d); return np.eye(n) - Dm[:, None] * Gw * Dm[None, :]
    if variant == "comb_rw":
        Gr = Gw / np.outer(d, d); L = np.diag(Gr.sum(1)) - Gr; return L * (n / np.trace(L))
    if variant == "comb":
        L = np.diag(d) - Gw; return L * (n / np.trace(L))
    raise ValueError(variant)

def pinv_sym(L, tol=1e-9):
    lam, U = np.linalg.eigh(L); keep = lam > tol * lam.max()
    return (U[:, keep] / lam[keep]) @ U[:, keep].T, lam

wgrid = np.round(np.linspace(0, 1, 11), 2)
variants = args.variants.split(",")
Pw = {}
t = time.time()
for v in variants:
    for w in wgrid:
        Pw[(v, w)], lam = pinv_sym(laplacian(w * G + (1 - w) * A, v))
print(f"{len(Pw)} pseudo-inverses in {time.time()-t:.0f} s", flush=True)

# model C(r) implied by each kernel at w = 1, 0.5, 0 (centred, normalised) vs empirical 2026 C(r)
def model_Cr(P):
    Hc = np.eye(n) - 1 / n; Pc = Hc @ P @ Hc; v = np.diag(Pc); Cn = Pc / np.sqrt(np.outer(v, v))
    iu = np.triu_indices(n, 1); d = D[iu]; c = Cn[iu]; b = np.digitize(d, cf.R_EDGES) - 1
    return np.array([c[b == i].mean() if (b == i).sum() > 30 else np.nan for i in range(len(cf.R_EDGES) - 1)])
Cr_emp = pd.read_csv(cf.PROC / "stage1_Cr.csv"); Cr_emp = Cr_emp[(Cr_emp.tag == "2026") & (Cr_emp.obs == "tau") & (Cr_emp.set == "full")]
kernel_Cr = {f"{v}_w{w}": model_Cr(Pw[(v, w)]) for v in variants for w in (1.0, 0.5, 0.0)}
for k_, c_ in kernel_Cr.items():
    print(f"  model C(r) {k_:14s}: " + " ".join(f"{x:5.2f}" for x in c_))
print("  empirical 2026     : " + " ".join(f"{x:5.2f}" for x in Cr_emp.C.to_numpy()))

# ------------------------------------------------------------------ likelihood
def nll(theta, P, taus):
    s2, smu = np.exp(theta)
    S = s2 * P + np.diag(smu + binom)
    try: Lc = np.linalg.cholesky(S)
    except np.linalg.LinAlgError: return 1e30
    logdet = 2 * np.log(np.diag(Lc)).sum()
    Y = solve_triangular(Lc, np.asarray(taus).T, lower=True, check_finite=False)
    return 0.5 * (len(taus) * logdet + (Y * Y).sum())
_warm = {}
def profile(taus, P, s2_0, key=None):
    x0 = _warm.get(key, np.log([s2_0, args.sigma_mu2]))
    r = minimize(nll, x0, args=(P, taus), method="Nelder-Mead", options=dict(xatol=0.03, fatol=0.3, maxfev=60, initial_simplex=np.array([x0, x0 + [0.3, 0], x0 + [0, 0.3]])))
    if key is not None: _warm[key] = r.x
    return r.fun, np.exp(r.x)

rng = np.random.default_rng(31)
rows = []; profiles = {}
for v in ([] if args.lc_profile else variants):
    for w_true in (0.0, 0.5, 1.0):
        P = Pw[(v, w_true)]; dP = np.diag(P); s2 = args.sigma_phi2 / dP.mean()
        ev, U = np.linalg.eigh(P); sq = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
        for rep in range(args.sets):
            taus = [np.sqrt(s2) * (sq @ rng.standard_normal(n)) + np.sqrt(args.sigma_mu2 + binom) * rng.standard_normal(n) for _ in range(args.nel)]
            t = time.time()
            prof = np.array([profile(taus, Pw[(v, w)], s2, key=(v, 'j'))[0] for w in wgrid]); prof -= prof.min()
            prof1 = np.array([profile(taus[:1], Pw[(v, w)], s2, key=(v, 's'))[0] for w in wgrid]); prof1 -= prof1.min()
            i = int(round(w_true * 10)); d2 = min(prof[j] for j in (i - 2, i + 2) if 0 <= j <= 10); d1 = min(prof[j] for j in (i - 1, i + 1) if 0 <= j <= 10)
            wh, wh1 = wgrid[prof.argmin()], wgrid[prof1.argmin()]
            rows.append(dict(variant=v, w_true=w_true, rep=rep, w_hat_joint5=wh, w_hat_single=wh1, dNLL_pm02=d2, dNLL_pm01=d1, seconds=time.time() - t))
            profiles[(v, w_true, rep)] = prof
            print(f"  {v:8s} w_true={w_true:.1f} rep={rep}: w_hat(5 el)={wh:.1f} w_hat(1 el)={wh1:.1f} dNLL(±0.2)={d2:.1f} dNLL(±0.1)={d1:.1f} "
                  f"profile: " + " ".join(f"{x:.0f}" for x in prof) + f" [{time.time()-t:.0f} s]", flush=True)
if rows: tab = pd.DataFrame(rows); tab.to_csv(cf.PROC / f"stage3a_recovery{args.suffix}.csv", index=False)

# ------------------------------------------------------------------ l_c profile on synthetic data: does the profile return the generating l_c or run to the pseudo-inverse cut?
if args.lc_profile:
    LCG = (1.25, 1.5, 2.0, 3.0, 4.5, 7.0, 10.0); v = "comb_rw"; PL = {}
    for lc in LCG:
        Gl, Al, kl = kernels(lc); PL[lc] = {w: pinv_sym(laplacian(w * Gl + (1 - w) * Al, v))[0] for w in wgrid}; print(f"  pinv grid l_c={lc} (k={kl})", flush=True)
    res_ = []
    if args.landscape:   # truth = the single-election landscape model of Table 3 (2026 fit: sg K_r(20 km) + sBC K_BC + smu I + binomial), no Laplacian term
        m4 = __import__("json").load(open(cf.PROC / "stage4d_model_comparison.json"))["2026"]["M2_static_r20_BC"]
        Eu = las["Eu"][np.ix_(sel, sel)].astype(float); KBC = np.clip(1 - Eu ** 2, 0, 1); np.fill_diagonal(KBC, 1.0); KR = np.exp(-D / 20.0)
        Sf = m4["sg"] * KR + m4["ss_BC"] * KBC; ev, U = np.linalg.eigh(Sf); sqL = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T; smuL = m4["smu"]
        print(f"  landscape truth: sg={m4['sg']:.4f} (l_r=20 km) sBC={m4['ss_BC']:.4f} smu={smuL:.4f}; field var at mean unit {np.diag(Sf).mean():.4f}", flush=True)
    truths = [("landscape", None)] if args.landscape else [(w, w) for w in (0.5, 1.0)]
    for w_lab, w_true in truths:
        if w_true is not None: P = Pw[(v, w_true)]; dP = np.diag(P); s2 = args.sigma_phi2 / dP.mean(); ev, U = np.linalg.eigh(P); sq = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
        else: sq = sqL; s2 = 1.0
        for rep in range(max(args.sets, 1)):
            taus = [np.sqrt(s2) * (sq @ rng.standard_normal(n)) + np.sqrt((smuL if w_true is None else args.sigma_mu2) + binom) * rng.standard_normal(n) for _ in range(args.nel)]; t = time.time(); prof = {}
            for lc in LCG:
                pw = np.array([profile(taus, PL[lc][w], s2 if w_true is not None else 0.08 / np.diag(PL[lc][w]).mean(), key=(v, "p"))[0] for w in wgrid]); prof[lc] = (float(pw.min()), float(wgrid[pw.argmin()]))
            m0 = min(x[0] for x in prof.values()); lch = min(prof, key=lambda l: prof[l][0])
            res_.append(dict(w_true=w_lab, rep=rep, lc_gen=("landscape" if args.landscape else args.lc), lc_hat=lch, w_hat_at_lc_hat=prof[lch][1], profile={str(l): dict(dNLL=prof[l][0] - m0, w_hat=prof[l][1]) for l in LCG}))
            print(f"  l_c profile: generated {'landscape' if args.landscape else 'l_c='+str(args.lc)} w={w_lab} rep={rep}: l_c_hat={lch} w_hat={prof[lch][1]:.1f} | dNLL by l_c: " + " ".join(f"{l}:{prof[l][0]-m0:.0f}(w={prof[l][1]:.1f})" for l in LCG) + f" [{time.time()-t:.0f} s]", flush=True)
    cf.save_json(dict(lc_gen=("landscape" if args.landscape else args.lc), grid=list(LCG), sigma_phi2=args.sigma_phi2, sigma_mu2=args.sigma_mu2, n=n, landscape_truth=(dict(set="2026", lr_km=20.0, sg=m4["sg"], sBC=m4["ss_BC"], smu=smuL) if args.landscape else None), results=res_), cf.PROC / f"stage3a_lcprofile{args.suffix}.json"); print("done (l_c profile)"); raise SystemExit
# ------------------------------------------------------------------ mismatch check: generate at l_c, fit at 2 l_c (sym and comb_rw)
mism = []
G2, A2, k2 = kernels(2 * args.lc)
for v in [x for x in ("sym", "comb_rw") if x in variants]:
    P2 = {w: pinv_sym(laplacian(w * G2 + (1 - w) * A2, v))[0] for w in wgrid}
    for w_true in (0.0, 0.5, 1.0):
        P = Pw[(v, w_true)]; dP = np.diag(P); s2 = args.sigma_phi2 / dP.mean()
        ev, U = np.linalg.eigh(P); sq = (U * np.sqrt(np.clip(ev, 0, None))) @ U.T
        taus = [np.sqrt(s2) * (sq @ rng.standard_normal(n)) + np.sqrt(args.sigma_mu2 + binom) * rng.standard_normal(n) for _ in range(args.nel)]
        prof = np.array([profile(taus, P2[w], s2, key=(v, 'm'))[0] for w in wgrid]); prof -= prof.min()
        mism.append(dict(variant=v, w_true=w_true, w_hat_wrong_lc=float(wgrid[prof.argmin()]), lc_gen=args.lc, lc_fit=2 * args.lc, k_fit=k2))
        print(f"  mismatch {v}: generated l_c={args.lc}, fitted l_c={2*args.lc} (k={k2}): w_true={w_true:.1f} -> w_hat={wgrid[prof.argmin()]:.1f}")

# ------------------------------------------------------------------ gate + real-data quick look (descriptive only, not the Stage 3 fit)
gate = {}
for v in variants:
    t = tab[tab.variant == v]
    gate[v] = dict(max_abs_error_joint5=float((t.w_hat_joint5 - t.w_true).abs().max()), min_dNLL_pm02=float(t.dNLL_pm02.min()),
                   passed=bool(((t.w_hat_joint5 - t.w_true).abs().max() <= 0.15) and (t.dNLL_pm02.min() >= 3)))
    print(f"gate G3a [{v}]: max |w_hat - w_true| (5 el) = {gate[v]['max_abs_error_joint5']:.2f}, min dNLL(±0.2) = {gate[v]['min_dNLL_pm02']:.1f} -> {'PASS' if gate[v]['passed'] else 'FAIL'}")
real = {}
m = np.isfinite(tau_real)
for v in variants:
    prof = np.array([profile([tau_real[m]] if m.all() else [np.where(m, tau_real, 0)], Pw[(v, w)], args.sigma_phi2 / np.diag(Pw[(v, w)]).mean(), key=(v, 'r'))[0] for w in wgrid])
    prof -= prof.min(); real[v] = dict(w_hat_2026_single=float(wgrid[prof.argmin()]), profile=[float(x) for x in prof])
    print(f"  provisional real-data profile 2026 [{v}] (single election, l_c fixed, descriptive only): w_hat={wgrid[prof.argmin()]:.1f}  " + " ".join(f"{x:.0f}" for x in prof))
cf.save_json(dict(profiles_json={f"{v}|{w}|{r}": [float(x) for x in pr] for (v, w, r), pr in profiles.items()}, n=n, k=kdeg, lc=args.lc, h=h, sigma_phi2=args.sigma_phi2, sigma_mu2=args.sigma_mu2, gate_G3a=gate, mismatch=mism,
                  kernel_Cr={k_: [float(x) for x in c_] for k_, c_ in kernel_Cr.items()}, r_mid=[float(x) for x in np.sqrt(cf.R_EDGES[:-1] * cf.R_EDGES[1:])],
                  provisional_real_2026=real, wall_s=time.time() - T0), cf.PROC / f"stage3a_summary{args.suffix}.json")

# ------------------------------------------------------------------ figures
fig, axes = plt.subplots(1, len(variants), figsize=(4.2 * len(variants), 3.8), squeeze=False)
for ax, v in zip(axes[0], variants):
    for w_true, col in zip((0.0, 0.5, 1.0), ("C0", "C1", "C2")):
        for rep in range(args.sets):
            ax.plot(wgrid, profiles[(v, w_true, rep)], color=col, alpha=0.7, label=f"w_true={w_true}" if rep == 0 else None)
        ax.axvline(w_true, color=col, ls=":", lw=0.8)
    ax.set_ylim(0, 60); ax.set_xlabel("w (1 = geographic)"); ax.set_ylabel("ΔNLL"); ax.set_title(v); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(cf.FIG / f"stage3a_recovery{args.suffix}.png", dpi=150)
fig, ax = plt.subplots(figsize=(6, 4))
rm = np.sqrt(cf.R_EDGES[:-1] * cf.R_EDGES[1:])
ax.errorbar(Cr_emp.r_mid, Cr_emp.C, Cr_emp.C_se, color="k", marker="o", ms=3, label="empirical τ' 2026")
for k_, c_ in kernel_Cr.items():
    ax.plot(rm, c_, "--", lw=1, label=k_)
ax.axhline(0, color="gray", lw=0.5); ax.set_xscale("log"); ax.set_xlabel("r (km)"); ax.set_ylabel("C(r)"); ax.legend(fontsize=6, ncol=2)
fig.tight_layout(); fig.savefig(cf.FIG / f"stage3a_kernel_Cr{args.suffix}.png", dpi=150)
print(f"done in {time.time()-T0:.0f} s")
