"""Stage 6 — joint six-election likelihood: static landscape vs fluctuating/diffusing field.

Common panel (N >= 200, tau' finite in all six sets, n = 1556), stacked Y = (tau'_2010, ..., tau'_2026), nT = 9336.
Covariance (block (t, t') of n x n):
    Sigma = Sigma_static (x) J_T  +  Sigma_dyn(Delta_tt')  +  diag_t(smu_t + h * binom_t)
    Sigma_static = sg K_r(l_r) + sBC K_BC + su I                       (perfectly persistent across elections)
    Sigma_dyn(Delta) = sd * U diag(c_k exp(-gamma_k Delta)) U^T,  gamma_k = kappa + D lambda_k,  c_k = (1/lambda_k) / mean diag
                       (fluctuating field on the graph Laplacian L_w; D > 0 = diffusion signature: short modes decay faster)
                     + sfBC K_BC exp(-kappa_BC Delta)   [optional fluctuating kinship term]
                     + sfg K_r  exp(-kappa_g  Delta)    [optional fluctuating geo-static term, Kronecker]
Mean: per-election GLS on the Stage-3 controls (province, density class, ln N, GAM). Per-election smu_t.
Models: S (static only) -> SK (static + Kronecker fluctuating K_r, K_BC) -> SD (static + diffusive on geography, w = 1),
        SDprof (profile over fixed D incl. D = 0: mode-independent decay) -> SD at w = 0 and 0.5 (surname graph) -> SDK (SD + fluctuating K_BC).
After the static fit the six smu_t are frozen up to one common scale factor (smu_scale) to halve the gradient cost.
Outputs: data/processed/stage6_joint_dynamics.json (incremental), stage6_joint_dynamics.log via stdout.
Usage: python3 scripts/stage6_joint_dynamics.py [--quick] [--models S,SK,SD,SD0,SDw0,SDw05,SDK]
"""
import sys, json, time, argparse, itertools
from pathlib import Path
import numpy as np, pandas as pd
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import minimize
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--models", default="S,SD,SDprof,SDw0,SK,SDK,SDw05")
ap.add_argument("--maxiter", type=int, default=40); ap.add_argument("--resummarise", action="store_true"); ap.add_argument("--out", default="stage6_joint_dynamics.json")
ap.add_argument("--synthetic", default="", help="replace Y by a draw from a fitted model of stage6_joint_dynamics.json: SK, or SKD5 (SK + 5%% diffusive, D = 0.9)")
ap.add_argument("--subsample", type=int, default=0, help="contiguous spatial subsample: the N districts nearest the median seat (recovery runs)"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--lc", type=float, default=1.5, help="coupling length of the geographic graph (km)"); ap.add_argument("--k", type=int, default=13, help="k of the surname kNN graph")
ap.add_argument("--h", type=float, default=0.0, help="override the herding factor h (0 = value from stage1_summary.json)")
ap.add_argument("--social", action="store_true", help="add the district-level social covariates (IDS 2023, 2011 census schooling) to the mean, election by election")
args = ap.parse_args(); T0 = time.time(); LC, K = args.lc, args.k
YEARS = {"2010": 2010.1, "2014": 2014.1, "2018": 2018.1, "2022r1": 2022.1, "2022r2": 2022.25, "2026": 2026.1}
TAGS = cf.TAGS; T = len(TAGS)
s1 = json.load(open(cf.PROC / "stage1_summary.json")); h = float(np.clip(s1["variance_decomposition"]["2022r1-2026"]["h"], 1, 3))
if args.h > 0: h = args.h
las = np.load(cf.PROC / "lasker_2026_pooled.npz", allow_pickle=True); res = pd.read_csv(cf.PROC / "stage1_residuals.csv", dtype={"codelec": str}).set_index("codelec")
cov = pd.read_csv(cf.PROC / "de_covariates.csv", dtype={"codelec": str}).set_index("codelec"); codes_all = las["codelec"].astype(str)
m = (las["N"] / 2 >= 200) & np.isin(codes_all, res.index)
for t in TAGS: m &= np.isin(codes_all, res.index[np.isfinite(res[f"tau_{t}"])])
codes = codes_all[m]
if args.quick: rng = np.random.default_rng(0); codes = np.sort(rng.choice(codes, 300, replace=False)); m = np.isin(codes_all, codes)
if args.subsample:
    rr = res.loc[codes]; E0, N0 = rr.E_crtm05.median(), rr.N_crtm05.median(); dd = np.hypot(rr.E_crtm05.to_numpy(float) - E0, rr.N_crtm05.to_numpy(float) - N0)
    codes = np.sort(codes[np.argsort(dd)[:args.subsample]]); m = np.isin(codes_all, codes); print(f"contiguous subsample: {len(codes)} districts within {dd[np.argsort(dd)[args.subsample-1]]/1e3:.0f} km of the median seat")
Lm = las["L"][np.ix_(m, m)].astype(float); Eu = las["Eu"][np.ix_(m, m)].astype(float); r = res.loc[codes]; c_ = cov.loc[codes]; n = len(codes)
D = cf.pair_distance_km(r.E_crtm05.to_numpy(float), r.N_crtm05.to_numpy(float))
Y = np.concatenate([r[f"tau_{t}"].to_numpy(float) for t in TAGS]); Y -= np.repeat([r[f"tau_{t}"].mean() for t in TAGS], n)
binom = np.concatenate([h / (r[f"N_{t}"].to_numpy(float) * r[f"p_{t}"].to_numpy(float) * (1 - r[f"p_{t}"].to_numpy(float))) for t in TAGS])
X1 = pd.get_dummies(c_[["provincia", "density_class"]], drop_first=True).astype(float); X1["lnN"] = c_.lnN_2026; X1["gam"] = c_.gam
if args.social:
    soc = pd.read_csv(cf.PROC / "de_covariates_social.csv", dtype={"codelec": str}).set_index("codelec").loc[codes]
    for col in ("ids2023", "ids_edu", "ids_eco", "school2011", "illit2011", "secondary2011"): v = soc[col].to_numpy(float); X1[col] = (v - v.mean()) / v.std()
    print("social covariates in the mean:", list(X1.columns[-6:]), flush=True)
X1.insert(0, "c", 1.0); X1 = X1.to_numpy()
X = np.zeros((n * T, X1.shape[1] * T))
for t in range(T): X[t * n:(t + 1) * n, t * X1.shape[1]:(t + 1) * X1.shape[1]] = X1
KBC = np.clip(1 - Eu ** 2, 0, 1); np.fill_diagonal(KBC, 1.0); KR = {lr: np.exp(-D / lr) for lr in (20.0, 50.0)}
G = cf.geo_kernel(D, LC); A = cf.knn_graph(Lm, K)
yrs = np.array([YEARS[t] for t in TAGS]); DT = np.abs(yrs[:, None] - yrs[None, :]); dts = np.unique(DT)
print(f"n={n} T={T} nT={n*T} h={h:.2f} distinct dT={np.round(dts,2)}  [{time.time()-T0:.0f} s]", flush=True)
spec = {}
def graph_modes(w):
    if w not in spec:
        lam, U = np.linalg.eigh(cf.laplacian(cf.mix_kernels(G, A, w), "comb_rw")); keep = lam > 1e-9 * lam.max(); lam, U = lam[keep], U[:, keep]
        c = (1 / lam); c *= n / c.sum(); spec[w] = (lam, U, c)
    return spec[w]
NEV = [0]; SMU_BASE = [np.full(T, 0.02)]
def smu_vec(p):
    """per-election district noise: free in the static fit (smu0..smu5), afterwards one scale factor times the static-fit vector."""
    return np.array([p[f"smu{t}"] for t in range(T)]) if "smu0" in p else p["smu_scale"] * SMU_BASE[0]
def modes(p, mdl):
    """eigenpairs of L_w, mode rates gamma_k = kappa + D lambda_k, and spatial weights c_k (mean 1):
    c_k ~ 1/lambda_k (Laplacian pseudo-inverse; separable ansatz of the original fit) or, with mdl["ew"], c_k ~ 1/gamma_k,
    the stationary covariance of the Edwards-Wilkinson field dphi = -(kappa + D L) phi dt + white noise (massive EW)."""
    lam, U, c = graph_modes(mdl["w"]); gam = p["kappa"] + p.get("D", mdl.get("D_fixed", 0.0)) * lam
    if mdl.get("ew"): c = 1 / gam; c = c * n / c.sum()
    return lam, U, c, gam
def build_sigma(p, mdl):
    """p: dict of parameters (variances, rates). Returns the nT x nT covariance."""
    S = np.empty((n * T, n * T)); Ss = p["sg"] * KR[mdl["lr"]] + p["sBC"] * KBC + (0.0 if "ku" in p else p["su"]) * np.eye(n)
    dyn = {}
    for dt in dts:
        M = np.zeros((n, n))
        if "sd" in p:
            lam, U, c, gam = modes(p, mdl)
            M += p["sd"] * ((U * (c * np.exp(-gam * dt))) @ U.T)
        if "sfBC" in p: M += p["sfBC"] * np.exp(-p["kBC"] * dt) * KBC
        if "sfg" in p: M += p["sfg"] * np.exp(-p["kg"] * dt) * KR[mdl["lr"]]
        if "ku" in p: M += p["su"] * np.exp(-p["ku"] * dt) * np.eye(n)   # drifting district-specific term (SKU)
        dyn[dt] = M
    for t in range(T):
        for u in range(T):
            S[t * n:(t + 1) * n, u * n:(u + 1) * n] = Ss + dyn[DT[t, u]]
    S[np.diag_indices(n * T)] += np.repeat(smu_vec(p), n) + p.get("hb", 1.0) * binom   # hb: free scale of the binomial diagonal (h = 1.43 hb)
    return S
def nll(theta, names, mdl):
    p = dict(zip(names, np.exp(theta))); p.update(mdl.get("fix", {}))
    S = build_sigma(p, mdl)
    try: Lc, _ = cho_factor(S, lower=True, overwrite_a=True, check_finite=False)
    except np.linalg.LinAlgError: return 1e30
    Xi = solve_triangular(Lc, X, lower=True, check_finite=False); yi = solve_triangular(Lc, Y, lower=True, check_finite=False)
    beta = np.linalg.lstsq(Xi, yi, rcond=None)[0]; rr = yi - Xi @ beta; NEV[0] += 1
    return 0.5 * (2 * np.log(np.diag(Lc)).sum() + rr @ rr)
def fit(names, x0, mdl, label):
    NEV[0] = 0; t0 = time.time(); best = [np.inf, None]
    def f(th):
        v = nll(th, names, mdl)
        if v < best[0]: best[0], best[1] = v, th.copy()
        return v
    r_ = minimize(f, x0, method="L-BFGS-B", options=dict(maxiter=2 if args.quick else args.maxiter, eps=0.02, ftol=1e-10, gtol=0.5, maxfun=12 if args.quick else 40 * args.maxiter))
    p = dict(zip(names, np.exp(best[1]))); p.update(mdl.get("fix", {})); print(f"  {label}: nll={best[0]:.1f} evals={NEV[0]} {time.time()-t0:.0f} s | " + " ".join(f"{k}={v:.4g}" for k, v in p.items()), flush=True)
    return float(best[0]), p, best[1]
def summarise(p, mdl):
    """variance shares at the mean unit, model-implied cross-election correlation vs dT, dynamic-field decay."""
    out = {}
    comp = {"static_geo": p["sg"] * KR[mdl["lr"]].diagonal().mean(), "static_kinship": p["sBC"], "static_unit": 0.0 if "ku" in p else p["su"]}
    if "ku" in p: comp["dyn_unit"] = p["su"]
    if "sd" in p: comp["dyn_field"] = p["sd"]
    if "sfBC" in p: comp["dyn_kinship"] = p["sfBC"]
    if "sfg" in p: comp["dyn_geo"] = p["sfg"]
    comp["election_noise"] = float(np.mean(smu_vec(p))); out["smu_per_election"] = [float(x) for x in smu_vec(p)]; comp["binomial"] = float(p.get("hb", 1.0) * binom.mean())
    tot = sum(comp.values()); out["variance"] = {k: float(v) for k, v in comp.items()}; out["share"] = {k: float(v / tot) for k, v in comp.items()}
    stat = comp["static_geo"] + comp["static_kinship"] + comp["static_unit"]; dyn0 = sum(v for k, v in comp.items() if k.startswith("dyn"))
    out["persistent_share_of_field"] = float(stat / (stat + dyn0)) if stat + dyn0 > 0 else None
    # model-implied Pearson correlation per election pair (mean-unit covariance block / sqrt of the two mean variances), averaged by dT key
    def dynvar(dt):
        v = 0.0
        if "sd" in p:
            lam, U, c, gam = modes(p, mdl); v += p["sd"] * float(np.mean((U ** 2) @ (c * np.exp(-gam * dt))))
        if "sfBC" in p: v += p["sfBC"] * np.exp(-p["kBC"] * dt)
        if "sfg" in p: v += p["sfg"] * np.exp(-p["kg"] * dt)
        if "ku" in p: v += p["su"] * np.exp(-p["ku"] * dt)
        return v
    smu = smu_vec(p); bm = p.get("hb", 1.0) * binom.reshape(T, n).mean(1); var_t = stat + dynvar(0.0) + smu + bm; acc = {}
    for i, j in itertools.combinations(range(T), 2):
        acc.setdefault(f"{DT[i, j]:.2f}", []).append((stat + dynvar(DT[i, j])) / np.sqrt(var_t[i] * var_t[j]))
    out["model_corr_vs_dT"] = {k: float(np.mean(v)) for k, v in acc.items()}
    if "sd" in p and "D" in p:
        lam, U, c, gam = modes(p, mdl); out["dyn_field"] = dict(D=p["D"], kappa=p["kappa"], tau_slowest_y=float(1 / (p["kappa"] + p["D"] * lam.min())), tau_median_mode_y=float(1 / (p["kappa"] + p["D"] * np.median(lam))),
                                                                    lambda_half_rate=float(p["kappa"] / p["D"]) if p["D"] > 0 else None, var_share_modes_slower_than_4y=float(np.sum(c[(p["kappa"] + p["D"] * lam) < 0.25]) / c.sum()))
    return out
if args.synthetic:   # recovery run: Y <- draw from a fitted model (real-data parameters), zero mean
    real = json.load(open(cf.PROC / "stage6_joint_dynamics.json"))["models"]; src_ = "SD" if args.synthetic == "SD" else "SK"; pt = dict(real[src_]["params"]); mt = dict(real[src_]["model"])
    SMU_BASE[0] = np.array([real["S"]["params"][f"smu{t}"] for t in range(T)])
    if args.synthetic == "SKD5": pt.update(sd=0.01, kappa=0.03, D=0.9)
    elif args.synthetic == "SKD2": pt.update(sd=0.0035, kappa=0.03, D=0.9)   # ~2.5 % of the field variance
    elif args.synthetic == "SKD2s": pt.update(sd=0.0035, kappa=0.004, D=0.7)   # 2.5 %, slow relaxation (the regime of the real-data SKD fit)
    elif args.synthetic == "SKD1": pt.update(sd=0.0014, kappa=0.03, D=0.9)   # ~1 % of the field variance
    elif args.synthetic not in ("SK", "SD"): raise SystemExit("--synthetic must be SK, SD, SKD5, SKD2, SKD2s or SKD1")   # SD: diffusive truth (reverse gate: do drifting kernels manufacture a drift preference?)
    St = build_sigma(pt, mt); Lt = np.linalg.cholesky(St); Y = Lt @ np.random.default_rng(args.seed).standard_normal(n * T); del St, Lt
    truth = dict(name=args.synthetic, params=pt, model=mt, smu=[float(x) for x in SMU_BASE[0]]); print(f"synthetic Y from {args.synthetic}: {pt}", flush=True)
emp = {f"{abs(YEARS[b]-YEARS[a]):.2f}": [] for a, b in itertools.combinations(TAGS, 2)}
Yc = Y.reshape(T, n) - (X1 @ np.linalg.lstsq(X1, Y.reshape(T, n).T, rcond=None)[0]).T
for (i, a), (j, b) in itertools.combinations(enumerate(TAGS), 2): emp[f"{abs(YEARS[b]-YEARS[a]):.2f}"].append(float(np.corrcoef(Yc[i], Yc[j])[0, 1]))
out = {"n": n, "T": T, "h": h, "lc": LC, "k": K, "tags": TAGS, "truth": truth if args.synthetic else None, "subsample": args.subsample, "empirical_corr_vs_dT": {k: float(np.mean(v)) for k, v in emp.items()}, "models": {}}
print("empirical residual corr vs dT:", {k: round(float(np.mean(v)), 3) for k, v in emp.items()}, flush=True)
lam1 = graph_modes(1.0)[0]; out["lambda_quantiles_w1"] = {str(q): float(np.quantile(lam1, q)) for q in (0, .1, .25, .5, .75, .9, 1)}; print("lambda quantiles (w=1):", out["lambda_quantiles_w1"], flush=True)
OUT = cf.PROC / args.out; todo = args.models.split(","); R = out["models"]
smu0 = {f"smu{t}": 0.02 for t in range(T)}; base = dict(sg=0.05, sBC=0.07, su=0.03, **smu0)
def run(name, names, x0, mdl):
    f, p, th = fit(names, x0, mdl, name); R[name] = dict(nll=f, params=p, model=mdl, n_params=len(names), **summarise(p, mdl)); cf.save_json(out, OUT); return f, p, th
# --- S: static only, l_r profiled
if "S" in todo:
    bestS = None
    for lr in (20.0, 50.0):
        f, p, th = run(f"S_r{lr:.0f}", list(base), np.log(list(base.values())), dict(lr=lr, w=1.0))
        if bestS is None or f < bestS[0]: bestS = (f, p, th, lr)
    R["S"] = dict(R[f"S_r{bestS[3]:.0f}"]); cf.save_json(out, OUT); lr_best = bestS[3]; pS = bestS[1]
else:
    if OUT.exists(): prev = json.load(open(OUT)); out.update({k: v for k, v in prev.items() if k not in out}); R = prev["models"]; out["models"] = R
    else:   # new output file without a static fit: the static model does not depend on (l_c, k, synthetic draw), copy it from the main run
        main = json.load(open(cf.PROC / "stage6_joint_dynamics.json"))["models"]; R.update({k: v for k, v in main.items() if k in ("S", "S_r20", "S_r50")}); print("static fit copied from stage6_joint_dynamics.json", flush=True)
    lr_best = R["S"]["model"]["lr"]; pS = R["S"]["params"]
SMU_BASE[0] = np.array([pS[f"smu{t}"] for t in range(T)]); pS = {k: v for k, v in pS.items() if not k.startswith("smu")}; pS["smu_scale"] = 1.0
print(f"static base: l_r = {lr_best} km; smu per election = {np.round(SMU_BASE[0], 4)}", flush=True)
def warm(extra):
    p = dict(pS); p.update(extra); return list(p), np.log(list(p.values()))
if args.resummarise:
    for k, v in R.items():
        if "params" in v and "model" in v: v.update(summarise(v["params"], v["model"]))
    cf.save_json(out, OUT); print("re-summarised", list(R)); todo = []
if "SK" in todo:
    names, x0 = warm(dict(sfg=0.01, sfBC=0.01, kBC=0.3, kg=0.3)); run("SK", names, x0, dict(lr=lr_best, w=1.0))
if "SD" in todo:
    names, x0 = warm(dict(sd=0.02, kappa=0.2, D=0.05)); fSD, pSD, thSD = run("SD", names, x0, dict(lr=lr_best, w=1.0))
for key, w in (("SDw0", 0.0), ("SDw05", 0.5)):
    if key in todo:
        st = R.get("SD", {}).get("params", {}); names, x0 = warm(dict(sd=st.get("sd", 0.02), kappa=st.get("kappa", 0.2), D=st.get("D", 0.05))); run(key, names, x0, dict(lr=lr_best, w=w))
if "SDprof" in todo:   # profile likelihood over D (diffusion signature), other parameters re-fitted
    st = R.get("SD", {}).get("params", {}); prof = {}
    for Dfix in (0.0, 0.01, 0.03, 0.1, 0.3):
        names, x0 = warm(dict(sd=st.get("sd", 0.02), kappa=st.get("kappa", 0.2))); f, p, th = run(f"SD_D{Dfix:g}", names, x0, dict(lr=lr_best, w=1.0, D_fixed=Dfix)); prof[str(Dfix)] = f
    out["D_profile_w1"] = prof; cf.save_json(out, OUT)
out["D_profile_w1"] = {k[4:]: v["nll"] for k, v in R.items() if k.startswith("SD_D")} if any(k.startswith("SD_D") for k in R) else out.get("D_profile_w1")
if "SDK" in todo:
    st = R.get("SD", {}).get("params", {}); names, x0 = warm(dict(sd=st.get("sd", 0.02), kappa=st.get("kappa", 0.2), D=st.get("D", 0.05), sfBC=0.01, kBC=0.3)); run("SDK", names, x0, dict(lr=lr_best, w=1.0))
if "SKD" in todo:   # everything: slowly drifting kernels (SK) + diffusive field on geography
    sk = R.get("SK", {}).get("params", {}); sd_ = R.get("SD", {}).get("params", {})
    names, x0 = warm(dict(sfg=sk.get("sfg", 0.02), sfBC=sk.get("sfBC", 0.06), kBC=sk.get("kBC", 0.013), kg=sk.get("kg", 0.07), sd=0.01, kappa=sd_.get("kappa", 0.03), D=sd_.get("D", 0.9)))
    pS.update({k: sk[k] for k in ("sg", "sBC", "su", "smu_scale") if k in sk}); names, x0 = warm(dict(sfg=sk.get("sfg", 0.02), sfBC=sk.get("sfBC", 0.06), kBC=sk.get("kBC", 0.013), kg=sk.get("kg", 0.07), sd=0.01, kappa=sd_.get("kappa", 0.03), D=sd_.get("D", 0.9)))
    run("SKD", names, x0, dict(lr=lr_best, w=1.0))
if "hfree" in todo:   # binomial scale h free (hb = h / 1.43) in S (six noise variances free), SK and SKU, warm-started from the fixed-h fits
    for key, src in (("S_hfree", "S"), ("SK_hfree", "SK"), ("SKU_hfree", "SKU")):
        if src not in R: continue
        st = dict(R[src]["params"])
        if "smu_scale" in st: sc = st.pop("smu_scale"); st.update({f"smu{t}": sc * SMU_BASE[0][t] for t in range(T)})
        st["hb"] = 1.0; run(key, list(st), np.log(list(st.values())), dict(R[src]["model"]))
if "SKf" in todo or "SKDf" in todo:   # election noise variances free (six parameters) in the dynamic models
    for key, src in (("SKf", "SK"), ("SKDf", "SKD")):
        if key not in todo: continue
        st = dict(R[src]["params"]); sc = st.pop("smu_scale"); st.update({f"smu{t}": sc * SMU_BASE[0][t] for t in range(T)}); run(key, list(st), np.log(list(st.values())), dict(R[src]["model"]))
if "SDe" in todo or "SKDe" in todo:   # consistent Edwards-Wilkinson weights c_k ~ 1/(kappa + D lambda_k)
    for key, src in (("SDe", "SD"), ("SKDe", "SKD")):
        if key not in todo: continue
        st = dict(R[src]["params"]); mdl = dict(R[src]["model"]); mdl["ew"] = True; run(key, list(st), np.log(list(st.values())), mdl)
if "SKprof" in todo:   # profile likelihood in the two drift rates of SK, other parameters refitted
    sk = R["SK"]["params"]
    for par, grid in (("kBC", (0.005, 0.025, 0.04)), ("kg", (0.035, 0.12))):
        for val in grid:
            st = {k: v for k, v in sk.items() if k != par}; mdl = dict(R["SK"]["model"]); mdl["fix"] = {par: val}; run(f"SK_{par}{val:g}", list(st), np.log(list(st.values())), mdl)
if "SKprof2" in todo:   # second pass: tighter bracket of the kinship rate and the geographic rate
    sk = R["SK"]["params"]
    for par, grid in (("kBC", (0.002, 0.008, 0.018)), ("kg", (0.05, 0.09, 0.2))):
        for val in grid:
            st = {k: v for k, v in sk.items() if k != par}; mdl = dict(R["SK"]["model"]); mdl["fix"] = {par: val}; run(f"SK_{par}{val:g}", list(st), np.log(list(st.values())), mdl)
if "SKU" in todo:   # SK with the district-specific term drifting (rate ku): the D = 0 limit of SKDe written with the exact identity
    st = dict(R["SK"]["params"]); st["ku"] = 0.01; run("SKU", list(st), np.log(list(st.values())), dict(R["SK"]["model"]))
if "SKUD" in todo:   # SKU + diffusive field with pseudo-inverse weights: the diffusive test against the best reference
    st = dict(R["SKU"]["params"]); st.update(sd=0.005, kappa=0.03, D=0.9); run("SKUD", list(st), np.log(list(st.values())), dict(R["SKU"]["model"]))
if "SKr50" in todo:   # SK with the geographic kernel range at 50 km instead of the 20 km selected in S
    st = dict(R["SK"]["params"]); run("SK_r50", list(st), np.log(list(st.values())), dict(lr=50.0, w=1.0))
if "EWprof" in todo:   # EW-consistent weights: profile in D (kappa and the rest refitted) to check that D -> 0 is not a local optimum
    for src, grid in (("SKDe", (0.1, 0.3, 1.0)), ("SDe", (0.9,))):
        sk = R[src]["params"]
        for val in grid:
            st = {k: v for k, v in sk.items() if k != "D"}; mdl = dict(R[src]["model"]); mdl["fix"] = {"D": val}; run(f"{src}_D{val:g}", list(st), np.log(list(st.values())), mdl)
# --- summary table
print("\nmodel  nll  n_params  LR_vs_S  persistent_share  D  kappa")
fS = R.get("S", {}).get("nll", np.nan)
for k, v in R.items():
    d = v.get("dyn_field", {}); print(f"{k:8s} {v['nll']:10.1f} {v['n_params']:3d} {2*(fS-v['nll']):9.1f}  {v.get('persistent_share_of_field') or 0:.3f}  {d.get('D', float('nan')):.4g} {d.get('kappa', v['params'].get('kappa', float('nan'))):.4g}")
cf.save_json(out, OUT); print(f"done [{time.time()-T0:.0f} s]")
