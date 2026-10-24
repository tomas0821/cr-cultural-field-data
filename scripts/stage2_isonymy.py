"""Stage 2 — isonymy structure of Costa Rica from the 2026 roll, at the electoral-district level.

Series: pooled (both surnames), paternal, maternal.
Within unit: unbiased random isonymy I = sum n(n-1) / [N(N-1)], alpha = 1/I.
Between units: R_ij = sum_s p_i(s) p_j(s) / 2 ; Lasker L = -ln(2R) ; Nei = -ln(R_ij / sqrt(R_ii R_jj)) with
R_ii = I_i/2 ; Euclidean E = sqrt(1 - sum sqrt(p_i p_j)).
Sensitivity: N-cuts (all / >=200 / >=500), rarefaction to the median district size, paternal vs maternal.
Gate G2: Spearman(L_full, L_rarefied) > 0.9 on N>=200 pairs.
Outputs: data/processed/lasker_2026_<series>.npz, isonymy_2026_units.csv, isonymy_2026_cantons.csv,
         stage2_summary.json, figures/stage2_*.png
"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, scipy.sparse as sp
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
units = pd.read_csv(cf.PROC / "padron_2026_units.csv", dtype={"codelec": str})
master = cf.load_master().set_index("codelec")
keep = units.codelec.isin(master.index).to_numpy()
units = units[keep].reset_index(drop=True)
P0 = sp.load_npz(cf.PROC / "padron_2026_paternal.npz").tocsr()[np.flatnonzero(keep)]
M0 = sp.load_npz(cf.PROC / "padron_2026_maternal.npz").tocsr()[np.flatnonzero(keep)]
E = master.loc[units.codelec, "E_crtm05"].to_numpy(float); Nc = master.loc[units.codelec, "N_crtm05"].to_numpy(float)
D = cf.pair_distance_km(E, Nc); n_units = len(units)
print(f"units in spatial set: {n_units}; surname vocabulary: {P0.shape[1]}")

def within(C):
    """unbiased random isonymy per row, number of surnames, distinct surnames."""
    N = np.asarray(C.sum(1)).ravel().astype(float)
    s2 = np.asarray(C.multiply(C).sum(1)).ravel().astype(float)
    I = (s2 - N) / (N * (N - 1))
    return N, I, np.diff(C.indptr)

def between(C, I):
    N = np.asarray(C.sum(1)).ravel().astype(float)
    Pf = sp.diags(1 / N) @ C.astype(float)
    R = (Pf @ Pf.T).toarray() / 2.0
    L = -np.log(np.clip(2 * R, 1e-12, None))
    Rii = I / 2.0
    Nei = -np.log(np.clip(R / np.sqrt(np.outer(Rii, Rii)), 1e-12, None)); np.fill_diagonal(Nei, 0)
    Sq = Pf.sqrt(); Eu = np.sqrt(np.clip(1 - (Sq @ Sq.T).toarray(), 0, None)); np.fill_diagonal(Eu, 0)
    return R, L, Nei, Eu

def rarefy(C, target):
    """subsample each row with more than `target` surnames to exactly `target` (hypergeometric)."""
    C = C.tocsr(); out = C.copy().astype(np.int64)
    for i in range(C.shape[0]):
        a, b = C.indptr[i], C.indptr[i + 1]; cnt = C.data[a:b].astype(np.int64)
        if cnt.sum() > target:
            out.data[a:b] = rng.multivariate_hypergeometric(cnt, target)
    out.eliminate_zeros(); return out

series = {"pooled": (P0 + M0).tocsr(), "paternal": P0, "maternal": M0}
summary = {}; res = {}; unit_tab = units[["codelec", "provincia", "canton", "distrito_electoral"]].copy()
iu = np.triu_indices(n_units, 1)
for name, C in series.items():
    N, I, S = within(C)
    R, L, Nei, Eu = between(C, I)
    res[name] = dict(N=N, I=I, L=L, R=R, Nei=Nei, Eu=Eu)
    unit_tab[f"N_{name}"] = N.astype(int); unit_tab[f"S_{name}"] = S; unit_tab[f"I_{name}"] = I; unit_tab[f"alpha_{name}"] = 1 / I
    np.savez_compressed(cf.PROC / f"lasker_2026_{name}.npz", codelec=units.codelec.to_numpy(), N=N, I=I,
                        L=L.astype(np.float32), R=R.astype(np.float32), **({"Nei": Nei.astype(np.float32), "Eu": Eu.astype(np.float32)} if name == "pooled" else {}))
    summary[name] = {}
    for cut in (1, 200, 500):
        m = N >= cut * (2 if name == "pooled" else 1)   # cut on electors: pooled has 2 surnames per elector
        idx = np.flatnonzero(m); ju = np.triu_indices(len(idx), 1)
        Ls = L[np.ix_(idx, idx)][ju]; d = D[np.ix_(idx, idx)][ju]
        stats = dict(n_units=int(len(idx)), L_median=float(np.median(Ls)), L_p5=float(np.percentile(Ls, 5)), L_p95=float(np.percentile(Ls, 95)),
                     frac_L_inf=float(np.mean(~np.isfinite(Ls) | (Ls > 20))),
                     corr_L_r=float(np.corrcoef(Ls, d)[0, 1]), corr_L_lnr=float(np.corrcoef(Ls, np.log(d + 0.5))[0, 1]),
                     spearman_L_r=float(spearmanr(Ls, d).correlation))
        if name == "pooled":
            stats["corr_Nei_r"] = float(np.corrcoef(Nei[np.ix_(idx, idx)][ju], d)[0, 1]); stats["corr_Eu_r"] = float(np.corrcoef(Eu[np.ix_(idx, idx)][ju], d)[0, 1])
            stats["corr_L_Nei"] = float(np.corrcoef(Ls, Nei[np.ix_(idx, idx)][ju])[0, 1]); stats["corr_L_Eu"] = float(np.corrcoef(Ls, Eu[np.ix_(idx, idx)][ju])[0, 1])
        edges = [0, 5, 15, 40, 100, 200, 400]
        stats["L_by_r"] = [dict(r_lo=a, r_hi=b, L_mean=float(Ls[(d >= a) & (d < b)].mean()), n=int(((d >= a) & (d < b)).sum())) for a, b in zip(edges[:-1], edges[1:])]
        summary[name][f"N>={cut}"] = stats
        print(f"{name:8s} N>={cut:3d}: units={len(idx)} L median={stats['L_median']:.2f} [{stats['L_p5']:.2f},{stats['L_p95']:.2f}] "
              f"corr(L,r)={stats['corr_L_r']:.3f} corr(L,ln r)={stats['corr_L_lnr']:.3f} rho_s={stats['spearman_L_r']:.3f}"
              + (f" | Nei-r {stats['corr_Nei_r']:.3f} Eu-r {stats['corr_Eu_r']:.3f}" if name == "pooled" else ""))
    summary[name]["alpha_median"] = float(np.median(1 / I)); summary[name]["alpha_p5_p95"] = [float(np.percentile(1 / I, 5)), float(np.percentile(1 / I, 95))]
unit_tab.to_csv(cf.PROC / "isonymy_2026_units.csv", index=False)

# ---------------------------------------------------------------- sensitivity: rarefaction, paternal vs maternal
Np = res["pooled"]["N"]; med = int(np.median(Np[Np >= 400]))
m200 = Np >= 400; idx = np.flatnonzero(m200); ju = np.triu_indices(len(idx), 1)
Cr = rarefy(series["pooled"], med); Nr, Ir, _ = within(Cr); _, Lr, _, _ = between(Cr, Ir)
sp_rar = spearmanr(res["pooled"]["L"][np.ix_(idx, idx)][ju], Lr[np.ix_(idx, idx)][ju]).correlation
pe_rar = np.corrcoef(res["pooled"]["L"][np.ix_(idx, idx)][ju], Lr[np.ix_(idx, idx)][ju])[0, 1]
sp_pm = spearmanr(res["paternal"]["L"][np.ix_(idx, idx)][ju], res["maternal"]["L"][np.ix_(idx, idx)][ju]).correlation
sp_pp = spearmanr(res["paternal"]["L"][np.ix_(idx, idx)][ju], res["pooled"]["L"][np.ix_(idx, idx)][ju]).correlation
d = D[np.ix_(idx, idx)][ju]
summary["sensitivity"] = dict(rarefaction_target_surnames=med, n_units_N200=int(len(idx)), spearman_L_full_vs_rarefied=float(sp_rar), pearson_L_full_vs_rarefied=float(pe_rar),
                              corr_L_rarefied_r=float(np.corrcoef(Lr[np.ix_(idx, idx)][ju], d)[0, 1]),
                              spearman_L_paternal_vs_maternal=float(sp_pm), spearman_L_paternal_vs_pooled=float(sp_pp))
summary["gate_G2"] = dict(passed=bool(sp_rar > 0.9), criterion="Spearman(L_full, L_rarefied to median) > 0.9 on N>=200 pairs")
print(f"rarefaction to {med} surnames: Spearman full-vs-rarefied {sp_rar:.3f} (Pearson {pe_rar:.3f}); paternal-vs-maternal {sp_pm:.3f}; paternal-vs-pooled {sp_pp:.3f}; gate G2 {'PASSED' if sp_rar > 0.9 else 'FAILED'}")

# ---------------------------------------------------------------- PCoA of Lasker (pooled, N>=200)
Lm = res["pooled"]["L"][np.ix_(idx, idx)].astype(float); n = len(idx)
A = -0.5 * Lm ** 2; H = np.eye(n) - 1 / n; B = H @ A @ H
w, V = np.linalg.eigh(B); order = np.argsort(w)[::-1]; w = w[order]; V = V[:, order]
pos = w > 0; expl = w[pos] / w[pos].sum()
axes = V[:, :5] * np.sqrt(np.clip(w[:5], 0, None))
pc = pd.DataFrame(axes, columns=[f"PCo{i+1}" for i in range(5)]); pc.insert(0, "codelec", units.codelec.to_numpy()[idx])
pc["E_crtm05"] = E[idx]; pc["N_crtm05"] = Nc[idx]; pc.to_csv(cf.PROC / "isonymy_2026_pcoa.csv", index=False)
summary["pcoa"] = dict(n=int(n), explained_first5=[float(x) for x in expl[:5]], frac_negative_eigenvalues=float(np.mean(w < 0)),
                       corr_PCo1_E=float(np.corrcoef(axes[:, 0], E[idx])[0, 1]), corr_PCo1_N=float(np.corrcoef(axes[:, 0], Nc[idx])[0, 1]),
                       corr_PCo2_E=float(np.corrcoef(axes[:, 1], E[idx])[0, 1]), corr_PCo2_N=float(np.corrcoef(axes[:, 1], Nc[idx])[0, 1]))
print(f"PCoA: explained {np.round(expl[:5], 3)}; negative eigenvalues {np.mean(w < 0):.2f}; PCo1 corr with E/N = {summary['pcoa']['corr_PCo1_E']:.2f}/{summary['pcoa']['corr_PCo1_N']:.2f}")

# ---------------------------------------------------------------- canton-level check vs Saenz & Barrantes 2009 (Table 2, 1990 roll)
cant = units.assign(prov=units.codelec.str[0].astype(int), cnum=units.codelec.str[1:3].astype(int))
grp = cant.groupby(["prov", "cnum", "provincia", "canton"]).indices
rows = []
Cp = series["pooled"]
for (pr, cn, pv, ca), ii in grp.items():
    c = np.asarray(Cp[ii].sum(0)).ravel(); Nn = c.sum(); I_u = (np.sum(c * (c - 1))) / (Nn * (Nn - 1)); I_sb = np.sum((c / Nn) ** 2) - 1 / Nn
    rows.append(dict(prov=pr, cnum=cn, provincia=pv, canton=ca, N_surnames=int(Nn), I_unbiased=I_u, alpha=1 / I_u, I_saenz_form=I_sb, alpha_saenz_form=1 / I_sb))
ct = pd.DataFrame(rows).sort_values(["prov", "cnum"]); ct.to_csv(cf.PROC / "isonymy_2026_cantons.csv", index=False)
ref = {"SAN JOSE|CENTRAL": 171, "SAN JOSE|DESAMPARADOS": 131, "ALAJUELA|CENTRAL": 105, "CARTAGO|CENTRAL": 101, "HEREDIA|CENTRAL": 118,
       "ALAJUELA|GRECIA": 62, "ALAJUELA|PALMARES": 50, "SAN JOSE|ACOSTA": 41, "SAN JOSE|DOTA": 55, "SAN JOSE|TARRAZU": 52, "SAN JOSE|LEON CORTES CASTRO": 48}
chk = []
for k, a1990 in ref.items():
    pv, ca = k.split("|"); r = ct[(ct.provincia.str.upper() == pv) & (ct.canton.str.upper().str.replace("Ó", "O").str.replace("É", "E") == ca)]
    if len(r): chk.append(dict(canton=ca, alpha_1990_saenz=a1990, alpha_2026=float(r.alpha.iloc[0]), N_2026=int(r.N_surnames.iloc[0])))
summary["saenz_check"] = chk
for c in chk: print(f"  alpha {c['canton']:14s} Saenz 1990 = {c['alpha_1990_saenz']:4d}   2026 roll = {c['alpha_2026']:6.1f}")
if chk: summary["saenz_check_corr"] = float(np.corrcoef([c["alpha_1990_saenz"] for c in chk], [c["alpha_2026"] for c in chk])[0, 1])
cf.save_json(summary, cf.PROC / "stage2_summary.json")

# ---------------------------------------------------------------- figures
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
Ls = res["pooled"]["L"][np.ix_(idx, idx)][ju]
h = ax[0].hexbin(d, Ls, gridsize=60, bins="log", xscale="log", cmap="viridis"); ax[0].set_xlabel("r (km)"); ax[0].set_ylabel("Lasker L (pooled, N≥200)")
for name in ("pooled", "paternal", "maternal"):
    t = summary[name]["N>=200"]["L_by_r"]; ax[1].plot([np.sqrt(max(x["r_lo"], 1) * x["r_hi"]) for x in t], [x["L_mean"] for x in t], "o-", label=name)
ax[1].set_xscale("log"); ax[1].set_xlabel("r (km)"); ax[1].set_ylabel("mean L in bin"); ax[1].legend()
fig.tight_layout(); fig.savefig(cf.FIG / "stage2_L_vs_r.png", dpi=150)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
for k in range(2):
    s = ax[k].scatter(E[idx] / 1e3, Nc[idx] / 1e3, c=axes[:, k], s=6, cmap="RdBu_r"); ax[k].set_aspect("equal"); ax[k].set_title(f"PCo{k+1} ({expl[k]*100:.1f} %)"); plt.colorbar(s, ax=ax[k])
    ax[k].set_xlabel("E (km)"); ax[k].set_ylabel("N (km)")
fig.tight_layout(); fig.savefig(cf.FIG / "stage2_pcoa_map.png", dpi=150)
print("done:", cf.PROC / "stage2_summary.json")
