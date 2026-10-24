"""Stage 4b — JRV-level within-district term (2022 R1, 2022 R2, 2026).

Within each electoral district, the variance of logit turnout across its JRVs (polling stations) is compared with the
binomial expectation <1/(n p (1-p))> over its JRVs: the ratio is a direct estimate of the herding / within-unit factor h
(Borghesi & Bouchaud 2010 §3), independent of the cross-election argument of Stage 1. JRV electors from the padron map.
Also: the between-JRV correlation inside districts (intra-class correlation of tau at JRV level) = the sigma_mu^2 share that lives
below the district scale. Output: data/processed/stage4_jrv_within.json, stage4_jrv_within_by_district.csv
"""
import sys, io, csv, zipfile, collections, json
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf
raw = cf.ROOT / "data" / "raw" / "tse_juntas"
JOBS = [("2022r1", "DEFINITIVO_juntas_TSE_2022_ronda1.zip", "_consolidado_presidenciales.csv", "2022"),
        ("2022r2", "juntas_TSE_2022_ronda2_provisional_y_definitivo.zip", "_consolidado_definitivo.csv", "2022"),
        ("2026", "DEFINITIVO_juntas_TSE_2026.zip", "_consolidado_presidenciales.csv", "2026")]
master = cf.load_master().set_index("codelec")
out = {}; rows = []
for tag, zname, member, py in JOBS:
    jm = pd.read_csv(cf.PROC / f"padron_{py}_jrv_map.csv", dtype={"codelec": str}); jmap = dict(zip(jm.jrv.astype(int), jm.codelec)); emap = dict(zip(jm.jrv.astype(int), jm.electors.astype(int)))
    with zipfile.ZipFile(raw / zname) as z: txt = z.read(member).decode("utf-8-sig")
    per = {}
    for r in csv.DictReader(io.StringIO(txt)):
        if r["tipo_territorio"].strip('"') != "NACIONAL": continue
        j = int(r["junta"])
        if j not in per: per[j] = int(r["total_votos_recibidos"])
    d = pd.DataFrame([(jmap.get(j), emap.get(j, 0), v) for j, v in per.items()], columns=["codelec", "n", "v"]).dropna()
    d = d[(d.n >= 50) & (d.v > 0) & (d.v < d.n) & d.codelec.isin(master.index)]
    d["p"] = d.v / d.n; d["tau"] = cf.logit(d.p); d["binom"] = 1 / (d.n * d.p * (1 - d.p))
    g = d.groupby("codelec")
    t = pd.DataFrame(dict(n_jrv=g.size(), N=g.n.sum(), var_tau=g.tau.var(ddof=1), binom_mean=g.binom.mean(), p_mean=g.p.mean()))
    t = t[t.n_jrv >= 3]; t["h_within"] = t.var_tau / t.binom_mean; t["tag"] = tag
    rows.append(t.reset_index())
    # pooled estimate: sum of within-district SS / sum of expected SS (weights by df)
    h_pooled = float(np.sum((t.n_jrv - 1) * t.var_tau) / np.sum((t.n_jrv - 1) * t.binom_mean))
    # excess variance beyond binomial, absolute (this is the within-unit sigma^2 at JRV scale)
    excess = float(np.average(t.var_tau - t.binom_mean, weights=t.n_jrv - 1))
    # district-size dependence of h_within
    q = pd.qcut(t.N, 4, labels=["Q1 small", "Q2", "Q3", "Q4 large"]); hq = {str(k): float(np.sum((v.n_jrv - 1) * v.var_tau) / np.sum((v.n_jrv - 1) * v.binom_mean)) for k, v in t.groupby(q, observed=True)}
    out[tag] = dict(n_jrv=int(len(d)), n_districts=int(len(t)), h_within_pooled=h_pooled, h_within_median=float(t.h_within.median()), h_within_by_size_quartile=hq,
                    excess_var_within=excess, median_var_tau_within=float(t.var_tau.median()), median_binom=float(t.binom_mean.median()))
    print(f"{tag}: {len(d)} JRVs in {len(t)} districts (>=3 JRVs); within-district var(tau_jrv) median {t.var_tau.median():.4f} vs binomial {t.binom_mean.median():.4f}; "
          f"h_within pooled = {h_pooled:.2f} (median {t.h_within.median():.2f}); by size quartile {', '.join(f'{k}: {v:.2f}' for k, v in hq.items())}; excess within-unit variance {excess:.4f}")
# compare with Stage 1 cross-election h
s1 = json.load(open(cf.PROC / "stage1_summary.json"))
out["stage1_h_cross_election"] = {k: v["h"] for k, v in s1["variance_decomposition"].items()}
pd.concat(rows).to_csv(cf.PROC / "stage4_jrv_within_by_district.csv", index=False)
cf.save_json(out, cf.PROC / "stage4_jrv_within.json"); print("done")
