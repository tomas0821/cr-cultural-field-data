import csv, numpy as np, scipy.sparse as sp
rows=[r for r in csv.DictReader(open("data/processed/de_master.csv")) if r["E_crtm05"] and r["institutional"]=="0" and r["offshore"]=="0"]
def corr_fn(tag):
    sel=[r for r in rows if r.get(f"electores_{tag}") and int(r[f"electores_{tag}"])>=100]
    E=np.array([int(r["E_crtm05"]) for r in sel],float); N=np.array([int(r["N_crtm05"]) for r in sel],float)
    Ne=np.array([int(r[f"electores_{tag}"]) for r in sel],float); V=np.array([int(r[f"recibidos_{tag}"]) for r in sel],float)
    p=np.clip(V/Ne,1e-3,1-1e-3); tau=np.log(p/(1-p))
    # size-bin residuals (bins of ~100 units by N)
    order=np.argsort(Ne); res=np.empty_like(tau)
    for k in range(0,len(order),100):
        idx=order[k:k+100]; m=tau[idx].mean(); s=tau[idx].std()+1e-9
        res[idx]=tau[idx]-m
    keep=np.abs(res)<5*res.std(); res=res[keep]; E=E[keep]; N=N[keep]; Ne=Ne[keep]
    D=np.hypot(E[:,None]-E[None,:],N[:,None]-N[None,:])/1e3
    iu=np.triu_indices(len(res),1); d=D[iu]; prod=(res[:,None]*res[None,:])[iu]/res.var()
    edges=np.array([0,2,4,7,10,15,20,30,45,65,90,130,180,250,350])
    print(f"\n{tag}: units={len(res)} var(tau')={res.var():.4f} mean binomial var={np.mean(1/(Ne*p[keep]*(1-p[keep]))):.4f}")
    print("  r-bin[km]   n_pairs   C(r)   ±se")
    for a,b in zip(edges[:-1],edges[1:]):
        m=(d>=a)&(d<b)
        if m.sum()<50: continue
        c=prod[m].mean(); se=prod[m].std()/np.sqrt(m.sum())
        print(f"  {a:4d}-{b:<4d} {m.sum():9d} {c:7.4f} {se:7.4f}")
    return sel
for t in ("2026","2022r1","2018","2014","2010"): corr_fn(t)
# ---- Lasker distance (2026 padrón, both surnames pooled) vs geography
units=[r["codelec"] for r in csv.DictReader(open("data/processed/padron_2026_units.csv"))]
P=sp.load_npz("data/processed/padron_2026_paternal.npz"); M=sp.load_npz("data/processed/padron_2026_maternal.npz")
C=(P+M).astype(float)
idx={c:i for i,c in enumerate(units)}
sel=[r for r in rows if r["codelec"] in idx and int(r["N_2026"])>=200]
I=np.array([idx[r["codelec"]] for r in sel]); Cs=C[I]
n=np.asarray(Cs.sum(1)).ravel(); Pf=sp.diags(1/n)@Cs
R=(Pf@Pf.T).toarray()/2.0
L=-np.log(np.clip(2*R,1e-12,None))
E=np.array([int(r["E_crtm05"]) for r in sel],float); N=np.array([int(r["N_crtm05"]) for r in sel],float)
D=np.hypot(E[:,None]-E[None,:],N[:,None]-N[None,:])/1e3
iu=np.triu_indices(len(sel),1)
print(f"\nLasker (pooled surnames, N>=200): units={len(sel)} L range {L[iu].min():.2f}-{L[iu].max():.2f} median {np.median(L[iu]):.2f}")
print(f"  corr(L, r)={np.corrcoef(L[iu],D[iu])[0,1]:.3f}   corr(L, ln r)={np.corrcoef(L[iu],np.log(D[iu]+0.5))[0,1]:.3f}")
for a,b in [(0,5),(5,15),(15,40),(40,100),(100,200),(200,400)]:
    m=(D[iu]>=a)&(D[iu]<b); print(f"  r {a:3d}-{b:<3d} km: mean L={L[iu][m].mean():.3f} (n={m.sum()})")
np.savez_compressed("data/processed/lasker_2026_pooled_N200.npz", codelec=np.array([r["codelec"] for r in sel]), L=L.astype(np.float32), R=R.astype(np.float32), E=E, N=N)
