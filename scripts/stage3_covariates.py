"""Stage 3 prep — district covariates: GAM membership (cabecera inside the MIVAH Plan GAM polygon, CRTM05),
local density proxy (districts within 5 km of the cabecera), ln N_2026, province. Output: data/processed/de_covariates.csv"""
import sys, struct
from pathlib import Path
import numpy as np, pandas as pd
from matplotlib.path import Path as MPath
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import crfield as cf

def read_polygons(shp):
    """minimal ESRI shapefile reader for polygon (type 5) records -> list of rings (arrays of x,y)."""
    b = open(shp, "rb").read(); pos = 100; rings = []
    while pos < len(b):
        rec_no, clen = struct.unpack(">ii", b[pos:pos + 8]); pos += 8; start = pos
        stype = struct.unpack("<i", b[pos:pos + 4])[0]
        if stype == 5:
            nparts, npts = struct.unpack("<ii", b[pos + 36:pos + 44])
            parts = struct.unpack(f"<{nparts}i", b[pos + 44:pos + 44 + 4 * nparts])
            pts = np.frombuffer(b[pos + 44 + 4 * nparts:pos + 44 + 4 * nparts + 16 * npts], dtype="<f8").reshape(npts, 2)
            for i, p0 in enumerate(parts):
                p1 = parts[i + 1] if i + 1 < nparts else npts; rings.append(pts[p0:p1])
        pos = start + 2 * clen
    return rings
rings = read_polygons(cf.ROOT / "data/raw/gam/gam_limite/GAM_LIMITE_1.shp")
print(f"GAM polygon: {len(rings)} rings, x range {min(r[:,0].min() for r in rings):.0f}-{max(r[:,0].max() for r in rings):.0f}, y range {min(r[:,1].min() for r in rings):.0f}-{max(r[:,1].max() for r in rings):.0f}")
m = cf.load_master()
pts = m[["E_crtm05", "N_crtm05"]].to_numpy(float)
inside = np.zeros(len(m), int)
for r in rings: inside += MPath(r).contains_points(pts)
gam = (inside % 2 == 1).astype(int)
D = cf.pair_distance_km(pts[:, 0], pts[:, 1]); np.fill_diagonal(D, np.inf)
dens5 = (D < 5).sum(1); nn1 = D.min(1)
cov = pd.DataFrame(dict(codelec=m.codelec, provincia=m.provincia, gam=gam, n_within_5km=dens5, nn_dist_km=nn1, lnN_2026=np.log(m.N_2026.clip(lower=1))))
cov["density_class"] = pd.qcut(dens5.astype(float) + 1e-6 * np.arange(len(m)), 3, labels=["low", "mid", "high"]).astype(str)
cov.to_csv(cf.PROC / "de_covariates.csv", index=False)
print(f"GAM districts: {gam.sum()} of {len(m)} (electors {m.N_2026[gam==1].sum():,} of {m.N_2026.sum():,}); density tertile cut-points: {np.percentile(dens5,[33.3,66.7])}; "
      f"GAM share by density class: {cov.groupby('density_class').gam.mean().round(2).to_dict()}")
print(cov.groupby("provincia").agg(n=("gam","size"), gam=("gam","sum")).T.to_string())
