"""Stage 0 add-on (2026-10-24): district-level social covariates for the confound check.
Sources (downloaded into data/raw/ids and data/raw/censo2011):
  - MIDEPLAN Indice de Desarrollo Social 2023, district shapefile (CODIGO_DTA = INEC 5-digit district code; IDS_2023 and its five dimensions)
  - INEC Censo 2011, Cuadro 48 (Indicadores educativos segun provincia, canton y distrito): illiteracy %, mean schooling (15+), attendance, % secondary+
Electoral districts (CODELEC) are assigned to administrative districts by point-in-polygon of the district seat into the HDX admin-3 polygons
(data/processed/de_admin3_crosswalk.csv); 2011 districts are matched to 2024 admin codes by canton/district name,
with canton renames (Alfaro Ruiz->Zarcero, Valverde Vega->Sarchi, Aguirre->Quepos); admin districts created after 2011 inherit nothing (NaN).
Output: data/processed/de_covariates_social.csv (codelec, pcode, ids2023, ids_edu, ids_eco, school2011, illit2011, secondary2011)."""
import struct, unicodedata, sys, pandas as pd, numpy as np
from pathlib import Path
root = Path(__file__).parents[1]; PROC = root / "data/processed"
def read_dbf(path, enc):
    f = open(path, 'rb'); hdr = f.read(32); n, hl, rl = struct.unpack('<xxxxIHH', hdr[:12]); fields = []
    while True:
        b = f.read(32)
        if b[0] == 0x0D: break
        fields.append((b[:11].split(b'\0')[0].decode('latin1'), chr(b[11]), b[16]))
    f.seek(hl); recs = []
    for i in range(n):
        r = f.read(rl); pos = 1; d = {}
        for name, typ, ln in fields: d[name] = r[pos:pos + ln].decode(enc, 'replace').strip(); pos += ln
        recs.append(d)
    return recs
norm = lambda s: unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower().replace("  ", " ").strip()
H = pd.DataFrame(read_dbf(root / "data/raw/boundaries/extracted/cri_admin3.dbf", "utf-8")); H["pcode"] = H.adm3_pcode.str[2:]
H["canton_n"] = H.adm2_name.map(norm); H["dist_n"] = H.adm3_name.map(norm); H["prov_n"] = H.adm1_name.map(norm)
I = pd.DataFrame(read_dbf(root / "data/raw/ids/extracted/ids_2023_distrito.dbf", "utf-8")); I["pcode"] = I.CODIGO_DTA.str.zfill(5)
for c in ["INDICE_SAL", "INDICE_PAR", "INDICE_SEG", "INDICE_EDU", "INDICE_ECO", "IDS_2023"]: I[c] = I[c].astype(float)
rows = pd.read_csv(root / "data/raw/censo2011/cuadro48_rows_raw.csv").rename(columns={"pop": "popn"}); rows["n"] = rows.name.map(norm)
provs = ["san jose", "alajuela", "cartago", "heredia", "guanacaste", "puntarenas", "limon"]
alias = {"sarchi": "valverde vega", "quepos": "aguirre"}
cant = H.drop_duplicates("adm2_pcode").sort_values("adm2_pcode"); cant = cant[~cant.canton_n.isin(["rio cuarto", "monteverde", "puerto jimenez"])]
exp = [(r.adm2_pcode[2:], alias.get(r.canton_n, r.canton_n), r.prov_n) for r in cant.itertuples()]
out = []; ptr = 0; cur = None; i = 0; prov = None
while i < len(rows):
    r = rows.iloc[i]
    if r.n in provs and ptr < len(exp) and exp[ptr][2] == r.n and (prov is None or exp[ptr][2] != exp[ptr - 1][2]) and r.n != prov:
        prov = r.n; i += 1; continue
    if ptr < len(exp) and r.n == exp[ptr][1] and exp[ptr][2] == prov:
        cur = dict(code=exp[ptr][0], canton=exp[ptr][1], prov=prov, popn=r.popn, dists=[]); out.append(cur); ptr += 1; i += 1; continue
    if cur is not None: cur["dists"].append(dict(name=r.n, popn=r.popn, illit=r.illit, school=r.school, attend=r.attend, secondary=r.secondary))
    i += 1
bad = [(c["canton"], c["popn"], sum(d["popn"] for d in c["dists"])) for c in out if c["popn"] != sum(d["popn"] for d in c["dists"])]
print("cantons parsed", len(out), "of", len(exp), "; canton-population mismatches:", bad); print("districts parsed", sum(len(c["dists"]) for c in out))
C = []
for c in out:
    for d in c["dists"]:
        m = H[(H.adm2_pcode.str[2:] == c["code"]) & (H.dist_n == d["name"])]
        if len(m) == 0: m = H[(H.prov_n == c["prov"]) & (H.dist_n == d["name"])]
        C.append(dict(canton2011=c["canton"], dist2011=d["name"], pcode=(m.pcode.iloc[0] if len(m) else ""), nmatch=len(m), **{k: d[k] for k in ("popn", "illit", "school", "attend", "secondary")}))
C = pd.DataFrame(C); print("2011 districts matched:", (C.pcode != "").sum(), "unmatched:", C[C.pcode == ""][["canton2011", "dist2011"]].values.tolist())
Cm = C[C.pcode != ""].copy(); print("2011 rows sharing one admin code (fallback matches):", Cm.pcode.duplicated().sum()); Cm = Cm[~Cm.pcode.duplicated(keep="first")]
H2 = H[["pcode", "adm1_name", "adm2_name", "adm3_name"]].merge(Cm[["pcode", "popn", "illit", "school", "attend", "secondary"]], on="pcode", how="left")
miss = H2[H2.school.isna()]; print("admin districts without 2011 values:", len(miss), miss[["adm2_name", "adm3_name"]].values.tolist())
H2.to_csv(PROC / "admin3_census2011.csv", index=False)
I[["pcode", "PROVINCIA", "CANTON", "DISTRITO", "INDICE_SAL", "INDICE_PAR", "INDICE_SEG", "INDICE_EDU", "INDICE_ECO", "IDS_2023"]].to_csv(PROC / "admin3_ids2023.csv", index=False)
X = pd.read_csv(PROC / "de_admin3_crosswalk.csv", dtype={"codelec": str}); X["pcode"] = X.adm3_pcode.str[2:]
S = X[["codelec", "pcode"]].merge(I[["pcode", "IDS_2023", "INDICE_EDU", "INDICE_ECO"]], on="pcode", how="left").merge(H2[["pcode", "school", "illit", "secondary"]], on="pcode", how="left")
S = S.rename(columns={"IDS_2023": "ids2023", "INDICE_EDU": "ids_edu", "INDICE_ECO": "ids_eco", "school": "school2011", "illit": "illit2011", "secondary": "secondary2011"})
S["imputed2011"] = S.school2011.isna().astype(int)
for c in ("school2011", "illit2011", "secondary2011"):   # admin districts created after 2011: canton mean of the 2011 values
    cm = S.groupby(S.pcode.str[:3])[c].transform("mean"); S[c] = S[c].fillna(cm); S[c] = S[c].fillna(S.groupby(S.pcode.str[:1])[c].transform("mean"))   # cantons created after 2011: province mean
S.to_csv(PROC / "de_covariates_social.csv", index=False)
print("electoral districts:", len(S), "with IDS:", S.ids2023.notna().sum(), "with 2011 schooling:", S.school2011.notna().sum(), "(imputed by canton mean:", int(S.imputed2011.sum()), ")")
print(S.describe().round(2).to_string())
