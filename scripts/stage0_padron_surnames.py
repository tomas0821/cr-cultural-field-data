"""Stage 0a — padrón → surname counts per electoral district (CODELEC) + JRV→CODELEC map.

Input : PADRON_COMPLETO.txt (ISO-8859-1, comma-separated, fixed-width fields)
        fields: CEDULA, CODELEC, RELLENO, FECHACADUC, JUNTA, NOMBRE, APELLIDO1, APELLIDO2
Output: data/processed/padron_<year>_{paternal,maternal}.npz (sparse counts: units × surnames;
        pooled = paternal + maternal), padron_<year>_units.csv (CODELEC, N, names),
        padron_<year>_jrv_map.csv (JRV → CODELEC, electors), padron_<year>_vocab.txt
Usage : python3 stage0_padron_surnames.py <year> <path/to/PADRON_COMPLETO.txt> <distelec.txt>
"""
import sys, csv, unicodedata, collections, numpy as np, scipy.sparse as sp
from pathlib import Path

year, padron, distelec = sys.argv[1], sys.argv[2], sys.argv[3]
out = Path(__file__).resolve().parents[1] / "data" / "processed"
out.mkdir(parents=True, exist_ok=True)

def norm(s):
    s = unicodedata.normalize("NFKD", s.strip().upper())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())

# unit and surname indices
unit_idx, units = {}, []
vocab_idx, vocab = {}, []
def uid(c):
    if c not in unit_idx:
        unit_idx[c] = len(units); units.append(c)
    return unit_idx[c]
def sid(s):
    if s not in vocab_idx:
        vocab_idx[s] = len(vocab); vocab.append(s)
    return vocab_idx[s]

rows_p, cols_p, rows_m, cols_m = [], [], [], []
N = collections.Counter()
jrv_map = collections.defaultdict(collections.Counter)   # jrv -> Counter(codelec)
bad = 0
with open(padron, encoding="latin-1", newline="") as fh:
    for line in fh:
        parts = line.rstrip("\r\n").split(",")
        if len(parts) < 8:
            bad += 1; continue
        codelec = parts[1].strip()
        jrv = parts[4].strip()
        a1, a2 = norm(parts[6]), norm(parts[7])
        u = uid(codelec)
        N[codelec] += 1
        jrv_map[jrv][codelec] += 1
        if a1:
            rows_p.append(u); cols_p.append(sid(a1))
        if a2:
            rows_m.append(u); cols_m.append(sid(a2))

nu, nv = len(units), len(vocab)
P = sp.coo_matrix((np.ones(len(rows_p), dtype=np.int32), (rows_p, cols_p)), shape=(nu, nv)).tocsr()
M = sp.coo_matrix((np.ones(len(rows_m), dtype=np.int32), (rows_m, cols_m)), shape=(nu, nv)).tocsr()
P.sum_duplicates(); M.sum_duplicates()
sp.save_npz(out / f"padron_{year}_paternal.npz", P)
sp.save_npz(out / f"padron_{year}_maternal.npz", M)
(out / f"padron_{year}_vocab.txt").write_text("\n".join(vocab), encoding="utf-8")

# distelec names
names = {}
with open(distelec, encoding="latin-1") as fh:
    for line in fh:
        p = line.rstrip("\r\n").split(",")
        if len(p) >= 4:
            names[p[0].strip()] = (norm(p[1]), norm(p[2]), norm(p[3]))
with open(out / f"padron_{year}_units.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["codelec", "N", "provincia", "canton", "distrito_electoral"])
    for c in units:
        w.writerow([c, N[c], *names.get(c, ("", "", ""))])
multi = 0
with open(out / f"padron_{year}_jrv_map.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["jrv", "codelec", "electors", "n_codelec"])
    for jrv, cnt in sorted(jrv_map.items()):
        c, n = cnt.most_common(1)[0]
        if len(cnt) > 1: multi += 1
        w.writerow([jrv, c, sum(cnt.values()), len(cnt)])

Ns = np.array([N[c] for c in units])
print(f"{year}: records={Ns.sum():,} bad={bad} units={nu} surnames={nv} "
      f"paternal_entries={P.sum():,} maternal_entries={M.sum():,} jrvs={len(jrv_map)} "
      f"jrv_multi_codelec={multi} N min/med/max={Ns.min()}/{int(np.median(Ns))}/{Ns.max()} "
      f"units<200={int((Ns<200).sum())} units<500={int((Ns<500).sum())}")
