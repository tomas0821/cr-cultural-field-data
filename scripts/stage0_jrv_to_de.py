"""Stage 0b — per-JRV consolidados (2022 R1, 2022 R2, 2026) → electoral-district (CODELEC) tables.

JRV → CODELEC comes from the padrón JRV map of the matching year (every JRV maps to one CODELEC).
Output: data/processed/results_<tag>_de.csv with columns
        codelec, n_jrv, electores, recibidos, validos, nulos, blancos, <party columns...>
plus a coverage report on stdout.
"""
import sys, csv, io, zipfile, collections, unicodedata
from pathlib import Path
root = Path(__file__).resolve().parents[1]
proc = root / "data" / "processed"
raw = root / "data" / "raw" / "tse_juntas"

def norm(s):
    s = unicodedata.normalize("NFKD", s.strip().upper())
    return " ".join("".join(c for c in s if not unicodedata.combining(c)).split())

def load_map(year):
    m, e = {}, {}
    with open(proc / f"padron_{year}_jrv_map.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            m[int(r["jrv"])] = r["codelec"]; e[int(r["jrv"])] = int(r["electors"])
    return m, e

JOBS = [
    ("2022r1", "DEFINITIVO_juntas_TSE_2022_ronda1.zip", "_consolidado_presidenciales.csv", "2022"),
    ("2022r2", "juntas_TSE_2022_ronda2_provisional_y_definitivo.zip", "_consolidado_definitivo.csv", "2022"),
    ("2026",   "DEFINITIVO_juntas_TSE_2026.zip", "_consolidado_presidenciales.csv", "2026"),
]
for tag, zname, member, py in JOBS:
    jmap, emap = load_map(py)
    with zipfile.ZipFile(raw / zname) as z:
        txt = z.read(member).decode("utf-8-sig")
    rd = csv.DictReader(io.StringIO(txt))
    per_jrv = {}          # jrv -> dict(totals) (repeated on every party row)
    votes = collections.defaultdict(lambda: collections.Counter())  # jrv -> party -> votes
    for r in rd:
        if r["tipo_territorio"].strip('"') != "NACIONAL":
            continue
        j = int(r["junta"])
        if j not in per_jrv:
            per_jrv[j] = dict(electores=int(r["electores_inscritos"]), validos=int(r["votos_validos"]),
                              nulos=int(r["votos_nulos"]), blancos=int(r["votos_en_blanco"]),
                              recibidos=int(r["total_votos_recibidos"]),
                              prov=norm(r["provincia_pais"]), canton=norm(r["canton_ciudad"]), dist=norm(r["distrito"]))
        votes[j][norm(r["partido"])] += int(r["votos"])
    parties = sorted({p for j in votes for p in votes[j]})
    agg = collections.defaultdict(lambda: collections.Counter())
    unmapped, jrv_used = [], 0
    for j, t in per_jrv.items():
        c = jmap.get(j)
        if c is None:
            unmapped.append(j); continue
        jrv_used += 1
        a = agg[c]
        a["n_jrv"] += 1
        a["electores_padron"] += emap[j]
        if t["electores"] == 0: a["n_jrv_zero_elec"] += 1
        for k in ("electores", "validos", "nulos", "blancos", "recibidos"):
            a[k] += t[k]
        for p in parties:
            a["P:" + p] += votes[j][p]
    outp = proc / f"results_{tag}_de.csv"
    with open(outp, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["codelec", "n_jrv", "n_jrv_zero_elec", "electores_padron", "electores_results", "recibidos", "validos", "nulos", "blancos"] + parties)
        for c in sorted(agg):
            a = agg[c]
            w.writerow([c, a["n_jrv"], a["n_jrv_zero_elec"], a["electores_padron"], a["electores"], a["recibidos"], a["validos"], a["nulos"], a["blancos"]] + [a["P:" + p] for p in parties])
    tot_e = sum(a["electores_padron"] for a in agg.values()); tot_r = sum(t["recibidos"] for t in per_jrv.values())
    print(f"{tag}: national JRVs={len(per_jrv)} mapped={jrv_used} unmapped={len(unmapped)} "
          f"DEs={len(agg)} parties={len(parties)} electores(padron)={tot_e:,} recibidos={tot_r:,} turnout={tot_r/tot_e:.4f} "
          f"JRVs_with_zero_elec_in_results={sum(a['n_jrv_zero_elec'] for a in agg.values())}"
          + (f" unmapped_sample={unmapped[:8]}" if unmapped else ""))
