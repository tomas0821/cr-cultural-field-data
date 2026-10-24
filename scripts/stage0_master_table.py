"""Stage 0d — master electoral-district table: one row per domestic CODELEC (2026 padrón),
with names, padrón sizes, decree coordinates, and per-election electorate/turnout where present.
Output: data/processed/de_master.csv + coverage report.
"""
import csv, re, unicodedata, math
from pathlib import Path
root=Path(__file__).resolve().parents[1]; proc=root/"data"/"processed"; dte=root/"data"/"raw"/"dte"
def norm(s):
    s=unicodedata.normalize("NFKD",str(s)).encode("ascii","ignore").decode().upper()
    return re.sub(r"\s+"," ",re.sub(r"[^A-Z0-9 ]"," ",s)).strip()
INST=re.compile(r"^(C P |H A |C S M |CAI |CAE |HLE |CENTRO PENAL|HOGAR|CAPEMCOL|UNIDAD DE ATENCION|CENTRO DE ATENCION)")
units={}
for r in csv.DictReader(open(proc/"padron_2026_units.csv")):
    c=r["codelec"]
    if int(c)>=800000: continue
    units[c]=dict(codelec=c, provincia=r["provincia"], canton=r["canton"], distrito_electoral=r["distrito_electoral"],
                  prov=int(c[0]), cnum=int(c[1:3]), institutional=int(bool(INST.match(norm(r["distrito_electoral"])))),
                  N_2026=int(r["N"]), N_2022="")
for r in csv.DictReader(open(proc/"padron_2022_units.csv")):
    if r["codelec"] in units: units[r["codelec"]]["N_2022"]=int(r["N"])
# coordinates (Wayback edition first, live edition as fallback)
coords={}
for f in ("dte2026_live_coords.csv","dte2026_wayback_coords.csv"):
    for r in csv.DictReader(open(dte/f)):
        coords[r["codelec6"]]=(int(r["E_crtm05"]),int(r["N_crtm05"]))
# Documented corrections to the decree coordinates (checked 2026-09-20 against the CRTM05 mainland box):
#  505011 EL COCO (Carrillo): decree prints N=166896, a dropped leading digit; Playas del Coco is at N≈1166896.
#  601105 ISLA DEL COCO: genuine (Cocos Island, ~500 km offshore, 16 electors) — flagged offshore=1 and excluded from spatial analyses.
coords["505011"]=(coords.get("505011",(314055,0))[0], 1166896)
OFFSHORE={"601105"}
for c,u in units.items():
    u["offshore"]=int(c in OFFSHORE)
    e,n=coords.get(c,("",""))
    u["E_crtm05"]=e; u["N_crtm05"]=n
# elections
def add(tag, path, ecol, rcol, vcol):
    for r in csv.DictReader(open(path)):
        c=r.get("codelec","")
        if c in units:
            u=units[c]; u[f"electores_{tag}"]=r[ecol]; u[f"recibidos_{tag}"]=r[rcol]; u[f"validos_{tag}"]=r[vcol]
add("2010", proc/"results_2010_de.csv","electorado","recibidos","validos")
add("2014", proc/"results_2014_de.csv","electorado","recibidos","validos")
add("2018", proc/"results_2018_de.csv","electorado","recibidos","validos")
add("2022r1", proc/"results_2022r1_de.csv","electores_padron","recibidos","validos")
add("2022r2", proc/"results_2022r2_de.csv","electores_padron","recibidos","validos")
add("2026", proc/"results_2026_de.csv","electores_padron","recibidos","validos")
tags=["2010","2014","2018","2022r1","2022r2","2026"]
fields=["codelec","provincia","canton","distrito_electoral","prov","cnum","institutional","offshore","N_2026","N_2022","E_crtm05","N_crtm05"]+[f"{k}_{t}" for t in tags for k in ("electores","recibidos","validos")]
with open(proc/"de_master.csv","w",newline="") as fh:
    w=csv.DictWriter(fh,fieldnames=fields); w.writeheader()
    for c in sorted(units):
        u=units[c]; w.writerow({f:u.get(f,"") for f in fields})
dom=[u for u in units.values()]; nonin=[u for u in dom if not u["institutional"] and not u["offshore"]]
print(f"domestic DEs={len(dom)} institutional={len(dom)-len(nonin)} with_coords={sum(1 for u in dom if u['E_crtm05']!='')} "
      f"nonInst_with_coords={sum(1 for u in nonin if u['E_crtm05']!='')}")
for t in tags:
    have=[u for u in nonin if u.get(f"electores_{t}")]
    print(f"  {t}: non-institutional DEs with results={len(have)} ({100*len(have)/len(nonin):.1f}%)")
p5=[u for u in nonin if all(u.get(f"electores_{t}") for t in tags) and u["E_crtm05"]!=""]
print(f"P5 panel (all six result sets + coords, non-institutional): {len(p5)}  ({100*len(p5)/len(nonin):.1f}% of non-institutional)")
big=[u for u in p5 if u["N_2026"]>=200]; print(f"  of which N_2026>=200: {len(big)}; >=500: {sum(1 for u in p5 if u['N_2026']>=500)}")
# distance range check
import statistics
xs=[(u["E_crtm05"],u["N_crtm05"]) for u in p5]
d=[math.hypot(xs[i][0]-xs[j][0],xs[i][1]-xs[j][1]) for i in range(0,len(xs),37) for j in range(i+1,len(xs),41)]
print(f"  pair distance sample: min={min(d)/1e3:.1f} km median={statistics.median(d)/1e3:.1f} km max={max(d)/1e3:.1f} km (n={len(d)})")
