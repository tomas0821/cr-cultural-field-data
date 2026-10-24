"""Stage 0c — 2010/2014/2018 cómputo workbooks → electoral-district rows with party votes,
crosswalked to padrón CODELEC by (province#, canton#, normalised DE name).
Output: data/processed/results_<year>_de.csv (codelec may be empty when unmatched),
        data/processed/crosswalk_<year>_unmatched.csv
"""
import re, csv, zipfile, unicodedata, collections, sys
from pathlib import Path
import openpyxl
root = Path(__file__).resolve().parents[1]; proc = root/"data"/"processed"; rawz = root/"data"/"raw"/"tse_computo"
def norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii","ignore").decode().upper()
    s = re.sub(r"\d+/\s*$", "", s); s = re.sub(r"[^A-Z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()
INSTITUTIONAL = re.compile(r"^(C P |H A |C S M |CAI |CAE |HLE |C A I |CENTRO PENAL|HOGAR|UNIDAD DE ATENCION|CENTRO DE ATENCION)")
FILES = {"2010": ("computovotos_febrero2010.zip", "ComputoVotos_ febrero2010_Presidente.xlsm", [f"C1.{i}" for i in range(10,17)]),
         "2014": ("computovotos_febrero_abril_2014.zip", "Computo de Votos Elecciones Febrero y Abril 2014.xlsm", [f"C1.{i}" for i in range(11,18)]),
         "2018": ("computovotos_febrero_abril_2018.zip", "Computo de Votos Elecciones 2018.xlsm", [f"C1.{i}" for i in range(11,18)])}
# padrón key → codelec (2026 distelec; fall back to 2022)
padkey = collections.defaultdict(list)
for y in ("2026","2022"):
    with open(proc/f"padron_{y}_units.csv") as fh:
        for r in csv.DictReader(fh):
            c=r["codelec"]
            if int(c) >= 800000: continue
            k=(int(c[0]), int(c[1:3]), norm(r["distrito_electoral"]))
            if c not in padkey[k]: padkey[k].append(c)
for year,(zname, member, sheets) in FILES.items():
    with zipfile.ZipFile(rawz/zname) as z:
        m=[n for n in z.namelist() if n.endswith(member.split("/")[-1])][0]
        xl = z.open(m)
        wb = openpyxl.load_workbook(xl, read_only=True, data_only=True)
    rows=[]
    for pnum, sh in enumerate(sheets, 1):
        ws = wb[sh]; grid=[list(r) for r in ws.iter_rows(values_only=True)]
        # locate header rows
        hj = next(i for i,r in enumerate(grid) if any(isinstance(c,str) and norm(c)=="JUNTAS" for c in r))
        hp = next(i for i,r in enumerate(grid) if i>hj and any(isinstance(c,str) and norm(c)=="TOTAL" and r.index(c)>=4 for c in r))
        hdr = [norm(c) if isinstance(c,str) else "" for c in grid[hp]]
        # merge upper header rows for the non-party columns
        for i in range(hj, hp):
            for k,c in enumerate(grid[i]):
                if isinstance(c,str) and not hdr[k]: hdr[k]=norm(c)
        col_tot = hdr.index("TOTAL")
        parties=[]; end=None
        for k in range(col_tot+1, len(hdr)):
            h=hdr[k]
            if h in ("NULOS","BLANCOS","ABSTENCIONISMO","VOTOS NULOS","VOTOS EN BLANCO") or h.startswith("ABSTENCION"):
                end=k; break
            if h: parties.append((k,h))
        col_nulos = next((k for k in range(col_tot+1,len(hdr)) if hdr[k] in ("NULOS","VOTOS NULOS")), None)
        col_blancos = next((k for k in range(col_tot+1,len(hdr)) if hdr[k] in ("BLANCOS","VOTOS EN BLANCO")), None)
        canton=cnum=dist=dnum=None
        for r in grid[hp+1:]:
            a=r[0]
            if not isinstance(a,str): continue
            u=norm(a)
            mm=re.match(r"^CANTON\s*:?\s*(\d+)\s+(.*)$",u)
            if mm: cnum=int(mm.group(1)); canton=mm.group(2); continue
            mm=re.match(r"^DISTRITO\s*:?\s*(\d+)\s+(.*)$",u)
            if mm: dnum=int(mm.group(1)); dist=mm.group(2); continue
            if u.startswith(("PROVINCIA","CUADRO","SEGUN","TOTAL","NOTA","FUENTE")) or "FUENTE" in u or re.match(r"^\d+/",u): continue
            if cnum is None or not isinstance(r[1],(int,float)): continue
            key=(pnum,cnum,u); codes=padkey.get(key,[])
            rows.append(dict(year=year, prov=pnum, cnum=cnum, canton=canton, dnum=dnum, dist_admin=dist, de_name=u,
                             codelec=(codes[0] if len(codes)==1 else ""), n_codes=len(codes),
                             institutional=int(bool(INSTITUTIONAL.match(u))),
                             juntas=round(r[1]), electorado=round(r[2]), recibidos=round(r[3]), validos=round(r[col_tot]),
                             nulos=round(r[col_nulos]) if col_nulos is not None and isinstance(r[col_nulos],(int,float)) else "",
                             blancos=round(r[col_blancos]) if col_blancos is not None and isinstance(r[col_blancos],(int,float)) else "",
                             **{ "P:"+h: (round(r[k]) if isinstance(r[k],(int,float)) else 0) for k,h in parties}))
    fields=list(rows[0].keys())
    for rr in rows:
        for f in fields: rr.setdefault(f,0)
    with open(proc/f"results_{year}_de.csv","w",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=fields); w.writeheader(); w.writerows(rows)
    matched=[x for x in rows if x["codelec"]]; inst=[x for x in rows if x["institutional"]]
    amb=[x for x in rows if x["n_codes"]>1]
    with open(proc/f"crosswalk_{year}_unmatched.csv","w",newline="") as fh:
        w=csv.writer(fh); w.writerow(["prov","cnum","canton","dnum","dist_admin","de_name","institutional","electorado"])
        for x in rows:
            if not x["codelec"]: w.writerow([x["prov"],x["cnum"],x["canton"],x["dnum"],x["dist_admin"],x["de_name"],x["institutional"],x["electorado"]])
    pcols=[f for f in fields if f.startswith("P:")]
    print(f"{year}: DE rows={len(rows)} matched={len(matched)} ({100*len(matched)/len(rows):.1f}%) ambiguous={len(amb)} institutional={len(inst)} "
          f"parties={len(pcols)} electorado={sum(x['electorado'] for x in rows):,} recibidos={sum(x['recibidos'] for x in rows):,} "
          f"turnout={sum(x['recibidos'] for x in rows)/sum(x['electorado'] for x in rows):.4f} | non-institutional unmatched={sum(1 for x in rows if not x['codelec'] and not x['institutional'])}")
    print("   parties:", [p[2:] for p in pcols])
