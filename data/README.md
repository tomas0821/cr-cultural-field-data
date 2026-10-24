# data/

`raw/` — `tse_padron`, `tse_juntas`, `boundaries`, `gam` are symlinks into
`../../IsingCR/data/raw/` (not tracked; see that project's README for provenance).
`raw/tse_computo/` — TSE cómputo-de-votos workbooks: computovotos_febrero2010.zip,
computovotos_febrero_abril_2014.zip, computovotos_febrero_abril_2018.zip
(https://www.tse.go.cr/zip/elecciones/…, downloaded 2026-09-20 with a browser UA).
`raw/dte/` — División Territorial Electoral 2026 decree (Decreto 9-2025) PDF: Wayback capture
2026-01-12 and the live infoconsulados edition; pdftotext output; parsed coordinate CSVs
(`code8, codelec6, name, E_crtm05, N_crtm05`); 2022 decree (no coordinates).

`processed/`
- `padron_<year>_{paternal,maternal}.npz` — sparse (units × surnames) counts; row order =
  `padron_<year>_units.csv`; column order = `padron_<year>_vocab.txt`.
- `padron_<year>_jrv_map.csv` — JRV → CODELEC, electors per JRV (1:1 in both years).
- `results_2022r1_de.csv`, `results_2022r2_de.csv`, `results_2026_de.csv` — per-DE
  aggregates from per-JRV files (`electores_padron` is the registry count to use).
- `results_2010_de.csv`, `results_2014_de.csv`, `results_2018_de.csv` — per-DE rows from the
  workbooks with party columns (`P:<party>`), `codelec` filled where the name crosswalk is
  unique; `crosswalk_<year>_unmatched.csv` lists the rest.
- `de_master.csv` — one row per domestic 2026 CODELEC with names, `institutional`,
  `offshore`, N_2026, N_2022, coordinates, and electores/recibidos/validos per election.
- `lasker_2026_pooled_N200.npz` — preview Lasker/R matrices for N ≥ 200 districts.
- `de_rows_20xx.csv`, `unmatched_20xx.txt` — deep-plan round-2 parse tests (superseded).
