# Cultural field of electoral turnout in Costa Rica: data and code

Derived data and analysis code for the manuscript *Where does the cultural field live? A slowly drifting kinship landscape, not a diffusing field, in Costa Rican turnout, 2010–2026*.

The study tests the diffusive "cultural field" model of electoral correlations on the Costa Rican electoral district (*distrito electoral*, ~2 100 units) using six presidential result sets (2010, 2014, 2018, 2022 first round and run-off, 2026) and a kinship kernel computed from the surnames of all electors in the national roll (Lasker isonymy distance and Bhattacharyya surname-overlap between districts).

## Layout

- `scripts/` — one script per stage, run from the repository root:
  - `stage0_*.py` ingestion (roll surnames, polling-station maps, result workbooks, master table, social covariates)
  - `stage1_replication.py`, `stage2_isonymy.py` — spatial statistics of turnout; isonymy matrices
  - `stage3*.py` — Gaussian-field likelihood profiles over the propagation graph, synthetic recovery panels, distance-matrix inference, mode decay
  - `stage4*.py` — robustness, static kinship kernel, single-election model comparison
  - `stage5_figures.py`, `stage6_joint_dynamics.py`, `stage6_figures.py` — figures and the joint six-election likelihood
  - `stage6_round*.sh` — sequential drivers of the robustness and recovery runs
- `src/crfield.py` — shared loaders, kernels, Laplacians and likelihood helpers
- `data/processed/` — all derived tables (CSV), summaries (JSON) and matrices (NPZ); see `data/README.md`
- `data/raw/` — the parts of the raw inputs that are small and derived from public documents (parsed district coordinates from the 2026 territorial decree; the 2011 census education table as extracted text; the MIDEPLAN social development index shapefile as downloaded). The TSE result and roll files are not redistributed; sources are listed below.
- `figures/` — the manuscript figures

## Sources

- Tribunal Supremo de Elecciones (TSE): per-polling-station results 2022 and 2026, district-level result workbooks 2010–2018, the electoral roll (*padrón*) 2022 and 2026, and the 2026 decree of electoral districts with seat coordinates.
- Instituto Nacional de Estadística y Censos (INEC): Censo 2011, *Características sociales y demográficas*, cuadro 48 (district education indicators).
- MIDEPLAN: Índice de Desarrollo Social 2023, district shapefile.
- Administrative boundaries: Humanitarian Data Exchange (HDX) Costa Rica subnational boundaries.
- Greater Metropolitan Area polygon: Plan GAM.

## Reproduction

```bash
pip install -r requirements.txt
python3 scripts/preview_stage1_stage2.py        # first look, about one minute
python3 scripts/stage6_joint_dynamics.py        # joint six-election likelihood (hours)
```

The stage scripts print their own coverage and sanity reports. Cluster or long runs pin `OMP_NUM_THREADS`.

## Licence

Code: MIT. Derived data: CC BY 4.0. Original data remain under the terms of their publishers (TSE, INEC, MIDEPLAN, HDX).
