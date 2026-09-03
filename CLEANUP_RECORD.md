# Cleanup Record — files deleted to free storage

Date: 2026-09-03
Scope: removed recoverable data products and generated outputs to free disk space.
Everything here can be recovered later via one of the documented paths below.

## How to recover

1. **Git-tracked outputs / source** — run on this repo's `main` branch:
   ```
   git log --all --oneline -- <path>        # find the commit that had it
   git checkout <commit> -- <path>          # restore
   ```
   (Recent commits: `0b01302` HEAD, `c231b86`, `6f8743b`, `a4482f1`, …)

2. **New large imagery products** (TMC ortho TIFF, IIR cube, OHRC NRP img) —
   these were committed only on the local Cline checkpoint ref
   `87f3cd6` (tree = `a4482f1`). Restore from git:
   ```
   git cat-file -e 87f3cd6:<path>           # verify blob exists (currently true)
   git checkout 87f3cd6 -- <path>
   ```
   They are ALSO available for re-download from the PDS/CNSA Chang'e-2 archive
   by their product id (see filename). Prefer re-download for a fresh copy.

3. **Product bundles (OHRC / TMC-2 directories + extracted zip)** — these were
   never committed (gitignored); recover by re-downloading from the PDS archive
   by product id and re-extracting.

## Deleted items (largest first)

### A. New large imagery data (recoverable via checkpoint ref + re-download)
| Item | Size | Recovery |
|---|---|---|
| `ch2_tmc_ndn_20231101T0125121377_d_oth_d18.tif` | 18.2 GB | git `87f3cd6` / PDS re-download |
| `ch2_iir_nci_20231225T1904122779_d_img_d18.qub` | 3.1 GB | git `87f3cd6` / PDS re-download |
| `ch2_ohr_nrp_20200827T0030107497_d_img_d18.img` | 1.2 GB | git `87f3cd6` / PDS re-download |
| `ch2_ohr_nrp_20200827T0030107497_b_brw_d18.png` | 7.6 MB | git `87f3cd6` / PDS re-download |
| (their `.xml` sidecars) | <0.2 MB | git `87f3cd6` / PDS re-download |

### B. Extracted archive (re-downloadable)
| Item | Size | Recovery |
|---|---|---|
| `ch2_tmc_ncn_20250707T1853051045_d_img_d18.zip` | 715.7 MB | PDS re-download |

### C. Downloaded product bundles (PDS re-download; never committed)
| Item | Size | Recovery |
|---|---|---|
| `ch2_ohr_ncp_20231004T0406038822_d_img_d18/` | 1.17 GB | PDS re-download |
| `ch2_ohr_ncp_20260103T0609041371_d_img_d18/` | 1.17 GB | PDS re-download |
| `ch2_ohr_ncp_20260103T1005176450_d_img_d18/` | 1.17 GB | PDS re-download |
| `ch2_tmc_ncn_20250707T1853051045_d_img_d18/` | 1.73 GB | PDS re-download |
| `ch2_tmc_ncn_20260813T0627378557_d_img_d18/` | 1.15 GB | PDS re-download |
| `ch2_tmc_ncn_20260813T1023298745_d_img_d18/` | 1.15 GB | PDS re-download |
| `OHRC_ShapeFiles/`, `TMC2_ShapeFiles/` | ~32 MB | PDS/shapefile re-download |

### D. Generated outputs (regenerable by re-running the relevant phase; several tracked in git)
| Item | Size |
|---|---|
| `baseline_output/`, `baseline_output_relaxed/` | ~7 MB |
| `comparison_output/`, `crater_output/`, `crater_graph_comparison*` | ~18 MB |
| `phase3_eval_output/`, `phase3d_eval_output/`, `phase5_labels/` | ~26 MB |
| `phase6_ohrc_ohrc_output/`, `phase7_fmt_output/` | ~9 MB |
| `phase8_aniso_output/`, `phase9_output/`, `phase10_output/` | ~3 MB |
| `phase11_geoprior_output/`, `phase11_audit_output/` | ~0.3 MB |
| `phase12_transfer_output/`, `phase12_verify_output/`, `phase13_correspondence_export/` | ~4 MB |
| `data/` | ~4 MB |
| `.pytest_cache/`, `__pycache__/` | ~0.2 MB |

## Preserved / NOT deleted
- `lunar_data_pipeline/` (source code), all run scripts, all `*_findings.md` docs
- `.git/` (history; after `git gc`)
- `.venv/` temporarily (can be regenerated with `pip install -r lunar_data_pipeline/requirements.txt` if needed)
- Shapefiles under `OHRC_ShapeFiles/` / `TMC2_ShapeFiles/` only if still wanted for catalog queries
