# Worklog & Next-Step Instructions (remastered branch)

Date: 2025-10-06

## 1. Summary of Work Completed Today

### Hypsometry & Bathymetry Pipeline
- Implemented `reconstruct_hypsometry.py` enabling physical rebuild of area–volume curve from bathymetry + DEM with:
  - Unified reprojection & grid synthesis
  - Shoreline + NDWI mask ingestion (single or multiple) and union
  - Auto outward buffering to match target surface area (Amax ≈ 93.7 km²)
  - Generation of reconstructed curve (`area_volume_curve_reconstructed.csv`) and plots
  - Framework hooks for future vertical calibration (depth scaling / shoreline elevation alignment)
  - Implemented conservative masking logic: shoreline polygon rasterization → NDWI mask union → optional iterative outward buffering with convergence condition (relative tolerance) ensuring controlled area growth rather than a single large dilation. Logged each iteration's area for auditability (visible in stdout during runs).
  - Added adaptive NDWI threshold heuristic: if NDWI dynamic range detected within [-1,1] choose 0.3 water threshold, else treat as already binarized; prevents misclassification for pre-thresholded masks.
  - Added CRS selection heuristic (compute centroid lon/lat → derive UTM zone; fallback EPSG:6933) guaranteeing near-equal-area behavior for volumetric integration across widely separated rasters.
  - Added shoreline elevation estimation via upper quantile (p=0.985) of merged DEM to approximate datum alignment before vertical calibration phase.
- Enhanced multi-resolution bathymetry generation logic in `Bathymetry/Main/Analysis_Isobars_Full.py`:
  - Added argparse interface (`--grid_steps`, `--out_prefix`, `--stats_csv`, `--no_plots`)
  - Loop over arbitrary resolution set (e.g. 100,50,20 m)
  - Per-resolution statistics (min/max/mean depth, pixel count, coarse volume proxy) → CSV summary
  - GeoTIFF export for each resolution with diagnostic orientation checks

### Data & Modeling Infrastructure
- Added modular database ingestion utility `wbm/db.py` with partitioned or unified SQLite build from CSV artifacts.
- Introduced lightweight path & data loader scaffolding (`wbm/paths.py`, `wbm/ui/data_loader.py`) to decouple from missing original implementations and enable robust fallback ingestion.
- Implemented ensemble, forecast, and trend analysis core modules:
  - `wbm/ensemble.py` (block bootstrap residual ensemble for volume projection)
  - `wbm/forecast.py` (robust season + Theil–Sen trend deterministic extender)
  - `wbm/trends.py` (Theil–Sen + Kendall tau + uncertainty & aggregated comparison tooling)
- Ensured statistical robustness by preferring medians (seasonal cycle) and Theil–Sen slopes over OLS to mitigate leverage from short anomalous warm/cold or wet/dry sequences.
- Established consistent time index sanitation (sorting + duplicate drop) in loaders, setting a foundation for deterministic climatology and reproducible trend diagnostics.
- Added partition-aware SQLite builder with table classification (core / era5 / outputs) enabling incremental data layering and potential future remote sync without monolithic DB rebuild.

### Documentation
- Added comprehensive Russian technical documentation:
  - `Документация для создания/Project_Documentation.md` (system architecture, data flow, module catalog)
  - `Документация для создания/Документ по логике модели.md` (hydrologic model technical plan & governing equations)
- Compiled today’s consolidated worklog (this file) summarizing implemented features and pending calibration workflow.
- Provided two-tier documentation approach: (1) System-level architecture & stakeholder narrative; (2) Formal hydrologic process specification with mathematical decomposition and calibration levers—supports both managerial review and technical extension.
- Embedded explicit unit conventions (mm/day ↔ MCM conversion) to prevent ambiguity in subsequent parameter calibration tasks.

## 2. Current State Assessment
- Spatial extent of reservoir polygon successfully tuned (A ≈ target). Volume remains vastly inflated (≈16,064 MCM vs design 791 MCM) indicating vertical scaling / datum inconsistency.
- Multi-resolution bathymetry execution still pending (blocked previously by missing `geopandas` runtime on this machine). Code changes are in place; only dependency installation + run required.
- Vertical calibration logic (scaling factor search to enforce Vmax) not yet implemented in `reconstruct_hypsometry.py`; structural placeholder present (`calibrate_vertical`).
- Risk identified: Raw reconstructed curve indicates average depth unrealistically high relative to design storage (implied mean depth ≈ 169 m vs expected ~ 791 MCM / 93.7 km² ≈ 8.44 m). Confirms vertical exaggeration factor on order ~20x requiring controlled scaling rather than spatial edits.

## 3. Immediate Next Steps
1. Install geospatial dependencies required for multi-resolution generation (if not already installed):
   - geopandas, fiona, shapely, pyproj, rasterio (ensure compiled wheels available for Windows)
2. Run multi-resolution bathymetry script:
   ```bash
   python Bathymetry/Main/Analysis_Isobars_Full.py --grid_steps 100,50,20 --no_plots --out_prefix bathymetry_hh --stats_csv bathymetry_resolution_stats.csv
   ```
3. Inspect `bathymetry_resolution_stats.csv` for volume convergence; decide canonical grid (likely 50 m or 20 m depending on stability vs cost).
4. Implement vertical calibration:
   - Add iterative search (binary or secant) over depth scaling factor α applied to (negative) depths prior to integration.
   - Optionally adjust shoreline elevation offset ΔL.
   - Objective: minimize |V_calibrated_max - 791| with A_max constraint preserved (within ±0.5%).
5. Produce calibrated curve → `processing_output/area_volume_curve_calibrated.csv` plus diagnostic plot overlay (original vs calibrated volume & area). 
6. Update application logic (in `app.py` or loader) to prefer calibrated curve file if present (fallback to legacy).
7. Archive provenance: write `processing_output/hypsometry_calibration_report.md` summarizing factors (α, ΔL, pre/post stats).

### Detailed Execution Outline for Next Steps
| Step | Input Artifacts | Operation | Output / Decision | Acceptance |
|------|-----------------|-----------|-------------------|------------|
| 3.1 | Existing modified isobars script | Install geo deps & run multi-res (100,50,20) | `bathymetry_hh_100m.tif`, `...50m.tif`, `...20m.tif`, stats CSV | Runtime < 15 min; no NaN-dominated grid |
| 3.2 | Stats CSV | Evaluate volume_mcm vs step | Choose canonical resolution (first point where |ΔV|/V < 2% when going finer) | Document choice rationale |
| 3.3 | Chosen raster + masks | Re-run reconstruction (no scaling) | Raw calibrated-ready curve | A_max error < 1% |
| 3.4 | Raw curve | Binary search α depth scale | Calibrated volume match | |V_max−791| ≤ 1 MCM |
| 3.5 | Calibrated grid/curve | (Optional) micro ΔL adjustment if area drift | Final curve | Area within ±0.5% |
| 3.6 | Curves (raw vs calibrated) | Diagnostics: incremental area/volume, mean depth | Report + plot | Monotonic & smooth |
| 3.7 | App & loader code | Add preference chain (calibrated > reconstructed > legacy) | Updated app behavior | Hot-swap without crash |
| 3.8 | Simulation tests | Run sample scenarios (baseline year & dry/wet) | Validation notes | No instability / volume blow-up |

### Binary Search Calibration Pseudocode
```
target = 791.0  # MCM
lo, hi = 0.01, 1.0  # Start with aggressive bounds; expand hi until V_hi > target
while volume(scaled_depths(hi)) < target:
  hi *= 2
for _ in range(max_iter):
  mid = 0.5*(lo+hi)
  curve = integrate(depths * mid)
  vmax = curve.volume_mcm.max()
  if abs(vmax - target) < tol: break
  if vmax > target: hi = mid
  else: lo = mid
return mid
```
Safeguard: abort if hi exceeds 100 or if scaling collapses min depth below numerical resolution (< cell_size / 10).

## 4. Vertical Calibration Design (Planned)
- Let original integrated depth grid (relative to provisional shoreline) yield volume function V_raw(L).
- Introduce scaling α > 0 applied to submerged depth magnitudes: d' = α * d.
- Optionally introduce vertical shift ΔL to align shoreline elevation reference (if DEM vs bathymetry mismatch).
- Recompute cumulative volume curve for candidate (α, ΔL). 
- Optimization approach:
  1. Hold ΔL = 0 initially; solve for α satisfying V_max_target.
  2. If residual area drift emerges due to numerical clipping, small ΔL refinement (±0.5 m) to preserve A_max (rarely needed if mask-based area fixed).
  3. Use monotonicity of volume vs α to apply binary search (tolerance 0.1 MCM).
- Safeguards:
  - Reject α yielding max area deviation > 1%.
  - Report α, resulting max depth, mean depth, and RMSE vs design volume if intermediate reference points known.
    - Add secondary diagnostic: compute implied mean depth (V_max / A_max) and compare with regional bathymetric plausibility range; flag if deviation > ±30%.

## 5. Risk & Validation Checklist
| Aspect | Risk | Mitigation |
|--------|------|------------|
| Depth sign inversion | Double inversion leading to positive depths | Assert median(depth) < 0 before scaling |
| Mask underfill | NDWI gap leaves cavities reducing usable area | Fill small holes via binary closing before integration |
| Over-buffering | Auto-buffer overshoots area target | Implement final erosive refinement (negative buffer) if > +0.7% |
| Volume calibration instability | Non-linear artifacts from DEM noise | Pre-smooth depths (Gaussian kernel) before scaling pass |
| CRS distortion | Non-equal-area projection inflates area | Force equal-area (UTM zone or EPSG:6933 fallback) |
| Mask edge aliasing | Jagged polygon raster edges alter perimeter & area | Apply morphological opening+closing (1 px) pre-integration |
| Large depth outliers | Spikes dominate volume after scaling | Winsorize depths at p99 before scaling (retain extremes log) |

## 6. Future Enhancements (Post-Calibration)
- Integrate runoff (ERA5 ro) partitioned into direct inflow vs basin retention via simple runoff coefficient model.
- Add snowmelt augmentation: accumulate SWE increases; release positive ΔSWE * melt_factor on warming days.
- Develop automated parameter fitting (P_scale, ET_scale, runoff_coeff) against historical volume via objective minimization (MAE / NSE) using differential evolution.
- Add ensemble driver using stochastic perturbations of P/ET consistent with historical variance (beyond residual bootstrap for deterministic trend case).
- Integrate hindcast automatic scoring (Nash–Sutcliffe Efficiency, Kling-Gupta Efficiency) to complement MAE/RMSE for volume pathway evaluation.
- Introduce parameter provenance ledger (JSON) tying each calibrated curve to α, ΔL, mask hash, resolution, smoothing kernel, timestamp.

## 7. File Inventory Added / Modified Today
- Added: `reconstruct_hypsometry.py` (physical rebuild pipeline)
- Modified: `Bathymetry/Main/Analysis_Isobars_Full.py` (multi-resolution & stats)
- Added Core Modules: `wbm/db.py`, `wbm/ensemble.py`, `wbm/forecast.py`, `wbm/paths.py`, `wbm/trends.py`, `wbm/ui/data_loader.py`
- Added Documentation: `Документация для создания/Project_Documentation.md`, `Документация для создания/Документ по логике модели.md`
- Added: `WORKLOG_REM_AHEAD.md` (this file)
  - (Will be extended tomorrow with calibration parameter table once α finalized.)

## 8. Quick Usage Reference
Reconstruct (spatial + buffer, no vertical calibration yet):
```bash
python reconstruct_hypsometry.py \
  --bathy Bathymetry/Main/bathymetry_hh.tif \
  --dem "Hillshade + DEM/output_hh.tif" \
  --shoreline shoreline.shp \
  --ndwi_mask processing_output/ndwi_mask_0275.tif \
  --auto_buffer --amax 93.7 --vmax 791 --hmax 19.8 \
  --out processing_output --overwrite
```
Multi-resolution bathymetry stats:
```bash
python Bathymetry/Main/Analysis_Isobars_Full.py \
  --grid_steps 100,50,20 --no_plots --out_prefix bathymetry_hh \
  --stats_csv bathymetry_resolution_stats.csv
```

## 9. Acceptance Targets for Calibration
- Max area error: |A_max_calibrated - 93.7| ≤ 0.5 km²
- Max volume error: |V_max_calibrated - 791| ≤ 1 MCM
- Monotonicity: area and volume strictly non-decreasing with elevation
- Smoothness: no negative incremental area increments > tolerance (floating jitter) after median filter (3-level window)

## 10. Actionable TODO Snapshot
- [ ] Run multi-resolution generation & review stats
- [ ] Implement α depth scaling search
- [ ] Produce calibrated curve + report
- [ ] Wire calibrated curve into application preference order
- [ ] Validate simulation outputs with new curve (spot check evaporation sensitivity)
  - [ ] Add NSE / KGE evaluation vs historical after curve swap
  - [ ] Record α, depth stats, mean depth plausibility check

---
Prepared for branch: remastered
