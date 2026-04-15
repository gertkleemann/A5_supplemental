# A₅ Cell Framework — Computational Supplement

**Paper:** *Does Fundamental Physics Follow from Pure Geometry?*
**Authors:** Gert Kleemann, Hendrik M. Michalsky
**DOI:** [10.5281/zenodo.19582689](https://doi.org/10.5281/zenodo.19582689)

All scripts and data files required to reproduce every computation in the paper. Every numerical result is backed by a machine-readable JSON file. No cluster access, no proprietary software, no special expertise beyond undergraduate linear algebra.

---

## Requirements

```
Python 3.9+
pip install numpy numba mpmath
```

Optional: `scipy` (for eigenvector analysis scripts).

All scripts run on a standard laptop. The heavy transfer matrices (1024×1024 and 3125×3125) take hours to days of CPU time but are resumable — progress files are saved incrementally.

---

## Quick Start

**Reproduce all algebraic results in < 1 second:**

```
python3 a5f_compute_all_tables.py
```

This regenerates all 27 algebraic tables (character tables, 3j/5j/6j symbols, class algebra, Laplacian eigenvalues, α formula verification) from scratch. No input files needed.

**Reproduce the full 3125×3125 transfer matrix spectrum:**

```
python3 analyse_3125.py
```

Requires `qcd3125_progress.json` (pre-computed transfer matrix rows, included as `qcd3125_progress.zip` — unzip first).

---

## Scripts

### Core algebraic computation

| Script | Purpose | Input | Output | Time |
|--------|---------|-------|--------|------|
| `a5f_compute_all_tables.py` | All 27 algebraic tables | None | `a5f_*.json` | < 1 sec |

### Transfer matrices (heavy computation, resumable)

| Script | Purpose | Input | Output | Time |
|--------|---------|-------|--------|------|
| `qcd1024_numba.py` | Visible sector 1024×1024 | None (builds from scratch) | `qcd1024_progress.json` | ~20 CPU-hours |
| `dark_sector_1024.py` | Dark sector 1024×1024 | None | `dark_sector_1024_progress.json` | ~20 CPU-hours |
| `dodec_subsectors.py` | All six dodecahedral subsectors (32×32, 243×243) | None | `dodec_*_progress.json` | Minutes–hours |
| `dodec_subsectors_numba.py` | Same with Numba optimisation | None | Same | Faster |
| `icosa_boundary_2I.py` | 2I boundary transfer matrix (Linux) | None | `icosa_2I_*_progress.json` | Hours per sector |
| `icosa_boundary_2I_win.py` | Same (Windows-compatible) | None | Same | Hours per sector |
| `icosa_boundary_A5.py` | A₅ icosahedral boundary | None | `icosa_A5_*_progress.json` | Hours |
| `icosa_boundary_z3.py` | Z₃ boundary projection | None | Output to stdout | Minutes |
| `run_all_2I_sectors.sh` | Shell script: all 2I boundary sectors | None | All `icosa_2I_*` files | Hours |

### Analysis scripts (require pre-computed transfer matrices)

| Script | Purpose | Input required |
|--------|---------|----------------|
| `analyse_3125.py` | Full spectrum analysis | `qcd3125_progress.json` (unzip `qcd3125_progress.zip`) |
| `analyse_3125_T2.py` | Two-cell (T²) analysis | `qcd3125_progress.json` |
| `analyse_3125_projected.py` | Projected HVP analysis | `qcd3125_progress.json` |
| `alpha_from_1024.py` | α extraction from visible sector | `qcd1024_progress.json` |
| `compute_alpha_running.py` | α running to M_Z | `qcd1024_progress.json` |
| `hadron_spectrum.py` | Hadron spectrum from subgroups | `dodec_*_progress.json` |
| `qcd_generations.py` | Subgroup transfer matrices (A₄, Z₃, Z₂) | None (builds subsectors) |
| `generation_masses.py` | Generation mass ratios | Subgroup results |
| `qcd_vacuum_1024.py` | Vacuum composition | `qcd1024_progress.json` |
| `extract_mW.py` | W boson mass | `qcd1024_progress.json` |
| `dark_visible_portal.py` | Dark-visible portal states | `qcd3125_progress.json` eigenvectors |
| `klein_nishina_v2.py` | Thomson scattering (1+cos²θ) | None (uses Laplacian) |
| `discrete_dirac.py` | Dirac operator and propagator | None (uses dodecahedral graph) |
| `entanglement_entropy.py` | Vacuum entanglement S = 2α | `qcd3125_progress.json` eigenvectors |
| `entanglement_entropy_universal.py` | Entropy for any sector | Any eigenvector file |
| `schmidt_coefficients.py` | Schmidt decomposition | `qcd3125_progress.json` eigenvectors |
| `analyse_2I_eigensystems.py` | Boundary analysis, Theorem 20c | `icosa_2I_*_eigen.npz` |
| `five_quick_analyses.py` | Proton dark content, gap ratios | `qcd3125_progress.json` eigenvectors |
| `six_quick_analyses.py` | Pion composition, Wigner-Dyson | `qcd3125_progress.json` eigenvectors |
| `boundary_filtration_cosmology.py` | Face-rep projection, cosmology | `qcd3125_progress.json` eigenvectors |
| `source_operator_spectroscopy.py` | Meson/baryon source operators | `qcd3125_progress.json` eigenvectors |
| `multicell_correlator.py` | Multi-cell C(N) correlators | `qcd3125_progress.json` |
| `four_final_analyses.py` | Connected correlator, Dirac, entanglement | `qcd3125_progress.json` eigenvectors |

---

## Result files

All results are JSON. Filename conventions:

| Prefix | Meaning |
|--------|---------|
| `a5f_` | Algebraic/analytical (regenerable in < 1 second) |
| `dodec_` | Dodecahedral bulk subsectors |
| `qcd` | Dodecahedral bulk (full or visible sector) |
| `dark` | Dark sector |
| `icosa_2I_` | Icosahedral 2I boundary |
| `icosa_A5_` | Icosahedral A₅ boundary |

Eigensystem files (`.npz`) contain eigenvalues and eigenvectors for boundary transfer matrices.

Progress files store transfer matrix rows as JSON arrays with row indices, enabling incremental computation and independent verification:
```json
{"rows": [{"row": 0, "data": [T[0,0], T[0,1], ...]}, ...]}
```

---

## Reproducing key results

**Fine structure constant (< 1 second):**
```
python3 a5f_compute_all_tables.py
# Check a5f_alpha_formula.json → α⁻¹ = 137.035999260424...
```

**Thomson scattering (< 1 minute):**
```
python3 klein_nishina_v2.py
# Reproduces (1+cos²θ) to 10⁻¹⁶
```

**Dirac propagator (< 1 minute):**
```
python3 discrete_dirac.py
# S = −(d̂·σ)/3, eigenvalue E = √3 exact
```

**Full hadron spectrum (requires pre-computed matrices):**
```
# Unzip qcd3125_progress.zip first
python3 analyse_3125.py
# Proton 938 MeV (1.9%), pion 135 MeV (blind), etc.
```

---

## License

MIT — see [LICENSE](LICENSE).

## Citation

```
Kleemann, G. & Michalsky, H.M. (2026). Does Fundamental Physics Follow from Pure Geometry?
A zero-parameter construction from the combination of a dodecahedron, an icosahedron,
and an observer. Zenodo. https://doi.org/10.5281/zenodo.19582689
```
