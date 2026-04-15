#!/usr/bin/env python3
"""
DARK-VISIBLE MIXING PORTAL
===========================
Extracts dark-visible portal amplitudes from the 3125x3125 eigenvectors.
Finds states that straddle the chi4 (dark) and chi5 (visible) sectors.

Requires: numpy
Usage: python dark_visible_portal.py qcd3125_progress_seeded.json
"""

import numpy as np
import json, sys, time, math

phi = (1 + math.sqrt(5)) / 2
Lambda = 332.0

# ================================================================
# LOAD MATRIX
# ================================================================
fname = 'qcd3125_progress_seeded.json'
if len(sys.argv) > 1:
    fname = sys.argv[1]

print(f"Loading {fname}...")
t0 = time.time()
with open(fname) as f:
    data = json.load(f)

rows = data.get('rows', [])
n = max(r['row'] for r in rows) + 1
print(f"  Matrix size: {n}x{n} ({len(rows)} rows loaded)")
print(f"  Load time: {time.time()-t0:.1f}s")

if n == 1024:
    k = 4
    print("  WARNING: This is a 1024x1024 (4-irrep) matrix.")
    print("  Dark-visible mixing requires the full 3125x3125 (5-irrep) matrix.")
    print("  The 1024 matrix has either chi4 OR chi5, not both.")
    sys.exit(1)
elif n == 3125:
    k = 5
    print("  Full 5-irrep matrix detected. Proceeding.")
else:
    k = round(n ** 0.2)
    print(f"  Detected k={k} irreps (k^5 = {k**5})")

# Build matrix
print("Building matrix...", end=' ', flush=True)
T = np.zeros((n, n), dtype=np.float64)
for r in rows:
    T[r['row'], :len(r['data'])] = r['data']
T = (T + T.T) / 2
print("done.")

# Check completeness
zero_rows = np.sum(np.all(T == 0, axis=1))
if zero_rows > 0:
    print(f"  WARNING: {zero_rows} zero rows — matrix may be incomplete!")

# ================================================================
# DIAGONALIZE
# ================================================================
print("Diagonalizing...", end=' ', flush=True)
t0 = time.time()
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
print(f"done ({time.time()-t0:.1f}s)")

T_max = evals[0]
n_pos = np.sum(evals > 0)
print(f"  T_max = {T_max:.4e}")
print(f"  Positive eigenvalues: {n_pos}")

# ================================================================
# IRREP COMPOSITION
# ================================================================
# For 5 irreps: chi1(0), chi3(1), chi3'(2), chi4(3), chi5(4)
# Config index = sum(label[j] * k^j) for j in 0..4

irrep_names = ['chi1', 'chi3', "chi3'", 'chi4', 'chi5']
irrep_dims = [1, 3, 3, 4, 5]

def config_to_labels(idx, k=5):
    labels = []
    for j in range(5):
        labels.append(idx % k)
        idx //= k
    return labels

# Precompute: for each config, which irreps are on which edges
print("Computing irrep projections...", end=' ', flush=True)
# Build projection masks: for each irrep r, mask[r] = set of configs with r on any edge
edge_irrep = np.zeros((n, 5), dtype=np.int32)  # edge_irrep[cfg, edge] = irrep label
for cfg in range(n):
    labels = config_to_labels(cfg, k)
    for j in range(5):
        edge_irrep[cfg, j] = labels[j]

# For each eigenstate and each irrep: fraction of that irrep
def compute_compositions(evecs, n_states, k=5):
    """Compute irrep composition for each eigenstate."""
    comps = np.zeros((n_states, k), dtype=np.float64)
    for i in range(n_states):
        v = evecs[:, i]
        v2 = v ** 2
        for r in range(k):
            # Sum |v(cfg)|^2 over all cfgs where any edge has irrep r
            for j in range(5):
                mask = edge_irrep[:, j] == r
                comps[i, r] += np.sum(v2[mask])
        comps[i] /= 5.0  # average over 5 edges
    return comps

n_analyse = min(200, n_pos)  # analyse top 200 states
comps = compute_compositions(evecs, n_analyse, k)
print("done.")

# ================================================================
# DARK-VISIBLE PORTAL STATES
# ================================================================
print(f"\n{'='*70}")
print("DARK-VISIBLE PORTAL ANALYSIS")
print(f"{'='*70}")

# Portal states: significant content in BOTH chi4 (dark) and chi5 (visible)
chi4_idx = 3  # dark matter
chi5_idx = 4  # photon/gluon

print(f"\n  Threshold: >5% in both chi4 AND chi5")
print(f"  {'State':>5} {'Mass(MeV)':>10} {'chi1%':>6} {'chi3%':>6} {'chi3p':>6} {'chi4%':>6} {'chi5%':>6} {'Portal':>8}")
print(f"  {'-'*58}")

portal_states = []
for i in range(n_analyse):
    if evals[i] <= 0 or evals[i] >= T_max * 0.9999:
        continue
    
    gap = -math.log(evals[i] / T_max)
    mass = gap * Lambda
    
    c4 = comps[i, chi4_idx]
    c5 = comps[i, chi5_idx]
    
    is_portal = c4 > 0.05 and c5 > 0.05
    
    if is_portal or i < 30:
        portal_mark = "PORTAL" if is_portal else ""
        print(f"  {i:>5} {mass:>10.1f} {comps[i,0]*100:>6.1f} {comps[i,1]*100:>6.1f} "
              f"{comps[i,2]*100:>6.1f} {comps[i,3]*100:>6.1f} {comps[i,4]*100:>6.1f} {portal_mark:>8}")
    
    if is_portal:
        portal_states.append({
            'state': i,
            'mass_MeV': round(mass, 1),
            'chi4_pct': round(c4 * 100, 1),
            'chi5_pct': round(c5 * 100, 1),
            'chi3_pct': round(comps[i, 1] * 100, 1),
            'mixing_amplitude': round(math.sqrt(c4 * c5), 6),
        })

print(f"\n  Total portal states (>5% in both chi4 and chi5): {len(portal_states)}")

# ================================================================
# MIXING AMPLITUDE ANALYSIS
# ================================================================
print(f"\n{'='*70}")
print("MIXING AMPLITUDES")
print(f"{'='*70}")

if portal_states:
    print(f"\n  Portal state details:")
    print(f"  {'State':>5} {'Mass':>8} {'chi4':>6} {'chi5':>6} {'sqrt(c4*c5)':>12} {'Interpretation':>20}")
    print(f"  {'-'*60}")
    
    for ps in portal_states[:20]:
        # Interpretation
        mass = ps['mass_MeV']
        interp = ""
        if abs(mass - 1604) < 50: interp = "pi1(1600)"
        elif abs(mass - 977) < 50: interp = "dark proton?"
        elif abs(mass - 1336) < 50: interp = "glueball/f0(1370)"
        elif abs(mass - 775) < 50: interp = "rho(775)"
        elif abs(mass - 1020) < 50: interp = "phi(1020)"
        
        print(f"  {ps['state']:>5} {mass:>8.1f} {ps['chi4_pct']:>5.1f}% {ps['chi5_pct']:>5.1f}% "
              f"{ps['mixing_amplitude']:>12.6f} {interp:>20}")
    
    # Overall mixing strength
    mix_amps = [ps['mixing_amplitude'] for ps in portal_states]
    print(f"\n  Average mixing amplitude: {np.mean(mix_amps):.6f}")
    print(f"  Max mixing amplitude:     {max(mix_amps):.6f}")
    print(f"  Min mixing amplitude:     {min(mix_amps):.6f}")

# ================================================================
# SECTOR PROJECTIONS
# ================================================================
print(f"\n{'='*70}")
print("SECTOR PROJECTIONS")  
print(f"{'='*70}")

# Project eigenstates onto pure-dark and pure-visible sectors
# Pure dark: configs with ALL edges in {chi1, chi3, chi3', chi4} (no chi5)
# Pure visible: configs with ALL edges in {chi1, chi3, chi3', chi5} (no chi4)

dark_cfgs = []
visible_cfgs = []
mixed_cfgs = []

for cfg in range(n):
    labels = config_to_labels(cfg, k)
    has_chi4 = chi4_idx in labels
    has_chi5 = chi5_idx in labels
    
    if has_chi4 and not has_chi5:
        dark_cfgs.append(cfg)
    elif has_chi5 and not has_chi4:
        visible_cfgs.append(cfg)
    elif has_chi4 and has_chi5:
        mixed_cfgs.append(cfg)
    # else: neither (pure chi1/chi3/chi3' only)

print(f"\n  Configuration counts:")
print(f"    Pure dark (has chi4, no chi5):     {len(dark_cfgs)}")
print(f"    Pure visible (has chi5, no chi4):  {len(visible_cfgs)}")
print(f"    Mixed (has both chi4 AND chi5):    {len(mixed_cfgs)}")
print(f"    Neither (chi1/chi3/chi3' only):    {n - len(dark_cfgs) - len(visible_cfgs) - len(mixed_cfgs)}")

# For each eigenstate: what fraction lives in the mixed configs?
print(f"\n  Mixed-config content of top eigenstates:")
print(f"  {'State':>5} {'Mass':>8} {'Dark%':>7} {'Visible%':>9} {'Mixed%':>7} {'Neither%':>9}")
print(f"  {'-'*50}")

for i in range(min(30, n_analyse)):
    if evals[i] <= 0 or evals[i] >= T_max * 0.9999:
        continue
    
    v2 = evecs[:, i] ** 2
    dark_frac = np.sum(v2[dark_cfgs])
    vis_frac = np.sum(v2[visible_cfgs])
    mix_frac = np.sum(v2[mixed_cfgs])
    neither_frac = 1 - dark_frac - vis_frac - mix_frac
    
    gap = -math.log(evals[i] / T_max)
    mass = gap * Lambda
    
    print(f"  {i:>5} {mass:>8.1f} {dark_frac*100:>7.1f} {vis_frac*100:>9.1f} "
          f"{mix_frac*100:>7.1f} {neither_frac*100:>9.1f}")

# ================================================================
# BULLET CLUSTER BOUND
# ================================================================
print(f"\n{'='*70}")
print("SELF-INTERACTION ESTIMATE")
print(f"{'='*70}")

# The dark matter self-interaction cross section
# sigma/m < 1 cm^2/g from the Bullet Cluster
# 
# The framework gives N(chi4,chi4,chi4) = 1 (self-coupling exists)
# The self-interaction amplitude ~ mixing_amplitude * alpha_dark
# where alpha_dark ~ N(chi4,chi4,chi4) * dim(chi4)^2 / |A5|

alpha_dark = 1.0 * 16 / 60  # N * dim^2 / |A5|
print(f"\n  Dark self-coupling: N(chi4,chi4,chi4) = 1")
print(f"  alpha_dark ~ N * dim(chi4)^2 / |A5| = 1 * 16/60 = {alpha_dark:.4f}")
print(f"  Compare to alpha_QCD ~ dim(chi5)^2 / |A5| = 25/60 = {25/60:.4f}")
print(f"  Dark coupling is {alpha_dark / (25/60):.1f}x weaker than QCD")
print(f"")
print(f"  The dark sector self-interaction is WEAKER than QCD.")
print(f"  With dark proton mass 977 MeV and alpha_dark = {alpha_dark:.3f},")
print(f"  the dark matter self-interaction cross-section is expected to be")
print(f"  suppressed relative to hadronic cross-sections by (alpha_dark/alpha_QCD)^2")
print(f"  = ({alpha_dark:.3f}/{25/60:.3f})^2 = {(alpha_dark/(25/60))**2:.3f}")

# ================================================================
# SAVE
# ================================================================
output = {
    'n_matrix': n,
    'n_positive_evals': int(n_pos),
    'T_max': float(T_max),
    'n_portal_states': len(portal_states),
    'portal_states': portal_states,
    'config_counts': {
        'pure_dark': len(dark_cfgs),
        'pure_visible': len(visible_cfgs),
        'mixed': len(mixed_cfgs),
        'neither': n - len(dark_cfgs) - len(visible_cfgs) - len(mixed_cfgs),
    },
    'alpha_dark': alpha_dark,
}

with open('dark_visible_portal_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to dark_visible_portal_results.json")
print("Done.")
