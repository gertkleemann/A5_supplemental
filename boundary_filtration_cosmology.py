#!/usr/bin/env python3
"""
BOUNDARY FILTRATION: WHAT DOES THE OBSERVER SEE?
=================================================
Projects the 3125 vacuum eigenvector onto the face representation
(configs with no χ₄ edges) to test the cosmology mapping.

Usage: python boundary_filtration_cosmology.py [path_to_qcd3125_progress_seeded.json]
"""
import numpy as np
import json, sys, os, math

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

fname = sys.argv[1] if len(sys.argv) > 1 else 'qcd3125_progress_seeded.json'

print(f"Loading {fname}...")
with open(fname) as f:
    data = json.load(f)

rows = data['rows']
n = max(r['row'] for r in rows) + 1
T = np.zeros((n, n), dtype=np.float64)
for r in rows:
    T[r['row'], :len(r['data'])] = r['data']
T = (T + T.T) / 2

print("Diagonalizing...", flush=True)
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

psi0 = evecs[:, 0]  # vacuum eigenvector
T_max = evals[0]
print(f"T_max = {T_max:.6e}")

k = 5
n_edges = 5
irr_names = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']
dims = [1, 3, 3, 4, 5]

# ================================================================
# 1. FULL VACUUM COMPOSITION
# ================================================================
print(f"\n{'='*65}")
print(f"1. FULL VACUUM COMPOSITION (all 3125 configs)")
print(f"{'='*65}")

full_fracs = {}
for r in range(k):
    weight = 0.0
    for c in range(n):
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        count = labels.count(r)
        weight += psi0[c]**2 * count / n_edges
    full_fracs[r] = weight
    print(f"  {irr_names[r]:>4}: {weight*100:.4f}%")

# ================================================================
# 2. FACE-REP PROJECTION (no χ₄ on any edge)
# ================================================================
print(f"\n{'='*65}")
print(f"2. FACE-REP PROJECTION (configs with NO χ₄ edges)")
print(f"{'='*65}")

face_indices = []
for c in range(n):
    labels = []
    tmp = c
    for i in range(n_edges):
        labels.append(tmp % k)
        tmp //= k
    if 3 not in labels:  # no χ₄ (index 3)
        face_indices.append(c)

print(f"  Face-rep configs: {len(face_indices)} (expected: 4^5 = {4**5})")

face_weight = sum(psi0[c]**2 for c in face_indices)
print(f"  Vacuum probability in face rep: {face_weight*100:.6f}%")

# Composition within face rep (normalized)
face_fracs = {}
for r in [0, 1, 2, 4]:  # no χ₄
    weight = 0.0
    for c in face_indices:
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        count = labels.count(r)
        weight += psi0[c]**2 * count / n_edges
    face_fracs[r] = weight / face_weight if face_weight > 0 else 0
    print(f"  {irr_names[r]:>4}: {face_fracs[r]*100:.4f}% (of face rep)")

# ================================================================
# 3. CONFIGS BY χ₄ COUNT (0, 1, 2, 3, 4, 5 dark edges)
# ================================================================
print(f"\n{'='*65}")
print(f"3. PROBABILITY BY NUMBER OF DARK (χ₄) EDGES")
print(f"{'='*65}")

by_n_dark = {i: 0.0 for i in range(6)}
count_by_n_dark = {i: 0 for i in range(6)}
for c in range(n):
    labels = []
    tmp = c
    for i in range(n_edges):
        labels.append(tmp % k)
        tmp //= k
    n_chi4 = labels.count(3)
    by_n_dark[n_chi4] += psi0[c]**2
    count_by_n_dark[n_chi4] += 1

print(f"  {'# dark edges':>12} {'# configs':>10} {'Probability':>12} {'Per config':>12}")
for nd in range(6):
    per = by_n_dark[nd] / count_by_n_dark[nd] if count_by_n_dark[nd] > 0 else 0
    print(f"  {nd:>12} {count_by_n_dark[nd]:>10} {by_n_dark[nd]*100:>11.6f}% {per:>12.4e}")

# ================================================================
# 4. PORTAL ANALYSIS
# ================================================================
print(f"\n{'='*65}")
print(f"4. PORTAL ANALYSIS")
print(f"{'='*65}")

print(f"  Face rep (0 dark edges):  {by_n_dark[0]*100:.4f}%")
print(f"  Portal (1-4 dark edges):  {sum(by_n_dark[i] for i in range(1,5))*100:.4f}%")
print(f"  Pure dark (5 dark edges): {by_n_dark[5]*100:.6f}%")

# ================================================================
# 5. THE COSMOLOGY MAPPING
# ================================================================
print(f"\n{'='*65}")
print(f"5. THE COSMOLOGY MAPPING")
print(f"{'='*65}")

chi1 = full_fracs[0]
chi3 = full_fracs[1]
chi3p = full_fracs[2]
chi4 = full_fracs[3]
chi5 = full_fracs[4]
matter = chi3 + chi3p
total = chi1 + matter + chi4 + chi5

print(f"\n  Framework vacuum composition:")
print(f"    χ₁ (vacuum):    {chi1*100:.2f}%")
print(f"    χ₃+χ₃' (matter): {matter*100:.2f}%")
print(f"    χ₄ (dark):      {chi4*100:.2f}%")
print(f"    χ₅ (force):     {chi5*100:.2f}%")

print(f"\n  Planck 2018:")
print(f"    Dark energy:    68.3%")
print(f"    Dark matter:    26.8%")
print(f"    Visible matter:  4.9%")

print(f"\n  Mapping 1: χ₄ → DM (matches to {abs(chi4*100-26.8)/26.8*100:.1f}%)")
print(f"  Mapping 2: χ₅ → DE? ({chi5*100:.1f}% vs 68.3% — off by {abs(chi5*100-68.3)/68.3*100:.1f}%)")
print(f"  Mapping 3: χ₃+χ₃' → visible? ({matter*100:.1f}% vs 4.9% — off)")

print(f"\n  FACE-REP FRACTIONS (what observer projects out):")
print(f"    Prob in face rep: {face_weight*100:.4f}%")
print(f"    This is the 'visible universe' fraction of vacuum energy")

# The key: what fraction of TOTAL energy is visible vs dark vs DE?
# Gravitational: observer infers χ₄ from missing mass
# Visible: observer sees face-rep content
# Dark energy: the rest?

print(f"\n  GRAVITATIONAL ACCOUNTING:")
print(f"    Observable (face rep): {face_weight*100:.4f}%")
print(f"    Dark (non-face-rep):   {(1-face_weight)*100:.4f}%")
print(f"    Ratio dark/observable: {(1-face_weight)/face_weight:.4f}")

# ================================================================
# 6. SAVE
# ================================================================
results = {
    'full_vacuum': {irr_names[r]: round(full_fracs[r]*100, 4) for r in range(k)},
    'face_rep_weight_pct': round(face_weight*100, 6),
    'face_rep_composition': {irr_names[r]: round(face_fracs[r]*100, 4) for r in [0,1,2,4]},
    'prob_by_n_dark_edges': {str(nd): round(by_n_dark[nd]*100, 6) for nd in range(6)},
    'matter_fraction': round(matter*100, 4),
    'dark_fraction': round(chi4*100, 4),
    'force_fraction': round(chi5*100, 4),
    'planck_DM': 26.8,
    'planck_DE': 68.3,
    'planck_visible': 4.9,
    'chi4_vs_planck_DM_pct': round(abs(chi4*100-26.8)/26.8*100, 2),
}

outfile = 'boundary_filtration_cosmology_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")
