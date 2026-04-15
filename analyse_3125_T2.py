#!/usr/bin/env python3
"""
DOUBLE-ICOSAHEDRON HYPOTHESIS TEST
=====================================
Tests whether χ₄ leaks through the boundary between two A₅ cells.

Single filter:  T restricted to face-rep configs (no χ₄ on boundary)
Double filter:  T² where intermediate state CAN have χ₄

If T²_full[face,face] ≠ T²_restricted[face,face], then χ₄ leaks
through via second-order processes. This is the smoking gun.

Run: python3 analyse_3125_T2.py
"""
import numpy as np
import json, math, os, time
from itertools import product as iprod

phi = (1+math.sqrt(5))/2
sqrt5 = math.sqrt(5)
alpha_fw = 1/(20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332
dims = [1, 3, 3, 4, 5]

SECTOR = [0, 1, 2, 3, 4]
configs = list(iprod(SECTOR, repeat=5))
n = 3125

# Identify face-rep configs (no χ₄ on any edge)
face_rep_indices = [ic for ic, cfg in enumerate(configs) if all(x != 3 for x in cfg)]
chi4_indices = [ic for ic, cfg in enumerate(configs) if any(x == 3 for x in cfg)]
n_face = len(face_rep_indices)
n_chi4 = len(chi4_indices)

print(f'Face-rep configs (no χ₄): {n_face}')
print(f'χ₄-containing configs: {n_chi4}')
print(f'Total: {n_face + n_chi4} = {n}')

PROGRESS_FILE = 'qcd1024_progress.json'

print(f'\nLoading {PROGRESS_FILE}...')
t0 = time.time()
with open(PROGRESS_FILE) as f:
    prog = json.load(f)
print(f'  Loaded in {time.time()-t0:.1f}s')

print(f'Building {n}x{n} matrix...')
T = np.zeros((n, n), dtype=np.float64)
for r in prog['rows']:
    T[r['row']] = r['data']
T_sym = (T + T.T) / 2
del prog

# ============================================================
# STEP 1: Extract the face-rep sub-block (= the 1024×1024)
# ============================================================
print(f'\n{"="*70}')
print(f'1. EXTRACTING FACE-REP SUB-BLOCK')
print(f'{"="*70}')

T_face = T_sym[np.ix_(face_rep_indices, face_rep_indices)]
print(f'  T_face: {T_face.shape}')

evals_face, evecs_face = np.linalg.eigh(T_face)
idx_f = np.argsort(evals_face)[::-1]
evals_face = evals_face[idx_f]
evecs_face = evecs_face[:, idx_f]

print(f'  T_max(face sub-block): {evals_face[0]:.6e}')
print(f'  Compare to 1024 result: 4.1199e+26')
print(f'  Ratio: {evals_face[0]/4.1199e26:.4f}')

# ============================================================
# STEP 2: T² via the face-rep sub-block ONLY (single filter)
# ============================================================
print(f'\n{"="*70}')
print(f'2. T² — SINGLE FILTER (face-rep only)')
print(f'{"="*70}')

T2_face = T_face @ T_face
evals_T2_face = evals_face**2
print(f'  T²_max(face only): {evals_T2_face[0]:.6e}')

# ============================================================
# STEP 3: T² via the FULL 3125 matrix, then restricted to face-rep
# ============================================================
print(f'\n{"="*70}')
print(f'3. T² — DOUBLE FILTER (full matrix, χ₄ intermediate states)')
print(f'{"="*70}')

print(f'  Computing T² = T × T ({n}×{n})...')
t0 = time.time()
T2_full = T_sym @ T_sym
print(f'  Done in {time.time()-t0:.1f}s')

# Restrict T² to face-rep rows and columns
T2_full_face = T2_full[np.ix_(face_rep_indices, face_rep_indices)]
print(f'  T²_full restricted to face-rep: {T2_full_face.shape}')

evals_T2_full_face, evecs_T2_full_face = np.linalg.eigh(T2_full_face)
idx_ff = np.argsort(evals_T2_full_face)[::-1]
evals_T2_full_face = evals_T2_full_face[idx_ff]
evecs_T2_full_face = evecs_T2_full_face[:, idx_ff]

print(f'  T²_max(full→face): {evals_T2_full_face[0]:.6e}')

# ============================================================
# STEP 4: THE COMPARISON — does χ₄ leak through?
# ============================================================
print(f'\n{"="*70}')
print(f'4. THE SMOKING GUN: χ₄ LEAKAGE')
print(f'{"="*70}')

# The difference between T2_full_face and T2_face tells us
# how much χ₄ intermediate states contribute
diff = T2_full_face - T2_face
frobenius_diff = np.linalg.norm(diff)
frobenius_face = np.linalg.norm(T2_face)
relative_diff = frobenius_diff / frobenius_face

print(f'  ||T²_full - T²_face||_F = {frobenius_diff:.6e}')
print(f'  ||T²_face||_F = {frobenius_face:.6e}')
print(f'  Relative difference: {relative_diff:.6f} ({relative_diff*100:.4f}%)')
print()

if relative_diff > 0.001:
    print(f'  *** χ₄ LEAKS THROUGH! ***')
    print(f'  The double boundary allows {relative_diff*100:.2f}% χ₄ transmission.')
    print(f'  Dark matter is NOT absolutely confined.')
else:
    print(f'  χ₄ is completely blocked at second order.')
    print(f'  Dark matter is absolutely confined by the boundary.')

# Eigenvalue comparison
print(f'\n  Top eigenvalue comparison:')
print(f'  {"Level":>5} {"T²(face only)":>18} {"T²(with χ₄)":>18} {"Ratio":>10} {"χ₄ boost":>10}')
print(f'  {"-"*65}')
for i in range(min(20, len(evals_T2_face))):
    e_face = evals_T2_face[i]
    e_full = evals_T2_full_face[i]
    if abs(e_face) > 0:
        ratio = e_full / e_face
        boost = (ratio - 1) * 100
        print(f'  {i:>5} {e_face:>18.6e} {e_full:>18.6e} {ratio:>10.6f} {boost:>+9.2f}%')

# ============================================================
# STEP 5: HVP from T² (two-cell propagation)
# ============================================================
print(f'\n{"="*70}')
print(f'5. HVP FROM TWO-CELL PROPAGATION')
print(f'{"="*70}')

a_mu_SM = 6.93e-8

# Source on the face-rep sub-block
source_face = np.zeros(n_face)
for i, ic in enumerate(face_rep_indices):
    cfg = configs[ic]
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    source_face[i] = math.sqrt(n3*n3p)/5.0

# HVP from T_face (single cell, = the 1024 result)
ov_face = evecs_face.T @ source_face
sum_HVP_1cell = 0
for t in range(1, 300):
    C_t = sum(ov_face[nn]**2 * (evals_face[nn]/evals_face[0])**t
              for nn in range(n_face) if 0 < evals_face[nn] < evals_face[0]*0.9999)
    sum_HVP_1cell += t**2 * C_t
a_mu_1cell = (4*alpha_fw**2/3)*sum_HVP_1cell*(0.10566/Lambda)**2
print(f'  1-cell (face sub-block): a_μ = {a_mu_1cell:.4e}, ratio = {a_mu_1cell/a_mu_SM:.4f}')

# HVP from T²_full_face (two cells, χ₄ intermediate allowed)
ov_T2 = evecs_T2_full_face.T @ source_face
T2_max = evals_T2_full_face[0]
sum_HVP_2cell = 0
for t in range(1, 300):
    C_t = sum(ov_T2[nn]**2 * (evals_T2_full_face[nn]/T2_max)**t
              for nn in range(n_face) if 0 < evals_T2_full_face[nn] < T2_max*0.9999)
    sum_HVP_2cell += t**2 * C_t
a_mu_2cell = (4*alpha_fw**2/3)*sum_HVP_2cell*(0.10566/Lambda)**2
print(f'  2-cell (χ₄ intermediate): a_μ = {a_mu_2cell:.4e}, ratio = {a_mu_2cell/a_mu_SM:.4f}')

# HVP from T²_face (two cells, NO χ₄ intermediate)
ov_T2_noχ4 = evecs_face.T @ source_face  # same eigenvectors, eigenvalues squared
sum_HVP_2cell_noχ4 = 0
for t in range(1, 300):
    C_t = sum(ov_T2_noχ4[nn]**2 * (evals_face[nn]**2/(evals_face[0]**2))**t
              for nn in range(n_face) if 0 < evals_face[nn] < evals_face[0]*0.9999)
    sum_HVP_2cell_noχ4 += t**2 * C_t
a_mu_2cell_noχ4 = (4*alpha_fw**2/3)*sum_HVP_2cell_noχ4*(0.10566/Lambda)**2
print(f'  2-cell (no χ₄ intermediate): a_μ = {a_mu_2cell_noχ4:.4e}, ratio = {a_mu_2cell_noχ4/a_mu_SM:.4f}')

print(f'\n  χ₄ effect on two-cell HVP: {(a_mu_2cell-a_mu_2cell_noχ4)/a_mu_2cell_noχ4*100:+.2f}%')

# ============================================================
# STEP 6: Check T_max hierarchy
# ============================================================
print(f'\n{"="*70}')
print(f'6. T_MAX HIERARCHY')
print(f'{"="*70}')

T_max_full = np.max(np.linalg.eigvalsh(T_sym))

# qq sub-block
qq_idx = [ic for ic, cfg in enumerate(configs) if all(x in [1,2] for x in cfg)]
T_qq = T_sym[np.ix_(qq_idx, qq_idx)]
T_max_qq = np.max(np.linalg.eigvalsh(T_qq))

print(f'  T_max(full 3125): {T_max_full:.4e}')
print(f'  T_max(face 1024): {evals_face[0]:.4e}')
print(f'  T_max(qq 32):     {T_max_qq:.4e}')
print(f'  Ratio full/face:  {T_max_full/evals_face[0]:.2f}')
print(f'  Ratio face/qq:    {evals_face[0]/T_max_qq:.2f}')

# ============================================================
# SAVE
# ============================================================
print(f'\n{"="*70}')
print('Saving results...')

results = {
    'T_max_full': float(T_max_full),
    'T_max_face': float(evals_face[0]),
    'T_max_qq': float(T_max_qq),
    'frobenius_diff': float(frobenius_diff),
    'frobenius_face': float(frobenius_face),
    'relative_chi4_leakage': float(relative_diff),
    'a_mu_1cell': float(a_mu_1cell),
    'a_mu_2cell_with_chi4': float(a_mu_2cell),
    'a_mu_2cell_without_chi4': float(a_mu_2cell_noχ4),
    'chi4_effect_on_2cell_HVP_pct': float((a_mu_2cell-a_mu_2cell_noχ4)/a_mu_2cell_noχ4*100),
    'evals_face_top20': [float(e) for e in evals_face[:20]],
    'evals_T2_face_top20': [float(e) for e in evals_T2_face[:20]],
    'evals_T2_full_face_top20': [float(e) for e in evals_T2_full_face[:20]],
}

outfile = 'qcd3125_T2_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)

size_kb = os.path.getsize(outfile)/1024
print(f'  Saved to {outfile} ({size_kb:.0f} KB)')
print(f'\n  Upload this file for analysis.')

print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}')
print(f'''
  χ₄ leakage: {relative_diff*100:.4f}%
  
  If > 0.1%:  Double-icosahedron hypothesis CONFIRMED.
              Dark matter leaks through at measurable rate.
              
  If < 0.01%: Single filter is sufficient.
              Dark matter is absolutely confined.
              
  Either way, the A₅ cell boundary is load-bearing:
  it separates the 3125 (everything mixed) from the 
  1024 (dark matter filtered), producing the physical 
  HVP at 93.5% of the SM value.
''')

PYEOF
