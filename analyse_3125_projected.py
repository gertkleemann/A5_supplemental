#!/usr/bin/env python3
"""
CHI4-PROJECTED SOURCE ANALYSIS
================================
Recomputes HVP from the 3125×3125 matrix with the physical
EM source that excludes χ₄ edges (because χ₄(C₂) = 0).

Also tests: what if we use the 3125 MATRIX but 1024-like SOURCE?
And: Λ recalibration from the full spectrum.

Run: python3 analyse_3125_projected.py
"""
import numpy as np
import json, math, os, time
from itertools import product as iprod

phi = (1+math.sqrt(5))/2
sqrt5 = math.sqrt(5)
alpha_fw = 1/(20*phi**4 - (3+5*sqrt5)/308)
dims = [1, 3, 3, 4, 5]

SECTOR = [0, 1, 2, 3, 4]
configs = list(iprod(SECTOR, repeat=5))
n = 3125

PROGRESS_FILE = 'qcd1024_progress.json'

print(f'Loading {PROGRESS_FILE}...')
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

print(f'Computing eigendecomposition...')
t0 = time.time()
evals_raw, evecs_raw = np.linalg.eigh(T_sym)
idx = np.argsort(evals_raw)[::-1]
evals = evals_raw[idx]
evecs = evecs_raw[:, idx]
T_max = evals[0]
print(f'  Done in {time.time()-t0:.1f}s, T_max = {T_max:.6e}')

# ============================================================
# DEFINE ALL SOURCE VARIANTS
# ============================================================

# Source A: ORIGINAL (unprojected) — same as before
source_orig = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    source_orig[ic] = math.sqrt(n3*n3p)/5.0

# Source B: χ₄-PROJECTED — exclude configs with ANY χ₄ edge
source_no_chi4 = np.zeros(n)
for ic, cfg in enumerate(configs):
    if any(x == 3 for x in cfg):
        continue  # skip configs with χ₄
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    source_no_chi4[ic] = math.sqrt(n3*n3p)/5.0

# Source C: WEIGHTED projection — reduce χ₄ contribution by χ₄(C₂)=0
# This means: weight each config by the fraction of non-χ₄ edges
source_weighted = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    n4 = sum(1 for x in cfg if x==3)
    # Weight by fraction of edges that are EM-visible
    em_visible = (5 - n4) / 5.0
    source_weighted[ic] = math.sqrt(n3*n3p)/5.0 * em_visible

# Source D: BARYON (for comparison)
source_baryon = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    if n3 >= 3:
        source_baryon[ic] = n3*(n3-1)*(n3-2)/6.0

# Source E: BARYON χ₄-projected
source_baryon_no_chi4 = np.zeros(n)
for ic, cfg in enumerate(configs):
    if any(x == 3 for x in cfg):
        continue
    n3 = sum(1 for x in cfg if x==1)
    if n3 >= 3:
        source_baryon_no_chi4[ic] = n3*(n3-1)*(n3-2)/6.0

sources = {
    'A_original': source_orig,
    'B_no_chi4': source_no_chi4,
    'C_weighted': source_weighted,
    'D_baryon': source_baryon,
    'E_baryon_no_chi4': source_baryon_no_chi4,
}

print(f'\nSource vector statistics:')
for name, src in sources.items():
    n_nonzero = int(np.sum(np.abs(src) > 0))
    print(f'  {name:>20}: {n_nonzero:>5} nonzero configs, norm={np.linalg.norm(src):.3f}')

# ============================================================
# COMPUTE HVP FOR EACH SOURCE
# ============================================================
print(f'\n{"="*70}')
print(f'HVP WITH DIFFERENT SOURCE OPERATORS')
print(f'{"="*70}')

a_mu_SM = 6.93e-8

for name, src in sources.items():
    if 'baryon' in name.lower():
        continue  # skip baryons for HVP
    
    ov = evecs.T @ src
    
    sum_HVP = 0
    for t in range(1, 300):
        C_t = 0
        for nn in range(n):
            ev = evals[nn]
            if ev <= 0: continue
            ratio = ev/T_max
            if 0 < ratio < 1:
                C_t += ov[nn]**2 * ratio**t
        sum_HVP += t**2 * C_t
    
    a_mu = (4*alpha_fw**2/3)*sum_HVP*(0.10566/0.332)**2
    print(f'  {name:>20}: a_μ = {a_mu:.4e}, ratio to SM = {a_mu/a_mu_SM:.4f} ({a_mu/a_mu_SM*100:.1f}%)')

# ============================================================
# Λ RECALIBRATION
# ============================================================
print(f'\n{"="*70}')
print(f'Λ RECALIBRATION')
print(f'{"="*70}')

# The qq sub-block: configs with only χ₃ and χ₃'
qq_idx = [ic for ic, cfg in enumerate(configs) if all(x in [1,2] for x in cfg)]
T_qq = T_sym[np.ix_(qq_idx, qq_idx)]
evals_qq = sorted(np.linalg.eigvalsh(T_qq), reverse=True)
T_max_qq = evals_qq[0]

print(f'  T_max(full 3125): {T_max:.4e}')
print(f'  T_max(qq 32):     {T_max_qq:.4e}')
print(f'  Ratio: {T_max/T_max_qq:.1f}')

# The qq sub-block should give the SAME masses as the 32×32
# because it IS the same sub-block
print(f'\n  qq sub-block masses (should match 32×32):')
known = [(135,'π'),(494,'K'),(548,'η'),(775,'ρ'),(782,'ω'),(958,"η'")]
Lambda_qq = 332
for i, ev in enumerate(evals_qq[:6]):
    if ev > 0 and ev < T_max_qq*0.9999:
        gap = -math.log(ev/T_max_qq)
        mass = gap * Lambda_qq
        best = min(known, key=lambda h: abs(h[0]-mass))
        err = abs(best[0]-mass)/best[0]*100
        print(f'    {mass:.0f} MeV -> {best[1]} ({best[0]}) {err:.1f}%')

# What Λ would put the ρ at 775?
# From the FULL matrix, find the state that best matches the ρ
# The ρ should be the lightest state with large qq overlap
ov_nochi4 = evecs.T @ source_no_chi4
print(f'\n  Finding ρ in the full 3125 spectrum:')
for i in range(20):
    ev = evals[i]
    if ev <= 0 or ev >= T_max*0.9999: continue
    o = abs(ov_nochi4[i])
    if o > 0.1:
        gap = -math.log(ev/T_max)
        Lambda_rho = 775.0 / gap
        mass_332 = gap * 332
        print(f'    ψ_{i}: gap={gap:.4f}, mass@332={mass_332:.0f}, '
              f'|ov|={o:.3f}, Λ_for_ρ={Lambda_rho:.0f}')

# ============================================================
# BARYON SPECTRUM WITH PROJECTED SOURCE
# ============================================================
print(f'\n{"="*70}')
print(f'BARYON SPECTRUM: PROJECTED vs UNPROJECTED')
print(f'{"="*70}')

for name in ['D_baryon', 'E_baryon_no_chi4']:
    src = sources[name]
    ov = evecs.T @ src
    
    states = []
    for nn in range(200):
        ev = evals[nn]
        if ev <= 0 or ev >= T_max*0.9999: continue
        o = abs(ov[nn])
        if o > 0.01:
            gap = -math.log(ev/T_max)
            states.append((gap*332, o, nn))
    states.sort()
    
    print(f'\n  {name}:')
    for mass, o, nn in states[:8]:
        print(f'    {mass:7.0f} MeV  |ov|={o:.4f}')
    
    if len(states) >= 2:
        split = states[1][0] - states[0][0]
        print(f'  N-Δ splitting: {split:.0f} MeV (observed: 294)')

# ============================================================
# HVP WITH DIFFERENT Λ VALUES
# ============================================================
print(f'\n{"="*70}')
print(f'HVP SENSITIVITY TO Λ (using χ₄-projected source)')
print(f'{"="*70}')

ov_proj = evecs.T @ source_no_chi4

for Lambda_test in [180, 200, 220, 250, 280, 300, 332, 360, 400]:
    sum_HVP = 0
    for t in range(1, 300):
        C_t = 0
        for nn in range(n):
            ev = evals[nn]
            if ev <= 0: continue
            ratio = ev/T_max
            if 0 < ratio < 1:
                C_t += ov_proj[nn]**2 * ratio**t
        sum_HVP += t**2 * C_t
    
    a_mu = (4*alpha_fw**2/3)*sum_HVP*(0.10566/Lambda_test)**2
    ratio = a_mu/a_mu_SM
    mark = ' <-- best' if abs(ratio-1) < 0.1 else ''
    print(f'  Λ={Lambda_test:>3}: a_μ = {a_mu:.4e}, ratio = {ratio:.4f}{mark}')

# ============================================================
# DARK MATTER CONTENT OF VACUUM
# ============================================================
print(f'\n{"="*70}')
print(f'VACUUM COMPOSITION (full 3125)')
print(f'{"="*70}')

psi0 = evecs[:, 0]
for n_irrep, irrep_name in enumerate(['χ₁','χ₃','χ₃\'','χ₄','χ₅']):
    avg = sum(psi0[ic]**2 * sum(1 for x in cfg if x==n_irrep)
              for ic, cfg in enumerate(configs))
    frac = avg/5.0
    print(f'  <n_{irrep_name}>/5 = {frac:.4f} ({frac*100:.1f}% of edges)')

# ============================================================
# SAVE COMPACT RESULTS
# ============================================================
print(f'\n{"="*70}')
print(f'Saving results...')

results = {}

# Recompute all HVP variants
for name, src in sources.items():
    if 'baryon' in name.lower():
        continue
    ov = evecs.T @ src
    sum_HVP = 0
    for t in range(1, 300):
        C_t = sum(ov[nn]**2 * (evals[nn]/T_max)**t 
                  for nn in range(n) if 0 < evals[nn] < T_max*0.9999)
        sum_HVP += t**2 * C_t
    a_mu = (4*alpha_fw**2/3)*sum_HVP*(0.10566/0.332)**2
    results[f'a_mu_{name}'] = float(a_mu)
    results[f'ratio_{name}'] = float(a_mu/a_mu_SM)

# Source overlaps for projected source
results['source_overlaps_no_chi4'] = [float(x) for x in (evecs.T @ source_no_chi4)[:200]]

outfile = 'qcd3125_projected_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)

size_kb = os.path.getsize(outfile)/1024
print(f'  Saved to {outfile} ({size_kb:.0f} KB)')
print(f'\n  Upload this file for further analysis.')
