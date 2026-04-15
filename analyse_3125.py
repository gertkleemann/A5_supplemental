#!/usr/bin/env python3
"""
ANALYSE THE 3125×3125 TRANSFER MATRIX
======================================
Run this AFTER qcd1024_numba.py --sector 0,1,2,3,4 completes.
Produces a compact results file (~1MB) with everything needed.

Run: python3 analyse_3125.py
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

PROGRESS_FILE = 'qcd1024_progress.json'

print(f'Loading {PROGRESS_FILE}...')
t0 = time.time()
with open(PROGRESS_FILE) as f:
    prog = json.load(f)
print(f'  Loaded in {time.time()-t0:.1f}s')

print(f'Building {n}x{n} matrix...')
T = np.zeros((n, n), dtype=np.float64)
loaded = 0
for r in prog['rows']:
    T[r['row']] = r['data']
    loaded += 1
print(f'  {loaded} rows loaded')

if loaded < n:
    print(f'  WARNING: only {loaded}/{n} rows! Matrix is incomplete.')
    print(f'  Proceeding with partial analysis...')

# Symmetrise
T_sym = (T + T.T) / 2

# Free memory
del prog

print(f'Computing eigendecomposition ({n}x{n})...')
t0 = time.time()
evals_raw, evecs_raw = np.linalg.eigh(T_sym)
idx = np.argsort(evals_raw)[::-1]
evals = evals_raw[idx]
evecs = evecs_raw[:, idx]
T_max = evals[0]
print(f'  Done in {time.time()-t0:.1f}s')
print(f'  T_max = {T_max:.6e}')
print(f'  Positive eigenvalues: {sum(1 for e in evals if e > 0)}')

# ============================================================
# 1. MASS SPECTRUM
# ============================================================
print('\n1. MASS SPECTRUM')
masses = []
for i, ev in enumerate(evals[:500]):
    if ev > 0 and ev < T_max * 0.9999:
        gap = -math.log(ev/T_max)
        masses.append({'level': int(i), 'eigenvalue': float(ev),
                       'gap': float(gap), 'mass_MeV': float(gap*Lambda)})

known = [(135,'pi'),(494,'K'),(548,'eta'),(600,'sigma'),(775,'rho'),(782,'omega'),
         (958,"eta'"),(1020,'phi'),(1232,'Delta'),(1275,'f2'),(1525,"f2'"),(1680,'rho3')]

for m in masses[:20]:
    best = min(known, key=lambda h: abs(h[0]-m['mass_MeV']))
    err = abs(best[0]-m['mass_MeV'])/best[0]*100
    mark = '***' if err<3 else '**' if err<5 else '*' if err<10 else ''
    print(f'  {m["mass_MeV"]:8.0f} MeV -> {best[1]:>8} ({best[0]}) {err:5.1f}% {mark}')

# ============================================================
# 2. SOURCE VECTORS AND OVERLAPS
# ============================================================
print('\n2. SOURCE VECTORS')

# Meson source: sqrt(n3*n3')/5
source_meson = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    source_meson[ic] = math.sqrt(n3*n3p)/5.0

# Baryon source: C(n3,3)
source_baryon = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    if n3 >= 3:
        source_baryon[ic] = n3*(n3-1)*(n3-2)/6.0

# Dark matter source: C(n4,3)
source_dark = np.zeros(n)
for ic, cfg in enumerate(configs):
    n4 = sum(1 for x in cfg if x==3)
    if n4 >= 3:
        source_dark[ic] = n4*(n4-1)*(n4-2)/6.0

# Glueball source: C(n5,2)
source_glueball = np.zeros(n)
for ic, cfg in enumerate(configs):
    n5 = sum(1 for x in cfg if x==4)
    if n5 >= 2:
        source_glueball[ic] = n5*(n5-1)/2.0

# Axial current (for f_pi)
source_axial = np.zeros(n)
for ic, cfg in enumerate(configs):
    n3 = sum(1 for x in cfg if x==1)
    n3p = sum(1 for x in cfg if x==2)
    source_axial[ic] = (n3-n3p)/5.0

sources = {
    'meson': source_meson,
    'baryon': source_baryon,
    'dark_baryon': source_dark,
    'glueball': source_glueball,
    'axial': source_axial,
}

# Compute overlaps with eigenstates
overlaps = {}
for name, src in sources.items():
    ov = evecs.T @ src
    overlaps[name] = ov
    print(f'  {name}: norm={np.linalg.norm(src):.3f}, '
          f'top overlap={max(abs(ov)):.4f}')

# ============================================================
# 3. HADRONIC VP
# ============================================================
print('\n3. HADRONIC VP')

ov_meson = overlaps['meson']
sum_HVP = 0
for t in range(1, 300):
    C_t = 0
    for nn in range(n):
        ev = evals[nn]
        if ev <= 0: continue
        ratio = ev/T_max
        if 0 < ratio < 1:
            C_t += ov_meson[nn]**2 * ratio**t
    sum_HVP += t**2 * C_t

a_e_HVP = (4*alpha_fw**2/3)*sum_HVP*(0.000511/0.332)**2
a_mu_HVP = (4*alpha_fw**2/3)*sum_HVP*(0.10566/0.332)**2

print(f'  a_e(HVP) = {a_e_HVP:.6e} (SM: 1.67e-12)')
print(f'  a_mu(HVP) = {a_mu_HVP:.6e} (SM: 6.93e-8)')
print(f'  Ratio to SM: {a_mu_HVP/6.93e-8:.4f}')
print(f'  THIS IS THE EXACT SINGLE-CELL ANSWER (no truncation)')

# ============================================================
# 4. VACUUM STRUCTURE
# ============================================================
print('\n4. VACUUM STRUCTURE')
psi0 = evecs[:, 0]

for n_irrep, irrep_name in enumerate(['chi1','chi3','chi3p','chi4','chi5']):
    avg = sum(psi0[ic]**2 * sum(1 for x in cfg if x==n_irrep)
              for ic, cfg in enumerate(configs))
    print(f'  <n_{irrep_name}> = {avg:.4f} (out of 5)')

chiral = sum(psi0[ic]**2 * (sum(1 for x in cfg if x==1) - sum(1 for x in cfg if x==2))
             for ic, cfg in enumerate(configs))
qq_cond = sum(psi0[ic]**2 * sum(1 for x in cfg if x==1) * sum(1 for x in cfg if x==2)
              for ic, cfg in enumerate(configs))
print(f'  Chiral order parameter: {chiral:.8f}')
print(f'  Quark condensate <n3*n3p>: {qq_cond:.4f}')

# ============================================================
# 5. SECTOR DECOMPOSITION OF TOP EIGENSTATES
# ============================================================
print('\n5. EIGENVECTOR SECTOR CONTENT')

sector_content = []
for i in range(min(50, n)):
    psi = evecs[:, i]
    content = {}
    for irrep_idx, irrep_name in enumerate(['chi1','chi3','chi3p','chi4','chi5']):
        w = sum(psi[ic]**2 for ic, cfg in enumerate(configs)
                if sum(1 for x in cfg if x==irrep_idx) >= 3)
        content[irrep_name] = float(w)
    sector_content.append(content)
    if i < 20:
        gap = -math.log(evals[i]/T_max) if 0 < evals[i] < T_max*0.9999 else 0
        mass = gap*Lambda
        print(f'  psi_{i}: {mass:7.0f} MeV  '
              f'q={content["chi3"]*100+content["chi3p"]*100:4.0f}% '
              f'g={content["chi5"]*100:4.0f}% '
              f'DM={content["chi4"]*100:4.0f}%')

# ============================================================
# 6. SOURCE SPECTRA (mass states for each particle type)
# ============================================================
print('\n6. PARTICLE SPECTRA FROM DIFFERENT SOURCES')

source_spectra = {}
for name, ov in overlaps.items():
    states = []
    for nn in range(min(200, n)):
        ev = evals[nn]
        if ev <= 0 or ev >= T_max*0.9999: continue
        o = float(abs(ov[nn]))
        if o > 0.01:
            gap = -math.log(ev/T_max)
            states.append({'level': int(nn), 'mass_MeV': float(gap*Lambda),
                           'overlap': o, 'overlap_sq': float(o**2)})
    source_spectra[name] = states[:30]
    
    print(f'\n  {name} spectrum:')
    for s in states[:8]:
        print(f'    {s["mass_MeV"]:7.0f} MeV  |ov|={s["overlap"]:.4f}')

# ============================================================
# 7. CORRELATOR DECAY (confinement check)
# ============================================================
print('\n7. CORRELATOR DECAY')
C_vals = []
for t in range(1, 20):
    C_t = 0
    for nn in range(n):
        ev = evals[nn]
        if ev <= 0: continue
        ratio = ev/T_max
        if 0 < ratio < 1:
            C_t += ov_meson[nn]**2 * ratio**t
    C_vals.append(float(C_t))
    if C_t > 0 and len(C_vals) > 1 and C_vals[-2] > 0:
        m_eff = -math.log(C_t/C_vals[-2]) * Lambda
        print(f'  C(t={t:2d}) = {C_t:.4e}, m_eff = {m_eff:.0f} MeV')
    else:
        print(f'  C(t={t:2d}) = {C_t:.4e}')

# ============================================================
# 8. T^2 CHECK (double icosahedron hypothesis)
# ============================================================
print('\n8. T^2: DOUBLE ICOSAHEDRON TEST')
# T^2 eigenvalues = evals^2 (since we have the eigendecomposition)
# The question: do chi4-dominated eigenstates appear with significant weight?
# Check: which eigenstates have >50% chi4 content?
print('  chi4-dominated eigenstates (>30% chi4 content):')
chi4_states = []
for i in range(min(200, n)):
    psi = evecs[:, i]
    w4 = sum(psi[ic]**2 for ic, cfg in enumerate(configs)
             if sum(1 for x in cfg if x==3) >= 2)
    if w4 > 0.3:
        gap = -math.log(evals[i]/T_max) if 0 < evals[i] < T_max*0.9999 else 0
        mass = gap*Lambda
        chi4_states.append({'level': int(i), 'mass_MeV': float(mass),
                            'chi4_content': float(w4),
                            'eigenvalue': float(evals[i])})
        print(f'    psi_{i}: {mass:7.0f} MeV, chi4={w4*100:.0f}%')

# ============================================================
# 9. f_pi
# ============================================================
print('\n9. PION DECAY CONSTANT')
ov_axial = overlaps['axial']
for i in range(min(20, n)):
    o = abs(ov_axial[i])
    if o > 0.05:
        f_raw = o * math.sqrt(3) * Lambda
        f_corr = f_raw / sqrt5
        gap = -math.log(evals[i]/T_max) if 0 < evals[i] < T_max*0.9999 else 0
        mass = gap*Lambda
        print(f'  psi_{i}: |ov|={o:.4f}, mass={mass:.0f}, '
              f'f_pi={f_raw:.0f}/sqrt5={f_corr:.1f} MeV (obs 92.3)')

# ============================================================
# 10. EIGENVALUE STATISTICS
# ============================================================
print('\n10. EIGENVALUE STATISTICS')
pos_evals = sorted([float(e) for e in evals if e > 0], reverse=True)
gaps = [pos_evals[i-1]-pos_evals[i] for i in range(1, min(300, len(pos_evals)))]
if gaps:
    mean_g = np.mean(gaps)
    std_g = np.std(gaps)
    ratio = std_g/mean_g
    print(f'  Gap std/mean = {ratio:.3f} (Wigner-Dyson=0.52, Poisson=1.0)')
    print(f'  -> {"CHAOTIC" if ratio < 0.7 else "INTEGRABLE" if ratio > 0.85 else "MIXED"}')

# ============================================================
# SAVE COMPACT RESULTS
# ============================================================
print('\nSaving compact results...')

results = {
    'n_configs': n,
    'n_loaded': loaded,
    'sector_irreps': SECTOR,
    'T_max': float(T_max),
    'eigenvalues_top200': [float(e) for e in evals[:200]],
    'eigenvalues_bottom50': [float(e) for e in evals[-50:]],
    'n_positive': int(sum(1 for e in evals if e > 0)),
    'n_negative': int(sum(1 for e in evals if e < 0)),
    
    'a_e_HVP': float(a_e_HVP),
    'a_mu_HVP': float(a_mu_HVP),
    'a_mu_ratio_to_SM': float(a_mu_HVP/6.93e-8),
    'sum_HVP': float(sum_HVP),
    
    'masses_top60': masses[:60],
    'sector_content_top50': sector_content[:50],
    
    'source_spectra': source_spectra,
    
    'vacuum_structure': {
        'chiral_order': float(chiral),
        'quark_condensate': float(qq_cond),
    },
    
    'correlator_C_t': C_vals,
    
    'chi4_states': chi4_states,
    
    'eigenvalue_gap_ratio': float(ratio) if gaps else None,
    
    'source_overlaps': {
        name: [float(x) for x in ov[:200]]
        for name, ov in overlaps.items()
    },
}

outfile = 'qcd3125_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)

size_kb = os.path.getsize(outfile) / 1024
print(f'  Saved to {outfile} ({size_kb:.0f} KB)')
print(f'\n  Upload this file (NOT the 200MB progress file).')
print(f'  It contains everything needed for analysis.')
