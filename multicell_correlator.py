#!/usr/bin/env python3
"""
MULTI-CELL CORRELATOR: PION AND PROTON FROM T^N
=================================================
Uses C(N) = Σᵢ λᵢ^N × |⟨S|ψᵢ⟩|² to extract hadron masses
from the exponential decay of channel-specific correlators.

No new matrix computation — uses existing eigendecomposition.
C(N) for N=1..1000 takes milliseconds.

Usage: python multicell_correlator.py [path_to_qcd3125_progress_seeded.json]
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

print("Diagonalizing 3125×3125...", flush=True)
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

T_max = evals[0]
print(f"T_max = {T_max:.6e}")

# ================================================================
# CONFIG LABELS
# ================================================================
k = 5
n_edges = 5

def config_labels(c):
    labels = []
    tmp = c
    for i in range(n_edges):
        labels.append(tmp % k)
        tmp //= k
    return labels

all_labels = [config_labels(c) for c in range(n)]

# ================================================================
# SOURCE OPERATORS (same as spectroscopy, but keep full vectors)
# ================================================================
def make_source(name, condition):
    src = np.zeros(n, dtype=np.float64)
    count = 0
    for c in range(n):
        if condition(all_labels[c]):
            src[c] = 1.0
            count += 1
    if np.linalg.norm(src) > 0:
        src /= np.linalg.norm(src)
    print(f"  {name}: {count} configs")
    return src

print(f"\nBuilding sources...")

sources = {}

sources['meson'] = make_source('meson (chi3-chi3p adjacent)',
    lambda l: any(
        (l[i] == 1 and l[(i+1)%5] == 2) or (l[i] == 2 and l[(i+1)%5] == 1)
        for i in range(5)))

sources['meson_pure'] = make_source('meson pure (chi3-chi3p adj, rest chi5)',
    lambda l: any(
        ((l[i] == 1 and l[(i+1)%5] == 2) or (l[i] == 2 and l[(i+1)%5] == 1))
        and all(l[j] == 4 for j in range(5) if j != i and j != (i+1)%5)
        for i in range(5)))

sources['baryon'] = make_source('baryon (3x chi3 consecutive)',
    lambda l: any(
        l[i] == 1 and l[(i+1)%5] == 1 and l[(i+2)%5] == 1
        for i in range(5)))

sources['baryon_glue'] = make_source('baryon+glue (3x chi3, rest chi5)',
    lambda l: l.count(1) >= 3 and all(x in [1, 4] for x in l))

sources['glueball'] = make_source('glueball (all chi5)',
    lambda l: all(x == 4 for x in l))

sources['pseudoscalar'] = make_source('pseudoscalar (chi3-chi3p non-adj)',
    lambda l: any(
        (l[i] == 1 and l[(i+2)%5] == 2) or (l[i] == 2 and l[(i+2)%5] == 1)
        for i in range(5)))

sources['vector'] = make_source('vector (chi3-chi5 adjacent)',
    lambda l: any(
        (l[i] == 1 and l[(i+1)%5] == 4) or (l[i] == 4 and l[(i+1)%5] == 1)
        for i in range(5)))

# ================================================================
# COMPUTE ALL OVERLAPS: |⟨S|ψᵢ⟩|² for ALL eigenvectors
# ================================================================
print(f"\nComputing overlaps for all {n} eigenvectors...", flush=True)

overlaps = {}
for src_name, src_vec in sources.items():
    ov = np.array([np.dot(src_vec, evecs[:, i])**2 for i in range(n)])
    overlaps[src_name] = ov
    n_nonzero = np.sum(ov > 1e-20)
    print(f"  {src_name}: {n_nonzero} nonzero overlaps")

# ================================================================
# CORRELATOR C(N) = Σᵢ λᵢ^N × |⟨S|ψᵢ⟩|²
# ================================================================
print(f"\n{'='*65}")
print(f"COMPUTING CORRELATORS C(N) FOR N = 1..500")
print(f"{'='*65}")

# Normalize eigenvalues: r_i = λ_i / λ_0
ratios = evals / T_max  # includes negative eigenvalues

N_max = 500
N_values = list(range(1, N_max + 1))

results = {}

known_hadrons = [
    (135, 'pion'), (498, 'kaon'), (548, 'eta'), (775, 'rho'),
    (782, 'omega'), (938, 'proton'), (958, "eta'"), (1019, 'phi'),
    (1232, 'Delta'), (1370, 'f0(1370)'), (1440, 'Roper'),
]

for src_name in sources:
    ov = overlaps[src_name]
    
    # C(N) = Σᵢ (λᵢ/λ₀)^N × |⟨S|ψᵢ⟩|²
    # Use log space to avoid overflow: log C(N) = log Σᵢ exp(N log|rᵢ| + log ovᵢ)
    # But we need to handle negative eigenvalues (they oscillate with (-1)^N)
    
    # Split into positive and negative eigenvalues
    pos_mask = evals > 0
    
    correlator = np.zeros(N_max)
    eff_mass = np.zeros(N_max)
    
    for idx_N, N in enumerate(N_values):
        # For positive eigenvalues: r^N
        # For negative eigenvalues: |r|^N × (-1)^N
        C = 0.0
        for i in range(n):
            if ov[i] > 1e-30 and abs(ratios[i]) > 1e-30:
                if evals[i] > 0:
                    # Use log to avoid overflow
                    log_term = N * math.log(ratios[i]) 
                    if log_term > -300:  # avoid underflow
                        C += ov[i] * math.exp(log_term)
                else:
                    log_term = N * math.log(abs(ratios[i]))
                    if log_term > -300:
                        C += ov[i] * math.exp(log_term) * ((-1)**N)
        correlator[idx_N] = C
    
    # Effective mass: m_eff(N) = -Λ × ln(C(N+1)/C(N))
    for idx_N in range(N_max - 1):
        if correlator[idx_N] > 0 and correlator[idx_N + 1] > 0:
            eff_mass[idx_N] = -Lambda * math.log(correlator[idx_N + 1] / correlator[idx_N])
        elif correlator[idx_N] != 0 and correlator[idx_N + 1] != 0:
            # Oscillating — use |C|
            eff_mass[idx_N] = -Lambda * math.log(abs(correlator[idx_N + 1]) / abs(correlator[idx_N]))
        else:
            eff_mass[idx_N] = float('nan')
    
    # Find plateau
    print(f"\n  --- {src_name} ---")
    print(f"  {'N':>5} {'C(N)':>14} {'m_eff MeV':>10}")
    print(f"  {'-'*35}")
    
    # Print key N values
    for N in [1, 2, 3, 5, 10, 20, 50, 100, 200, 500]:
        idx_N = N - 1
        if idx_N < N_max:
            m = eff_mass[idx_N] if idx_N < N_max - 1 else float('nan')
            C = correlator[idx_N]
            if not math.isnan(m) and m > 0:
                best = min(known_hadrons, key=lambda h: abs(h[0] - m))
                err = abs(best[0] - m) / best[0] * 100
                mark = f" → {best[1]} ({err:.1f}%)" if err < 20 else ""
                print(f"  {N:>5} {C:>14.6e} {m:>10.1f}{mark}")
            else:
                print(f"  {N:>5} {C:>14.6e} {'---':>10}")
    
    # Find the plateau value (average m_eff over N = 100..200)
    plateau_masses = [eff_mass[i] for i in range(99, 200) 
                      if not math.isnan(eff_mass[i]) and eff_mass[i] > 0 and eff_mass[i] < 20000]
    
    if plateau_masses:
        m_plateau = np.mean(plateau_masses)
        m_std = np.std(plateau_masses)
        best = min(known_hadrons, key=lambda h: abs(h[0] - m_plateau))
        err = abs(best[0] - m_plateau) / best[0] * 100
        print(f"\n  PLATEAU (N=100-200): {m_plateau:.1f} ± {m_std:.1f} MeV")
        print(f"  Closest hadron: {best[1]} ({best[0]} MeV), error: {err:.1f}%")
    else:
        m_plateau = None
        print(f"\n  NO STABLE PLATEAU FOUND")
    
    results[src_name] = {
        'correlator_N1': float(correlator[0]),
        'correlator_N10': float(correlator[9]),
        'correlator_N100': float(correlator[99]) if N_max >= 100 else None,
        'eff_mass_N1': round(float(eff_mass[0]), 1) if not math.isnan(eff_mass[0]) else None,
        'eff_mass_N10': round(float(eff_mass[9]), 1) if not math.isnan(eff_mass[9]) else None,
        'eff_mass_N50': round(float(eff_mass[49]), 1) if N_max >= 50 and not math.isnan(eff_mass[49]) else None,
        'eff_mass_N100': round(float(eff_mass[99]), 1) if N_max >= 100 and not math.isnan(eff_mass[99]) else None,
        'plateau_mass_MeV': round(float(m_plateau), 1) if m_plateau else None,
        'plateau_std_MeV': round(float(m_std), 1) if m_plateau else None,
        'plateau_match': best[1] if m_plateau else None,
        'plateau_error_pct': round(err, 2) if m_plateau else None,
        'eff_mass_series': [round(float(eff_mass[i]), 2) if not math.isnan(eff_mass[i]) else None
                           for i in range(min(200, N_max-1))],
    }

# ================================================================
# SUMMARY
# ================================================================
print(f"\n{'='*65}")
print(f"SUMMARY: PLATEAU MASSES")
print(f"{'='*65}")

print(f"\n  {'Channel':>20} {'Plateau MeV':>12} {'±':>4} {'Match':>12} {'Err':>6}")
print(f"  {'-'*60}")
for src_name, res in results.items():
    if res['plateau_mass_MeV']:
        print(f"  {src_name:>20} {res['plateau_mass_MeV']:>12.1f} {res['plateau_std_MeV']:>4.0f} {res['plateau_match']:>12} {res['plateau_error_pct']:>5.1f}%")
    else:
        print(f"  {src_name:>20} {'no plateau':>12}")

# ================================================================
# SAVE
# ================================================================
outfile = 'multicell_correlator_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {outfile}")
