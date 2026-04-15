#!/usr/bin/env python3
"""
ENTANGLEMENT ENTROPY — UNIVERSAL (any sector)
===============================================
Works on 3125 (k=5), 1024 (k=4), 243 (k=3), 32 (k=2).
Auto-detects k from matrix size (n = k^5).

Usage:
  python entanglement_entropy_universal.py qcd3125_progress_seeded.json
  python entanglement_entropy_universal.py qcd1024_progress.json
  python entanglement_entropy_universal.py dark1024_progress.json
"""
import numpy as np
import json, sys, math, os

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)

fname = sys.argv[1] if len(sys.argv) > 1 else 'qcd1024_progress.json'

print(f"Loading {fname}...")
with open(fname) as f:
    data = json.load(f)

rows = data.get('rows', [])
n = max(r['row'] for r in rows) + 1

# Auto-detect k (number of irreps in sector)
n_edges = 5
k = round(n ** (1.0/n_edges))
assert k**n_edges == n, f"Matrix size {n} is not k^5 for any integer k"

print(f"  Matrix: {n}x{n}, k={k} irreps, {n_edges} edges")

T = np.zeros((n, n), dtype=np.float64)
for r in rows:
    T[r['row'], :len(r['data'])] = r['data']
T = (T + T.T) / 2

print("Diagonalizing...", end=' ', flush=True)
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
print("done")

T_max = evals[0]
psi0 = evecs[:, 0]
print(f"  T_max = {T_max:.6e}")

def von_neumann_entropy(psi, shape_A, shape_B):
    M = psi.reshape(shape_A, shape_B)
    sv = np.linalg.svd(M, compute_uv=False)
    p = sv**2
    p = p[p > 1e-30]
    p = p / p.sum()
    S = -np.sum(p * np.log(p))
    return S, len(p), p

# ================================================================
# BIPARTITIONS
# ================================================================
print(f"\n{'='*65}")
print(f"MULTI-EDGE BIPARTITIONS")
print(f"{'='*65}")

entropies = {}
for n_traced in range(1, n_edges):
    n_kept = n_edges - n_traced
    shape_A = k ** n_traced
    shape_B = k ** n_kept
    S, n_schmidt, p = von_neumann_entropy(psi0, shape_A, shape_B)
    S_max = np.log(min(shape_A, shape_B))
    entropies[f'{n_traced}|{n_kept}'] = {
        'S_nats': float(S),
        'S_bits': float(S / np.log(2)),
        'S_over_Smax': float(S / S_max),
        'schmidt_rank': int(n_schmidt),
    }
    print(f"  {n_traced}|{n_kept}: S = {S:.8f} nats = {S/np.log(2):.6f} bits")
    print(f"         S/S_max = {S/S_max:.6f} ({S/S_max*100:.2f}%)")
    print(f"         Schmidt rank = {n_schmidt}")

# ================================================================
# KEY COMPARISONS
# ================================================================
S_14 = entropies[f'1|{n_edges-1}']['S_nats']

print(f"\n{'='*65}")
print(f"COMPARISON WITH α")
print(f"{'='*65}")
print(f"  S(1|{n_edges-1}) = {S_14:.10f}")
print(f"  2α        = {2*alpha:.10f}")
print(f"  Ratio     = {S_14/(2*alpha):.8f}")
print(f"  Match     = {abs(S_14-2*alpha)/(2*alpha)*100:.4f}%")

m_e_Lambda = 0.511 / 332.0
predicted = 2*alpha * (1 - m_e_Lambda)
print(f"  2α(1-m_e/Λ) = {predicted:.10f}")
print(f"  Match     = {abs(S_14-predicted)/predicted*100:.4f}%")

# Sector-specific checks
print(f"\n  This is the {'FULL' if k==5 else f'{k}-irrep SUBSECTOR'} computation")
if k == 4:
    print(f"  Sector has {k} irreps out of 5")
    print(f"  If S ≈ 2α: dark matter NOT needed for the relationship")
    print(f"  If S ≠ 2α: dark matter IS essential")

# ================================================================
# SAVE
# ================================================================
sector_name = os.path.basename(fname).replace('_progress.json', '').replace('_results.json', '')
outfile = f'entropy_{sector_name}.json'

results = {
    'source_file': fname,
    'matrix_size': n,
    'n_irreps': k,
    'T_max': float(T_max),
    'entropies': entropies,
    'S_1_vs_rest': float(S_14),
    'two_alpha': float(2*alpha),
    'ratio_S_over_2alpha': float(S_14 / (2*alpha)),
    'two_alpha_screened': float(predicted),
    'ratio_S_over_2alpha_screened': float(S_14 / predicted),
}

with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")
