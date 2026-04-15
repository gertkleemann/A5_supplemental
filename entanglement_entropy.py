#!/usr/bin/env python3
"""
ENTANGLEMENT ENTROPY OF THE A₅ CELL VACUUM
============================================
Computes the von Neumann entropy of the ground state eigenvector
of the 3125×3125 transfer matrix, under various bipartitions.

The ground state |ψ₀⟩ lives in a 3125 = 5⁵ dimensional space,
where each of the 5 face edges carries one of 5 irrep labels.
Tracing out one or more edges gives a reduced density matrix
whose entropy measures the entanglement between the traced-out
subsystem and the rest.

Physical interpretation: the observer occupies one face of the
dodecahedron. The 5 edges of that face are the observation channels.
Tracing out k edges is analogous to the 12→(12-k) projection.

Usage: python entanglement_entropy.py qcd3125_progress_seeded.json

Requirements: numpy
"""

import numpy as np
import json, sys, time, math

Lambda = 332.0
phi = (1 + math.sqrt(5)) / 2

# ================================================================
# LOAD AND DIAGONALIZE
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
print(f"  Matrix: {n}x{n}, rows loaded: {len(rows)}")
print(f"  Load time: {time.time()-t0:.1f}s")

T = np.zeros((n, n), dtype=np.float64)
for r in rows:
    T[r['row'], :len(r['data'])] = r['data']
T = (T + T.T) / 2

print("Diagonalizing...", end=' ', flush=True)
t0 = time.time()
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]
print(f"done ({time.time()-t0:.1f}s)")

T_max = evals[0]
psi0 = evecs[:, 0]  # ground state eigenvector
print(f"  T_max = {T_max:.4e}")
print(f"  |ψ₀| = {np.linalg.norm(psi0):.6f}")

# ================================================================
# HELPER: von Neumann entropy from singular values
# ================================================================
def von_neumann_entropy(psi, shape_A, shape_B):
    """Compute entanglement entropy of bipartition A|B.
    
    psi: state vector of length shape_A * shape_B
    Reshape into matrix shape_A × shape_B, SVD, entropy from Schmidt values.
    """
    M = psi.reshape(shape_A, shape_B)
    sv = np.linalg.svd(M, compute_uv=False)
    # Schmidt coefficients squared = eigenvalues of reduced density matrix
    p = sv**2
    p = p[p > 1e-30]  # remove numerical zeros
    p = p / p.sum()  # normalize
    S = -np.sum(p * np.log(p))
    return S, len(p), p

# ================================================================
# 1. SINGLE-EDGE BIPARTITIONS
# ================================================================
k = 5  # number of irreps
n_edges = 5

print(f"\n{'='*65}")
print(f"SINGLE-EDGE BIPARTITIONS (trace out 1 of 5 edges)")
print(f"{'='*65}")

# The 3125 configs are ordered as: c = a0 + a1*5 + a2*25 + a3*125 + a4*625
# To trace out edge j, we need to reshape so that edge j is separated.

for j in range(n_edges):
    # Rearrange indices so edge j is first
    # Config index: c = sum(a_i * 5^i)
    # We need to permute so edge j is in the first position
    
    # Build permuted state vector
    psi_perm = np.zeros(n)
    for c in range(n):
        # Extract labels
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        
        # Permute: move edge j to position 0
        label_j = labels[j]
        others = [labels[i] for i in range(n_edges) if i != j]
        
        # New index: label_j + 5 * (others as base-5 number)
        other_idx = 0
        for i, l in enumerate(others):
            other_idx += l * (k ** i)
        new_c = label_j + k * other_idx
        psi_perm[new_c] = psi0[c]
    
    S, n_schmidt, p = von_neumann_entropy(psi_perm, k, k**(n_edges-1))
    S_max = np.log(k)
    
    print(f"  Edge {j}: S = {S:.6f} nats = {S/np.log(2):.4f} bits")
    print(f"           S/ln(5) = {S/np.log(5):.6f}, S/ln(2) = {S/np.log(2):.6f}")
    print(f"           Schmidt rank = {n_schmidt}, S_max = ln(5) = {S_max:.4f}")
    print(f"           Top Schmidt values: {p[:5]}")

# ================================================================
# 2. MULTI-EDGE BIPARTITIONS (1|4, 2|3, 3|2, 4|1)
# ================================================================
print(f"\n{'='*65}")
print(f"MULTI-EDGE BIPARTITIONS")
print(f"{'='*65}")

# For k edges traced out: reshape psi into (5^k) × (5^(5-k))
# Using the natural ordering (first k edges | last 5-k edges)

for n_traced in range(1, 5):
    n_kept = n_edges - n_traced
    shape_A = k ** n_traced
    shape_B = k ** n_kept
    
    S, n_schmidt, p = von_neumann_entropy(psi0, shape_A, shape_B)
    S_max = np.log(min(shape_A, shape_B))
    
    print(f"  {n_traced}|{n_kept} split: S = {S:.6f} nats = {S/np.log(2):.4f} bits")
    print(f"           S/ln(5) = {S/np.log(5):.6f}")
    print(f"           S_max = ln({min(shape_A, shape_B)}) = {S_max:.4f}")
    print(f"           S/S_max = {S/S_max:.4f} (entanglement fraction)")
    print(f"           Schmidt rank = {n_schmidt}")

# ================================================================
# 3. IRREP-SECTOR ENTROPY
# ================================================================
print(f"\n{'='*65}")
print(f"IRREP-SECTOR DECOMPOSITION")
print(f"{'='*65}")

irrep_names = ['chi1', 'chi3', "chi3'", 'chi4', 'chi5']
irrep_dims = [1, 3, 3, 4, 5]

# For each irrep: what fraction of |ψ₀|² lives in configs containing that irrep?
for r in range(k):
    # Configs where at least one edge has irrep r
    weight = 0.0
    for c in range(n):
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        if r in labels:
            weight += psi0[c]**2
    print(f"  {irrep_names[r]}: {weight*100:.2f}% of vacuum")

# For each irrep: entropy of "has irrep r" vs "doesn't have irrep r"
print(f"\n  Binary entropy (has/hasn't each irrep):")
for r in range(k):
    p_has = 0.0
    for c in range(n):
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        if r in labels:
            p_has += psi0[c]**2
    p_not = 1.0 - p_has
    if p_has > 0 and p_not > 0:
        S_bin = -p_has * np.log(p_has) - p_not * np.log(p_not)
    else:
        S_bin = 0
    print(f"    {irrep_names[r]}: S_bin = {S_bin:.6f} nats (p_has = {p_has:.4f})")

# ================================================================
# 4. CHECK FOR CLEAN A₅ QUANTITIES
# ================================================================
print(f"\n{'='*65}")
print(f"CHECKING FOR A₅ QUANTITIES")
print(f"{'='*65}")

# Compute the 1|4 entropy (most natural: trace out observer edge)
S_14, _, _ = von_neumann_entropy(psi0, k, k**4)

candidates = {
    'ln(2)': np.log(2),
    'ln(3)': np.log(3),
    'ln(4)': np.log(4),
    'ln(5)': np.log(5),
    'ln(11)': np.log(11),
    'ln(12)': np.log(12),
    'ln(60)': np.log(60),
    'ln(phi)': np.log(phi),
    'phi': phi,
    '1/phi': 1/phi,
    'sqrt(5)/2': math.sqrt(5)/2,
    'ln(5)/2': np.log(5)/2,
    '(3-sqrt(5))/2': (3-math.sqrt(5))/2,
    'ln(phi^2)': np.log(phi**2),
    '5*ln(phi)': 5*np.log(phi),
    '2*ln(phi)': 2*np.log(phi),
    'ln(5)/phi': np.log(5)/phi,
    'phi*ln(phi)': phi*np.log(phi),
}

print(f"\n  S(1|4) = {S_14:.8f} nats")
print(f"\n  Comparison with A₅ quantities:")
for name, val in sorted(candidates.items(), key=lambda x: abs(x[1] - S_14)):
    err = abs(val - S_14) / S_14 * 100 if S_14 > 0 else 999
    mark = " <---" if err < 1 else ""
    print(f"    {name:>15} = {val:.8f}  ({err:6.2f}% off){mark}")

# ================================================================
# 5. AREA LAW CHECK (single cell)
# ================================================================
print(f"\n{'='*65}")
print(f"AREA LAW: S vs boundary size")
print(f"{'='*65}")

entropies = []
for n_traced in range(1, 5):
    n_kept = n_edges - n_traced
    shape_A = k ** n_traced
    shape_B = k ** n_kept
    S, _, _ = von_neumann_entropy(psi0, shape_A, shape_B)
    boundary = min(n_traced, n_kept)  # boundary = min(A, B) edges
    entropies.append((n_traced, n_kept, boundary, S))
    print(f"  {n_traced}|{n_kept}: boundary = {boundary} edge(s), S = {S:.6f}")

print(f"\n  If area law holds: S should be proportional to boundary size")
for i in range(len(entropies)-1):
    if entropies[i][2] > 0 and entropies[i+1][2] > 0:
        ratio_S = entropies[i+1][3] / entropies[i][3]
        ratio_A = entropies[i+1][2] / entropies[i][2]
        print(f"  S({entropies[i+1][0]}|{entropies[i+1][1]})/S({entropies[i][0]}|{entropies[i][1]}) = {ratio_S:.4f}, "
              f"boundary ratio = {ratio_A:.4f}")

# ================================================================
# SAVE
# ================================================================
results = {
    'T_max': float(T_max),
    'single_edge_entropies': {},
    'multi_edge_entropies': {},
}

for n_traced in range(1, 5):
    n_kept = n_edges - n_traced
    shape_A = k ** n_traced
    shape_B = k ** n_kept
    S, n_schmidt, _ = von_neumann_entropy(psi0, shape_A, shape_B)
    results['multi_edge_entropies'][f'{n_traced}|{n_kept}'] = {
        'S_nats': float(S),
        'S_bits': float(S / np.log(2)),
        'schmidt_rank': int(n_schmidt),
    }

with open('entanglement_entropy_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to entanglement_entropy_results.json")
print("Done.")
