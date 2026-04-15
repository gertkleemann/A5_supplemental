#!/usr/bin/env python3
"""
SCHMIDT COEFFICIENTS OF THE VACUUM EIGENVECTOR
================================================
Extracts the 5 Schmidt coefficients from the 1|4 bipartition
of the 3125×3125 transfer matrix ground state.

If these are clean A₅ quantities, m_e is derivable.

Usage: python schmidt_coefficients.py qcd3125_progress_seeded.json
"""
import numpy as np
import json, sys, math

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

fname = 'qcd3125_progress_seeded.json'
if len(sys.argv) > 1:
    fname = sys.argv[1]

print(f"Loading {fname}...")
with open(fname) as f:
    data = json.load(f)

rows = data.get('rows', [])
n = max(r['row'] for r in rows) + 1
T = np.zeros((n, n), dtype=np.float64)
for r in rows:
    T[r['row'], :len(r['data'])] = r['data']
T = (T + T.T) / 2

print("Diagonalizing...")
evals, evecs = np.linalg.eigh(T)
idx = np.argsort(np.abs(evals))[::-1]
evals = evals[idx]
evecs = evecs[:, idx]

psi0 = evecs[:, 0]
T_max = evals[0]
print(f"  T_max = {T_max:.6e}")

# ================================================================
# SCHMIDT DECOMPOSITION: 1|4 bipartition
# ================================================================
k = 5
M = psi0.reshape(k, k**4)  # 5 × 625 matrix
U, sv, Vh = np.linalg.svd(M, full_matrices=False)

# Schmidt coefficients (squared = eigenvalues of reduced density matrix)
p = sv**2
p = p / p.sum()  # normalize

print(f"\n{'='*65}")
print(f"SCHMIDT DECOMPOSITION (1|4 bipartition)")
print(f"{'='*65}")
print(f"\n  Schmidt rank: {len(sv[sv > 1e-15])}")
print(f"  Singular values: {sv}")
print(f"\n  Schmidt coefficients p_i = σ_i²/Σσ²:")
for i, pi in enumerate(p):
    print(f"    p[{i}] = {pi:.15f}")

S = -np.sum(p * np.log(p))
print(f"\n  S = -Σ p_i ln(p_i) = {S:.15f}")
print(f"  2α = {2*alpha:.15f}")
print(f"  2α(1-m_e/Λ) = {2*alpha*(1-0.511/332):.15f}")

# ================================================================
# CHECK AGAINST A₅ QUANTITIES
# ================================================================
print(f"\n{'='*65}")
print(f"ARE THE SCHMIDT COEFFICIENTS A₅ QUANTITIES?")
print(f"{'='*65}")

dims = [1, 3, 3, 4, 5]
irr_names = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']

# Sort p descending
idx_sort = np.argsort(p)[::-1]
p_sorted = p[idx_sort]

print(f"\n  Sorted coefficients:")
for i, pi in enumerate(p_sorted):
    print(f"    p[{i}] = {pi:.15f}")

# Check ratios between coefficients
print(f"\n  Ratios p[i]/p[j]:")
for i in range(len(p_sorted)):
    for j in range(i+1, len(p_sorted)):
        if p_sorted[j] > 1e-15:
            r = p_sorted[i] / p_sorted[j]
            print(f"    p[{i}]/p[{j}] = {r:.8f}")

# Check candidate forms
candidates = {
    'dim²/60': [d**2/60 for d in dims],
    'dim/16': [d/16 for d in dims],
    'dim²/Σdim²': [d**2/sum(d**2 for d in dims) for d in dims],
    'dim⁴/Σdim⁴': [d**4/sum(d**4 for d in dims) for d in dims],
}

print(f"\n  Check against standard distributions:")
for name, dist in candidates.items():
    dist_sorted = sorted(dist, reverse=True)
    match = sum(abs(a-b) for a,b in zip(p_sorted, dist_sorted)) / 5
    print(f"    {name}: {dist_sorted}")
    print(f"      avg |Δ| = {match:.6f} {'<--- CLOSE' if match < 0.001 else ''}")

# Check individual values against A₅ numbers
print(f"\n  Individual coefficient checks:")
a5_candidates = {
    '1/|A₅| = 1/60': 1/60,
    'dim(χ₁)²/60': 1/60,
    'dim(χ₃)²/60': 9/60,
    'dim(χ₄)²/60': 16/60,
    'dim(χ₅)²/60': 25/60,
    '1/5': 1/5,
    '1/4': 1/4,
    '1/3': 1/3,
    'φ/5': phi/5,
    '(3-√5)/2': (3-sqrt5)/2,
    'φ²/5': phi**2/5,
    '1/φ⁴': 1/phi**4,
    '(φ+2)/20': (phi+2)/20,
    '1/(4φ)': 1/(4*phi),
    '3/(4×5)': 3/20,
    '√5/12': sqrt5/12,
}

for i, pi in enumerate(p_sorted):
    print(f"\n    p[{i}] = {pi:.10f}")
    for name, val in sorted(a5_candidates.items(), key=lambda x: abs(x[1]-pi)):
        err = abs(val - pi) / pi * 100 if pi > 1e-15 else 999
        if err < 5:
            print(f"      {name:>20} = {val:.10f}  ({err:.3f}%)")

# ================================================================
# WHICH IRREP DOMINATES EACH SCHMIDT COMPONENT?
# ================================================================
print(f"\n{'='*65}")
print(f"IRREP CONTENT OF EACH SCHMIDT COMPONENT")
print(f"{'='*65}")

# U is 5×5: rows = edge irrep label, cols = Schmidt component
# U[a, i] = amplitude of irrep a in Schmidt component i
print(f"\n  Left singular vectors U (rows = irreps, cols = Schmidt components):")
print(f"  {'':>6}", end='')
for i in range(5):
    print(f"  comp_{i:d}  ", end='')
print()
for a in range(5):
    print(f"  {irr_names[a]:>6}", end='')
    for i in range(5):
        print(f"  {U[a,i]:>8.5f}", end='')
    print()

print(f"\n  |U|² (probability of each irrep in each component):")
print(f"  {'':>6}", end='')
for i in range(5):
    print(f"  comp_{i:d}  ", end='')
print()
for a in range(5):
    print(f"  {irr_names[a]:>6}", end='')
    for i in range(5):
        print(f"  {U[a,i]**2:>8.5f}", end='')
    print()

# Dominant irrep per component
print(f"\n  Dominant irrep per Schmidt component:")
for i in range(5):
    dom = np.argmax(np.abs(U[:, i]))
    print(f"    Component {i}: {irr_names[dom]} ({U[dom,i]**2*100:.1f}%)")

# ================================================================
# SAVE
# ================================================================
results = {
    'schmidt_singular_values': sv.tolist(),
    'schmidt_coefficients_p': p.tolist(),
    'schmidt_coefficients_sorted': p_sorted.tolist(),
    'entropy_nats': float(S),
    'two_alpha': float(2*alpha),
    'ratio_S_over_2alpha': float(S/(2*alpha)),
    'U_matrix': U.tolist(),
    'dominant_irreps': [irr_names[np.argmax(np.abs(U[:, i]))] for i in range(5)],
}

with open('schmidt_coefficients_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to schmidt_coefficients_results.json")
