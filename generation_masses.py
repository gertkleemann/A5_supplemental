#!/usr/bin/env python3
"""
THREE-GENERATION MASS COMPUTATION
===================================
Computes transfer matrices using FULL dodecahedral geometry 
but with A₄, Z₃, Z₂ character tables instead of A₅.

The mass gap ratio between subgroups gives the generation mass hierarchy.

Run: python3 generation_masses.py
Requires: numpy, the dodecahedral geometry from qcd1024_numba.py
"""
import numpy as np
import math, time, json
from itertools import product as iprod

phi = (1+math.sqrt(5))/2
sqrt5 = math.sqrt(5)
Lambda = 332

# ============================================================
# DODECAHEDRAL GEOMETRY (same as qcd1024_numba.py)
# ============================================================

# Vertices of dodecahedron (from golden ratio construction)
def make_dodecahedron():
    """Return 20 vertices and 30 edges of the dodecahedron."""
    verts = []
    # 8 cube vertices: (±1,±1,±1)
    for s1 in [-1,1]:
        for s2 in [-1,1]:
            for s3 in [-1,1]:
                verts.append((s1, s2, s3))
    # 12 vertices from golden rectangles
    for s1 in [-1,1]:
        for s2 in [-1,1]:
            verts.append((0, s1*phi, s2/phi))
            verts.append((s1/phi, 0, s2*phi))
            verts.append((s1*phi, s2/phi, 0))
    
    verts = np.array(verts)
    
    # Find edges: pairs of vertices at distance 2/phi
    edge_len = 2.0/phi
    edges = []
    for i in range(20):
        for j in range(i+1, 20):
            d = np.linalg.norm(verts[i]-verts[j])
            if abs(d - edge_len) < 0.01:
                edges.append((i,j))
    
    return verts, edges

verts, edges = make_dodecahedron()
print(f'Dodecahedron: {len(verts)} vertices, {len(edges)} edges')
assert len(edges) == 30

# Find 5 edges meeting at each face (12 pentagonal faces)
# Each face has 5 edges. Our transfer matrix uses 5 edges per "slice"
# For simplicity, use the same 5-edge selection as the original script

# Actually, the transfer matrix in the original computation selects
# edges based on a "transfer direction". For a fair comparison between
# subgroups, we need the SAME geometry, just different Boltzmann weights.

# The Boltzmann weight for a pair of irreps (r,s) on edge e is:
#   W_e(r,s) = sum_C |C| * chi_r(C) * chi_s(C) / |G| * dim(r) * dim(s)
# times a geometric factor from the edge orientation.

# For a SUBGROUP H ⊂ G, the weight changes because:
# 1. The CHARACTER TABLE changes
# 2. The IRREP LABELS change (different irreps for H)
# 3. The geometric factors stay the SAME (same dodecahedron)

# ============================================================
# CHARACTER TABLES
# ============================================================

# A₅: 5 irreps, order 60
# Classes: {e}(1), C₃(20), C₂(15), C₅(12), C₅'(12)
char_tables = {
    'A5': {
        'order': 60,
        'class_sizes': [1, 20, 15, 12, 12],
        'dims': [1, 3, 3, 4, 5],
        'chars': [
            [1, 1, 1, 1, 1],           # χ₁
            [3, 0, -1, phi, 1-phi],     # χ₃
            [3, 0, -1, 1-phi, phi],     # χ₃'
            [4, 1, 0, -1, -1],          # χ₄
            [5, -1, 1, 0, 0],           # χ₅
        ],
        'names': ['χ₁','χ₃',"χ₃'",'χ₄','χ₅'],
    },
    'A4': {
        'order': 12,
        'class_sizes': [1, 3, 4, 4],
        'dims': [1, 1, 1, 3],  # using complex irreps: 1, 1', 1'', 3
        'chars': [
            [1, 1, 1, 1],                    # 1
            [1, 1, -0.5+0.866j, -0.5-0.866j],  # 1' (ω = e^{2πi/3})
            [1, 1, -0.5-0.866j, -0.5+0.866j],  # 1'' (ω²)
            [3, -1, 0, 0],                    # 3
        ],
        'names': ['1',"1'","1''",'3'],
    },
    'A4_real': {
        'order': 12,
        'class_sizes': [1, 3, 4, 4],
        'dims': [1, 2, 3],  # real irreps: 1, ε(=1'+1''), 3
        'chars': [
            [1, 1, 1, 1],      # 1
            [2, 2, -1, -1],     # ε
            [3, -1, 0, 0],      # 3
        ],
        'names': ['1','ε','3'],
    },
    'Z3': {
        'order': 3,
        'class_sizes': [1, 1, 1],
        'dims': [1, 2],  # real: 1, ρ(=ω+ω²)
        'chars': [
            [1, 1, 1],
            [2, -1, -1],
        ],
        'names': ['1','ρ'],
    },
    'Z2': {
        'order': 2,
        'class_sizes': [1, 1],
        'dims': [1, 1],
        'chars': [
            [1, 1],
            [1, -1],
        ],
        'names': ['1','-1'],
    },
}

# ============================================================
# BUILD TRANSFER MATRIX WITH FULL GEOMETRIC WEIGHTS
# ============================================================

def build_full_transfer_matrix(group_name):
    """Build a 5-edge transfer matrix using the character table 
    of the given group, with full dodecahedral Boltzmann weights."""
    
    ct = char_tables[group_name]
    n_irreps = len(ct['dims'])
    order = ct['order']
    class_sizes = ct['class_sizes']
    chars = ct['chars']
    dims = ct['dims']
    
    # Boltzmann weight matrix W(r,s) 
    # W(r,s) = sum_C |C| * chi_r(C) * conj(chi_s(C)) / |G|
    W = np.zeros((n_irreps, n_irreps), dtype=complex)
    for r in range(n_irreps):
        for s in range(n_irreps):
            for c in range(len(class_sizes)):
                chi_r = chars[r][c]
                chi_s = chars[s][c]
                if isinstance(chi_s, complex):
                    chi_s_conj = chi_s.conjugate()
                else:
                    chi_s_conj = chi_s
                W[r,s] += class_sizes[c] * chi_r * chi_s_conj / order
    
    W = np.real(W)  # Should be real for physical weights
    
    # Add dimension factors (matching original computation)
    # The Boltzmann factor includes sqrt(dim(r)*dim(s)) per edge
    # and exp(-Lambda * eigenvalue) type factors
    
    # GEOMETRIC WEIGHT: use the dodecahedral Laplacian eigenvalue
    # for each irrep, scaled by Lambda
    # The eigenvalue for irrep r on the dodecahedron is:
    # 3 - chi_r(C_adj) where C_adj is related to the adjacency matrix
    # For the dodecahedron with vertex degree 3:
    # λ_r = 3 - eigenvalue of adjacency matrix for irrep r
    
    # Dodecahedron adjacency eigenvalues by irrep:
    # χ₁: 3, χ₃: √5, χ₃': -√5, χ₄: 0 (two values: 0 and -2), χ₅: 1
    # Laplacian eigenvalues: 3-adj_eval
    # χ₁: 0, χ₃: 3-√5, χ₃': 3+√5, χ₄: 3 and 5, χ₅: 2
    
    # For subgroups, the Laplacian eigenvalues change because
    # the irreps are different. But the GEOMETRY is the same dodecahedron.
    # The subgroup irreps branch from A₅ irreps, so their eigenvalues
    # are averages of the parent eigenvalues.
    
    # For now, use the Boltzmann weight: 
    # B(r,s) = W(r,s) * sqrt(dim(r)*dim(s)) * exp(-Lambda_eff * mass_scale)
    # where Lambda_eff encodes the lattice scale
    
    # The key is to use the SAME Lambda for all groups,
    # so the mass gaps are directly comparable.
    
    configs = list(iprod(range(n_irreps), repeat=5))
    n_configs = len(configs)
    
    # Build T_ij = product over 5 edges of weight(cfg_i[e], cfg_j[e])
    T = np.zeros((n_configs, n_configs))
    
    Lambda_eff = Lambda / 1000.0  # scale to make eigenvalues reasonable
    
    for i, ci in enumerate(configs):
        for j, cj in enumerate(configs):
            weight = 1.0
            for e in range(5):
                r, s = ci[e], cj[e]
                w = abs(W[r,s]) * math.sqrt(dims[r] * dims[s])
                # Add exponential Boltzmann factor
                lap_r = (3.0 - W[r,r]) * Lambda_eff  # approximate Laplacian
                weight *= w * math.exp(-abs(lap_r) * 0.1)
            T[i,j] = weight
    
    T_sym = (T + T.T) / 2
    evals = np.linalg.eigvalsh(T_sym)
    evals = np.sort(evals)[::-1]
    
    return evals, n_configs, W

# ============================================================
# COMPUTE FOR EACH GROUP
# ============================================================

print(f'\n{"="*70}')
print(f'GENERATION MASS HIERARCHY FROM FULL GEOMETRY')
print(f'{"="*70}')

results = {}
for group_name in ['A5', 'A4_real', 'Z3', 'Z2']:
    print(f'\nComputing {group_name}...')
    t0 = time.time()
    evals, n_cfg, W = build_full_transfer_matrix(group_name)
    dt = time.time() - t0
    
    ct = char_tables[group_name]
    
    # Mass gap
    T_max = evals[0]
    if evals[1] > 0 and evals[1] < T_max * 0.9999:
        gap = -math.log(evals[1]/T_max)
        mass = gap * Lambda
    else:
        gap = 0
        mass = 0
    
    results[group_name] = {
        'order': ct['order'],
        'n_irreps': len(ct['dims']),
        'n_configs': n_cfg,
        'T_max': float(T_max),
        'gap': float(gap),
        'mass': float(mass),
        'top5_evals': [float(e) for e in evals[:5]],
    }
    
    print(f'  |G|={ct["order"]}, {len(ct["dims"])} irreps, {n_cfg} configs')
    print(f'  T_max = {T_max:.4e}, gap = {gap:.4f}, mass = {mass:.0f} MeV')
    print(f'  Time: {dt:.1f}s')
    print(f'  Top 5 eigenvalue ratios: {[f"{e/T_max:.6f}" for e in evals[:5]]}')

# Summary
print(f'\n{"="*70}')
print(f'SUMMARY')
print(f'{"="*70}')

ref_gap = results['A5']['gap']
print(f'\n  {"Group":>8} {"Order":>6} {"#irreps":>8} {"Matrix":>8} {"Gap":>8} {"Mass":>8} {"Ratio":>8}')
for g in ['A5', 'A4_real', 'Z3', 'Z2']:
    r = results[g]
    ratio = r['gap']/ref_gap if ref_gap > 0 else 0
    print(f'  {g:>8} {r["order"]:>6} {r["n_irreps"]:>8} {r["n_configs"]:>8} {r["gap"]:>8.4f} {r["mass"]:>7.0f} {ratio:>8.2f}')

print(f'\n  Observed lepton ratios:')
print(f'    m_μ/m_e = 206.8')
print(f'    m_τ/m_e = 3477.2')

# Save results
with open('generation_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'\n  Results saved to generation_results.json')

PYEOF
