#!/usr/bin/env python3
"""
DISCRETE DIRAC OPERATOR ON THE DODECAHEDRON
=============================================
Builds D = Σ_edges d̂·σ (40×40 matrix: 20 vertices × 2 spin components)
Inverts to get the lattice Dirac propagator S = D⁻¹
Tests whether the continuum Dirac equation emerges.

Key questions:
1. Does S(i,j) fall as 1/distance with spin structure?
2. Does the mass gap match m_e/Λ?
3. Is the propagator structure consistent with continuum (γ·p + m)⁻¹?
"""
import numpy as np
import math

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

# ================================================================
# BUILD DODECAHEDRON
# ================================================================
verts = []
for s1 in [1,-1]:
    for s2 in [1,-1]:
        for s3 in [1,-1]:
            verts.append([s1, s2, s3])
for s1 in [1,-1]:
    for s2 in [1,-1]:
        verts.append([0, s1/phi, s2*phi])
        verts.append([s1/phi, s2*phi, 0])
        verts.append([s1*phi, 0, s2/phi])
verts = np.array(verts, dtype=np.float64)
Nv = 20

# Find edges (distance = 2/phi)
edge_len = 2.0/phi
edges = []
adj = [[] for _ in range(Nv)]
for i in range(Nv):
    for j in range(i+1, Nv):
        if abs(np.linalg.norm(verts[i]-verts[j]) - edge_len) < 0.01:
            edges.append((i,j))
            adj[i].append(j)
            adj[j].append(i)
Ne = len(edges)

print(f"Dodecahedron: {Nv} vertices, {Ne} edges, degree {len(adj[0])}")

# BFS distances
from collections import deque
dist_matrix = np.zeros((Nv, Nv), dtype=int)
for start in range(Nv):
    visited = [-1]*Nv
    visited[start] = 0
    q = deque([start])
    while q:
        v = q.popleft()
        for nb in adj[v]:
            if visited[nb] < 0:
                visited[nb] = visited[v] + 1
                q.append(nb)
    dist_matrix[start] = visited

# ================================================================
# PAULI MATRICES (from inscribed cube)
# ================================================================
sigma = np.array([
    [[0, 1], [1, 0]],      # σ_x
    [[0, -1j], [1j, 0]],   # σ_y
    [[1, 0], [0, -1]],     # σ_z
], dtype=np.complex128)

# ================================================================
# BUILD DISCRETE DIRAC OPERATOR (40×40)
# ================================================================
D = np.zeros((2*Nv, 2*Nv), dtype=np.complex128)

for i, j in edges:
    # Unit direction vector along edge
    d_vec = verts[j] - verts[i]
    d_hat = d_vec / np.linalg.norm(d_vec)
    
    # d̂ · σ (2×2 matrix)
    dsig = sum(d_hat[k] * sigma[k] for k in range(3))
    
    # Place in the 40×40 matrix (both directions)
    D[2*i:2*i+2, 2*j:2*j+2] += dsig
    D[2*j:2*j+2, 2*i:2*i+2] += -dsig  # antisymmetric (Dirac, not Laplacian)

print(f"\nDirac operator D: {D.shape}")
print(f"  Hermitian: {np.allclose(D, D.conj().T)}")
print(f"  Anti-Hermitian check (D = -D†): {np.allclose(D, -D.conj().T)}")

# ================================================================
# EIGENVALUES
# ================================================================
evals = np.linalg.eigvalsh(D) if np.allclose(D, D.conj().T) else np.linalg.eigvals(D)
evals_sorted = np.sort(np.real(evals))

print(f"\n{'='*65}")
print(f"DIRAC EIGENVALUE SPECTRUM")
print(f"{'='*65}")

# Check for ±E pairing
pos_evals = sorted([e for e in evals_sorted if e > 0.001])
neg_evals = sorted([-e for e in evals_sorted if e < -0.001])

print(f"\n  Positive eigenvalues: {len(pos_evals)}")
print(f"  Negative eigenvalues: {len(neg_evals)}")
print(f"  Zero (|e| < 0.001): {sum(1 for e in evals_sorted if abs(e) < 0.001)}")

print(f"\n  Eigenvalues (sorted):")
unique_pos = sorted(set(round(e, 4) for e in pos_evals))
for ep in unique_pos:
    mult = sum(1 for e in pos_evals if abs(e - ep) < 0.01)
    # Check degeneracy
    neg_match = sum(1 for e in neg_evals if abs(e - ep) < 0.01)
    print(f"    ±{ep:.4f}  (multiplicity: +{mult}, −{neg_match})")

print(f"\n  ±E PAIRING: ", end="")
paired = all(any(abs(ne - pe) < 0.01 for ne in neg_evals) for pe in pos_evals)
print(f"{'✓ Every positive eigenvalue has a negative partner' if paired else '✗ BROKEN'}")

# Mass gap
if pos_evals:
    mass_gap = min(pos_evals)
    print(f"\n  Mass gap (smallest |E|): {mass_gap:.6f}")
    print(f"  In MeV: {mass_gap * Lambda:.1f}")
    print(f"  Compare: m_e/Λ = {0.511/Lambda:.6f}")
    print(f"  Compare: 3-√5 = {3-sqrt5:.6f} (χ₃ Laplacian eigenvalue)")

# ================================================================
# DIRAC PROPAGATOR S = D⁻¹ (or pseudoinverse if singular)
# ================================================================
print(f"\n{'='*65}")
print(f"DIRAC PROPAGATOR S = D⁻¹")
print(f"{'='*65}")

# Check if D is singular
det = np.linalg.det(D)
print(f"\n  det(D) = {det:.4e}")
singular = abs(det) < 1e-10

if singular:
    print(f"  D is singular — using pseudoinverse")
    S = np.linalg.pinv(D)
else:
    print(f"  D is invertible")
    S = np.linalg.inv(D)

# ================================================================
# PROPAGATOR vs DISTANCE
# ================================================================
print(f"\n{'='*65}")
print(f"PROPAGATOR vs GRAPH DISTANCE")
print(f"{'='*65}")

# For each pair of vertices, compute |S(i,j)| (Frobenius norm of 2×2 block)
prop_by_dist = {d: [] for d in range(6)}

for i in range(Nv):
    for j in range(Nv):
        if i == j:
            continue
        d = dist_matrix[i,j]
        # Extract 2×2 block
        S_block = S[2*i:2*i+2, 2*j:2*j+2]
        norm = np.linalg.norm(S_block)  # Frobenius
        prop_by_dist[d].append(norm)

print(f"\n  {'Distance':>8} {'|S| mean':>12} {'|S| std':>10} {'N pairs':>8} {'1/d':>8} {'Ratio':>8}")
print(f"  {'-'*60}")

d1_mean = None
for d in range(1, 6):
    if prop_by_dist[d]:
        vals = prop_by_dist[d]
        mean = np.mean(vals)
        std = np.std(vals)
        n = len(vals)
        inv_d = 1.0/d
        if d == 1:
            d1_mean = mean
        ratio = mean/d1_mean if d1_mean else 0
        print(f"  {d:>8} {mean:>12.6f} {std:>10.6f} {n:>8} {inv_d:>8.4f} {ratio:>8.4f}")

# Check: does |S| ∝ 1/d?
print(f"\n  If |S| ∝ 1/d, ratio column should match 1/d column")
print(f"  If |S| ∝ exp(-md), check log plot:")
for d in range(1, 6):
    if prop_by_dist[d] and d1_mean:
        mean = np.mean(prop_by_dist[d])
        if mean > 0:
            print(f"    d={d}: ln(|S|/|S₁|) = {math.log(mean/d1_mean):.4f}, expected -m×(d-1) for massive")

# ================================================================
# SPIN STRUCTURE
# ================================================================
print(f"\n{'='*65}")
print(f"SPIN STRUCTURE OF PROPAGATOR")
print(f"{'='*65}")

# For nearest neighbors, decompose S into scalar + d̂·σ
print(f"\n  Nearest-neighbor propagator decomposition:")
nn_count = 0
scalar_parts = []
vector_parts = []

for i, j in edges[:5]:  # first 5 edges as sample
    S_block = S[2*i:2*i+2, 2*j:2*j+2]
    
    # Decompose: S = a·I + b·(d̂·σ)
    d_vec = verts[j] - verts[i]
    d_hat = d_vec / np.linalg.norm(d_vec)
    dsig = sum(d_hat[k] * sigma[k] for k in range(3))
    
    # a = Tr(S)/2, b = Tr(S·(d̂·σ)†)/2
    I2 = np.eye(2, dtype=complex)
    a = np.trace(S_block) / 2
    b = np.trace(S_block @ dsig.conj().T) / 2
    
    # Reconstruction error
    recon = a * I2 + b * dsig
    err = np.linalg.norm(S_block - recon) / np.linalg.norm(S_block)
    
    scalar_parts.append(a)
    vector_parts.append(b)
    
    print(f"  Edge ({i},{j}): scalar a = {a:.6f}, vector b = {b:.6f}, recon error = {err:.2e}")

print(f"\n  Mean scalar part: {np.mean(scalar_parts):.6f}")
print(f"  Mean vector part: {np.mean(vector_parts):.6f}")
print(f"  Ratio |vector/scalar|: {abs(np.mean(vector_parts)/np.mean(scalar_parts)):.4f}")
print(f"\n  Continuum Dirac: S(x) = (γ·x̂)/(4π|x|²) + m/(4π|x|)")
print(f"  → ratio |vector/scalar| = |x|/m at distance |x|")
print(f"  Framework ratio {abs(np.mean(vector_parts)/np.mean(scalar_parts)):.4f}")

# ================================================================
# CHIRAL STRUCTURE (γ₅ analogue)
# ================================================================
print(f"\n{'='*65}")
print(f"CHIRAL STRUCTURE")
print(f"{'='*65}")

# γ₅ on the dodecahedron = the matrix that distinguishes the two 
# pentagonal orientations (C₅ vs C₅')
# Simplest analogue: σ_z in the spin basis

gamma5 = np.zeros((2*Nv, 2*Nv), dtype=complex)
for i in range(Nv):
    gamma5[2*i, 2*i] = 1      # spin up
    gamma5[2*i+1, 2*i+1] = -1  # spin down

# Check anticommutation {D, γ₅} = ?
anticomm = D @ gamma5 + gamma5 @ D
anticomm_norm = np.linalg.norm(anticomm)
print(f"\n  ||{{D, γ₅}}|| = {anticomm_norm:.6f}")
print(f"  ||D|| = {np.linalg.norm(D):.6f}")
print(f"  Ratio: {anticomm_norm/np.linalg.norm(D):.6f}")
if anticomm_norm / np.linalg.norm(D) < 0.01:
    print(f"  → CHIRAL: D anticommutes with γ₅ (massless Dirac)")
elif anticomm_norm / np.linalg.norm(D) < 0.1:
    print(f"  → APPROXIMATELY chiral")
else:
    print(f"  → NOT chiral — the discrete geometry breaks chiral symmetry")
    print(f"    This is EXPECTED: the dodecahedron has a preferred handedness")
    print(f"    Chiral symmetry breaking from geometry, not from a Higgs field")

# ================================================================
# KEY A₅ QUANTITIES
# ================================================================
print(f"\n{'='*65}")
print(f"MATCHING TO A₅ QUANTITIES")
print(f"{'='*65}")

if pos_evals:
    unique = sorted(set(round(e, 3) for e in pos_evals))
    
    a5_candidates = {
        '3-√5': 3-sqrt5,
        '√(3-√5)': math.sqrt(3-sqrt5),
        '1/φ': 1/phi,
        'φ-1': phi-1,
        '√5-1': sqrt5-1,
        '1': 1.0,
        '√3': math.sqrt(3),
        '2': 2.0,
        '3': 3.0,
        'φ': phi,
        '√5': sqrt5,
        'φ²': phi**2,
        '3+√5': 3+sqrt5,
        '√(3+√5)': math.sqrt(3+sqrt5),
    }
    
    print(f"\n  Eigenvalue → A₅ quantity matching:")
    for ev in unique:
        best_name = min(a5_candidates.items(), key=lambda x: abs(x[1]-ev))
        err = abs(best_name[1]-ev)/ev*100 if ev > 0 else 999
        mark = " ← EXACT" if err < 0.1 else (" ← close" if err < 2 else "")
        print(f"    E = {ev:.4f} ≈ {best_name[0]} = {best_name[1]:.4f} ({err:.2f}%){mark}")

# ================================================================
# CONNECTION TO LAPLACIAN
# ================================================================
print(f"\n{'='*65}")
print(f"D² vs LAPLACIAN")
print(f"{'='*65}")

D2 = D @ D
# The Laplacian in 40×40 space (block diagonal with L in each spin sector)
A_mat = np.zeros((Nv, Nv))
for i in range(Nv):
    for j in adj[i]:
        A_mat[i,j] = 1
L = np.diag(A_mat.sum(axis=1)) - A_mat

# Build block diagonal L ⊗ I₂
L_block = np.zeros((2*Nv, 2*Nv), dtype=complex)
for i in range(Nv):
    for j in range(Nv):
        L_block[2*i, 2*j] = L[i,j]
        L_block[2*i+1, 2*j+1] = L[i,j]

# Compare D² with L ⊗ I₂
diff = D2 + L_block  # D² should be -L (up to sign convention)
ratio = np.linalg.norm(diff) / np.linalg.norm(L_block)
print(f"\n  ||D² + L⊗I₂|| / ||L⊗I₂|| = {ratio:.6f}")
if ratio < 0.01:
    print(f"  → D² = −L ⊗ I₂ (Dirac squares to Laplacian) ✓")
elif ratio < 0.5:
    print(f"  → D² ≈ −L ⊗ I₂ + corrections")
    # What are the corrections?
    corr = D2 + L_block
    corr_evals = np.sort(np.real(np.linalg.eigvals(corr)))
    print(f"    Correction eigenvalues range: [{corr_evals[0]:.4f}, {corr_evals[-1]:.4f}]")
    print(f"    These are the SPIN-ORBIT corrections from the dodecahedral geometry")
else:
    print(f"  → D² ≠ −L ⊗ I₂ (significant spin-orbit coupling)")

# Laplacian eigenvalues for comparison
L_evals = np.sort(np.linalg.eigvalsh(L))
print(f"\n  Laplacian eigenvalues: {np.round(L_evals, 3)}")
print(f"  Dirac eigenvalues²: {np.round(np.sort(evals_sorted**2)[:10], 3)}")

print(f"\n{'='*65}")
print(f"DONE")
print(f"{'='*65}")
