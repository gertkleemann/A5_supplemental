#!/usr/bin/env python3
"""
KLEIN-NISHINA v2: VERTEX-RESOLVED COMPTON SCATTERING
=====================================================
Computes the Compton scattering amplitude from the dodecahedral
vertex geometry and lattice Green's function. No QED input.

The angular distribution comes from:
  1. Photon absorbed at vertex v1 (coupling to edge direction d_a)
  2. Electron propagates from v1 to v2 via lattice Green's function
  3. Photon emitted at vertex v2 (coupling to edge direction d_b)
  4. Sum over all (v1,v2) pairs
  5. The interference between vertices produces the angular distribution

Requires: numpy (no numba needed)
Usage: python klein_nishina_v2.py
"""

import numpy as np
import json, math

phi = (1 + np.sqrt(5)) / 2

print("=" * 65)
print("KLEIN-NISHINA v2: VERTEX-RESOLVED COMPTON SCATTERING")
print("=" * 65)

# ================================================================
# DODECAHEDRON GEOMETRY
# ================================================================

def dodecahedron_vertices():
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            for s3 in [1, -1]:
                verts.append([s1, s2, s3])
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1/phi, s2*phi])
            verts.append([s1/phi, s2*phi, 0])
            verts.append([s1*phi, 0, s2/phi])
    return np.array(verts, dtype=np.float64)

VERTS = dodecahedron_vertices()

# Find edges (minimum distance pairs)
def find_edges(verts):
    n = len(verts)
    dists = []
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(verts[i] - verts[j])
            dists.append((d, i, j))
    dists.sort()
    edge_len = dists[0][0]
    edges = [(i, j) for d, i, j in dists if abs(d - edge_len) < 0.01]
    return edges, edge_len

EDGES, EDGE_LEN = find_edges(VERTS)
print(f"Dodecahedron: {len(VERTS)} vertices, {len(EDGES)} edges")
print(f"Edge length: {EDGE_LEN:.4f}")

# Build adjacency and vertex-edge structure
ADJ = {i: [] for i in range(20)}
VERT_EDGE_DIRS = {i: [] for i in range(20)}  # direction vectors of edges at each vertex

for ei, (u, v) in enumerate(EDGES):
    ADJ[u].append(v)
    ADJ[v].append(u)
    d_uv = VERTS[v] - VERTS[u]
    d_uv = d_uv / np.linalg.norm(d_uv)
    VERT_EDGE_DIRS[u].append(d_uv)
    VERT_EDGE_DIRS[v].append(-d_uv)

# Verify: degree 3 and edge angle 108°
for v in range(20):
    assert len(ADJ[v]) == 3, f"Vertex {v} degree {len(ADJ[v])}"
    dirs = VERT_EDGE_DIRS[v]
    for i in range(3):
        for j in range(i+1, 3):
            cos_a = np.dot(dirs[i], dirs[j])
            angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
            assert abs(angle - 108) < 0.1, f"Edge angle {angle} at vertex {v}"

print("All vertices degree 3, all edge angles 108° ✓")

# ================================================================
# GRAPH LAPLACIAN AND GREEN'S FUNCTION
# ================================================================

print("\n--- GRAPH LAPLACIAN ---")

# Adjacency matrix
A = np.zeros((20, 20))
for u, v in EDGES:
    A[u, v] = 1
    A[v, u] = 1

# Laplacian: L = D - A (degree matrix minus adjacency)
D = np.diag(np.sum(A, axis=1))  # D = 3*I for regular graph
L = D - A

# Eigendecomposition
evals_L, evecs_L = np.linalg.eigh(L)
print(f"Laplacian eigenvalues: {np.round(evals_L, 4)}")

# Pseudoinverse (Green's function): G = L⁺
# Zero out the null eigenvalue, invert the rest
evals_inv = np.zeros_like(evals_L)
for i in range(len(evals_L)):
    if abs(evals_L[i]) > 1e-10:
        evals_inv[i] = 1.0 / evals_L[i]

G = evecs_L @ np.diag(evals_inv) @ evecs_L.T

print(f"Green's function G(0,0) = {G[0,0]:.4f}")
print(f"Green's function G(0,1) = {G[0, ADJ[0][0]]:.4f}")

# Graph distances
from collections import deque
def bfs_distances(adj, start):
    dist = [-1] * 20
    dist[start] = 0
    q = deque([start])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist

# Shell structure
d0 = bfs_distances(ADJ, 0)
shells = {}
for v in range(20):
    shells.setdefault(d0[v], []).append(v)
print(f"Shell structure from vertex 0: {[(d, len(vs)) for d, vs in sorted(shells.items())]}")

# ================================================================
# VERTEX SCATTERING TENSOR
# ================================================================

print("\n--- VERTEX SCATTERING TENSOR ---")

# At each vertex v, construct the edge tensor:
#   E(v) = Σ_a d_a ⊗ d_a  (sum over 3 edge directions)
# This is a 3×3 matrix that couples incoming to outgoing polarisation.

# For the full scattering, the Compton tensor is:
#   S_ij = Σ_{v1,v2} Σ_{a,b} d_a^(v1)_i × G(v1,v2) × d_b^(v2)_j
#
# where d_a^(v1) is edge direction a at vertex v1.
#
# This is: S = Σ_{v1,v2} E(v1) × G(v1,v2) × E(v2)
# but since E(v) is 3×3 and we sum the dot products, it's:
#   S = Σ_{v1} D(v1)^T × G_row(v1) × D(v2) for v2
# where D(v) is the 3×N_edges matrix of edge directions at v.

# Build edge direction matrices (3×3, one column per edge direction)
D_v = {}  # D_v[v] is 3×3 matrix of edge direction columns
for v in range(20):
    dirs = VERT_EDGE_DIRS[v]
    D_v[v] = np.column_stack(dirs)  # 3×3

# Compute the scattering tensor S (3×3)
# S = Σ_{v1,v2} D(v1) @ D(v1)^T × G(v1,v2) (× D(v2) @ D(v2)^T for outgoing)
# Actually the correct contraction is:
# S_ij = Σ_{v1,v2} [Σ_a d_a^(v1)_i d_a^(v1)_k] × G(v1,v2) × [Σ_b d_b^(v2)_k d_b^(v2)_j]
# Hmm, this involves the electron propagator connecting the two vertices.
# The incoming photon couples at v1 through ε_in · (Σ_a d_a × ...) 
# The outgoing photon couples at v2 through ε_out · (Σ_b d_b × ...)

# Simplest form:
# M(ε_in, ε_out) = Σ_{v1,v2} (ε_in · p1(v1)) × G(v1,v2) × (ε_out · p2(v2))
# where p1, p2 are the photon coupling vectors at each vertex.
#
# The photon coupling at vertex v: the photon can enter on any of the 3 edges.
# The coupling is: p(v) = Σ_a d_a (sum of edge direction unit vectors)
# This is the TOTAL photon coupling vector at vertex v.

# Compute total photon coupling vectors
P = np.zeros((20, 3))
for v in range(20):
    P[v] = sum(VERT_EDGE_DIRS[v])

print(f"Photon coupling vectors (sample):")
print(f"  P(0) = {P[0].round(4)}, |P(0)| = {np.linalg.norm(P[0]):.4f}")
print(f"  P(1) = {P[1].round(4)}, |P(1)| = {np.linalg.norm(P[1]):.4f}")

# The scattering tensor
S = np.zeros((3, 3))
for v1 in range(20):
    for v2 in range(20):
        S += np.outer(P[v1], P[v2]) * G[v1, v2]

print(f"\nScattering tensor S (should be ~ proportional to I for isotropic):")
print(f"  {S.round(6)}")

evals_S, evecs_S = np.linalg.eigh(S)
print(f"  Eigenvalues: {evals_S.round(6)}")
print(f"  Anisotropy: {(max(evals_S)-min(evals_S))/np.mean(np.abs(evals_S))*100:.2f}%")

# ================================================================
# FULL VERTEX-RESOLVED CALCULATION
# ================================================================

print("\n--- FULL VERTEX-RESOLVED COMPTON AMPLITUDE ---")

# Better approach: don't sum P at each vertex. Instead, keep each edge separate.
# The amplitude for photon entering on edge a at v1, exiting on edge b at v2:
#   M_ab = d_a^(v1) · ε_in × G(v1,v2) × d_b^(v2) · ε_out
#
# Total: M = Σ_{v1,v2,a,b} (d_a · ε_in)(d_b · ε_out) G(v1,v2)
#
# This factorises:
#   M = ε_in · [Σ_{v1,a} d_a^(v1) × (Σ_{v2,b} G(v1,v2) × d_b^(v2))] · ε_out
#
# But for the CROSS-SECTION, we need:
#   dσ/dΩ ∝ Σ_pol |M|² = Σ_pol |ε_in · T · ε_out|²
#
# where T is the 3×3 scattering tensor.

# More refined: separate s-channel and u-channel
# s-channel: photon in at v1, electron propagates v1→v2, photon out at v2
# u-channel: photon out at v1, electron propagates v1→v2, photon in at v2
# (crossed diagram)

# For now, compute just the s-channel tensor and check angular distribution.

# The s-channel tensor:
# T^s_ij = Σ_{v1,v2} [D(v1) D(v1)^T]_ik × G(v1,v2) × [D(v2) D(v2)^T]_kj
# where [DD^T]_ik = Σ_a d_a_i d_a_k (the edge projector at each vertex)

# Edge projector at each vertex
E_proj = {}
for v in range(20):
    E_proj[v] = D_v[v] @ D_v[v].T  # 3×3

print("Edge projectors (sample):")
print(f"  E(0) eigenvalues: {np.linalg.eigvalsh(E_proj[0]).round(4)}")

# Full tensor: T_s[i,j] = Σ_{v1,v2,k} E(v1)_{ik} × G(v1,v2) × E(v2)_{kj}
T_s = np.zeros((3, 3))
for v1 in range(20):
    for v2 in range(20):
        T_s += E_proj[v1] @ E_proj[v2] * G[v1, v2]

# u-channel: swap v1 and v2 in the coupling
# T_u[i,j] = Σ_{v1,v2,k} E(v2)_{ik} × G(v1,v2) × E(v1)_{kj}
T_u = np.zeros((3, 3))
for v1 in range(20):
    for v2 in range(20):
        T_u += E_proj[v2] @ E_proj[v1] * G[v1, v2]

# Total Compton tensor (s + u channel)
T_total = T_s + T_u

print(f"\ns-channel tensor eigenvalues: {np.linalg.eigvalsh(T_s).round(6)}")
print(f"u-channel tensor eigenvalues: {np.linalg.eigvalsh(T_u).round(6)}")
print(f"Total tensor eigenvalues:     {np.linalg.eigvalsh(T_total).round(6)}")

# ================================================================
# ANGULAR DISTRIBUTION
# ================================================================

print(f"\n{'='*65}")
print("ANGULAR DISTRIBUTION: FRAMEWORK vs THOMSON")
print(f"{'='*65}")

# For incoming photon along z-axis:
# k_in = (0, 0, 1)
# k_out = (sin θ, 0, cos θ)
# ε_in can be (1,0,0) or (0,1,0) — two polarisations
# ε_out perpendicular to k_out

# For each θ, compute:
# dσ/dΩ = Σ_{ε_in, ε_out} |ε_in · T · ε_out|²

k_in = np.array([0, 0, 1.0])

print(f"\n  {'θ (deg)':>8} {'Framework':>12} {'Thomson':>12} {'ratio':>8}")
print(f"  {'-'*44}")

angles = np.linspace(0, 180, 37)  # every 5 degrees
framework_vals = []
thomson_vals = []

for theta_deg in angles:
    theta = np.radians(theta_deg)
    
    # Outgoing photon direction
    k_out = np.array([np.sin(theta), 0, np.cos(theta)])
    
    # Polarisation vectors (perpendicular to k)
    # For k_in along z: ε1 = x, ε2 = y
    eps_in = [np.array([1, 0, 0]), np.array([0, 1, 0])]
    
    # For k_out: need two vectors perpendicular to k_out
    if abs(np.sin(theta)) > 1e-10:
        eps_out_1 = np.array([np.cos(theta), 0, -np.sin(theta)])  # in xz plane
        eps_out_2 = np.array([0, 1, 0])  # y direction
    else:
        eps_out_1 = np.array([1, 0, 0])
        eps_out_2 = np.array([0, 1, 0])
    eps_out = [eps_out_1, eps_out_2]
    
    # Sum over polarisations
    dsigma = 0.0
    for ei in eps_in:
        for eo in eps_out:
            amp = ei @ T_total @ eo
            dsigma += amp**2
    
    # Thomson prediction: (1 + cos²θ)
    thomson = 1 + np.cos(theta)**2
    
    framework_vals.append(dsigma)
    thomson_vals.append(thomson)

# Normalise framework to match Thomson at θ=90°
fw = np.array(framework_vals)
th = np.array(thomson_vals)

# Normalise so that fw(90°) = th(90°) = 1
idx90 = np.argmin(np.abs(angles - 90))
if fw[idx90] > 0:
    fw_norm = fw / fw[idx90]
    th_norm = th / th[idx90]  # = (1+cos²θ)/(1+0) = (1+cos²θ)
else:
    fw_norm = fw / np.max(fw)
    th_norm = th / np.max(th)

for i, theta_deg in enumerate(angles):
    if i % 3 == 0:  # print every 15 degrees
        ratio = fw_norm[i] / th_norm[i] if th_norm[i] > 0 else 0
        print(f"  {theta_deg:>8.0f} {fw_norm[i]:>12.6f} {th_norm[i]:>12.6f} {ratio:>8.4f}")

# Chi-squared
residuals = fw_norm - th_norm
chi2 = np.sum(residuals**2) / len(residuals)
print(f"\n  Mean squared residual: {chi2:.8f}")
print(f"  RMS deviation: {np.sqrt(chi2):.6f}")

if np.sqrt(chi2) < 0.01:
    print(f"  RESULT: Thomson distribution (1+cos²θ) REPRODUCED ✓")
elif np.sqrt(chi2) < 0.1:
    print(f"  RESULT: Approximate match — deviations may be Klein-Nishina corrections")
else:
    print(f"  RESULT: Does NOT match Thomson — investigating structure...")
    print(f"  Forward/backward ratio: {fw_norm[0]/fw_norm[-1]:.4f} (Thomson: 1.0000)")
    print(f"  Forward/90° ratio: {fw_norm[0]/fw_norm[idx90]:.4f} (Thomson: 2.0000)")

# ================================================================
# MOMENTUM-DEPENDENT (KLEIN-NISHINA REGIME)
# ================================================================

print(f"\n{'='*65}")
print("MOMENTUM-DEPENDENT AMPLITUDES (KLEIN-NISHINA)")
print(f"{'='*65}")

# Add phase factors exp(i k · r_v) for finite photon momentum
# k = ω/c in natural units. The lattice spacing is the Planck length.
# For "laboratory" energies, k × a << 1 (Thomson limit).
# For Planck-scale energies, k × a ~ 1 (Klein-Nishina).
#
# We scan ka from 0 (Thomson) to π (lattice Nyquist).

print(f"\n  Scanning ka (lattice momentum) from 0 to 2:")
print(f"  {'ka':>6} {'fw(0°)/fw(90°)':>16} {'Thomson ratio':>14} {'KN deviation':>14}")
print(f"  {'-'*54}")

for ka in [0, 0.01, 0.1, 0.5, 1.0, 1.5, 2.0]:
    k_mag = ka / EDGE_LEN  # momentum in units of 1/vertex_spacing
    
    fw_ka = []
    for theta_deg in [0, 90, 180]:
        theta = np.radians(theta_deg)
        k_in_vec = np.array([0, 0, k_mag])
        k_out_vec = k_mag * np.array([np.sin(theta), 0, np.cos(theta)])
        q = k_out_vec - k_in_vec  # momentum transfer
        
        # Phase-weighted scattering tensor
        T_ka = np.zeros((3, 3))
        for v1 in range(20):
            for v2 in range(20):
                phase = np.exp(1j * np.dot(q, VERTS[v1] - VERTS[v2]))
                T_ka += E_proj[v1] @ E_proj[v2] * G[v1, v2] * phase.real
        
        # Polarisation sum
        if abs(np.sin(theta)) > 1e-10:
            eps_out_1 = np.array([np.cos(theta), 0, -np.sin(theta)])
            eps_out_2 = np.array([0, 1, 0])
        else:
            eps_out_1 = np.array([1, 0, 0])
            eps_out_2 = np.array([0, 1, 0])
        
        dsigma_ka = 0
        for ei in [np.array([1,0,0]), np.array([0,1,0])]:
            for eo in [eps_out_1, eps_out_2]:
                dsigma_ka += (ei @ T_ka @ eo)**2
        
        fw_ka.append(dsigma_ka)
    
    if fw_ka[1] > 0:
        ratio_fw = fw_ka[0] / fw_ka[1]
        ratio_th = 2.0  # Thomson: (1+1)/(1+0) = 2
        deviation = (ratio_fw - ratio_th) / ratio_th * 100
        print(f"  {ka:>6.2f} {ratio_fw:>16.6f} {ratio_th:>14.1f} {deviation:>13.2f}%")
    else:
        print(f"  {ka:>6.2f} {'N/A':>16}")

# ================================================================
# 5-DESIGN VERIFICATION
# ================================================================

print(f"\n{'='*65}")
print("5-DESIGN VERIFICATION")
print(f"{'='*65}")

# The dodecahedron is a spherical 5-design.
# This means: (1/20) Σ_v f(r_v/|r_v|) = (1/4π) ∫ f(n) dΩ
# for any polynomial f of degree ≤ 5.

# Test: Σ_v (n_v)_i (n_v)_j = (20/3) δ_ij
normals = VERTS / np.linalg.norm(VERTS, axis=1, keepdims=True)
test_2 = normals.T @ normals  # 3×3
print(f"\n  Σ n_i n_j (should be 20/3 × I = {20/3:.4f} × I):")
print(f"    Diagonal: {np.diag(test_2).round(4)}")
print(f"    Off-diag max: {np.max(np.abs(test_2 - np.diag(np.diag(test_2)))):.6f}")

# Test degree 4: Σ (n_i)^4
test_4 = np.sum(normals**4, axis=0)
# For 5-design: should equal 20 × 3/(4π) × ∫ n_z^4 dΩ = 20 × 1/5 = 4
print(f"  Σ n_i^4: {test_4.round(4)} (5-design predicts: {20*3/15:.4f})")

# ================================================================
# SAVE
# ================================================================

output = {
    'T_total_eigenvalues': np.linalg.eigvalsh(T_total).tolist(),
    'T_total': T_total.tolist(),
    'angles_deg': angles.tolist(),
    'framework_norm': fw_norm.tolist(),
    'thomson_norm': th_norm.tolist(),
    'rms_deviation': float(np.sqrt(chi2)),
    'laplacian_eigenvalues': evals_L.tolist(),
    'greens_function_diagonal': float(G[0,0]),
}

with open('klein_nishina_v2_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to klein_nishina_v2_results.json")
print("Done.")
