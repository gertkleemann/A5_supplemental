#!/usr/bin/env python3
"""
DODECAHEDRAL FRAMEWORK: LIGHT HADRON SPECTRUM
==============================================
Computes the 32×32 face transfer matrix in the χ₃/χ₃' sector.
Each face edge carries either χ₃ or χ₃', giving 2⁵=32 configurations.
The eigenvalues give the light hadron mass spectrum (pion, rho, etc.).

RUNTIME: ~25 minutes on a modern laptop
OUTPUT: eigenvalues, mass spectrum, f_π, hadronic VP

Run: python3 hadron_spectrum.py
"""
import numpy as np
import math
import time
import json
from itertools import product as iprod
from collections import deque

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / 137.036
Lambda_QCD = 332  # MeV
N_IRREPS = 5

dims = [1, 3, 3, 4, 5]  # χ₁, χ₃, χ₃', χ₄, χ₅
names = ['χ₁', 'χ₃', 'χ₃\'', 'χ₄', 'χ₅']
chars = [
    [1, 1, 1, 1, 1],
    [3, -1, 0, phi, 1-phi],
    [3, -1, 0, 1-phi, phi],
    [4, 0, 1, -1, -1],
    [5, 1, -1, 0, 0],
]
class_sizes = [1, 15, 20, 12, 12]
G_ORDER = 60

# ================================================================
# BUILD 3j SYMBOL TABLE
# ================================================================
print("Building 3j symbol table...")
threej = np.zeros((N_IRREPS, N_IRREPS, N_IRREPS))
for a in range(N_IRREPS):
    for b in range(N_IRREPS):
        for c in range(N_IRREPS):
            threej[a, b, c] = round(
                sum(class_sizes[k] * chars[a][k] * chars[b][k] * chars[c][k] 
                    for k in range(5)) / G_ORDER
            )

# ================================================================
# BUILD DODECAHEDRON
# ================================================================
print("Building dodecahedron...")
dv = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        for s3 in [1, -1]:
            dv.append([s1, s2, s3])
for s1 in [1, -1]:
    for s2 in [1, -1]:
        dv.append([0, s1/phi, s2*phi])
        dv.append([s1/phi, s2*phi, 0])
        dv.append([s2*phi, 0, s1/phi])
dv = np.array(dv)
Nv = 20

edges = []
adj = [[] for _ in range(Nv)]
for i in range(Nv):
    for j in range(i+1, Nv):
        if abs(np.linalg.norm(dv[i] - dv[j]) - 2/phi) < 0.01:
            eidx = len(edges)
            edges.append((i, j))
            adj[i].append(eidx)
            adj[j].append(eidx)
Ne = len(edges)
print(f"  Vertices: {Nv}, Edges: {Ne}")

# ================================================================
# FIND OPPOSITE FACES
# ================================================================
print("Finding pentagonal faces...")

def get_edge_idx(v1, v2):
    key = (min(v1, v2), max(v1, v2))
    for i, e in enumerate(edges):
        if e == key:
            return i
    return None

# Find faces by walking pentagonal cycles
faces = []
for start in range(Nv):
    nbs_s = set()
    for e in adj[start]:
        a, b = edges[e]
        nbs_s.add(b if a == start else a)
    for v1 in nbs_s:
        nbs_1 = set()
        for e in adj[v1]:
            a, b = edges[e]
            nbs_1.add(b if a == v1 else a)
        for v2 in (nbs_1 - {start}):
            nbs_2 = set()
            for e in adj[v2]:
                a, b = edges[e]
                nbs_2.add(b if a == v2 else a)
            for v3 in (nbs_2 - {start, v1}):
                nbs_3 = set()
                for e in adj[v3]:
                    a, b = edges[e]
                    nbs_3.add(b if a == v3 else a)
                for v4 in (nbs_3 - {start, v1, v2}):
                    if start in set():
                        pass
                    nbs_4 = set()
                    for e in adj[v4]:
                        a, b = edges[e]
                        nbs_4.add(b if a == v4 else a)
                    if start in nbs_4:
                        f = tuple(sorted([start, v1, v2, v3, v4]))
                        if f not in faces:
                            faces.append(f)

print(f"  Found {len(faces)} faces")

# Find opposite face pairs
opp_pairs = []
for i in range(len(faces)):
    for j in range(i+1, len(faces)):
        if len(set(faces[i]) & set(faces[j])) == 0:
            opp_pairs.append((i, j))
            break
    if opp_pairs:
        break

face_A_verts = faces[opp_pairs[0][0]]
face_B_verts = faces[opp_pairs[0][1]]

def face_edge_list(fverts):
    fv = sorted(fverts)
    result = []
    for i in range(len(fv)):
        for j in range(i+1, len(fv)):
            eidx = get_edge_idx(fv[i], fv[j])
            if eidx is not None:
                result.append(eidx)
    return result

edges_A = face_edge_list(face_A_verts)
edges_B = face_edge_list(face_B_verts)
boundary = set(edges_A + edges_B)

print(f"  Face A: vertices {sorted(face_A_verts)}, edges {edges_A}")
print(f"  Face B: vertices {sorted(face_B_verts)}, edges {edges_B}")
print(f"  Boundary: {len(boundary)} edges, Internal: {Ne - len(boundary)} edges")

# BFS vertex ordering from face A
visited = [False] * Nv
order = []
q = deque()
for v in sorted(face_A_verts):
    q.append(v)
    visited[v] = True
while q:
    v = q.popleft()
    order.append(v)
    for eidx in adj[v]:
        i, j = edges[eidx]
        nb = j if i == v else i
        if not visited[nb]:
            visited[nb] = True
            q.append(nb)

print(f"  BFS order: {order}")

# ================================================================
# TENSOR NETWORK CONTRACTION: SINGLE ELEMENT
# ================================================================
def compute_T(la, lb):
    """
    Compute transfer matrix element T[la, lb].
    la, lb: tuples of 5 irrep labels for faces A, B.
    Returns: scalar (the transfer amplitude).
    """
    fixed = {}
    for idx, eidx in enumerate(edges_A):
        fixed[eidx] = la[idx]
    for idx, eidx in enumerate(edges_B):
        fixed[eidx] = lb[idx]
    
    processed = set()
    open_edges = []
    state = {(): 1.0}
    
    for v in order:
        v_edges = adj[v]
        new_e = []
        closing_e = []
        
        for eidx in v_edges:
            i, j = edges[eidx]
            other = j if i == v else i
            typ = 'f' if eidx in fixed else 'r'  # fixed or free
            if other in processed:
                closing_e.append((typ, eidx))
            else:
                new_e.append((typ, eidx))
        
        edge_pos = {e: i for i, e in enumerate(open_edges)}
        cfp = sorted(
            [(eidx, edge_pos[eidx]) for typ, eidx in closing_e if typ == 'r'],
            key=lambda x: -x[1]
        )
        
        new_state = {}
        free_new = [eidx for typ, eidx in new_e if typ == 'r']
        
        for fl in iprod(range(N_IRREPS), repeat=len(free_new)):
            for ok, ow in state.items():
                if abs(ow) < 1e-15:
                    continue
                
                # Build edge labels
                el = {}
                fi = 0
                for typ, eidx in new_e:
                    if typ == 'f':
                        el[eidx] = fixed[eidx]
                    else:
                        el[eidx] = fl[fi]
                        fi += 1
                for typ, eidx in closing_e:
                    if typ == 'f':
                        el[eidx] = fixed[eidx]
                    else:
                        el[eidx] = ok[edge_pos[eidx]]
                
                # Vertex weight (3j symbol)
                jabc = tuple(el[e] for e in v_edges)
                vw = threej[jabc]
                if vw == 0:
                    continue
                
                # Edge weight for free new edges
                ew = 1.0
                for eidx in free_new:
                    ew *= dims[el[eidx]]
                
                # Build new state key
                nkl = list(ok)
                for _, pos in cfp:
                    nkl.pop(pos)
                for eidx in free_new:
                    nkl.append(el[eidx])
                nk = tuple(nkl)
                
                new_state[nk] = new_state.get(nk, 0.0) + ow * vw * ew
        
        new_open = [e for e in open_edges 
                     if not any(eidx == e for typ, eidx in closing_e if typ == 'r')]
        new_open.extend(free_new)
        open_edges = new_open
        processed.add(v)
        state = new_state
    
    return sum(state.values())

# ================================================================
# QUICK VALIDATION
# ================================================================
print("\nValidating tensor network...")
t0 = time.time()
test = compute_T((0, 0, 0, 0, 0), (0, 0, 0, 0, 0))
dt = time.time() - t0
print(f"  T(χ₁⁵, χ₁⁵) = {test:.2e} [{dt:.2f}s]")

t0 = time.time()
test2 = compute_T((1, 1, 1, 1, 1), (1, 1, 1, 1, 1))
dt = time.time() - t0
print(f"  T(χ₃⁵, χ₃⁵) = {test2:.2e} [{dt:.2f}s]")

# ================================================================
# COMPUTE 32×32 χ₃/χ₃' MIXED BLOCK
# ================================================================
# Each face edge = χ₃(1) or χ₃'(2). 2⁵ = 32 configs per face.
configs = list(iprod([1, 2], repeat=5))
n_configs = len(configs)

print(f"\n{'='*60}")
print(f"COMPUTING {n_configs}×{n_configs} = {n_configs**2} TRANSFER MATRIX ELEMENTS")
print(f"{'='*60}")

T_matrix = np.zeros((n_configs, n_configs))
t_start = time.time()
count = 0
total = n_configs ** 2

for ia, ca in enumerate(configs):
    for ib, cb in enumerate(configs):
        T_matrix[ia, ib] = compute_T(ca, cb)
        count += 1
    
    elapsed = time.time() - t_start
    rate = count / elapsed if elapsed > 0 else 0
    remaining = (total - count) / rate if rate > 0 else 0
    pct = count / total * 100
    
    config_str = ''.join(['3' if x == 1 else '3\'' for x in ca])
    print(f"  Row {ia:2d}/32 [{config_str}]: "
          f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining ({pct:.0f}%)")

total_time = time.time() - t_start
print(f"\nTotal computation time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# ================================================================
# EIGENVALUE ANALYSIS
# ================================================================
print(f"\n{'='*60}")
print("EIGENVALUE ANALYSIS")
print(f"{'='*60}")

eigenvalues = sorted(np.linalg.eigvalsh(T_matrix), reverse=True)
T_max = eigenvalues[0]

print(f"\nAll {n_configs} eigenvalues:")
for i, ev in enumerate(eigenvalues):
    if ev > 0:
        gap = -math.log(ev / T_max)
        mass_MeV = gap * Lambda_QCD
        print(f"  λ_{i:2d} = {ev:>14.4e}  gap = {gap:>8.4f}  "
              f"mass = {mass_MeV:>8.0f} MeV")
    else:
        print(f"  λ_{i:2d} = {ev:>14.4e}  (negative — interference)")

# ================================================================
# HADRON MASS SPECTRUM
# ================================================================
print(f"\n{'='*60}")
print("HADRON MASS SPECTRUM")
print(f"{'='*60}")

# Extract masses from positive eigenvalues
masses = []
for i, ev in enumerate(eigenvalues):
    if ev > 0 and ev < T_max * 0.9999:
        gap = -math.log(ev / T_max)
        mass = gap * Lambda_QCD
        masses.append((i, gap, mass, ev))

print(f"\n  {'Level':>6} {'Gap':>8} {'Mass (MeV)':>12} {'Possible ID':>20}")
print(f"  {'-'*50}")

known_hadrons = [
    (135, 'π⁰'), (140, 'π±'), (494, 'K±'), (498, 'K⁰'),
    (548, 'η'), (775, 'ρ'), (782, 'ω'), (958, 'η\''),
    (1020, 'φ'), (1232, 'Δ'), (1275, 'f₂')
]

for i, gap, mass, ev in masses[:15]:
    # Find closest known hadron
    closest = min(known_hadrons, key=lambda h: abs(h[0] - mass))
    match = f"{closest[1]} ({closest[0]} MeV)" if abs(closest[0] - mass) / closest[0] < 0.3 else "?"
    print(f"  {i:>6} {gap:>8.4f} {mass:>12.0f} {match:>20}")

# ================================================================
# PION DECAY CONSTANT f_π
# ================================================================
print(f"\n{'='*60}")
print("PION DECAY CONSTANT")
print(f"{'='*60}")

# f_π comes from the overlap of the pion eigenvector with the axial current
# The axial current is the antisymmetric combination of χ₃ and χ₃':
# J_A ∝ (χ₃ - χ₃') on each edge

# Build the axial current vector in the 32-config basis
axial_vec = np.zeros(n_configs)
for ic, cfg in enumerate(configs):
    # Count χ₃ minus χ₃' edges: axial current weight
    n3 = sum(1 for x in cfg if x == 1)
    n3p = sum(1 for x in cfg if x == 2)
    axial_vec[ic] = (n3 - n3p) / 5.0  # normalized

# Get eigenvectors
eigenvalues_full, eigenvectors = np.linalg.eigh(T_matrix)
# Sort by eigenvalue (descending)
idx_sort = np.argsort(eigenvalues_full)[::-1]
eigenvalues_sorted = eigenvalues_full[idx_sort]
eigenvectors_sorted = eigenvectors[:, idx_sort]

# f_π = overlap of first EXCITED state with axial current
# (ground state = vacuum, first excited = pion)
for i in range(min(5, n_configs)):
    overlap = abs(np.dot(axial_vec, eigenvectors_sorted[:, i]))
    print(f"  |⟨J_A|ψ_{i}⟩| = {overlap:.6f}")

# f_π in lattice units = overlap × normalization
# Physical f_π = f_lat × Λ_QCD
if len(masses) > 0:
    pion_idx = 1  # first excited state
    f_pi_overlap = abs(np.dot(axial_vec, eigenvectors_sorted[:, pion_idx]))
    f_pi_lattice = f_pi_overlap * math.sqrt(dims[1])  # χ₃ dimension factor
    f_pi_phys = f_pi_lattice * Lambda_QCD
    print(f"\n  f_π (lattice) = {f_pi_lattice:.6f}")
    print(f"  f_π (physical) = {f_pi_phys:.1f} MeV")
    print(f"  Observed: 93 MeV")
    if f_pi_phys > 0:
        print(f"  Agreement: {abs(f_pi_phys - 93) / 93 * 100:.0f}%")

# ================================================================
# PION MASS FROM GMOR
# ================================================================
print(f"\n{'='*60}")
print("PION MASS FROM GMOR RELATION")
print(f"{'='*60}")

m_q_current = alpha * math.sqrt(3 - sqrt5) * Lambda_QCD  # MeV
chiral_condensate_cube = (0.716 * Lambda_QCD) ** 3  # MeV³

print(f"  m_q(current) = {m_q_current:.2f} MeV (framework)")
print(f"  ⟨ψ̄ψ⟩^(1/3) = {0.716 * Lambda_QCD:.0f} MeV (framework)")

if f_pi_phys > 0:
    m_pi_sq = 2 * m_q_current * chiral_condensate_cube / f_pi_phys**2
    m_pi = math.sqrt(abs(m_pi_sq))
    print(f"  f_π = {f_pi_phys:.1f} MeV (from eigenvector)")
    print(f"  m_π(GMOR) = √(2 m_q ⟨ψ̄ψ⟩ / f_π²) = {m_pi:.0f} MeV")
    print(f"  Observed: 140 MeV")
    print(f"  Agreement: {abs(m_pi - 140) / 140 * 100:.0f}%")

# Also use the direct mass gap if available
if len(masses) > 0:
    print(f"\n  m_π(direct from gap) = {masses[0][2]:.0f} MeV")

# ================================================================
# HADRONIC VACUUM POLARISATION
# ================================================================
print(f"\n{'='*60}")
print("HADRONIC VACUUM POLARISATION FOR g-2")
print(f"{'='*60}")

# VP source vector: V(χ₃,χ₃'→intermediate) projected onto 32 configs
# The source = quark-antiquark current = configurations where 
# one edge is χ₃ and the rest are background
source_vec = np.zeros(n_configs)
for ic, cfg in enumerate(configs):
    # Weight by the "meson-ness": presence of both χ₃ and χ₃' on the face
    n3 = sum(1 for x in cfg if x == 1)
    n3p = sum(1 for x in cfg if x == 2)
    source_vec[ic] = math.sqrt(n3 * n3p) / 5.0

# Correlator: C(t) = Σ_n |⟨source|ψ_n⟩|² × (λ_n/λ_max)^t
print(f"\n  Current-current correlator C(t):")
sum_HVP = 0
for t in range(1, 50):
    C_t = 0
    for n in range(n_configs):
        ev = eigenvalues_sorted[n]
        if ev <= 0 or T_max <= 0:
            continue
        ratio = ev / T_max
        if ratio > 0:
            overlap = np.dot(source_vec, eigenvectors_sorted[:, n])
            C_t += overlap**2 * ratio**t
    sum_HVP += t**2 * C_t
    if t <= 10 or t % 10 == 0:
        print(f"    C(t={t:2d}) = {C_t:.6e}")

# Physical HVP
m_e = 0.000511  # GeV
m_mu = 0.10566  # GeV
a_lat = 1 / 0.332  # GeV⁻¹ (QCD lattice spacing)

a_e_HVP = (4 * alpha**2 / 3) * sum_HVP * (m_e * a_lat)**2
a_mu_HVP = (4 * alpha**2 / 3) * sum_HVP * (m_mu * a_lat)**2

print(f"\n  Sum Σ t² C(t) = {sum_HVP:.6e}")
print(f"\n  Electron g-2 hadronic VP:")
print(f"    a_e(HVP) = {a_e_HVP:.4e}")
print(f"    SM value: 1.67 × 10⁻¹²")
print(f"    Ratio: {a_e_HVP / 1.67e-12:.4f}")
print(f"\n  Muon g-2 hadronic VP:")
print(f"    a_μ(HVP) = {a_mu_HVP:.4e}")
print(f"    SM value: 6.93 × 10⁻⁸")
print(f"    Ratio: {a_mu_HVP / 6.93e-8:.6f}")

# ================================================================
# SAVE RESULTS
# ================================================================
results = {
    'eigenvalues': [float(e) for e in eigenvalues],
    'masses_MeV': [(float(g), float(m)) for _, g, m, _ in masses],
    'f_pi_MeV': float(f_pi_phys) if 'f_pi_phys' in dir() else None,
    'a_e_HVP': float(a_e_HVP),
    'a_mu_HVP': float(a_mu_HVP),
    'sum_HVP': float(sum_HVP),
    'computation_time_seconds': total_time,
    'T_matrix_diagonal': [float(T_matrix[i, i]) for i in range(n_configs)],
}

with open('hadron_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*60}")
print(f"Results saved to hadron_results.json")
print(f"Copy this file back to Claude for analysis.")
print(f"{'='*60}")
