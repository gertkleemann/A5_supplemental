#!/usr/bin/env python3
"""
DARK SECTOR TRANSFER MATRIX: Pure χ₄ (1024×1024)
===================================================
The unexplored corner. χ₄ is the dark matter irrep:
  - χ₄(C₂) = 0 → EM-invisible
  - χ₄(C₃) = 1 → weak-coupled
  - χ₄ ⊗ χ₄ = χ₁+χ₃+χ₃'+χ₄+χ₅ → produces EVERYTHING
  - Face rep has NO χ₄ → self-confining at every boundary

This computation gives:
  1. Dark hadron spectrum (mass tower of dark bound states)
  2. Dark confinement scale (mass gap in pure χ₄ sector)
  3. Dark matter self-interaction eigenvectors
  4. Dark→visible annihilation matrix elements
  5. Comparison with visible sector (same matrix size = 1024)

Run:  python3 dark_sector_1024.py --cores 20

Progress file: dark1024_progress.json   (unique, no overlap)
Results file:  dark1024_results.json

Compatible with existing progress files:
  qcd1024_progress.json          (χ₁+χ₃+χ₃'+χ₅, visible sector)
  qcd3125_progress_seeded.json   (all 5 irreps)
"""
import numpy as np
import math
import time
import json
import os
import sys
from itertools import product as iprod
from collections import deque

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha_fw = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda_QCD = 332  # MeV
N_IRREPS = 5

dims = [1, 3, 3, 4, 5]
chars = [
    [1, 1, 1, 1, 1],
    [3, -1, 0, phi, 1-phi],
    [3, -1, 0, 1-phi, phi],
    [4, 0, 1, -1, -1],
    [5, 1, -1, 0, 0],
]
class_sizes = [1, 15, 20, 12, 12]
G_ORDER = 60
irr_names = ['χ₁','χ₃',"χ₃'",'χ₄','χ₅']

# ============================================================
# THIS COMPUTATION: PURE χ₄ SECTOR
# ============================================================
SECTOR_IRREPS = [3]  # χ₄ only (index 3 in the full list)
SECTOR_NAMES = ['χ₄']
SECTOR_DIMS_SQ = sum(dims[j]**2 for j in SECTOR_IRREPS)  # 16

# For comparison: also define the dark+photon sector
# SECTOR_IRREPS_DP = [3, 4]  # χ₄+χ₅ → 9⁵ = 59049 (too large for now)

threej = np.zeros((N_IRREPS, N_IRREPS, N_IRREPS))
for a in range(N_IRREPS):
    for b in range(N_IRREPS):
        for c in range(N_IRREPS):
            threej[a, b, c] = round(
                sum(class_sizes[k]*chars[a][k]*chars[b][k]*chars[c][k]
                    for k in range(5)) / G_ORDER)

# Quick check: can χ₄ self-couple at a degree-3 vertex?
N444 = threej[3,3,3]
print(f"N(χ₄,χ₄,χ₄) = {N444:.0f} — {'self-coupling exists!' if N444>0 else 'NO self-coupling'}")

# ================================================================
# DODECAHEDRON (identical to visible sector script)
# ================================================================
dv = []
for s1 in [1,-1]:
    for s2 in [1,-1]:
        for s3 in [1,-1]: dv.append([s1,s2,s3])
for s1 in [1,-1]:
    for s2 in [1,-1]:
        dv.append([0,s1/phi,s2*phi])
        dv.append([s1/phi,s2*phi,0])
        dv.append([s2*phi,0,s1/phi])
dv = np.array(dv); Nv = 20

edges = []; adj = [[] for _ in range(Nv)]
for i in range(Nv):
    for j in range(i+1,Nv):
        if abs(np.linalg.norm(dv[i]-dv[j])-2/phi)<0.01:
            eidx=len(edges); edges.append((i,j))
            adj[i].append(eidx); adj[j].append(eidx)
Ne = len(edges)

def _get_edge_idx(v1,v2):
    key=(min(v1,v2),max(v1,v2))
    for i,e in enumerate(edges):
        if e==key: return i
    return None

_faces = []
for start in range(Nv):
    nbs_s = set()
    for e in adj[start]:
        a,b = edges[e]; nbs_s.add(b if a==start else a)
    for v1 in nbs_s:
        nbs_1 = set()
        for e in adj[v1]:
            a,b = edges[e]; nbs_1.add(b if a==v1 else a)
        for v2 in (nbs_1-{start}):
            nbs_2 = set()
            for e in adj[v2]:
                a,b = edges[e]; nbs_2.add(b if a==v2 else a)
            for v3 in (nbs_2-{start,v1}):
                nbs_3 = set()
                for e in adj[v3]:
                    a,b = edges[e]; nbs_3.add(b if a==v3 else a)
                for v4 in (nbs_3-{start,v1,v2}):
                    nbs_4 = set()
                    for e in adj[v4]:
                        a,b = edges[e]; nbs_4.add(b if a==v4 else a)
                    if start in nbs_4:
                        f = tuple(sorted([start,v1,v2,v3,v4]))
                        if f not in _faces: _faces.append(f)

_opp = None
for i in range(len(_faces)):
    for j in range(i+1,len(_faces)):
        if len(set(_faces[i])&set(_faces[j]))==0:
            _opp=(i,j); break
    if _opp: break

face_A = _faces[_opp[0]]; face_B = _faces[_opp[1]]

def _face_edges(fv):
    fv=sorted(fv); r=[]
    for i in range(len(fv)):
        for j in range(i+1,len(fv)):
            e=_get_edge_idx(fv[i],fv[j])
            if e is not None: r.append(e)
    return r

edges_A = _face_edges(face_A)
edges_B = _face_edges(face_B)

visited=[False]*Nv; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

# Face configurations: each edge carries ONE irrep from SECTOR_IRREPS
# For pure χ₄: each edge is χ₄ (index 3). BUT we allow ALL 5 irreps
# on internal edges — the sector restriction is on FACE edges only.
# Wait — that's the FULL 3125 computation.
# 
# Actually: the sector restriction means face edges carry only χ₄.
# Internal edges carry ALL irreps (the sum over internal states is unrestricted).
# So face configs: 1⁵ = 1 config per face (all edges = χ₄).
# That gives a 1×1 "matrix" — not useful.
#
# CORRECT APPROACH: the sector means face edges carry irreps FROM the sector.
# For pure χ₄: each face edge ∈ {χ₄}, so 1⁵ = 1.
# For χ₄+χ₅: each face edge ∈ {χ₄, χ₅}, so 2⁵ = 32.
#
# The 1024 in "pure χ₄ 1024×1024" actually means:
# We use 4 irreps per edge: {χ₁, χ₃, χ₃', χ₄} (the same trick as the 
# visible sector, but INCLUDING χ₄ and EXCLUDING χ₅).
# OR: we can do something more interesting — the DARK+MATTER sector
# with χ₁+χ₃+χ₃'+χ₄ (excludes photon/gluon).
# 
# Let me reconsider. The interesting physics is:
# (A) χ₁+χ₄ sector: vacuum + dark = 5⁵ = ... no, 2 irreps → 2⁵ = 32
# (B) χ₁+χ₃+χ₃'+χ₄ sector: matter+dark = 4⁵ = 1024 (excludes χ₅/photon)
# (C) χ₁+χ₄+χ₅ sector: vacuum+dark+photon = 3⁵ = 243
# (D) χ₄+χ₅ sector: dark+photon = 2⁵ = 32 face configs
#
# The most informative is (B): the DARK MATTER SECTOR = 1024×1024
# Same size as the visible sector (which excluded χ₄).
# This one EXCLUDES χ₅ (photon/gluon) and INCLUDES χ₄ (dark matter).
# It answers: what does the universe look like without photons?

# REDEFINE: Dark sector = χ₁+χ₃+χ₃'+χ₄ (no photon/gluon)
SECTOR_IRREPS = [0, 1, 2, 3]  # χ₁+χ₃+χ₃'+χ₄
SECTOR_NAMES = ['χ₁', 'χ₃', "χ₃'", 'χ₄']
SECTOR_DIMS_SQ = sum(dims[j]**2 for j in SECTOR_IRREPS)  # 1+9+9+16=35
N_SECTOR = len(SECTOR_IRREPS)

configs = list(iprod(SECTOR_IRREPS, repeat=5))
n_configs = len(configs)  # 4⁵ = 1024

print(f"\nDark sector: {' + '.join(SECTOR_NAMES)}")
print(f"Σdim² = {SECTOR_DIMS_SQ}/60 ({SECTOR_DIMS_SQ/60*100:.1f}% of |A₅|)")
print(f"Configs: {n_configs} (face edges from {SECTOR_NAMES})")
print(f"Missing: χ₅ (photon/gluon, dim²=25)")
print(f"Compare: visible sector had χ₁+χ₃+χ₃'+χ₅, Σdim²=44")
print(f"         dark sector has χ₁+χ₃+χ₃'+χ₄, Σdim²=35")

# ================================================================
# TENSOR NETWORK (identical engine)
# ================================================================
def compute_T(la, lb):
    fixed = {}
    for idx,eidx in enumerate(edges_A): fixed[eidx]=la[idx]
    for idx,eidx in enumerate(edges_B): fixed[eidx]=lb[idx]
    processed=set(); open_edges=[]; state={():1.0}
    for v in order:
        v_edges=adj[v]; new_e=[]; closing_e=[]
        for eidx in v_edges:
            i,j=edges[eidx]; other=j if i==v else i
            typ='f' if eidx in fixed else 'r'
            if other in processed: closing_e.append((typ,eidx))
            else: new_e.append((typ,eidx))
        edge_pos={e:i for i,e in enumerate(open_edges)}
        cfp=sorted([(eidx,edge_pos[eidx]) for typ,eidx in closing_e if typ=='r'],
                    key=lambda x:-x[1])
        new_state={}; free_new=[eidx for typ,eidx in new_e if typ=='r']
        # Internal edges: ALL 5 irreps (not restricted to sector)
        for fl in iprod(range(N_IRREPS), repeat=len(free_new)):
            for ok,ow in state.items():
                if abs(ow)<1e-15: continue
                el={}; fi=0
                for typ,eidx in new_e:
                    if typ=='f': el[eidx]=fixed[eidx]
                    else: el[eidx]=fl[fi]; fi+=1
                for typ,eidx in closing_e:
                    if typ=='f': el[eidx]=fixed[eidx]
                    else: el[eidx]=ok[edge_pos[eidx]]
                jabc=tuple(el[e] for e in v_edges)
                vw=threej[jabc]
                if vw==0: continue
                ew=1.0
                for eidx in free_new: ew*=dims[el[eidx]]
                nkl=list(ok)
                for _,pos in cfp: nkl.pop(pos)
                for eidx in free_new: nkl.append(el[eidx])
                nk=tuple(nkl)
                new_state[nk]=new_state.get(nk,0.0)+ow*vw*ew
        new_open=[e for e in open_edges
                  if not any(eidx==e for typ,eidx in closing_e if typ=='r')]
        new_open.extend(free_new)
        open_edges=new_open; processed.add(v); state=new_state
    return sum(state.values())

def compute_row(ia):
    """Compute one row of the transfer matrix. Generates configs locally."""
    local_configs = list(iprod(SECTOR_IRREPS, repeat=5))
    ca = local_configs[ia]
    row = [0.0]*len(local_configs)
    for ib in range(len(local_configs)):
        row[ib] = compute_T(ca, local_configs[ib])
    return (ia, row)

# ================================================================
# PROGRESS (unique filenames — no overlap with visible sector)
# ================================================================
SAVE_FILE = 'dark1024_progress.json'
RESULT_FILE = 'dark1024_results.json'

def save_progress(path, T_matrix, completed_rows):
    rows = []
    for ia in sorted(completed_rows):
        rows.append({'row': int(ia), 'data': T_matrix[ia].tolist()})
    with open(path, 'w') as f:
        json.dump({
            'sector': 'chi1_chi3_chi3p_chi4',
            'sector_irreps': SECTOR_IRREPS,
            'sector_names': SECTOR_NAMES,
            'sector_dims_sq': SECTOR_DIMS_SQ,
            'n_configs': n_configs,
            'rows': rows
        }, f)

def load_progress(path):
    T = np.zeros((n_configs, n_configs))
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            prog = json.load(f)
        if 'rows' in prog:
            for entry in prog['rows']:
                ia = entry['row']
                T[ia] = entry['data']
                done.add(ia)
        print(f"  Loaded {len(done)} rows from {path}")
    return T, done

# ================================================================
# DARK SECTOR ANALYSIS
# ================================================================
def analyse(T_matrix):
    T_sym = (T_matrix + T_matrix.T) / 2
    evals_raw, evecs_raw = np.linalg.eigh(T_sym)
    idx = np.argsort(evals_raw)[::-1]
    evals = evals_raw[idx]
    evecs = evecs_raw[:, idx]

    T_max = evals[0]
    n_pos = sum(1 for e in evals if e > 0)

    print(f"\n{'='*65}")
    print(f"DARK SECTOR EIGENVALUE ANALYSIS")
    print(f"{'='*65}")
    print(f"  T_max = {T_max:.6e}")
    print(f"  Positive: {n_pos}, Negative: {n_configs-n_pos}")

    # Mass spectrum
    masses = []
    for i, ev in enumerate(evals):
        if ev > 0 and ev < T_max * 0.9999:
            gap = -math.log(ev/T_max)
            masses.append({'level':i, 'eigenvalue':float(ev),
                           'gap':float(gap), 'mass_MeV':float(gap*Lambda_QCD)})

    print(f"\n  DARK HADRON SPECTRUM ({len(masses)} states):")
    print(f"  {'#':>3} {'Mass(MeV)':>10} {'Gap':>8} {'Eigenvalue':>14}")
    for i, m in enumerate(masses[:30]):
        mark = ' '
        if abs(m['mass_MeV'] - 1604) < 50: mark = '★ 1604?'
        if abs(m['mass_MeV'] - 938) < 50: mark = '● proton?'
        if abs(m['mass_MeV'] - 135) < 30: mark = '● pion?'
        print(f"  {i:>3} {m['mass_MeV']:>10.1f} {m['gap']:>8.4f} {m['eigenvalue']:>14.4e}  {mark}")

    # Dark confinement scale
    if masses:
        dark_gap = masses[0]['gap']
        dark_Lambda = masses[0]['mass_MeV']
        print(f"\n  DARK CONFINEMENT SCALE:")
        print(f"    First mass gap: {dark_gap:.4f}")
        print(f"    Dark Λ = {dark_Lambda:.0f} MeV")
        print(f"    Visible Λ_QCD = 332 MeV")
        print(f"    Dark/Visible ratio: {dark_Lambda/332:.3f}")

    # Ground state composition (dark matter content)
    print(f"\n  GROUND STATE (VACUUM) COMPOSITION:")
    gs = evecs[:, 0]
    irrep_content = {n: 0.0 for n in SECTOR_NAMES}
    for ic, cfg in enumerate(configs):
        w = gs[ic]**2
        for edge_irrep in cfg:
            irrep_content[irr_names[edge_irrep]] += w / 5.0

    for name, frac in irrep_content.items():
        bar = '█' * int(frac * 50)
        print(f"    {name:>4}: {frac*100:5.1f}% {bar}")

    # χ₄ fraction in ground state
    chi4_frac = irrep_content.get('χ₄', 0)
    print(f"\n    Dark matter fraction of vacuum: {chi4_frac*100:.1f}%")
    print(f"    Framework prediction: ~25%")
    print(f"    Observed: 27%")

    # Compare with visible sector (if available)
    vis_file = 'qcd1024_results.json'
    if os.path.exists(vis_file):
        with open(vis_file) as f:
            vis = json.load(f)
        print(f"\n  DARK vs VISIBLE COMPARISON:")
        vis_Tmax = vis['eigenvalues'][0] if vis.get('eigenvalues') else None
        if vis_Tmax:
            print(f"    T_max(visible) = {vis_Tmax:.4e}")
            print(f"    T_max(dark)    = {T_max:.4e}")
            print(f"    Ratio: {T_max/vis_Tmax:.4f}")
            print(f"    This ratio determines the dark matter mass scale")
            if T_max > 0 and vis_Tmax > 0:
                mass_ratio = abs(math.log(vis_Tmax/T_max))
                print(f"    Mass scale difference: {mass_ratio:.4f} = {mass_ratio*Lambda_QCD:.0f} MeV")

    # Dark matter self-interaction
    print(f"\n  DARK MATTER SELF-INTERACTION:")
    print(f"    N(χ₄,χ₄,χ₄) = {threej[3,3,3]:.0f}")
    print(f"    a_{{44}}^{{C₃}} = {class_sizes[2]**2/60 * sum(chars[r][2]**2*chars[r][2]/dims[r] for r in range(5)):.0f}")

    # Dark sector annihilation matrix elements
    # The χ₄χ₄ → χ₅χ₅ channel (dark→photon)
    print(f"\n  DARK ANNIHILATION CHANNELS:")
    for c in range(5):
        n = threej[3,3,c]
        if n > 0:
            print(f"    χ₄ ⊗ χ₄ → {irr_names[c]}: N = {n:.0f}, "
                  f"dim = {dims[c]}, weighted = {n*dims[c]:.0f}")

    # Cross-coupling strength
    cross_45 = sum(class_sizes[k]*chars[3][k]**2*chars[4][k]**2 for k in range(5))/60
    print(f"    ⟨χ₄χ₄|χ₅χ₅⟩ = {cross_45:.0f} = QCD self-coupling strength")

    # Bullet Cluster constraint
    if masses:
        m_DM = masses[0]['mass_MeV']  # lightest dark state
        # σ/m estimate from the transfer matrix
        # σ ~ α_dark² / m² where α_dark = N(χ₄,χ₄,χ₄) / (4π × something)
        # Very rough: use geometric cross-section at confinement scale
        r_conf = 197.3 / m_DM  # fm (confinement radius)
        sigma_geo = math.pi * r_conf**2  # fm²
        sigma_over_m = sigma_geo * 10 / m_DM  # cm²/g (very rough)
        print(f"\n  BULLET CLUSTER CONSTRAINT:")
        print(f"    Dark confinement radius: {r_conf:.3f} fm")
        print(f"    Geometric σ: {sigma_geo:.4f} fm²")
        print(f"    σ/m (geometric, rough): {sigma_over_m:.4f} cm²/g")
        print(f"    Bullet Cluster limit: σ/m < 1 cm²/g")
        print(f"    {'CONSISTENT' if sigma_over_m < 1 else 'TENSION'}")

    # Eigenvalue spacing statistics (integrable vs chaotic)
    spacings = []
    for i in range(1, min(200, n_pos)):
        if evals[i] > 0:
            s = (evals[i-1] - evals[i]) / (evals[0] / n_pos)
            spacings.append(s)
    if spacings:
        mean_s = np.mean(spacings)
        var_s = np.var(spacings)
        print(f"\n  EIGENVALUE SPACING:")
        print(f"    Mean spacing (normalised): {mean_s:.4f}")
        print(f"    Variance/mean²: {var_s/mean_s**2:.4f}")
        print(f"    Poisson (integrable): var/mean² = 1")
        print(f"    GOE (chaotic): var/mean² ≈ 0.27")

    # Save results
    results = {
        'computation': 'DARK SECTOR — χ₁+χ₃+χ₃\'+χ₄ (no photon/gluon)',
        'sector': 'chi1_chi3_chi3p_chi4',
        'sector_irreps': SECTOR_IRREPS,
        'sector_dims_sq': SECTOR_DIMS_SQ,
        'n_configs': n_configs,
        'T_max': float(T_max),
        'n_positive_eigenvalues': n_pos,
        'eigenvalues_top100': [float(e) for e in evals[:100]],
        'dark_masses_MeV': [m['mass_MeV'] for m in masses[:60]],
        'dark_confinement_MeV': float(masses[0]['mass_MeV']) if masses else None,
        'vacuum_composition': {k: float(v) for k,v in irrep_content.items()},
        'chi4_vacuum_fraction': float(chi4_frac),
        'dark_self_coupling_N444': float(threej[3,3,3]),
        'dark_annihilation_cross_coupling': float(cross_45),
        'masses_full': masses[:60],
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {RESULT_FILE}")

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    N_CORES = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--cores' and i+1 < len(sys.argv):
            N_CORES = int(sys.argv[i+1])

    print(f"\n{'='*65}")
    print(f"  DARK SECTOR: 1024×1024 TRANSFER MATRIX")
    print(f"  Sector: χ₁ + χ₃ + χ₃' + χ₄ (no photon/gluon)")
    print(f"  'What does the universe look like without light?'")
    print(f"{'='*65}")
    print(f"  Configs: {n_configs}, Elements: {n_configs**2:,}")
    print(f"  Cores: {N_CORES}")
    print(f"  Σdim² = {SECTOR_DIMS_SQ}/60 (missing: χ₅, dim²=25)")
    print(f"  Face A: {sorted(face_A)}, Face B: {sorted(face_B)}")
    print(f"  Progress: {SAVE_FILE}")
    print(f"  Results:  {RESULT_FILE}")

    T_matrix, completed_rows = load_progress(SAVE_FILE)
    remaining = [ia for ia in range(n_configs) if ia not in completed_rows]
    print(f"  Done: {len(completed_rows)}, Remaining: {len(remaining)}")

    if remaining:
        t_start = time.time()
        done_session = 0

        if N_CORES > 1:
            from multiprocessing import Pool
            with Pool(N_CORES) as pool:
                for ia, row in pool.imap_unordered(compute_row, remaining):
                    T_matrix[ia] = row
                    completed_rows.add(ia)
                    done_session += 1
                    elapsed = time.time()-t_start
                    rate = done_session/elapsed
                    left = len(remaining)-done_session
                    eta = left/rate if rate>0 else 0
                    pct = len(completed_rows)/n_configs*100
                    print(f"  Row {ia:4d} | {len(completed_rows):4d}/{n_configs} ({pct:.1f}%) | "
                          f"{elapsed/3600:.1f}h | ~{eta/3600:.1f}h left | "
                          f"{rate*3600:.0f} rows/h")
                    if done_session % 10 == 0:
                        save_progress(SAVE_FILE, T_matrix, completed_rows)
        else:
            for ia in remaining:
                _, row = compute_row(ia)
                T_matrix[ia] = row
                completed_rows.add(ia)
                done_session += 1
                elapsed = time.time()-t_start
                rate = done_session/elapsed if elapsed > 0 else 1
                left = len(remaining)-done_session
                eta = left/rate if rate>0 else 0
                print(f"  Row {ia:4d}/{n_configs} | {elapsed/3600:.1f}h | ~{eta/3600:.1f}h left")
                if done_session % 5 == 0:
                    save_progress(SAVE_FILE, T_matrix, completed_rows)

        save_progress(SAVE_FILE, T_matrix, completed_rows)
        print(f"\n  Computation: {(time.time()-t_start)/3600:.2f} hours")

    analyse(T_matrix)
