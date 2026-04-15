#!/usr/bin/env python3
"""
DODECAHEDRAL FRAMEWORK: 1024×1024 TRANSFER MATRIX
===================================================
Adds χ₁ (vacuum) to the QCD sector: χ₁ + χ₃ + χ₃' + χ₅
4 irreps per edge → 4⁵ = 1024 configurations per face.

This tests the |A₅|/Σdim² truncation correction:
  243×243 captured 43/60 → corrected HVP at 5%
  1024×1024 captures 44/60 → if raw HVP ≈ 44/60 × SM, 
  the correction formula is CONFIRMED.

Run:  python3 qcd_vacuum_1024.py --cores 20

Saves progress. Resumes if interrupted.
Estimated: ~5 days on 20 cores.
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
# CONSTANTS (module level for multiprocessing workers)
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha_fw = 1 / (20*phi**4 - (3+5*sqrt5)/308)  # FRAMEWORK α
Lambda_QCD = 332
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

# THIS COMPUTATION: χ₁(0) + χ₃(1) + χ₃'(2) + χ₅(4)
SECTOR_IRREPS = [0, 1, 2, 4]
SECTOR_NAMES = ['χ₁', 'χ₃', 'χ₃\'', 'χ₅']
SECTOR_DIMS_SQ = sum(dims[j]**2 for j in SECTOR_IRREPS)  # 1+9+9+25=44

threej = np.zeros((N_IRREPS, N_IRREPS, N_IRREPS))
for a in range(N_IRREPS):
    for b in range(N_IRREPS):
        for c in range(N_IRREPS):
            threej[a, b, c] = round(
                sum(class_sizes[k]*chars[a][k]*chars[b][k]*chars[c][k]
                    for k in range(5)) / G_ORDER)

# Dodecahedron
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

configs = list(iprod(SECTOR_IRREPS, repeat=5))
n_configs = len(configs)

# ================================================================
# TENSOR NETWORK
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
        for fl in iprod(range(N_IRREPS),repeat=len(free_new)):
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
    ca = configs[ia]
    row = [0.0]*n_configs
    for ib in range(n_configs):
        row[ib] = compute_T(ca, configs[ib])
    return (ia, row)

# ================================================================
# PROGRESS
# ================================================================
SAVE_FILE = 'qcd1024_progress.json'
RESULT_FILE = 'qcd1024_results.json'

def save_progress(path, T_matrix, completed_rows):
    rows = []
    for ia in sorted(completed_rows):
        rows.append({'row': int(ia), 'data': T_matrix[ia].tolist()})
    with open(path, 'w') as f:
        json.dump({'rows': rows}, f)

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
        elif 'completed_rows' in prog and 'data' in prog:
            n_done = prog['completed_rows']
            for ia in range(min(n_done, len(prog['data']))):
                T[ia] = prog['data'][ia]
                done.add(ia)
        print(f"  Loaded {len(done)} rows from {path}")
    return T, done

# ================================================================
# ANALYSIS
# ================================================================
def analyse(T_matrix):
    T_sym = (T_matrix + T_matrix.T) / 2

    print(f"\n{'='*60}")
    print("EIGENVALUE ANALYSIS")
    print(f"{'='*60}")

    evals_raw, evecs_raw = np.linalg.eigh(T_sym)
    idx = np.argsort(evals_raw)[::-1]
    evals = evals_raw[idx]
    evecs = evecs_raw[:, idx]

    T_max = evals[0]
    n_pos = sum(1 for e in evals if e > 0)
    print(f"Largest eigenvalue: {T_max:.4e}")
    print(f"Positive: {n_pos}, Negative: {n_configs-n_pos}")

    masses = []
    for i, ev in enumerate(evals):
        if ev > 0 and ev < T_max*0.9999:
            gap = -math.log(ev/T_max)
            masses.append({'level':i, 'eigenvalue':float(ev),
                           'gap':float(gap), 'mass_MeV':float(gap*Lambda_QCD)})

    known = [(135,'π'),(494,'K'),(548,'η'),(600,'σ/f₀'),(775,'ρ'),
             (782,'ω'),(958,"η'"),(1020,'φ'),(1232,'Δ'),(1275,'f₂'),
             (1525,"f₂'"),(1680,'ρ₃'),(1507,'glueball?')]

    print(f"\nMass spectrum ({len(masses)} states):")
    print(f"  {'#':>3} {'Mass':>8} {'Match':>12} {'Obs':>6} {'Err':>6}")
    n5 = 0
    for i, m in enumerate(masses[:30]):
        best = min(known, key=lambda h: abs(h[0]-m['mass_MeV']))
        err = abs(best[0]-m['mass_MeV'])/best[0]*100
        mark = '★' if err<3 else '●' if err<5 else '◦' if err<10 else ' '
        if err<5: n5+=1
        if i<25:
            print(f"  {i:>3} {m['mass_MeV']:>8.0f} {best[1]:>12} {best[0]:>6} {err:>5.1f}% {mark}")
    print(f"\n  Within 5%: {n5}")

    # f_π
    print(f"\n{'='*60}")
    print("PION DECAY CONSTANT")
    axial = np.zeros(n_configs)
    for ic, cfg in enumerate(configs):
        n3 = sum(1 for x in cfg if x==SECTOR_IRREPS[1])  # χ₃
        n3p = sum(1 for x in cfg if x==SECTOR_IRREPS[2])  # χ₃'
        axial[ic] = (n3-n3p)/5.0

    f_pi_raw = f_pi_corr = None
    for i in range(min(8, n_configs)):
        ov = abs(np.dot(axial, evecs[:,i]))
        if ov > 0.1:
            f_raw = ov*math.sqrt(3)*Lambda_QCD
            f_corr = f_raw/sqrt5
            print(f"  |<J_A|ψ_{i}>| = {ov:.6f}  → f_π = {f_raw:.0f}/√5 = {f_corr:.1f} MeV")
            if f_pi_raw is None:
                f_pi_raw = f_raw
                f_pi_corr = f_corr
        elif i < 5:
            print(f"  |<J_A|ψ_{i}>| = {ov:.6f}")

    if f_pi_corr and f_pi_corr > 0:
        m_q = alpha_fw*math.sqrt(3-sqrt5)*Lambda_QCD
        cond = 3*0.367*Lambda_QCD**3
        m_pi = math.sqrt(2*m_q*cond/f_pi_corr**2)
        print(f"\n  GMOR: m_π = {m_pi:.1f} MeV (obs 139.6)")

    # Hadronic VP
    print(f"\n{'='*60}")
    print("HADRONIC VACUUM POLARISATION")
    print(f"  Sector Σdim² = {SECTOR_DIMS_SQ}/60 = {SECTOR_DIMS_SQ/60:.4f} of |A₅|")
    print(f"  Truncation correction: 60/{SECTOR_DIMS_SQ} = {60/SECTOR_DIMS_SQ:.4f}")

    source = np.zeros(n_configs)
    for ic, cfg in enumerate(configs):
        n3 = sum(1 for x in cfg if x==SECTOR_IRREPS[1])
        n3p = sum(1 for x in cfg if x==SECTOR_IRREPS[2])
        source[ic] = math.sqrt(n3*n3p)/5.0

    sum_HVP = 0
    for t in range(1, 200):
        C_t = 0
        for nn in range(n_configs):
            ev = evals[nn]
            if ev<=0: continue
            ratio = ev/T_max
            if 0<ratio<1:
                ov = np.dot(source, evecs[:,nn])
                C_t += ov**2 * ratio**t
        sum_HVP += t**2*C_t
        if t<=5 or t%20==0:
            print(f"  C(t={t:3d}) = {C_t:.6e}")

    a_e_raw = (4*alpha_fw**2/3)*sum_HVP*(0.000511/0.332)**2
    a_mu_raw = (4*alpha_fw**2/3)*sum_HVP*(0.10566/0.332)**2
    correction = 60/SECTOR_DIMS_SQ
    a_e_corr = a_e_raw*correction
    a_mu_corr = a_mu_raw*correction

    print(f"\n  Raw (before correction):")
    print(f"    a_e(HVP) = {a_e_raw:.4e}")
    print(f"    a_μ(HVP) = {a_mu_raw:.4e}")
    print(f"    a_μ/a_μ(SM) = {a_mu_raw/6.93e-8:.4f} (should be ≈{SECTOR_DIMS_SQ/60:.4f})")
    print(f"\n  Corrected (× {correction:.4f}):")
    print(f"    a_e(HVP) = {a_e_corr:.4e} (SM: 1.67e-12, err: {abs(a_e_corr-1.67e-12)/1.67e-12*100:.1f}%)")
    print(f"    a_μ(HVP) = {a_mu_corr:.4e} (SM: 6.93e-8, err: {abs(a_mu_corr-6.93e-8)/6.93e-8*100:.1f}%)")

    # Save
    results = {
        'sector': 'chi1_chi3_chi3p_chi5',
        'sector_irreps': SECTOR_IRREPS,
        'sector_dims_sq': SECTOR_DIMS_SQ,
        'n_configs': n_configs,
        'eigenvalues': [float(e) for e in evals[:100]],
        'masses': masses[:60],
        'f_pi_raw_MeV': float(f_pi_raw) if f_pi_raw else None,
        'f_pi_corrected_MeV': float(f_pi_corr) if f_pi_corr else None,
        'a_e_HVP_raw': float(a_e_raw),
        'a_mu_HVP_raw': float(a_mu_raw),
        'a_e_HVP_corrected': float(a_e_corr),
        'a_mu_HVP_corrected': float(a_mu_corr),
        'truncation_correction': float(correction),
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULT_FILE}")

# ================================================================
if __name__ == '__main__':
    N_CORES = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--cores' and i+1 < len(sys.argv):
            N_CORES = int(sys.argv[i+1])

    print(f"1024×1024 Transfer Matrix (χ₁+χ₃+χ₃'+χ₅)")
    print(f"Configs: {n_configs}, Elements: {n_configs**2:,}")
    print(f"Cores: {N_CORES}")
    print(f"Sector Σdim² = {SECTOR_DIMS_SQ}/60 (missing: χ₄, dim²=16)")
    print(f"Face A: {sorted(face_A)}, Face B: {sorted(face_B)}")

    T_matrix, completed_rows = load_progress(SAVE_FILE)
    remaining = [ia for ia in range(n_configs) if ia not in completed_rows]
    print(f"Done: {len(completed_rows)}, Remaining: {len(remaining)}")

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
                rate = done_session/elapsed
                left = len(remaining)-done_session
                eta = left/rate if rate>0 else 0
                print(f"  Row {ia:4d}/{n_configs} | {elapsed/3600:.1f}h | ~{eta/3600:.1f}h left")
                if done_session % 5 == 0:
                    save_progress(SAVE_FILE, T_matrix, completed_rows)

        save_progress(SAVE_FILE, T_matrix, completed_rows)
        print(f"\nComputation: {(time.time()-t_start)/3600:.2f} hours")

    analyse(T_matrix)
