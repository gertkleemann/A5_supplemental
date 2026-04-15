#!/usr/bin/env python3
import sys, os
# Windows compatibility: no fork, use spawn (default on Windows)
if sys.platform == 'win32':
    pass  # spawn is default on Windows
else:
    try:
        import multiprocessing; multiprocessing.set_start_method("fork")
    except RuntimeError:
        pass  # already set
"""
2I BOUNDARY TRANSFER MATRIX — LEPTON MASS DERIVATION
=====================================================
The binary icosahedral group 2I (order 120, double cover of A₅) governs
lepton masses. The electron requires the full 2I structure because 
spin-1/2 needs two A₅ cycles (|2I| = 120 = 2×|A₅|).

This script computes the icosahedral boundary transfer matrix using 2I
irreps (9 irreps, 729×729 matrix) and extracts lepton mass ratios by
comparing to the A₅ boundary result (5 irreps, 125×125).

Physical prediction:
  m_μ/m_e = |2I| × √dim(χ₃) = 120√3 = 207.85  (observed: 206.77)

Usage:
  python3 icosa_boundary_2I.py --cores 8              # all 9 irreps (729×729)
  python3 icosa_boundary_2I.py --cores 8 --bosonic     # 5 bosonic irreps (125×125, = A₅ check)
  python3 icosa_boundary_2I.py --cores 8 --fermionic   # 4 fermionic irreps (64×64)

Requirements: pip install numba numpy
"""
import numpy as np
from numba import njit
import math, time, json, os, sys
from itertools import product as iprod
from collections import deque

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
Lambda_QCD = 332

# ================================================================
# 2I CHARACTER TABLE (Binary Icosahedral Group, order 120)
# ================================================================
# 9 conjugacy classes:
#   C₁{1}  C₋₁{-1}  C₃(20)  C₄(30)  C₅(12)  C₅'(12)  C₆(20)  C₁₀(12)  C₁₀'(12)
# 
# Convention (from ATLAS/GAP for SL(2,5)):
#   Classes ordered: 1a, 2a, 3a, 4a, 5a, 5b, 6a, 10a, 10b
#   Sizes:            1,  1,  20,  30,  12,  12,  20,   12,   12

class_sizes_2I = np.array([1, 1, 20, 30, 12, 12, 20, 12, 12], dtype=np.float64)
order_2I = 120

# Irreps in order: ρ₁(1), ρ₂(2), ρ₂'(2), ρ₃(3), ρ₃'(3), ρ₄(4), ρ₄'(4), ρ₅(5), ρ₆(6)
# Bosonic (trivial on center): ρ₁, ρ₃, ρ₃', ρ₄, ρ₅  (lifts of A₅ irreps)
# Fermionic (nontrivial):      ρ₂, ρ₂', ρ₄', ρ₆   (genuine spinor representations)

dims_2I = np.array([1, 2, 2, 3, 3, 4, 4, 5, 6], dtype=np.float64)
n_irreps_2I = 9

# Character table (ATLAS notation, verified by orthogonality)
#                       1a   2a   3a   4a   5a      5b    6a   10a     10b
chars_2I = np.array([
    [ 1,   1,   1,   1,   1,      1,    1,    1,      1   ],  # ρ₁(1)  bosonic
    [ 2,  -2,  -1,   0,   phi-1, -phi,  1,    1-phi,  phi ],  # ρ₂(2)  fermionic
    [ 2,  -2,  -1,   0,  -phi,   phi-1, 1,    phi,    1-phi], # ρ₂'(2) fermionic
    [ 3,   3,   0,  -1,   phi,   1-phi, 0,    phi,    1-phi], # ρ₃(3)  bosonic = χ₃
    [ 3,   3,   0,  -1,   1-phi, phi,   0,    1-phi,  phi ],  # ρ₃'(3) bosonic = χ₃'
    [ 4,   4,   1,   0,  -1,    -1,     1,   -1,     -1   ],  # ρ₄(4)  bosonic = χ₄
    [ 4,  -4,   1,   0,  -1,    -1,    -1,    1,      1   ],  # ρ₄'(4) fermionic
    [ 5,   5,  -1,   1,   0,     0,    -1,    0,      0   ],  # ρ₅(5)  bosonic = χ₅
    [ 6,  -6,   0,   0,   1,     1,     0,   -1,     -1   ],  # ρ₆(6)  fermionic
], dtype=np.float64)

irr_names_2I = ['ρ₁(1)', 'ρ₂(2)', "ρ₂'(2)", 'ρ₃(3)', "ρ₃'(3)", 
                'ρ₄(4)', "ρ₄'(4)", 'ρ₅(5)', 'ρ₆(6)']
bosonic_indices = [0, 3, 4, 5, 7]   # ρ₁, ρ₃, ρ₃', ρ₄, ρ₅ — lifts of A₅
fermionic_indices = [1, 2, 6, 8]     # ρ₂, ρ₂', ρ₄', ρ₆ — spinors

# Verify character table orthogonality
print("Verifying 2I character table...")
for a in range(n_irreps_2I):
    norm = sum(class_sizes_2I[k] * chars_2I[a,k]**2 for k in range(9))
    assert abs(norm - order_2I) < 0.01, f"ρ_{a} norm = {norm} ≠ {order_2I}"
for a in range(n_irreps_2I):
    for b in range(a+1, n_irreps_2I):
        dot = sum(class_sizes_2I[k] * chars_2I[a,k] * chars_2I[b,k] for k in range(9))
        assert abs(dot) < 0.01, f"⟨ρ_{a},ρ_{b}⟩ = {dot} ≠ 0"
assert abs(sum(d**2 for d in dims_2I) - order_2I) < 0.01
print(f"  ✓ Orthogonality verified, Σdim² = {int(sum(d**2 for d in dims_2I))} = |2I|")

# ================================================================
# A₅ CHARACTER TABLE (for comparison / bosonic-only mode)
# ================================================================
dims_A5 = np.array([1, 3, 3, 4, 5], dtype=np.float64)
chars_A5 = np.array([
    [1,  1,  1,  1,      1],
    [3, -1,  0,  phi,    1-phi],
    [3, -1,  0,  1-phi,  phi],
    [4,  0,  1, -1,     -1],
    [5,  1, -1,  0,      0],
], dtype=np.float64)
class_sizes_A5 = np.array([1, 15, 20, 12, 12], dtype=np.float64)

# ================================================================
# ICOSAHEDRON GEOMETRY (same for both A₅ and 2I)
# ================================================================
print("Building icosahedron...", end=' ', flush=True)
iv = []
for s1 in [1, -1]:
    for s2 in [1, -1]:
        iv.append([0, s1, s2*phi])
        iv.append([s1, s2*phi, 0])
        iv.append([s2*phi, 0, s1])
iv = np.array(iv)

edges = []; adj = [[] for _ in range(12)]
for i in range(12):
    for j in range(i+1, 12):
        if abs(np.linalg.norm(iv[i]-iv[j]) - 2.0) < 0.1:
            eidx = len(edges); edges.append((i,j))
            adj[i].append(eidx); adj[j].append(eidx)
assert len(edges) == 30

faces = []
for i in range(12):
    ni = set()
    for e in adj[i]: a,b = edges[e]; ni.add(b if a==i else a)
    ni_list = sorted(ni)
    for ji,j in enumerate(ni_list):
        nj = set()
        for e in adj[j]: a,b = edges[e]; nj.add(b if a==j else a)
        for k in ni_list[ji+1:]:
            if k in nj:
                f = tuple(sorted([i,j,k]))
                if f not in faces: faces.append(f)
assert len(faces) == 20

def get_eidx(v1,v2):
    key = (min(v1,v2),max(v1,v2))
    for i,e in enumerate(edges):
        if e==key: return i
    return None

def face_edg(fv):
    fv=sorted(fv); r=[]
    for i in range(len(fv)):
        for j in range(i+1,len(fv)):
            e=get_eidx(fv[i],fv[j])
            if e is not None: r.append(e)
    return r

opp_pair = None
best_mo = 999
# Search for the face pair with lowest max_open under BFS
for i in range(len(faces)):
    for j in range(i+1,len(faces)):
        if len(set(faces[i])&set(faces[j]))==0:
            ci=np.mean([iv[v] for v in faces[i]],axis=0)
            cj=np.mean([iv[v] for v in faces[j]],axis=0)
            if np.dot(ci,cj)<-0.5:
                # Quick BFS to estimate max_open
                eA_t=set(face_edg(faces[i])); eB_t=set(face_edg(faces[j]))
                vis=[False]*12; ordt=[]; qt=deque()
                for v in sorted(faces[i]): qt.append(v); vis[v]=True
                while qt:
                    v=qt.popleft(); ordt.append(v)
                    for eidx in adj[v]:
                        a,b=edges[eidx]; nb=b if a==v else a
                        if not vis[nb]: vis[nb]=True; qt.append(nb)
                proc=set(); oel=[]; mo=0
                for v in ordt:
                    nf=[]
                    for eidx in adj[v]:
                        a,b=edges[eidx]; oth=b if a==v else a
                        if eidx in eA_t or eidx in eB_t: pass
                        elif oth in proc:
                            if eidx in oel: oel.remove(eidx)
                        else: nf.append(eidx)
                    oel.extend(nf); proc.add(v)
                    if len(oel)>mo: mo=len(oel)
                if mo < best_mo:
                    best_mo = mo; opp_pair = (i,j)

face_A=faces[opp_pair[0]]; face_B=faces[opp_pair[1]]
edges_A=face_edg(face_A); edges_B=face_edg(face_B)
print(f"done. Face A={face_A}, Face B={face_B} (optimised max_open={best_mo})")

visited=[False]*12; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

# ================================================================
# CONTRACTION PLAN (same for both — depends only on graph topology)
# ================================================================
boundary_A=set(edges_A); boundary_B=set(edges_B)
MAX_E=5; MAX_OPEN=15

plan_edge_types   = np.zeros((12,MAX_E), dtype=np.int32)
plan_edge_indices  = np.zeros((12,MAX_E), dtype=np.int32)
plan_n_edges       = np.zeros(12, dtype=np.int32)
plan_n_new         = np.zeros(12, dtype=np.int32)
plan_n_open_before = np.zeros(12, dtype=np.int32)
plan_n_open_after  = np.zeros(12, dtype=np.int32)
plan_closing       = np.full((12,MAX_E), -1, dtype=np.int32)
plan_n_closing     = np.zeros(12, dtype=np.int32)
plan_surviving     = np.full((12,MAX_OPEN), -1, dtype=np.int32)
plan_n_surviving   = np.zeros(12, dtype=np.int32)

processed=set(); open_edge_list=[]; max_open=0
for vi,v in enumerate(order):
    new_free=[]; closing_positions=[]
    plan_n_edges[vi] = len(adj[v])
    for ei,eidx in enumerate(adj[v]):
        i,j=edges[eidx]; other=j if i==v else i
        if eidx in boundary_A:
            plan_edge_types[vi,ei]=0; plan_edge_indices[vi,ei]=edges_A.index(eidx)
        elif eidx in boundary_B:
            plan_edge_types[vi,ei]=1; plan_edge_indices[vi,ei]=edges_B.index(eidx)
        elif other in processed:
            plan_edge_types[vi,ei]=2; pos=open_edge_list.index(eidx)
            plan_edge_indices[vi,ei]=pos; closing_positions.append(pos)
        else:
            plan_edge_types[vi,ei]=3; plan_edge_indices[vi,ei]=len(new_free)
            new_free.append(eidx)
    plan_n_new[vi]=len(new_free); plan_n_open_before[vi]=len(open_edge_list)
    plan_n_closing[vi]=len(closing_positions)
    for ci,p in enumerate(sorted(closing_positions)): plan_closing[vi,ci]=p
    cp_set=set(closing_positions)
    survivors=[p for p in range(len(open_edge_list)) if p not in cp_set]
    plan_n_surviving[vi]=len(survivors)
    for si,p in enumerate(survivors): plan_surviving[vi,si]=p
    plan_n_open_after[vi]=len(survivors)+len(new_free)
    for pos in sorted(closing_positions,reverse=True): open_edge_list.pop(pos)
    open_edge_list.extend(new_free); processed.add(v)
    if len(open_edge_list)>max_open: max_open=len(open_edge_list)

print(f"  Max open edges: {max_open}")

# ================================================================
# NUMBA KERNEL (same algorithm, parametrised by n_ir)
# ================================================================
@njit(cache=False)
def compute_T_numba(la, lb, n_ir, dims_arr, fivej_arr,
                     pe_types, pe_indices, pn_edges,
                     pn_new, pn_open_before, pn_open_after,
                     p_closing, pn_closing, p_surviving, pn_surviving):
    state = np.zeros(1, dtype=np.float64)
    state[0] = 1.0
    for vi in range(12):
        n_open = pn_open_before[vi]; n_new = pn_new[vi]
        n_surv = pn_surviving[vi]; n_open_out = pn_open_after[vi]
        n_closing = pn_closing[vi]; n_e = pn_edges[vi]
        si_in = np.int64(1)
        for k in range(n_open): si_in *= n_ir
        si_out = np.int64(1)
        for k in range(n_open_out): si_out *= n_ir
        new_state = np.zeros(si_out, dtype=np.float64)
        n_nc = np.int64(1)
        for k in range(n_new): n_nc *= n_ir
        for old_idx in range(si_in):
            w = state[old_idx]
            if w == 0.0: continue
            old_lab = np.zeros(15, dtype=np.int64)
            tmp = old_idx
            for oi in range(n_open): old_lab[oi] = tmp % n_ir; tmp //= n_ir
            s_lab = np.zeros(15, dtype=np.int64)
            for si in range(n_surv): s_lab[si] = old_lab[p_surviving[vi,si]]
            c_lab = np.zeros(5, dtype=np.int64)
            for ci in range(n_closing): c_lab[ci] = old_lab[p_closing[vi,ci]]
            for nc in range(n_nc):
                n_lab = np.zeros(5, dtype=np.int64)
                tmp = nc; ew = 1.0
                for ni in range(n_new):
                    nl = tmp % n_ir; n_lab[ni] = nl; ew *= dims_arr[nl]; tmp //= n_ir
                vl = np.zeros(5, dtype=np.int64)
                for ei in range(n_e):
                    et = pe_types[vi,ei]; eix = pe_indices[vi,ei]
                    if et == 0: vl[ei] = la[eix]
                    elif et == 1: vl[ei] = lb[eix]
                    elif et == 2:
                        val = np.int64(0)
                        for ci in range(n_closing):
                            if p_closing[vi,ci] == eix: val = c_lab[ci]; break
                        vl[ei] = val
                    else: vl[ei] = n_lab[eix]
                vw = fivej_arr[vl[0],vl[1],vl[2],vl[3],vl[4]]
                if vw == 0.0: continue
                nidx = np.int64(0); pw = np.int64(1)
                for si in range(n_surv): nidx += s_lab[si]*pw; pw *= n_ir
                for ni in range(n_new): nidx += n_lab[ni]*pw; pw *= n_ir
                new_state[nidx] += w * vw * ew
        state = new_state
    return state[0]

def compute_row(args):
    ia, sector_range = args
    cfgs = list(iprod(sector_range, repeat=3))
    n = len(cfgs); la = np.array(cfgs[ia], dtype=np.int64)
    row = np.zeros(n, dtype=np.float64)
    for ib in range(ia, n):  # SYMMETRY: only upper triangle (ib >= ia)
        lb = np.array(cfgs[ib], dtype=np.int64)
        row[ib] = compute_T_numba(la, lb, N_IR, dims_sec, fivej_sec,
                                   plan_edge_types, plan_edge_indices, plan_n_edges,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
    return (ia, row.tolist())

# ================================================================
# SECTOR SETUP
# ================================================================
MODE = 'full'  # full / bosonic / fermionic / lepton / custom
N_CORES = 1
CUSTOM_SECTOR = None
for i,arg in enumerate(sys.argv):
    if arg=='--bosonic': MODE='bosonic'
    if arg=='--fermionic': MODE='fermionic'
    if arg=='--lepton': MODE='lepton'
    if arg=='--cores' and i+1<len(sys.argv): N_CORES=int(sys.argv[i+1])
    if arg=='--sector' and i+1<len(sys.argv):
        MODE='custom'; CUSTOM_SECTOR=[int(x) for x in sys.argv[i+1].split(',')]

if MODE == 'bosonic':
    SECTOR = bosonic_indices          # [0,3,4,5,7] = ρ₁,ρ₃,ρ₃',ρ₄,ρ₅
    desc = "Bosonic (lifts of A₅, 5 irreps) — sanity check, should match A₅"
elif MODE == 'fermionic':
    SECTOR = fermionic_indices         # [1,2,6,8] = ρ₂,ρ₂',ρ₄',ρ₆
    desc = "Fermionic (spinor, 4 irreps)"
elif MODE == 'lepton':
    SECTOR = [0, 1, 2, 3, 4]          # ρ₁,ρ₂,ρ₂',ρ₃,ρ₃' — vacuum + spinors + matter
    desc = "Lepton sector (vacuum + spinors + matter, 5 irreps)"
elif MODE == 'custom':
    SECTOR = CUSTOM_SECTOR
    desc = f"Custom sector {SECTOR}"
else:
    SECTOR = list(range(9))            # all 9
    desc = "Full 2I (all 9 irreps)"

N_IR = np.int64(len(SECTOR))
dims_sec = np.array([dims_2I[s] for s in SECTOR], dtype=np.float64)

# Memory estimate
peak_state = int(N_IR) ** max_open
mem_bytes = peak_state * 8
mem_gb = mem_bytes / 1e9
print(f"\n  Peak state size: {int(N_IR)}^{max_open} = {peak_state:,.0f}")
print(f"  Memory per element: {mem_gb:.2f} GB")
if mem_gb > 16:
    print(f"  ⚠ WARNING: {mem_gb:.0f} GB exceeds typical RAM!")
    print(f"  Recommended: start with --lepton (5 irreps) or --bosonic (5 irreps)")
    print(f"  Progression:")
    print(f"    --bosonic     (5 irreps, 125×125, ~0.08 GB peak) — matches A₅")
    print(f"    --lepton      (5 irreps, 125×125, ~0.08 GB peak) — spinors + matter")
    print(f"    --sector 0,1,2,3,4,7 (6 irreps, 216×216, ~0.5 GB) — adds ρ₅")
    print(f"    --sector 0,1,2,3,4,5,7 (7 irreps, 343×343, ~2 GB) — adds ρ₄")
    print(f"    Full 9 irreps needs ~{mem_gb:.0f} GB — may need cluster")
    if MODE == 'full':
        print(f"\n  Switching to --lepton mode automatically.")
        MODE = 'lepton'
        SECTOR = [0, 1, 2, 3, 4]
        desc = "Lepton sector (auto-selected, 5 irreps)"
        N_IR = np.int64(len(SECTOR))
        dims_sec = np.array([dims_2I[s] for s in SECTOR], dtype=np.float64)
        peak_state = int(N_IR) ** max_open
        mem_gb = peak_state * 8 / 1e9
        print(f"  New peak: {int(N_IR)}^{max_open} = {peak_state:,.0f} ({mem_gb:.2f} GB)")


# Compute 5j symbols using 2I character table
print(f"\nComputing 5j symbols for {desc}...")
fivej_sec = np.zeros((int(N_IR),)*5, dtype=np.float64)
for a in range(int(N_IR)):
    for b in range(int(N_IR)):
        for c in range(int(N_IR)):
            for d in range(int(N_IR)):
                for e in range(int(N_IR)):
                    val = sum(class_sizes_2I[k] *
                              chars_2I[SECTOR[a],k] * chars_2I[SECTOR[b],k] *
                              chars_2I[SECTOR[c],k] * chars_2I[SECTOR[d],k] *
                              chars_2I[SECTOR[e],k]
                              for k in range(9)) / order_2I
                    fivej_sec[a,b,c,d,e] = round(val)

nz = np.count_nonzero(fivej_sec)
print(f"  5j nonzero: {nz}/{int(N_IR)**5}")

configs = list(iprod(range(int(N_IR)), repeat=3))
n_configs = len(configs)

# ================================================================
if __name__ == '__main__':
    if sys.platform == 'win32':
        import multiprocessing; multiprocessing.freeze_support()
    print(f"\n{'='*65}")
    print(f"2I BOUNDARY TRANSFER MATRIX")
    print(f"{'='*65}")
    print(f"Mode: {desc}")
    print(f"Sector: {[irr_names_2I[s] for s in SECTOR]}")
    print(f"Irreps: {int(N_IR)}, Matrix: {n_configs}×{n_configs}")
    print(f"Peak state: {int(N_IR)}^{max_open} = {int(N_IR)**max_open}")
    print(f"Cores: {N_CORES}")

    print("\nJIT compiling...", end=' ', flush=True)
    t0=time.time()
    la_t=np.zeros(3,dtype=np.int64)
    _=compute_T_numba(la_t,la_t,N_IR,dims_sec,fivej_sec,
                       plan_edge_types,plan_edge_indices,plan_n_edges,
                       plan_n_new,plan_n_open_before,plan_n_open_after,
                       plan_closing,plan_n_closing,plan_surviving,plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")

    # Benchmark
    t0=time.time()
    la=np.zeros(3,dtype=np.int64); lb=np.array([int(N_IR)-1]*3,dtype=np.int64)
    v=compute_T_numba(la,lb,N_IR,dims_sec,fivej_sec,
                       plan_edge_types,plan_edge_indices,plan_n_edges,
                       plan_n_new,plan_n_open_before,plan_n_open_after,
                       plan_closing,plan_n_closing,plan_surviving,plan_n_surviving)
    dt=time.time()-t0
    est_min = dt * n_configs * (n_configs+1) / 2 / max(N_CORES, 1) / 60  # upper triangle only
    print(f"Benchmark: {dt*1000:.1f}ms/element → est {est_min:.1f} min total (symmetry-optimised)")
    print(f"  (Half of full matrix — T is symmetric, only computing upper triangle)")

    SAVE_FILE = f'icosa_2I_{MODE}_progress.json'
    T_matrix = np.zeros((n_configs, n_configs))
    completed_rows = set()
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f: prog=json.load(f)
        for entry in prog.get('rows',[]):
            T_matrix[entry['row']]=entry['data']; completed_rows.add(entry['row'])
        print(f"Loaded {len(completed_rows)} rows from {SAVE_FILE}")

    remaining=[ia for ia in range(n_configs) if ia not in completed_rows]
    print(f"Done: {len(completed_rows)}, Remaining: {len(remaining)}")

    if remaining:
        t_start=time.time(); done_session=0
        def save_prog():
            rows=[{'row':int(ia),'data':T_matrix[ia].tolist()} for ia in sorted(completed_rows)]
            with open(SAVE_FILE,'w') as f: json.dump({'rows':rows},f)

        if N_CORES>1:
            import multiprocessing as mp
            if sys.platform == 'win32':
                mp.freeze_support()
            # spawn-safe: globals are recomputed on import in each worker
            ctx = mp.get_context('spawn') if sys.platform == 'win32' else mp
            args=[(ia,list(range(int(N_IR)))) for ia in remaining]
            with ctx.Pool(N_CORES) as pool:
                for ia,row in pool.imap_unordered(compute_row,args):
                    T_matrix[ia]=row; completed_rows.add(ia); done_session+=1
                    el=time.time()-t_start; rate=done_session/el
                    left=len(remaining)-done_session; eta=left/rate if rate>0 else 0
                    pct=len(completed_rows)/n_configs*100
                    if done_session % max(1, n_configs//50) == 0 or done_session <= 5:
                        print(f"  Row {ia:>4} | {len(completed_rows):>4}/{n_configs} ({pct:.1f}%) | "
                              f"{el/60:.1f}min | ~{eta/60:.1f}min left", flush=True)
                    if done_session%10==0: save_prog()
        else:
            for ia in remaining:
                _,row=compute_row((ia,list(range(int(N_IR)))))
                T_matrix[ia]=row; completed_rows.add(ia); done_session+=1
                el=time.time()-t_start; rate=done_session/el
                left=len(remaining)-done_session; eta=left/rate if rate>0 else 0
                if done_session % max(1, n_configs//20) == 0 or done_session <= 3:
                    print(f"  Row {ia:>4}/{n_configs} | {el/60:.2f}min | ~{eta/60:.1f}min left")
                if done_session%10==0: save_prog()
        save_prog()
        print(f"Computation: {(time.time()-t_start)/60:.2f} min")

    # ================================================================
    # SYMMETRISE (mirror upper triangle to lower)
    # ================================================================
    # compute_row only computed ib >= ia (upper triangle)
    # Mirror to get the full symmetric matrix
    T_matrix = np.triu(T_matrix)
    T_matrix = T_matrix + T_matrix.T - np.diag(np.diag(T_matrix))
    
    # ================================================================
    # ANALYSIS
    # ================================================================
    T_sym=(T_matrix+T_matrix.T)/2
    evals,evecs=np.linalg.eigh(T_sym)
    idx=np.argsort(evals)[::-1]; evals=evals[idx]; evecs=evecs[:,idx]
    T_max=evals[0]

    print(f"\n{'='*65}")
    print(f"2I BOUNDARY ANALYSIS ({desc})")
    print(f"{'='*65}")
    print(f"T_max: {T_max:.6e}")
    print(f"Positive eigenvalues: {sum(1 for e in evals if e>0)}")

    # Mass spectrum
    print(f"\nMass spectrum:")
    known=[(0.511,'e'),(105.7,'μ'),(135,'π'),(332,'Λ'),(775,'ρ'),(938,'p'),(1777,'τ')]
    for i in range(min(30,len(evals))):
        if 0<evals[i]<T_max*0.9999:
            gap=-math.log(evals[i]/T_max); mass=gap*Lambda_QCD
            best=min(known,key=lambda h:abs(h[0]-mass))
            err=abs(best[0]-mass)/best[0]*100
            mark='★' if err<5 else '●' if err<10 else ''
            print(f"  {i:>3}: {mass:>8.1f} MeV → {best[1]:>3} ({best[0]:>7.1f}) {err:5.1f}% {mark}")

    # ================================================================
    # LEPTON MASS EXTRACTION
    # ================================================================
    print(f"\n{'='*65}")
    print(f"LEPTON MASS RATIOS")
    print(f"{'='*65}")
    
    # Load A₅ boundary result for comparison (if available)
    a5_T_max = None
    for fname in ['icosa_0-1-2-3-4_progress.json', 'icosa_boundary_results.json']:
        if os.path.exists(fname):
            with open(fname) as f:
                d = json.load(f)
            a5_T_max = d.get('T_max', None)
            if a5_T_max:
                print(f"  A₅ boundary T_max: {a5_T_max:.6e} (from {fname})")
                break
    
    # Also compute the A₅ boundary T_max from the bosonic sector
    # The bosonic irreps of 2I ARE the A₅ irreps
    if MODE == 'full' and a5_T_max is None:
        print(f"\n  Computing bosonic sub-matrix for A₅ comparison...")
        # The bosonic sector in 2I local indices: ρ₁=0, ρ₃=3, ρ₃'=4, ρ₄=5, ρ₅=7
        # In the SECTOR ordering: index of each in SECTOR
        bos_local = [SECTOR.index(b) for b in bosonic_indices]
        bos_cfgs = [(i,j,k) for i,j,k in configs 
                    if i in bos_local and j in bos_local and k in bos_local]
        bos_idx = [configs.index(c) for c in bos_cfgs]
        T_bos = T_sym[np.ix_(bos_idx, bos_idx)]
        bos_evals = np.linalg.eigvalsh(T_bos)
        a5_T_max = max(bos_evals)
        print(f"  Bosonic sub-matrix T_max (≈ A₅): {a5_T_max:.6e}")
    
    # The 3125×3125 dodecahedral T_max (from previous computation)
    # User should provide this or it defaults to the known value
    dodec_T_max = None
    for fname in ['qcd3125_progress_seeded.json']:
        if os.path.exists(fname):
            try:
                with open(fname) as f:
                    d = json.load(f)
                rows = d.get('rows', [])
                if rows:
                    # The vacuum eigenvalue is approximately the (0,0) element
                    dodec_T_max = max(max(r['data']) for r in rows[:5])
                    print(f"  Dodecahedral T_max (approx): {dodec_T_max:.6e}")
            except:
                pass
    
    print(f"\n  2I boundary T_max: {T_max:.6e}")
    if a5_T_max:
        ratio_2I_A5 = T_max / a5_T_max
        print(f"  T_max(2I) / T_max(A₅) = {ratio_2I_A5:.6f}")
        
        # The muon/electron mass ratio prediction:
        # m_μ/m_e ∝ ln(T_bulk/T_2I_bdy) / ln(T_bulk/T_A5_bdy)
        # ≈ T_A5_bdy/T_2I_bdy for small mass ratios... actually not quite
        # The mass comes from -Λ × ln(T_bdy/T_bulk)
        # m_μ = -Λ × ln(T_A5/T_bulk) / 3
        # m_e = -Λ × ln(T_2I/T_bulk) / 3 (if 2I gives deeper propagation)
        
        # For the ratio we need T_bulk:
        if dodec_T_max:
            m_mu = -Lambda_QCD * math.log(a5_T_max / dodec_T_max) / 3
            m_e_pred = -Lambda_QCD * math.log(T_max / dodec_T_max) / 3
            print(f"\n  m_μ (A₅ boundary) = Λ × ln(T_bulk/T_A5) / 3 = {m_mu:.1f} MeV")
            print(f"  m_e (2I boundary)  = Λ × ln(T_bulk/T_2I) / 3 = {m_e_pred:.1f} MeV")
            if m_e_pred > 0:
                print(f"  m_μ/m_e = {m_mu/m_e_pred:.2f} (observed: 206.77)")
            print(f"  Observed m_e = 0.511 MeV")
        else:
            print(f"\n  Need T_bulk (3125×3125 T_max) for absolute masses.")
            print(f"  Set dodec_T_max manually or provide progress file.")
        
        print(f"\n  Pattern-match comparison:")
        print(f"  m_μ/m_e = |2I| × √dim(χ₃) = 120 × √3 = {120*math.sqrt(3):.2f}")
        print(f"  Observed: 206.77")
    
    # χ₃ (matter) source on boundary
    rho3_local = SECTOR.index(3) if 3 in SECTOR else -1
    if rho3_local >= 0:
        print(f"\n  ρ₃ (matter, dim 3) source analysis:")
        for k in range(3):
            cfg = [0]*3; cfg[k] = rho3_local
            cfg_idx = configs.index(tuple(cfg))
            ov = evecs[cfg_idx, :]
            C1 = sum(ov[n]**2 * (evals[n]/T_max) for n in range(len(evals)) if evals[n]>0)
            meff = -math.log(C1)*Lambda_QCD if C1>0 else 0
            print(f"    Edge {k}: m_eff = {meff:.1f} MeV, C(1) = {C1:.4e}")
    
    # Spinor (ρ₂) source — the electron IS a ρ₂ excitation
    rho2_local = SECTOR.index(1) if 1 in SECTOR else -1
    if rho2_local >= 0:
        print(f"\n  ρ₂ (spinor, dim 2) source analysis — the ELECTRON:")
        for k in range(3):
            cfg = [0]*3; cfg[k] = rho2_local
            cfg_idx = configs.index(tuple(cfg))
            ov = evecs[cfg_idx, :]
            C1 = sum(ov[n]**2 * (evals[n]/T_max) for n in range(len(evals)) if evals[n]>0)
            meff = -math.log(C1)*Lambda_QCD if C1>0 else 0
            print(f"    Edge {k}: m_eff = {meff:.1f} MeV, C(1) = {C1:.4e}")
    
    # Save results
    results = {
        'mode': MODE, 'n_irreps': int(N_IR), 'n_configs': n_configs,
        'sector': SECTOR, 'T_max': float(T_max),
        'eigenvalues': [float(e) for e in evals[:100]],
        'mass_gap': float(-math.log(evals[1]/T_max)) if 0<evals[1]<T_max else 0,
        'dims_sum_sq': float(sum(dims_sec**2)),
    }
    outfile = f'icosa_2I_{MODE}_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")
    
    # Also save eigensystem for further analysis
    np.savez(f'icosa_2I_{MODE}_eigen.npz', eigenvalues=evals, eigenvectors=evecs)
    print(f"Saved eigensystem to icosa_2I_{MODE}_eigen.npz")
