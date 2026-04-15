#!/usr/bin/env python3
"""
THREE-GENERATION TRANSFER MATRIX
=================================
Same dodecahedral geometry as qcd1024_numba.py, but with the
character table swapped for A4, Z3, or Z2 subgroups.

Usage:
  python3 qcd_generations.py A5      # 5^5=3125 configs (original)
  python3 qcd_generations.py A4      # 3^5=243 configs
  python3 qcd_generations.py Z3      # 2^5=32 configs
  python3 qcd_generations.py Z2      # 2^5=32 configs

Requirements: pip install numba
"""
import numpy as np
import numba
from numba import njit
import math, time, json, os, sys
from itertools import product as iprod
from collections import deque

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha_fw = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda_QCD = 332

# ================================================================
# GROUP CHARACTER TABLES
# ================================================================
def get_group(name):
    """Return dims, chars, class_sizes, order, group_name for each group."""
    if name == 'A5':
        dims = np.array([1, 3, 3, 4, 5], dtype=np.float64)
        chars = np.array([
            [1, 1, 1, 1, 1],
            [3, -1, 0, phi, 1-phi],
            [3, -1, 0, 1-phi, phi],
            [4, 0, 1, -1, -1],
            [5, 1, -1, 0, 0],
        ], dtype=np.float64)
        class_sizes = np.array([1, 15, 20, 12, 12], dtype=np.float64)
        order = 60
    elif name == 'A4':
        # Real irreps: 1(dim1), epsilon(dim2), 3(dim3)
        dims = np.array([1, 2, 3], dtype=np.float64)
        chars = np.array([
            [1, 1, 1, 1],
            [2, 2, -1, -1],
            [3, -1, 0, 0],
        ], dtype=np.float64)
        class_sizes = np.array([1, 3, 4, 4], dtype=np.float64)
        order = 12
    elif name == 'Z3':
        # Real irreps: 1(dim1), rho(dim2)
        dims = np.array([1, 2], dtype=np.float64)
        chars = np.array([
            [1, 1, 1],
            [2, -1, -1],
        ], dtype=np.float64)
        class_sizes = np.array([1, 1, 1], dtype=np.float64)
        order = 3
    elif name == 'Z2':
        dims = np.array([1, 1], dtype=np.float64)
        chars = np.array([
            [1, 1],
            [1, -1],
        ], dtype=np.float64)
        class_sizes = np.array([1, 1], dtype=np.float64)
        order = 2
    else:
        raise ValueError(f"Unknown group: {name}")
    return dims, chars, class_sizes, order, name

# ================================================================
# DODECAHEDRON GEOMETRY (identical to qcd1024_numba.py)
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
dv = np.array(dv)
edges = []; adj = [[] for _ in range(20)]
for i in range(20):
    for j in range(i+1, 20):
        if abs(np.linalg.norm(dv[i]-dv[j])-2/phi) < 0.01:
            eidx = len(edges); edges.append((i,j))
            adj[i].append(eidx); adj[j].append(eidx)

def get_eidx(v1, v2):
    key = (min(v1,v2), max(v1,v2))
    for i, e in enumerate(edges):
        if e == key: return i
    return None

_faces = []
for start in range(20):
    nbs = set()
    for e in adj[start]: a,b = edges[e]; nbs.add(b if a==start else a)
    for v1 in nbs:
        n1 = set()
        for e in adj[v1]: a,b = edges[e]; n1.add(b if a==v1 else a)
        for v2 in (n1-{start}):
            n2 = set()
            for e in adj[v2]: a,b = edges[e]; n2.add(b if a==v2 else a)
            for v3 in (n2-{start,v1}):
                n3 = set()
                for e in adj[v3]: a,b = edges[e]; n3.add(b if a==v3 else a)
                for v4 in (n3-{start,v1,v2}):
                    n4 = set()
                    for e in adj[v4]: a,b = edges[e]; n4.add(b if a==v4 else a)
                    if start in n4:
                        f = tuple(sorted([start,v1,v2,v3,v4]))
                        if f not in _faces: _faces.append(f)
_opp = None
for i in range(len(_faces)):
    for j in range(i+1, len(_faces)):
        if len(set(_faces[i])&set(_faces[j])) == 0: _opp = (i,j); break
    if _opp: break
face_A = _faces[_opp[0]]; face_B = _faces[_opp[1]]
def face_edg(fv):
    fv = sorted(fv); r = []
    for i in range(len(fv)):
        for j in range(i+1, len(fv)):
            e = get_eidx(fv[i], fv[j])
            if e is not None: r.append(e)
    return r
edges_A = face_edg(face_A); edges_B = face_edg(face_B)
visited = [False]*20; order = []; q = deque()
for v in sorted(face_A): q.append(v); visited[v] = True
while q:
    v = q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j = edges[eidx]; nb = j if i==v else i
        if not visited[nb]: visited[nb] = True; q.append(nb)

# ================================================================
# CONTRACTION PLAN (identical to qcd1024_numba.py)
# ================================================================
boundary_A = set(edges_A)
boundary_B = set(edges_B)
plan_edge_types = np.zeros((20, 3), dtype=np.int32)
plan_edge_indices = np.zeros((20, 3), dtype=np.int32)
plan_n_new = np.zeros(20, dtype=np.int32)
plan_n_open_before = np.zeros(20, dtype=np.int32)
plan_n_open_after = np.zeros(20, dtype=np.int32)
plan_closing = np.full((20, 3), -1, dtype=np.int32)
plan_n_closing = np.zeros(20, dtype=np.int32)
plan_surviving = np.full((20, 8), -1, dtype=np.int32)
plan_n_surviving = np.zeros(20, dtype=np.int32)

processed = set()
open_edge_list = []
for vi, v in enumerate(order):
    v_edges = adj[v]
    new_free_edges = []
    closing_free_positions = []
    for ei, eidx in enumerate(v_edges):
        i, j = edges[eidx]; other = j if i==v else i
        if eidx in boundary_A:
            plan_edge_types[vi, ei] = 0
            plan_edge_indices[vi, ei] = edges_A.index(eidx)
        elif eidx in boundary_B:
            plan_edge_types[vi, ei] = 1
            plan_edge_indices[vi, ei] = edges_B.index(eidx)
        elif other in processed:
            plan_edge_types[vi, ei] = 2
            pos = open_edge_list.index(eidx)
            plan_edge_indices[vi, ei] = pos
            closing_free_positions.append(pos)
        else:
            plan_edge_types[vi, ei] = 3
            plan_edge_indices[vi, ei] = len(new_free_edges)
            new_free_edges.append(eidx)
    plan_n_new[vi] = len(new_free_edges)
    plan_n_open_before[vi] = len(open_edge_list)
    plan_n_closing[vi] = len(closing_free_positions)
    for ci, p in enumerate(sorted(closing_free_positions)):
        plan_closing[vi, ci] = p
    cp_set = set(closing_free_positions)
    survivors = [p for p in range(len(open_edge_list)) if p not in cp_set]
    plan_n_surviving[vi] = len(survivors)
    for si, p in enumerate(survivors):
        plan_surviving[vi, si] = p
    plan_n_open_after[vi] = len(survivors) + len(new_free_edges)
    for pos in sorted(closing_free_positions, reverse=True):
        open_edge_list.pop(pos)
    open_edge_list.extend(new_free_edges)
    processed.add(v)

# ================================================================
# NUMBA KERNEL — parameterised by N_IRREPS
# ================================================================
# We use N_MAX=5 for array sizes (safe upper bound) and pass
# the actual n_irreps as a parameter.

N_MAX = 5
POWERS = np.array([N_MAX**i for i in range(10)], dtype=np.int64)

@njit(cache=False)
def decode_label(idx, pos, n_irreps):
    p = np.int64(1)
    for k in range(pos):
        p *= n_irreps
    return (idx // p) % n_irreps

@njit(cache=False)
def compute_T(la, lb, threej_3d, dims_1d, n_irreps,
              plan_edge_types, plan_edge_indices,
              plan_n_new, plan_n_open_before, plan_n_open_after,
              plan_closing, plan_n_closing,
              plan_surviving, plan_n_surviving):
    
    state = np.zeros(1, dtype=np.float64)
    state[0] = 1.0
    
    for vi in range(20):
        n_open = plan_n_open_before[vi]
        n_new = plan_n_new[vi]
        n_closing = plan_n_closing[vi]
        n_surv = plan_n_surviving[vi]
        n_open_out = plan_n_open_after[vi]
        
        state_size_in = np.int64(1)
        for k in range(n_open):
            state_size_in *= n_irreps
        state_size_out = np.int64(1)
        for k in range(n_open_out):
            state_size_out *= n_irreps
        
        new_state = np.zeros(state_size_out, dtype=np.float64)
        
        n_new_combos = np.int64(1)
        for k in range(n_new):
            n_new_combos *= n_irreps
        
        for old_idx in range(state_size_in):
            w = state[old_idx]
            if w == 0.0:
                continue
            
            surv_labels = np.zeros(8, dtype=np.int64)
            for si in range(n_surv):
                surv_labels[si] = decode_label(old_idx, plan_surviving[vi, si], n_irreps)
            
            close_labels = np.zeros(3, dtype=np.int64)
            for ci in range(n_closing):
                close_labels[ci] = decode_label(old_idx, plan_closing[vi, ci], n_irreps)
            
            for new_combo in range(n_new_combos):
                new_labels = np.zeros(3, dtype=np.int64)
                tmp = new_combo
                ew = 1.0
                for ni in range(n_new):
                    nl = tmp % n_irreps
                    new_labels[ni] = nl
                    ew *= dims_1d[nl]
                    tmp //= n_irreps
                
                if ew == 0.0:
                    continue
                
                j0 = np.int64(0); j1 = np.int64(0); j2 = np.int64(0)
                
                for ei in range(3):
                    et = plan_edge_types[vi, ei]
                    eidx = plan_edge_indices[vi, ei]
                    if et == 0:
                        val = la[eidx]
                    elif et == 1:
                        val = lb[eidx]
                    elif et == 2:
                        val = np.int64(0)
                        for ci in range(n_closing):
                            if plan_closing[vi, ci] == eidx:
                                val = close_labels[ci]
                                break
                    else:
                        val = new_labels[eidx]
                    
                    if ei == 0: j0 = val
                    elif ei == 1: j1 = val
                    else: j2 = val
                
                vw = threej_3d[j0, j1, j2]
                if vw == 0.0:
                    continue
                
                new_idx = np.int64(0)
                pw = np.int64(1)
                for si in range(n_surv):
                    new_idx += surv_labels[si] * pw
                    pw *= n_irreps
                for ni in range(n_new):
                    new_idx += new_labels[ni] * pw
                    pw *= n_irreps
                
                new_state[new_idx] += w * vw * ew
        
        state = new_state
    
    return state[0]


# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    GROUP = sys.argv[1] if len(sys.argv) > 1 else 'A5'
    
    dims, chars, class_sizes, group_order, group_name = get_group(GROUP)
    n_irreps = len(dims)
    
    # Compute 3j symbols for this group
    n_classes = len(class_sizes)
    threej = np.zeros((n_irreps, n_irreps, n_irreps), dtype=np.float64)
    for a in range(n_irreps):
        for b in range(n_irreps):
            for c in range(n_irreps):
                threej[a, b, c] = round(
                    sum(class_sizes[k]*chars[a,k]*chars[b,k]*chars[c,k]
                        for k in range(n_classes)) / group_order)
    
    sector_irreps = list(range(n_irreps))
    configs = list(iprod(sector_irreps, repeat=5))
    n_configs = len(configs)
    
    print(f"THREE-GENERATION TRANSFER MATRIX")
    print(f"Group: {group_name} (order {group_order}), {n_irreps} irreps")
    print(f"Configs: {n_configs} ({n_irreps}^5)")
    print(f"Dims: {[int(d) for d in dims]}")
    print(f"3j nonzero: {np.count_nonzero(threej)}/{n_irreps**3}")
    
    # JIT warmup
    print("JIT compiling...", end=' ', flush=True)
    t0 = time.time()
    la_test = np.zeros(5, dtype=np.int64)
    n_ir = np.int64(n_irreps)
    _ = compute_T(la_test, la_test, threej, dims, n_ir,
                  plan_edge_types, plan_edge_indices,
                  plan_n_new, plan_n_open_before, plan_n_open_after,
                  plan_closing, plan_n_closing,
                  plan_surviving, plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")
    
    # Compute full matrix
    print(f"Computing {n_configs}x{n_configs} matrix...")
    T = np.zeros((n_configs, n_configs), dtype=np.float64)
    t_start = time.time()
    
    for ia in range(n_configs):
        ca = configs[ia]
        la = np.array(ca, dtype=np.int64)
        for ib in range(n_configs):
            cb = configs[ib]
            lb = np.array(cb, dtype=np.int64)
            T[ia, ib] = compute_T(la, lb, threej, dims, n_ir,
                                   plan_edge_types, plan_edge_indices,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
        
        elapsed = time.time() - t_start
        rate = (ia+1) / elapsed
        eta = (n_configs - ia - 1) / rate if rate > 0 else 0
        if (ia+1) % max(1, n_configs//20) == 0 or ia == n_configs-1:
            print(f"  Row {ia+1}/{n_configs} ({(ia+1)/n_configs*100:.0f}%) "
                  f"[{elapsed:.1f}s, ETA {eta:.0f}s]")
    
    T_sym = (T + T.T) / 2
    
    # Analysis
    print(f"\n{'='*60}")
    print(f"EIGENVALUE ANALYSIS: {group_name}")
    print(f"{'='*60}")
    
    evals, evecs = np.linalg.eigh(T_sym)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    T_max = evals[0]
    
    print(f"T_max: {T_max:.6e}")
    print(f"Top 10 eigenvalue ratios:")
    for i in range(min(10, len(evals))):
        r = evals[i]/T_max if T_max > 0 else 0
        gap = -math.log(r) if 0 < r < 1 else 0
        mass = gap * Lambda_QCD
        print(f"  λ_{i}: ratio={r:.8f}, gap={gap:.4f}, mass={mass:.0f} MeV")
    
    # Mass spectrum
    print(f"\nMass spectrum:")
    known = [(135,'π'),(494,'K'),(548,'η'),(775,'ρ'),(782,'ω'),
             (938,'p'),(958,"η'"),(1020,'φ_m'),(1232,'Δ'),(1275,'f₂'),
             (1525,"f₂'")]
    for i in range(min(20, len(evals))):
        if 0 < evals[i] < T_max * 0.9999:
            gap = -math.log(evals[i]/T_max)
            mass = gap * Lambda_QCD
            best = min(known, key=lambda h: abs(h[0]-mass))
            err = abs(best[0]-mass)/best[0]*100
            mark = '★' if err<3 else '●' if err<5 else ''
            print(f"  {i:>3}: {mass:>7.0f} MeV -> {best[1]:>6} ({best[0]}) {err:5.1f}% {mark}")
    
    # Save results
    results = {
        'group': group_name,
        'order': group_order,
        'n_irreps': n_irreps,
        'n_configs': n_configs,
        'dims': [int(d) for d in dims],
        'T_max': float(T_max),
        'eigenvalues': [float(e) for e in evals[:50]],
        'mass_gap': float(-math.log(evals[1]/T_max)) if evals[1] > 0 and evals[1] < T_max else 0,
    }
    
    outfile = f'generation_{group_name}_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")
