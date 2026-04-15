#!/usr/bin/env python3
"""
DODECAHEDRAL SUB-SECTORS — NUMBA-COMPILED
==========================================
Six fast computations using JIT-compiled tensor contraction.
~100× faster than pure Python version.

Usage: python3 dodec_subsectors_numba.py --cores 20

Requires: pip install numpy numba
"""
import numpy as np
import math, time, json, os, sys
from numba import njit
from itertools import product as iprod
from collections import deque

# ================================================================
# A₅ DATA
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
Lambda_QCD = 332
N_IRREPS = 5

dims = np.array([1.0, 3.0, 3.0, 4.0, 5.0])
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

# Full 3j symbols (all 5 irreps, always — internal edges are unrestricted)
threej = np.zeros((N_IRREPS, N_IRREPS, N_IRREPS), dtype=np.float64)
for a in range(N_IRREPS):
    for b in range(N_IRREPS):
        for c in range(N_IRREPS):
            threej[a, b, c] = round(
                sum(class_sizes[k]*chars[a][k]*chars[b][k]*chars[c][k]
                    for k in range(5)) / G_ORDER)

# ================================================================
# DODECAHEDRON (20 vertices, 30 edges, 12 faces, degree 3)
# ================================================================
print("Building dodecahedron...", end=' ', flush=True)
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
assert len(edges) == 30

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
assert len(_faces) == 12

def _get_edge_idx(v1,v2):
    key=(min(v1,v2),max(v1,v2))
    for i,e in enumerate(edges):
        if e==key: return i
    return None

def _face_edges(fv):
    fv=sorted(fv); r=[]
    for i in range(len(fv)):
        for j in range(i+1,len(fv)):
            e=_get_edge_idx(fv[i],fv[j])
            if e is not None: r.append(e)
    return r

# Find opposite face pair
_opp = None
for i in range(len(_faces)):
    for j in range(i+1,len(_faces)):
        if len(set(_faces[i])&set(_faces[j]))==0:
            _opp=(i,j); break
    if _opp: break

face_A = _faces[_opp[0]]; face_B = _faces[_opp[1]]
edges_A = _face_edges(face_A); edges_B = _face_edges(face_B)

# BFS vertex order from face A
visited=[False]*Nv; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

print(f"done. Face A={face_A}, Face B={face_B}")

# ================================================================
# CONTRACTION PLAN (precomputed numpy arrays for Numba)
# ================================================================
MAX_E = 3   # dodecahedron vertex degree = 3
MAX_OPEN = 10

plan_edge_types   = np.zeros((20, MAX_E), dtype=np.int32)
plan_edge_indices  = np.zeros((20, MAX_E), dtype=np.int32)
plan_n_edges       = np.zeros(20, dtype=np.int32)
plan_n_new         = np.zeros(20, dtype=np.int32)
plan_n_open_before = np.zeros(20, dtype=np.int32)
plan_n_open_after  = np.zeros(20, dtype=np.int32)
plan_closing       = np.full((20, MAX_E), -1, dtype=np.int32)
plan_n_closing     = np.zeros(20, dtype=np.int32)
plan_surviving     = np.full((20, MAX_OPEN), -1, dtype=np.int32)
plan_n_surviving   = np.zeros(20, dtype=np.int32)

boundary_A = set(edges_A); boundary_B = set(edges_B)
processed = set(); open_edge_list = []; max_open = 0

for vi, v in enumerate(order):
    new_free = []; closing_positions = []
    plan_n_edges[vi] = len(adj[v])
    for ei, eidx in enumerate(adj[v]):
        i, j = edges[eidx]; other = j if i == v else i
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
            closing_positions.append(pos)
        else:
            plan_edge_types[vi, ei] = 3
            plan_edge_indices[vi, ei] = len(new_free)
            new_free.append(eidx)
    plan_n_new[vi] = len(new_free)
    plan_n_open_before[vi] = len(open_edge_list)
    plan_n_closing[vi] = len(closing_positions)
    for ci, p in enumerate(sorted(closing_positions)):
        plan_closing[vi, ci] = p
    cp_set = set(closing_positions)
    survivors = [p for p in range(len(open_edge_list)) if p not in cp_set]
    plan_n_surviving[vi] = len(survivors)
    for si, p in enumerate(survivors):
        plan_surviving[vi, si] = p
    plan_n_open_after[vi] = len(survivors) + len(new_free)
    for pos in sorted(closing_positions, reverse=True):
        open_edge_list.pop(pos)
    open_edge_list.extend(new_free)
    processed.add(v)
    if len(open_edge_list) > max_open:
        max_open = len(open_edge_list)

print(f"Max open edges: {max_open}")
print(f"Peak state size: 5^{max_open} = {5**max_open:,}")

# ================================================================
# NUMBA KERNEL (degree-3 vertices → threej, 20 vertices)
# ================================================================
@njit(cache=True)
def compute_T_numba(la, lb, n_ir, dims_arr, threej_arr,
                     pe_types, pe_indices, pn_edges,
                     pn_new, pn_open_before, pn_open_after,
                     p_closing, pn_closing, p_surviving, pn_surviving):
    state = np.zeros(1, dtype=np.float64)
    state[0] = 1.0
    for vi in range(20):  # 20 dodecahedral vertices
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
            old_lab = np.zeros(10, dtype=np.int64)
            tmp = old_idx
            for oi in range(n_open): old_lab[oi] = tmp % n_ir; tmp //= n_ir
            s_lab = np.zeros(10, dtype=np.int64)
            for si in range(n_surv): s_lab[si] = old_lab[p_surviving[vi, si]]
            c_lab = np.zeros(3, dtype=np.int64)
            for ci in range(n_closing): c_lab[ci] = old_lab[p_closing[vi, ci]]
            for nc in range(n_nc):
                n_lab = np.zeros(3, dtype=np.int64)
                tmp = nc; ew = 1.0
                for ni in range(n_new):
                    nl = tmp % n_ir; n_lab[ni] = nl
                    ew *= dims_arr[nl]; tmp //= n_ir
                vl = np.zeros(3, dtype=np.int64)  # degree 3!
                for ei in range(n_e):
                    et = pe_types[vi, ei]; eix = pe_indices[vi, ei]
                    if et == 0: vl[ei] = la[eix]
                    elif et == 1: vl[ei] = lb[eix]
                    elif et == 2:
                        val = np.int64(0)
                        for ci in range(n_closing):
                            if p_closing[vi, ci] == eix: val = c_lab[ci]; break
                        vl[ei] = val
                    else: vl[ei] = n_lab[eix]
                vw = threej_arr[vl[0], vl[1], vl[2]]  # 3j, not 5j!
                if vw == 0.0: continue
                nidx = np.int64(0); pw = np.int64(1)
                for si in range(n_surv): nidx += s_lab[si]*pw; pw *= n_ir
                for ni in range(n_new): nidx += n_lab[ni]*pw; pw *= n_ir
                new_state[nidx] += w * vw * ew
        state = new_state
    return state[0]

# ================================================================
# WORKER (module-level, picklable, uses globals for plan arrays)
# ================================================================
# These are set once at module level and inherited by workers
_N_IR = np.int64(N_IRREPS)

def compute_row(args):
    ia, sector_indices_list = args
    configs = list(iprod(sector_indices_list, repeat=5))
    n = len(configs)
    la = np.array(configs[ia], dtype=np.int64)
    row = np.zeros(n, dtype=np.float64)
    for ib in range(n):
        lb = np.array(configs[ib], dtype=np.int64)
        row[ib] = compute_T_numba(la, lb, _N_IR, dims, threej,
                                   plan_edge_types, plan_edge_indices, plan_n_edges,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
    return (ia, row.tolist())

# ================================================================
# SECTOR COMPUTATION
# ================================================================
def run_sector(sector_indices, sector_name, result_file, n_cores):
    sector_names = [irr_names[i] for i in sector_indices]
    dims_sq = sum(int(dims[i])**2 for i in sector_indices)
    configs = list(iprod(sector_indices, repeat=5))
    n_configs = len(configs)

    print(f"\n{'='*65}")
    print(f"  {sector_name}")
    print(f"  Sector: {' + '.join(sector_names)}")
    print(f"  Σdim² = {dims_sq}/60 ({dims_sq/60*100:.1f}%)")
    print(f"  Matrix: {n_configs}×{n_configs}, Elements: {n_configs**2:,}")
    print(f"  Cores: {n_cores}")
    print(f"{'='*65}")

    # JIT warmup
    print("  JIT compiling...", end=' ', flush=True)
    t0 = time.time()
    la_t = np.zeros(5, dtype=np.int64)
    _ = compute_T_numba(la_t, la_t, _N_IR, dims, threej,
                         plan_edge_types, plan_edge_indices, plan_n_edges,
                         plan_n_new, plan_n_open_before, plan_n_open_after,
                         plan_closing, plan_n_closing,
                         plan_surviving, plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")

    # Benchmark
    t0 = time.time()
    la_b = np.array(configs[0], dtype=np.int64)
    lb_b = np.array(configs[-1], dtype=np.int64)
    _ = compute_T_numba(la_b, lb_b, _N_IR, dims, threej,
                         plan_edge_types, plan_edge_indices, plan_n_edges,
                         plan_n_new, plan_n_open_before, plan_n_open_after,
                         plan_closing, plan_n_closing,
                         plan_surviving, plan_n_surviving)
    dt = time.time()-t0
    est_min = dt * n_configs**2 / max(n_cores, 1) / 60
    print(f"  Benchmark: {dt*1000:.1f}ms/element → est {est_min:.1f} min total")

    # Compute
    T_matrix = np.zeros((n_configs, n_configs))
    t_start = time.time()
    args = [(ia, sector_indices) for ia in range(n_configs)]

    if n_cores > 1 and n_configs > 4:
        from multiprocessing import Pool
        done = 0
        with Pool(n_cores) as pool:
            for ia, row in pool.imap_unordered(compute_row, args):
                T_matrix[ia] = row
                done += 1
                elapsed = time.time()-t_start
                rate = done / elapsed * 3600 if elapsed > 0 else 0
                remaining = (n_configs-done)/rate*3600 if rate>0 else 0
                if done % max(1, n_configs//20) == 0 or done == n_configs:
                    print(f"    Row {ia:>4} | {done:>4}/{n_configs} "
                          f"({done/n_configs*100:.0f}%) | {elapsed:.0f}s | "
                          f"{rate:.0f} rows/h")
    else:
        for a in args:
            ia, row = compute_row(a)
            T_matrix[ia] = row

    elapsed = time.time()-t_start
    print(f"  Computed in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Analysis
    T_sym = (T_matrix + T_matrix.T) / 2
    evals_raw, evecs_raw = np.linalg.eigh(T_sym)
    idx = np.argsort(evals_raw)[::-1]
    evals = evals_raw[idx]
    evecs = evecs_raw[:, idx]
    T_max = evals[0]
    n_pos = int(sum(1 for e in evals if e > 0))

    masses = []
    for i, ev in enumerate(evals):
        if ev > 0 and ev < T_max * 0.9999:
            gap = -math.log(ev/T_max)
            masses.append({'level':i, 'eigenvalue':float(ev),
                           'gap':float(gap), 'mass_MeV':float(gap*Lambda_QCD)})

    print(f"  T_max = {T_max:.6e}")
    print(f"  Positive: {n_pos}, Negative: {n_configs-n_pos}")

    print(f"  Mass spectrum (MeV):")
    for i, m in enumerate(masses[:15]):
        print(f"    {i}: {m['mass_MeV']:8.1f} MeV (gap={m['gap']:.4f})")

    # Vacuum composition
    gs = evecs[:, 0]
    irrep_content = {}
    for ir_idx in range(N_IRREPS):
        irrep_content[irr_names[ir_idx]] = 0.0
    for ic, cfg in enumerate(configs):
        w = gs[ic]**2
        for edge_irrep in cfg:
            irrep_content[irr_names[edge_irrep]] += w / 5.0

    print(f"  Vacuum composition:")
    for name, frac in irrep_content.items():
        if frac > 0.001:
            bar = '█' * int(frac * 40)
            print(f"    {name:>4}: {frac*100:5.1f}% {bar}")

    # Save
    results = {
        'computation': sector_name,
        'sector_indices': sector_indices,
        'sector_names': sector_names,
        'dims_sq': dims_sq,
        'n_configs': n_configs,
        'T_max': float(T_max),
        'n_positive': n_pos,
        'eigenvalues_top50': [float(e) for e in evals[:50]],
        'masses_MeV': [m['mass_MeV'] for m in masses[:30]],
        'mass_gaps': [m['gap'] for m in masses[:30]],
        'vacuum_composition': {k: float(v) for k,v in irrep_content.items()},
        'T_matrix': T_matrix.tolist(),
        'time_sec': elapsed,
    }
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  → Saved {result_file}")
    return results

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()

    N_CORES = 1
    for i, arg in enumerate(sys.argv):
        if arg == '--cores' and i+1 < len(sys.argv):
            N_CORES = int(sys.argv[i+1])

    print(f"\nDODECAHEDRAL SUB-SECTORS (Numba-compiled)")
    print(f"Cores: {N_CORES}")

    sectors = [
        ([1, 4], "QED core: χ₃+χ₅",
         "dodec_QED_32_results.json"),
        ([3, 4], "Dark photon: χ₄+χ₅",
         "dodec_dark_photon_32_results.json"),
        ([1, 2], "Matter-antimatter: χ₃+χ₃'",
         "dodec_matter_antimatter_32_results.json"),
        ([1, 2, 3], "Matter-dark portal: χ₃+χ₃'+χ₄",
         "dodec_matter_dark_243_results.json"),
        ([0, 3, 4], "Dark annihilation: χ₁+χ₄+χ₅",
         "dodec_dark_annihil_243_results.json"),
        ([1, 2, 3, 4], "No-vacuum: χ₃+χ₃'+χ₄+χ₅",
         "dodec_no_vacuum_1024_results.json"),
        ([0, 3], "V₄ sector: χ₁+χ₄ (vacuum + dark matter)",
        "dodec_V4_32_results.json"),
    ]

    all_results = {}
    t_total = time.time()

    for sector_indices, name, result_file in sectors:
        if os.path.exists(result_file):
            print(f"\n  {result_file} exists, skipping. Delete to recompute.")
            continue
        r = run_sector(sector_indices, name, result_file, N_CORES)
        all_results[name] = r

    total_time = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*65}")

    print(f"\n{'Sector':45} {'T_max':>12} {'1st mass':>10}")
    print(f"{'-'*70}")
    for name, r in all_results.items():
        mass = r['masses_MeV'][0] if r['masses_MeV'] else 0
        print(f"{name:45} {r['T_max']:>12.4e} {mass:>8.0f} MeV")
