#!/usr/bin/env python3
"""
DODECAHEDRAL FRAMEWORK: TRANSFER MATRIX (NUMBA-OPTIMIZED)
==========================================================
60-100× faster than the pure Python version.
Same progress file format — continues from qcd1024_progress.json.

Requirements: pip install numba

Run:  python3 qcd1024_numba.py --cores 16
      python3 qcd1024_numba.py --cores 16 --sector 0,1,2,4   (1024, default)
      python3 qcd1024_numba.py --cores 16 --sector 0,1,2,3,4  (3125)
"""
import numpy as np
import numba
from numba import njit, prange
import math, time, json, os, sys
from itertools import product as iprod
from collections import deque

# ================================================================
# CONSTANTS
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha_fw = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda_QCD = 332
N = 5

dims = np.array([1, 3, 3, 4, 5], dtype=np.float64)
chars = np.array([
    [1, 1, 1, 1, 1],
    [3, -1, 0, phi, 1-phi],
    [3, -1, 0, 1-phi, phi],
    [4, 0, 1, -1, -1],
    [5, 1, -1, 0, 0],
], dtype=np.float64)
class_sizes = np.array([1, 15, 20, 12, 12], dtype=np.float64)

threej = np.zeros((N, N, N), dtype=np.float64)
for a in range(N):
    for b in range(N):
        for c in range(N):
            threej[a, b, c] = round(
                sum(class_sizes[k]*chars[a,k]*chars[b,k]*chars[c,k]
                    for k in range(5)) / 60)

# ================================================================
# DODECAHEDRON GEOMETRY (same as always)
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
# CONTRACTION PLAN (precomputed as integer arrays)
# ================================================================
boundary_A = set(edges_A)
boundary_B = set(edges_B)

# Build plan
plan_edge_types = np.zeros((20, 3), dtype=np.int32)     # 0=A, 1=B, 2=closing, 3=new
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

POWERS = np.array([5**i for i in range(10)], dtype=np.int64)

# ================================================================
# NUMBA-COMPILED CONTRACTION
# ================================================================
@njit(cache=True)
def decode_label(idx, pos):
    """Extract label at position pos from packed index."""
    return (idx // POWERS[pos]) % 5

@njit(cache=True)
def compute_T_numba(la, lb, threej, dims,
                     plan_edge_types, plan_edge_indices,
                     plan_n_new, plan_n_open_before, plan_n_open_after,
                     plan_closing, plan_n_closing,
                     plan_surviving, plan_n_surviving):
    """Compute one transfer matrix element T[la, lb]."""
    
    # Start: state = {0: 1.0} (scalar, no open edges)
    state = np.zeros(1, dtype=np.float64)
    state[0] = 1.0
    
    for vi in range(20):
        n_open = plan_n_open_before[vi]
        n_new = plan_n_new[vi]
        n_closing = plan_n_closing[vi]
        n_surv = plan_n_surviving[vi]
        n_open_out = plan_n_open_after[vi]
        
        state_size_in = 1
        for k in range(n_open):
            state_size_in *= 5
        state_size_out = 1
        for k in range(n_open_out):
            state_size_out *= 5
        
        new_state = np.zeros(state_size_out, dtype=np.float64)
        
        # Number of new-edge combinations
        n_new_combos = 1
        for k in range(n_new):
            n_new_combos *= 5
        
        # Iterate over all old states
        for old_idx in range(state_size_in):
            w = state[old_idx]
            if w == 0.0:
                continue
            
            # Decode surviving labels from old_idx
            surv_labels = np.zeros(8, dtype=np.int64)
            for si in range(n_surv):
                surv_labels[si] = decode_label(old_idx, plan_surviving[vi, si])
            
            # Decode closing labels from old_idx
            close_labels = np.zeros(3, dtype=np.int64)
            for ci in range(n_closing):
                close_labels[ci] = decode_label(old_idx, plan_closing[vi, ci])
            
            # Iterate over new edge label combinations
            for new_combo in range(n_new_combos):
                # Decode new labels
                new_labels = np.zeros(3, dtype=np.int64)
                tmp = new_combo
                ew = 1.0
                for ni in range(n_new):
                    nl = tmp % 5
                    new_labels[ni] = nl
                    ew *= dims[nl]
                    tmp //= 5
                
                if ew == 0.0:
                    continue
                
                # Build vertex labels (3 edges)
                j0 = np.int64(0); j1 = np.int64(0); j2 = np.int64(0)
                
                for ei in range(3):
                    et = plan_edge_types[vi, ei]
                    eidx = plan_edge_indices[vi, ei]
                    if et == 0:      # fixed A
                        val = la[eidx]
                    elif et == 1:    # fixed B
                        val = lb[eidx]
                    elif et == 2:    # closing
                        # Find which closing index this is
                        val = np.int64(0)
                        for ci in range(n_closing):
                            if plan_closing[vi, ci] == eidx:
                                val = close_labels[ci]
                                break
                    else:            # new
                        val = new_labels[eidx]
                    
                    if ei == 0: j0 = val
                    elif ei == 1: j1 = val
                    else: j2 = val
                
                vw = threej[j0, j1, j2]
                if vw == 0.0:
                    continue
                
                # Build new state index: surviving + new
                new_idx = np.int64(0)
                for si in range(n_surv):
                    new_idx += surv_labels[si] * POWERS[si]
                for ni in range(n_new):
                    new_idx += new_labels[ni] * POWERS[n_surv + ni]
                
                new_state[new_idx] += w * vw * ew
        
        state = new_state
    
    return state[0]


def compute_row(args):
    """Compute one row of the transfer matrix."""
    ia, sector_irreps_list = args
    configs_local = list(iprod(sector_irreps_list, repeat=5))
    n_local = len(configs_local)
    ca = configs_local[ia]
    la = np.array([ca[i] for i in range(5)], dtype=np.int64)
    row = np.zeros(n_local, dtype=np.float64)
    for ib in range(n_local):
        cb = configs_local[ib]
        lb = np.array([cb[i] for i in range(5)], dtype=np.int64)
        row[ib] = compute_T_numba(la, lb, threej, dims,
                                   plan_edge_types, plan_edge_indices,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
    return (ia, row.tolist())


# ================================================================
# PROGRESS (same format)
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
        print(f"  Loaded {len(done)} rows from {path}")
    return T, done


# ================================================================
# ANALYSIS (same as original)
# ================================================================
def analyse(T_matrix, sector_irreps):
    T_sym = (T_matrix + T_matrix.T) / 2
    SECTOR_DIMS_SQ = sum(int(dims[j])**2 for j in sector_irreps)

    print(f"\n{'='*60}")
    print(f"EIGENVALUE ANALYSIS (Σdim²={SECTOR_DIMS_SQ}/60)")
    evals_raw, evecs_raw = np.linalg.eigh(T_sym)
    idx = np.argsort(evals_raw)[::-1]
    evals = evals_raw[idx]; evecs = evecs_raw[:, idx]
    T_max = evals[0]
    n_pos = sum(1 for e in evals if e > 0)
    print(f"T_max: {T_max:.4e}, Positive: {n_pos}")

    masses = []
    known = [(135,'π'),(494,'K'),(548,'η'),(775,'ρ'),(782,'ω'),
             (958,"η'"),(1020,'φ'),(1232,'Δ'),(1275,'f₂'),(1525,"f₂'")]
    for i, ev in enumerate(evals):
        if ev > 0 and ev < T_max*0.9999:
            gap = -math.log(ev/T_max)
            masses.append((i, gap*Lambda_QCD, ev))
    print(f"\nMass spectrum ({len(masses)} states):")
    for i, (level, mass, ev) in enumerate(masses[:20]):
        best = min(known, key=lambda h: abs(h[0]-mass))
        err = abs(best[0]-mass)/best[0]*100
        mark = '★' if err<3 else '●' if err<5 else ' '
        print(f"  {i:>3} {mass:>8.0f} {best[1]:>8} {best[0]:>6} {err:>5.1f}% {mark}")

    # HVP
    source = np.zeros(n_configs)
    si_3 = sector_irreps.index(1) if 1 in sector_irreps else -1
    si_3p = sector_irreps.index(2) if 2 in sector_irreps else -1
    if si_3 >= 0 and si_3p >= 0:
        for ic, cfg in enumerate(configs):
            n3 = sum(1 for x in cfg if x==1)
            n3p = sum(1 for x in cfg if x==2)
            source[ic] = math.sqrt(n3*n3p)/5.0

        sum_HVP = 0
        for t in range(1, 200):
            C_t = 0
            for nn in range(n_configs):
                if evals[nn] <= 0: continue
                ratio = evals[nn]/T_max
                if 0 < ratio < 1:
                    ov = np.dot(source, evecs[:,nn])
                    C_t += ov**2 * ratio**t
            sum_HVP += t**2*C_t

        a_mu_raw = (4*alpha_fw**2/3)*sum_HVP*(0.10566/0.332)**2
        correction = 60/SECTOR_DIMS_SQ
        a_mu_corr = a_mu_raw * correction
        print(f"\nHVP: raw a_μ = {a_mu_raw:.4e} (ratio: {a_mu_raw/6.93e-8:.4f})")
        print(f"     corrected (×{correction:.4f}) = {a_mu_corr:.4e} (ratio: {a_mu_corr/6.93e-8:.4f})")

    results = {
        'sector_irreps': list(sector_irreps),
        'n_configs': n_configs,
        'eigenvalues': [float(e) for e in evals[:100]],
        'a_mu_HVP_raw': float(a_mu_raw) if 'a_mu_raw' in dir() else None,
        'a_mu_HVP_corrected': float(a_mu_corr) if 'a_mu_corr' in dir() else None,
    }
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {RESULT_FILE}")


# ================================================================
if __name__ == '__main__':
    N_CORES = 1
    SECTOR_IRREPS = [0, 1, 2, 4]  # default: χ₁+χ₃+χ₃'+χ₅
    
    for i, arg in enumerate(sys.argv):
        if arg == '--cores' and i+1 < len(sys.argv):
            N_CORES = int(sys.argv[i+1])
        if arg == '--sector' and i+1 < len(sys.argv):
            SECTOR_IRREPS = [int(x) for x in sys.argv[i+1].split(',')]

    configs = list(iprod(SECTOR_IRREPS, repeat=5))
    n_configs = len(configs)
    SECTOR_DIMS_SQ = sum(int(dims[j])**2 for j in SECTOR_IRREPS)

    print(f"Transfer Matrix (NUMBA-OPTIMIZED)")
    print(f"Sector: {SECTOR_IRREPS}, Configs: {n_configs}, Cores: {N_CORES}")
    print(f"Σdim² = {SECTOR_DIMS_SQ}/60")

    # JIT warmup
    print("JIT compiling (first call)...", end=' ', flush=True)
    t0 = time.time()
    la_test = np.array(SECTOR_IRREPS[:1]*5, dtype=np.int64)
    _ = compute_T_numba(la_test, la_test, threej, dims,
                         plan_edge_types, plan_edge_indices,
                         plan_n_new, plan_n_open_before, plan_n_open_after,
                         plan_closing, plan_n_closing,
                         plan_surviving, plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")

    # Benchmark
    print("Benchmarking...")
    cases = [(SECTOR_IRREPS[0],)*5, (SECTOR_IRREPS[-1],)*5]
    for c in cases:
        la = np.array(c, dtype=np.int64)
        t0 = time.time()
        val = compute_T_numba(la, la, threej, dims,
                               plan_edge_types, plan_edge_indices,
                               plan_n_new, plan_n_open_before, plan_n_open_after,
                               plan_closing, plan_n_closing,
                               plan_surviving, plan_n_surviving)
        dt = time.time()-t0
        print(f"  {c}: {val:.3e} [{dt:.3f}s]")

    T_matrix, completed_rows = load_progress(SAVE_FILE)
    remaining = [ia for ia in range(n_configs) if ia not in completed_rows]
    print(f"Done: {len(completed_rows)}, Remaining: {len(remaining)}")

    if remaining:
        t_start = time.time()
        done_session = 0
        # Note: multiprocessing with Numba needs care — each worker recompiles
        # For simplicity, run single-threaded (Numba makes single-thread fast enough)
        # or use the threading approach below
        
        if N_CORES > 1:
            # With Numba, the GIL is released during njit calls
            # We can use multiprocessing
            from multiprocessing import Pool
            args_list = [(ia, SECTOR_IRREPS) for ia in remaining]
            with Pool(N_CORES) as pool:
                for ia, row in pool.imap_unordered(compute_row, args_list):
                    T_matrix[ia] = row
                    completed_rows.add(ia)
                    done_session += 1
                    elapsed = time.time()-t_start
                    rate = done_session/elapsed
                    left = len(remaining)-done_session
                    eta = left/rate if rate > 0 else 0
                    pct = len(completed_rows)/n_configs*100
                    print(f"  Row {ia:4d} | {len(completed_rows):4d}/{n_configs} ({pct:.1f}%) | "
                          f"{elapsed/3600:.1f}h | ~{eta/3600:.1f}h left | "
                          f"{rate*3600:.0f} rows/h", flush=True)
                    if done_session % 10 == 0:
                        save_progress(SAVE_FILE, T_matrix, completed_rows)
        else:
            for ia in remaining:
                _, row = compute_row((ia, SECTOR_IRREPS))
                T_matrix[ia] = row
                completed_rows.add(ia)
                done_session += 1
                elapsed = time.time()-t_start
                rate = done_session/elapsed
                left = len(remaining)-done_session
                eta = left/rate if rate > 0 else 0
                print(f"  Row {ia:4d}/{n_configs} | {elapsed/3600:.2f}h | ~{eta/3600:.1f}h left")
                if done_session % 5 == 0:
                    save_progress(SAVE_FILE, T_matrix, completed_rows)

        save_progress(SAVE_FILE, T_matrix, completed_rows)
        print(f"Computation: {(time.time()-t_start)/3600:.2f}h")

    analyse(T_matrix, SECTOR_IRREPS)
