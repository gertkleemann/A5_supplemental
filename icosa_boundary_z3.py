#!/usr/bin/env python3
"""
ICOSAHEDRAL BOUNDARY TRANSFER MATRIX — Z₃ REDUCED
====================================================
Exploits the Z₃ rotational symmetry of each triangular face to reduce
the 125×125 matrix to 45×45 unique computations (7.7× speedup).

The icosahedron's face stabilizer is Z₃: rotating the 3 edges of a
triangular face by 120° is an isometry that also rotates the opposite face.
Therefore T[σ(a), σ(b)] = T[a,b] for σ ∈ Z₃, and configs in the same
Z₃ orbit give identical T values.

Usage:
  python3 icosa_boundary_z3.py --sector 0,1,2,3,4 --cores 20
  python3 icosa_boundary_z3.py --sector 0,1,2,4 --cores 20

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
# A₅ CHARACTER TABLE
# ================================================================
dims_full = np.array([1, 3, 3, 4, 5], dtype=np.float64)
chars = np.array([
    [1, 1, 1, 1, 1],
    [3, -1, 0, phi, 1-phi],
    [3, -1, 0, 1-phi, phi],
    [4, 0, 1, -1, -1],
    [5, 1, -1, 0, 0],
], dtype=np.float64)
class_sizes = np.array([1, 15, 20, 12, 12], dtype=np.float64)

# ================================================================
# ICOSAHEDRON GEOMETRY (same as original)
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
for i in range(len(faces)):
    for j in range(i+1,len(faces)):
        if len(set(faces[i])&set(faces[j]))==0:
            ci=np.mean([iv[v] for v in faces[i]],axis=0)
            cj=np.mean([iv[v] for v in faces[j]],axis=0)
            if np.dot(ci,cj)<-0.5: opp_pair=(i,j); break
    if opp_pair: break

face_A=faces[opp_pair[0]]; face_B=faces[opp_pair[1]]
edges_A=face_edg(face_A); edges_B=face_edg(face_B)
print(f"done. Face A={face_A}, Face B={face_B}")

# BFS ordering
visited=[False]*12; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

# ================================================================
# CONTRACTION PLAN (same as original)
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
# NUMBA KERNEL (identical to original)
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

# ================================================================
# Z₃ ORBIT COMPUTATION
# ================================================================
def compute_z3_orbits(n_ir):
    """Group face configs (a,b,c) into Z₃ orbits.
    Z₃ acts as (a,b,c) -> (b,c,a) -> (c,a,b).
    Returns list of (representative, orbit_size, [members])."""
    configs = list(iprod(range(n_ir), repeat=3))
    seen = set()
    orbits = []
    for cfg in configs:
        if cfg in seen:
            continue
        a, b, c = cfg
        rot1 = (b, c, a)
        rot2 = (c, a, b)
        members = list(set([cfg, rot1, rot2]))
        for m in members:
            seen.add(m)
        # Representative = lexicographically smallest
        rep = min(members)
        orbits.append((rep, len(members), members))
    return orbits

def compute_row_z3(args):
    """Compute one row of the Z₃-reduced matrix."""
    orb_idx_a, orbit_reps_a, orbit_reps_b, sector_range = args
    rep_a = orbit_reps_a[orb_idx_a]
    la = np.array(rep_a, dtype=np.int64)
    row = np.zeros(len(orbit_reps_b), dtype=np.float64)
    for jj, rep_b in enumerate(orbit_reps_b):
        lb = np.array(rep_b, dtype=np.int64)
        row[jj] = compute_T_numba(la, lb, N_IR, dims_sec, fivej_sec,
                                   plan_edge_types, plan_edge_indices, plan_n_edges,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
    return (orb_idx_a, row.tolist())

# ================================================================
# SECTOR SETUP
# ================================================================
SECTOR = [0, 1, 2, 3, 4]; N_CORES = 1
for i, arg in enumerate(sys.argv):
    if arg == '--sector' and i+1 < len(sys.argv):
        SECTOR = [int(x) for x in sys.argv[i+1].split(',')]
    if arg == '--cores' and i+1 < len(sys.argv):
        N_CORES = int(sys.argv[i+1])

N_IR = np.int64(len(SECTOR))
dims_sec = np.array([dims_full[s] for s in SECTOR], dtype=np.float64)
fivej_sec = np.zeros((int(N_IR),)*5, dtype=np.float64)
for a in range(int(N_IR)):
    for b in range(int(N_IR)):
        for c in range(int(N_IR)):
            for d in range(int(N_IR)):
                for e in range(int(N_IR)):
                    fivej_sec[a,b,c,d,e] = round(
                        sum(class_sizes[k]*chars[SECTOR[a],k]*chars[SECTOR[b],k]*
                            chars[SECTOR[c],k]*chars[SECTOR[d],k]*chars[SECTOR[e],k]
                            for k in range(5)) / 60)

configs_full = list(iprod(range(int(N_IR)), repeat=3))
n_configs_full = len(configs_full)

# Compute Z₃ orbits
orbits = compute_z3_orbits(int(N_IR))
n_orbits = len(orbits)
orbit_reps = [o[0] for o in orbits]
orbit_sizes = [o[1] for o in orbits]
orbit_members = [o[2] for o in orbits]

# Map each config to its orbit index
config_to_orbit = {}
for oi, (rep, sz, members) in enumerate(orbits):
    for m in members:
        config_to_orbit[m] = oi

# ================================================================
if __name__ == '__main__':
    print(f"\nICOSAHEDRAL BOUNDARY TRANSFER MATRIX — Z₃ REDUCED")
    print(f"Sector: {SECTOR} ({N_IR} irreps), Cores: {N_CORES}")
    print(f"Full matrix: {n_configs_full}×{n_configs_full} = {n_configs_full**2} elements")
    print(f"Z₃ orbits: {n_orbits} (orbit sizes: {sum(1 for s in orbit_sizes if s==1)} fixed + "
          f"{sum(1 for s in orbit_sizes if s==3)} triples)")
    print(f"Reduced matrix: {n_orbits}×{n_orbits} = {n_orbits**2} elements")
    print(f"Speedup: {n_configs_full**2/n_orbits**2:.1f}×")
    print(f"Peak state: {int(N_IR)}^{max_open} = {int(N_IR)**max_open}")
    print(f"5j nonzero: {np.count_nonzero(fivej_sec)}/{int(N_IR)**5}")

    # JIT compile
    print("JIT compiling...", end=' ', flush=True)
    t0 = time.time()
    la_t = np.zeros(3, dtype=np.int64)
    _ = compute_T_numba(la_t, la_t, N_IR, dims_sec, fivej_sec,
                         plan_edge_types, plan_edge_indices, plan_n_edges,
                         plan_n_new, plan_n_open_before, plan_n_open_after,
                         plan_closing, plan_n_closing, plan_surviving, plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")

    # Benchmark
    t0 = time.time()
    la = np.zeros(3, dtype=np.int64)
    lb = np.array([int(N_IR)-1]*3, dtype=np.int64)
    v = compute_T_numba(la, lb, N_IR, dims_sec, fivej_sec,
                         plan_edge_types, plan_edge_indices, plan_n_edges,
                         plan_n_new, plan_n_open_before, plan_n_open_after,
                         plan_closing, plan_n_closing, plan_surviving, plan_n_surviving)
    dt = time.time() - t0
    est_min = dt * n_orbits**2 / max(N_CORES, 1) / 60
    est_min_full = dt * n_configs_full**2 / max(N_CORES, 1) / 60
    print(f"Benchmark: {dt:.3f}s/element")
    print(f"  Full matrix:    {est_min_full:.0f} min = {est_min_full/60:.1f} hours")
    print(f"  Z₃ reduced:     {est_min:.0f} min = {est_min/60:.1f} hours")
    print(f"  Saving:          {est_min_full - est_min:.0f} min")

    # Progress file
    SAVE_FILE = f'icosa_z3_{"-".join(str(s) for s in SECTOR)}_progress.json'
    T_reduced = np.zeros((n_orbits, n_orbits))
    completed_rows = set()
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            prog = json.load(f)
        for entry in prog.get('rows', []):
            T_reduced[entry['row']] = entry['data']
            completed_rows.add(entry['row'])
        print(f"Loaded {len(completed_rows)} rows from {SAVE_FILE}")

    remaining = [ia for ia in range(n_orbits) if ia not in completed_rows]
    print(f"Done: {len(completed_rows)}, Remaining: {len(remaining)}")

    if remaining:
        t_start = time.time(); done_session = 0

        def save_prog():
            rows = [{'row': int(ia), 'data': T_reduced[ia].tolist()}
                    for ia in sorted(completed_rows)]
            with open(SAVE_FILE, 'w') as f:
                json.dump({'rows': rows, 'orbits': [
                    {'rep': list(o[0]), 'size': o[1]} for o in orbits
                ]}, f)

        if N_CORES > 1:
            from multiprocessing import Pool
            args = [(ia, orbit_reps, orbit_reps, list(range(int(N_IR)))) for ia in remaining]
            with Pool(N_CORES) as pool:
                for ia, row in pool.imap_unordered(compute_row_z3, args):
                    T_reduced[ia] = row; completed_rows.add(ia); done_session += 1
                    el = time.time() - t_start; rate = done_session / el
                    left = len(remaining) - done_session
                    eta = left / rate if rate > 0 else 0
                    pct = len(completed_rows) / n_orbits * 100
                    print(f"  Row {ia:>3} | {len(completed_rows):>3}/{n_orbits} ({pct:.1f}%) | "
                          f"{el/60:.1f}min | ~{eta/60:.1f}min left", flush=True)
                    if done_session % 3 == 0:
                        save_prog()
        else:
            for ia in remaining:
                _, row = compute_row_z3((ia, orbit_reps, orbit_reps, list(range(int(N_IR)))))
                T_reduced[ia] = row; completed_rows.add(ia); done_session += 1
                el = time.time() - t_start; rate = done_session / el
                left = len(remaining) - done_session
                eta = left / rate if rate > 0 else 0
                print(f"  Row {ia:>3}/{n_orbits} | {el/60:.2f}min | ~{eta/60:.1f}min left")
                if done_session % 3 == 0:
                    save_prog()
        save_prog()
        print(f"Z₃ computation: {(time.time()-t_start)/60:.2f} min")

    # ================================================================
    # RECONSTRUCT FULL 125×125 FROM Z₃-REDUCED 45×45
    # ================================================================
    print(f"\nReconstructing full {n_configs_full}×{n_configs_full} matrix from "
          f"{n_orbits}×{n_orbits} Z₃-reduced...")

    T_full = np.zeros((n_configs_full, n_configs_full))
    for i_full, cfg_a in enumerate(configs_full):
        oi = config_to_orbit[cfg_a]
        for j_full, cfg_b in enumerate(configs_full):
            oj = config_to_orbit[cfg_b]
            T_full[i_full, j_full] = T_reduced[oi, oj]

    # Symmetrize and diagonalize
    T_sym = (T_full + T_full.T) / 2
    evals, evecs = np.linalg.eigh(T_sym)
    idx = np.argsort(evals)[::-1]; evals = evals[idx]; evecs = evecs[:, idx]
    T_max = evals[0]

    # Also diagonalize the reduced matrix for comparison
    T_red_sym = (T_reduced + T_reduced.T) / 2
    evals_red, _ = np.linalg.eigh(T_red_sym)
    evals_red = np.sort(evals_red)[::-1]

    print(f"\n{'='*60}")
    print(f"ICOSAHEDRAL BOUNDARY ANALYSIS (Z₃ REDUCED)")
    print(f"{'='*60}")
    print(f"T_max (full): {T_max:.6e}")
    print(f"T_max (reduced): {evals_red[0]:.6e}")
    print(f"Positive evals: {sum(1 for e in evals if e > 0)}")
    print(f"Negative evals: {sum(1 for e in evals if e < 0)}")

    # Compare with 64×64 result
    T_max_64 = 2.972626072618213e+41  # known from sector [0,1,2,4]
    print(f"\nComparison with [0,1,2,4] sector:")
    print(f"  64×64 T_max:  {T_max_64:.6e}")
    print(f"  125×125 T_max: {T_max:.6e}")
    print(f"  Ratio: {T_max/T_max_64:.6f}")
    print(f"  χ₄ effect: {abs(T_max/T_max_64 - 1)*100:.4f}%")

    print(f"\nMass spectrum:")
    known = [(0.511,'e'), (105.7,'μ'), (135,'π'), (332,'Λ'),
             (775,'ρ'), (938,'p'), (1777,'τ')]
    for i in range(min(30, len(evals))):
        if 0 < evals[i] < T_max * 0.9999:
            gap = -math.log(evals[i] / T_max)
            mass = gap * Lambda_QCD
            best = min(known, key=lambda h: abs(h[0] - mass))
            err = abs(best[0] - mass) / best[0] * 100
            mark = '★' if err < 5 else '●' if err < 10 else ''
            print(f"  {i:>3}: {mass:>8.1f} MeV → {best[1]:>3} ({best[0]:>7}) {err:5.1f}% {mark}")

    # Lepton source analysis
    chi3_local = SECTOR.index(1) if 1 in SECTOR else -1
    if chi3_local >= 0:
        print(f"\nLepton source (χ₃={chi3_local}):")
        for k in range(3):
            cfg = [0]*3; cfg[k] = chi3_local
            cfg_idx = configs_full.index(tuple(cfg))
            ov = evecs[cfg_idx, :]
            C1 = sum(ov[n]**2 * (evals[n]/T_max) for n in range(len(evals)) if evals[n] > 0)
            meff = -math.log(C1) * Lambda_QCD if C1 > 0 else 0
            print(f"  Edge {k}: m_eff={meff:.1f} MeV, C(1)={C1:.4e}")

    # Check χ₄ content
    chi4_local = SECTOR.index(3) if 3 in SECTOR else -1
    if chi4_local >= 0:
        print(f"\nχ₄ (dark matter) analysis:")
        for i in range(min(10, len(evals))):
            if evals[i] > 0:
                chi4_frac = sum(evecs[j, i]**2
                               for j, cfg in enumerate(configs_full)
                               if chi4_local in cfg)
                label = "← χ₄ dominated" if chi4_frac > 0.3 else ""
                print(f"  State {i}: χ₄ fraction = {chi4_frac:.3f} {label}")

    # Save results
    results = {
        'sector': SECTOR,
        'n_configs_full': n_configs_full,
        'n_orbits': n_orbits,
        'T_max': float(T_max),
        'eigenvalues': [float(e) for e in evals[:50]],
        'mass_gap': float(-math.log(evals[1]/T_max)) if 0 < evals[1] < T_max else 0,
        'chi4_effect_pct': float(abs(T_max/T_max_64 - 1)*100) if SECTOR == [0,1,2,3,4] else None,
        'orbit_count': n_orbits,
        'speedup': float(n_configs_full**2 / n_orbits**2),
    }
    outfile = f'icosa_z3_{"_".join(str(s) for s in SECTOR)}_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {outfile}")
