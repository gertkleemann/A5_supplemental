#!/usr/bin/env python3
"""
ALPHA RUNNING: compare 32x32 (no VP) vs 1024x1024 (with VP)
Uses the EXISTING qcd1024_progress.json — no new computation needed.

Usage: python alpha_from_1024.py qcd1024_progress.json
"""
import numpy as np
import json, sys, time
from numba import njit, prange

phi = (1 + np.sqrt(5)) / 2
sqrt5 = np.sqrt(5)

# A5 data
dims_full = np.array([1, 3, 3, 4, 5], dtype=np.float64)
chars_full = np.array([
    [1,  1,  1,  1,      1     ],
    [3, -1,  0,  phi,    1-phi ],
    [3, -1,  0,  1-phi,  phi   ],
    [4,  0,  1, -1,     -1     ],
    [5,  1, -1,  0,      0     ],
])
class_sizes = np.array([1, 15, 20, 12, 12], dtype=np.float64)

def compute_3j():
    N = np.zeros((5,5,5), dtype=np.float64)
    for a in range(5):
        for b in range(5):
            for c in range(5):
                N[a,b,c] = round(sum(class_sizes[i]*chars_full[a,i]*chars_full[b,i]*chars_full[c,i] for i in range(5))/60.0)
    return N
N3j = compute_3j()

# Dodecahedron topology (same as main script)
EDGES = np.array([
    [0,1],[1,2],[2,3],[3,4],[4,0],
    [0,5],[1,6],[2,7],[3,8],[4,9],
    [5,10],[6,10],[6,11],[7,11],[7,12],
    [8,12],[8,13],[9,13],[9,14],[5,14],
    [10,15],[11,16],[12,17],[13,18],[14,19],
    [15,16],[16,17],[17,18],[18,19],[19,15],
], dtype=np.int32)
VE = np.full((20, 3), -1, dtype=np.int32)
_c = np.zeros(20, dtype=np.int32)
for eidx in range(30):
    u, v = EDGES[eidx]
    VE[u, _c[u]] = eidx; _c[u] += 1
    VE[v, _c[v]] = eidx; _c[v] += 1

# ================================================================
# BUILD 32x32 (chi3+chi5, no VP)
# ================================================================
@njit(cache=True)
def config_to_labels(idx, k):
    labels = np.empty(5, dtype=np.int32)
    for j in range(5):
        labels[j] = idx % k; idx //= k
    return labels

@njit(parallel=True, cache=True)
def build_T_32(sub_3j, sub_dims, ve):
    k = 2; n = 32
    T = np.zeros((n, n), dtype=np.float64)
    for ia in prange(n):
        lA = config_to_labels(ia, k)
        for ib in range(n):
            lB = config_to_labels(ib, k)
            el = np.zeros(30, dtype=np.int32)
            for j in range(5):
                el[j] = lA[j]; el[25+j] = lB[j]
            val = 0.0
            for ic in range(1048576):
                tmp = ic
                for j in range(20):
                    el[5+j] = tmp & 1; tmp >>= 1
                w = 1.0; ok = True
                for v in range(20):
                    s = sub_3j[el[ve[v,0]], el[ve[v,1]], el[ve[v,2]]]
                    if s == 0.0: ok = False; break
                    w *= s
                if ok:
                    for j in range(20):
                        w *= sub_dims[el[5+j]]
                    val += w
            T[ia, ib] = val
    return T

# ================================================================
# ANALYSIS
# ================================================================
def analyse(T, k, irrep_indices, label):
    """Extract photon propagator ratio at various nesting depths."""
    T = (T + T.T) / 2
    evals, evecs = np.linalg.eigh(T)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    
    print(f"\n  {label}")
    print(f"  Matrix size: {T.shape[0]}x{T.shape[0]}")
    print(f"  Top 5 eigenvalues: {evals[:5]}")
    print(f"  Eigenvalue ratio lambda1/lambda0: {evals[1]/evals[0]:.6e}")
    
    # Find chi5 position in the sub-irrep list
    chi5_pos = list(irrep_indices).index(4)  # chi5 = index 4 in full table
    photon_idx = sum(chi5_pos * k**j for j in range(5))
    
    c = evecs[photon_idx, :]
    print(f"  Photon config index: {photon_idx} (all edges = chi5)")
    print(f"  Photon overlap with ground state: {c[0]:.6f}")
    print(f"  Photon overlap with state 1: {c[1]:.6f}")
    
    # Also check: what fraction of the ground state is "photon-like"?
    print(f"  Ground state: photon component |c0|^2 = {c[0]**2:.6f}")
    
    lam0 = evals[0]
    ratios = evals / lam0
    
    R = {}
    for N in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]:
        R[N] = float(np.sum(c**2 * ratios**N))
    
    return R, evals, evecs

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    
    print("=" * 70)
    print("ALPHA RUNNING: 32x32 (no VP) vs 1024x1024 (with VP)")
    print("=" * 70)
    
    # --- Load 1024 matrix ---
    fname = 'qcd1024_progress.json'
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    
    print(f"\nLoading {fname}...")
    with open(fname) as f:
        data = json.load(f)
    
    rows = data.get('rows', data.get('data', []))
    if isinstance(rows, list) and len(rows) > 0:
        if isinstance(rows[0], dict):
            # Format: [{"row": i, "data": [...]}]
            n = max(r['row'] for r in rows) + 1
            T_1024 = np.zeros((n, n), dtype=np.float64)
            for r in rows:
                row_data = r['data']
                T_1024[r['row'], :len(row_data)] = row_data
            print(f"  Loaded {len(rows)} rows, matrix size {n}x{n}")
        else:
            # Format: flat 2D array
            T_1024 = np.array(rows, dtype=np.float64)
            n = T_1024.shape[0]
            print(f"  Loaded matrix {n}x{n}")
    
    T_1024 = (T_1024 + T_1024.T) / 2
    
    # Check completeness
    zero_rows = np.sum(np.all(T_1024 == 0, axis=1))
    print(f"  Zero rows: {zero_rows}/{n}")
    if zero_rows > 0:
        print(f"  WARNING: {zero_rows} rows are all zeros — matrix may be incomplete")
    
    # 1024 irreps: chi1(0), chi3(1), chi3'(2), chi5(4) — indices in full A5 table
    irreps_1024 = [0, 1, 2, 4]
    k_1024 = 4
    
    # --- Build 32x32 ---
    print("\n--- Building chi3+chi5 (32x32, no VP) ---")
    irreps_32 = [1, 4]  # chi3, chi5
    k_32 = 2
    s3j_32 = np.zeros((2,2,2), dtype=np.float64)
    sd_32 = np.zeros(2, dtype=np.float64)
    for ia, a in enumerate(irreps_32):
        sd_32[ia] = dims_full[a]
        for ib, b in enumerate(irreps_32):
            for ic, cc in enumerate(irreps_32):
                s3j_32[ia,ib,ic] = N3j[a,b,cc]
    
    t0 = time.time()
    T_32 = build_T_32(s3j_32, sd_32, VE)
    T_32 = (T_32 + T_32.T) / 2
    print(f"  Done in {time.time()-t0:.1f}s")
    
    # --- Analyse both ---
    R_32, ev_32, _ = analyse(T_32, 2, irreps_32, "QED sector (chi3+chi5, 32x32, NO vacuum polarisation)")
    R_1024, ev_1024, _ = analyse(T_1024, 4, irreps_1024, "Face-rep sector (chi1+chi3+chi3'+chi5, 1024x1024, WITH vacuum polarisation)")
    
    # ================================================================
    # THE MONEY TABLE
    # ================================================================
    print("\n" + "=" * 70)
    print("THE RUNNING OF ALPHA")
    print("=" * 70)
    
    Lambda = 332.0
    
    print(f"\n  {'N':>4} {'Q (MeV)':>12} {'32x32':>12} {'1024x1024':>12} {'VP ratio':>10} {'1/alpha':>10}")
    print(f"  {'-'*62}")
    
    running_data = []
    for N in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 64]:
        Q = Lambda * 60 ** ((N-1) / 11.0)
        r32 = R_32[N]
        r1024 = R_1024[N]
        
        # VP ratio: how much does the dressed propagator differ from bare?
        vp = r1024 / r32 if r32 > 0 else 0
        
        # 1/alpha_eff: if VP enhances the propagator, alpha increases (1/alpha decreases)
        # 1/alpha(N) = 137.036 * (bare/dressed) = 137.036 * r32/r1024
        ainv = 137.036 * r32 / r1024 if r1024 > 0 else 137.036
        
        running_data.append((N, Q, r32, r1024, vp, ainv))
        print(f"  {N:>4} {Q:>12.1f} {r32:>12.8f} {r1024:>12.8f} {vp:>10.6f} {ainv:>10.4f}")
    
    # ================================================================
    # BETA FUNCTION
    # ================================================================
    print(f"\n  DISCRETE BETA FUNCTION (Delta(1/alpha) per step):")
    print(f"  {'-'*50}")
    prev = None
    for N, Q, r32, r1024, vp, ainv in running_data:
        if N <= 32:
            if prev:
                d = ainv - prev[1]
                steps = N - prev[0]
                print(f"    N={prev[0]:>2} -> {N:>2}: Delta(1/alpha) = {d:+8.4f}, per step = {d/steps:+8.4f}")
            prev = (N, ainv)
    
    print(f"\n  SM expectation:")
    print(f"    1/alpha goes from 137.036 (Q ~ 0) to ~128 (Q = M_Z = 91.2 GeV)")
    print(f"    M_Z corresponds to N ~ {1 + np.log(91200/332)/np.log(60):.1f} nesting levels")
    print(f"    Total Delta(1/alpha) ~ -9 over ~3 nesting levels")
    print(f"    Per step: ~ -3")
    
    # ================================================================
    # EIGENVALUE COMPARISON
    # ================================================================
    print(f"\n  EIGENVALUE STRUCTURE:")
    print(f"  {'':>4} {'32x32':>18} {'1024x1024':>18} {'ratio':>12}")
    print(f"  {'-'*55}")
    for i in range(min(10, len(ev_32))):
        r = ev_1024[i] / ev_32[i] if i < len(ev_1024) and ev_32[i] != 0 else 0
        e32 = ev_32[i] if i < len(ev_32) else 0
        e1024 = ev_1024[i] if i < len(ev_1024) else 0
        print(f"  {i:>4} {e32:>18.6e} {e1024:>18.6e} {r:>12.4f}")
    
    # ================================================================
    # SAVE
    # ================================================================
    out = {
        'Lambda': Lambda,
        'alpha_inv_bare': 137.036,
        'running': [
            {'N': N, 'Q_MeV': round(Q,1), 'ratio_32': round(r32,10), 
             'ratio_1024': round(r1024,10), 'alpha_inv': round(ainv,4)}
            for N, Q, r32, r1024, vp, ainv in running_data
        ]
    }
    with open('alpha_running_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    
    print(f"\nSaved to alpha_running_results.json")
    print("Done.")
