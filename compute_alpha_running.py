#!/usr/bin/env python3
"""
RUNNING OF alpha FROM TRANSFER MATRIX NESTING
Requires: numpy, numba
Usage: python compute_alpha_running.py
"""
import numpy as np
import json, time
from numba import njit, prange

phi = (1 + np.sqrt(5)) / 2
sqrt5 = np.sqrt(5)
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

N3j_full = compute_3j()

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
print("Topology OK:", all(_c[v]==3 for v in range(20)))

@njit(cache=True)
def config_to_labels(idx, k):
    labels = np.empty(5, dtype=np.int32)
    for j in range(5):
        labels[j] = idx % k
        idx //= k
    return labels

# ================================================================
# 32x32 BRUTE FORCE
# ================================================================
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
# 243x243 FACTORED — split into two functions
# ================================================================

@njit(cache=True)
def second_half(k, w_first, e14, e19, e20, e21, Tv8, Tv9, sub_3j, sub_dims, ve, n, T_row):
    for e15 in range(k):
        for e16 in range(k):
            w8 = Tv8[e15, e16]
            if w8 == 0.0: continue
            for e22 in range(k):
                w12 = sub_3j[e14, e15, e22]
                if w12 == 0.0: continue
                for e17 in range(k):
                    for e18 in range(k):
                        w9 = Tv9[e17, e18]
                        if w9 == 0.0: continue
                        for e23 in range(k):
                            w13 = sub_3j[e16, e17, e23]
                            if w13 == 0.0: continue
                            for e24 in range(k):
                                w14 = sub_3j[e18, e19, e24]
                                if w14 == 0.0: continue
                                wt = w_first * w8 * w9 * w12 * w13 * w14
                                wd = sub_dims[e20]*sub_dims[e21]*sub_dims[e22]*sub_dims[e23]*sub_dims[e24]
                                wt *= wd
                                eu = np.array([e20,e21,e22,e23,e24], dtype=np.int32)
                                for ib in range(n):
                                    lB = config_to_labels(ib, k)
                                    el = np.zeros(30, dtype=np.int32)
                                    for j in range(5):
                                        el[20+j] = eu[j]; el[25+j] = lB[j]
                                    wB = 1.0; ok = True
                                    for v in range(15, 20):
                                        s = sub_3j[el[ve[v,0]], el[ve[v,1]], el[ve[v,2]]]
                                        if s == 0.0: ok = False; break
                                        wB *= s
                                    if ok:
                                        T_row[ib] += wt * wB

@njit(cache=True)
def build_row_243(ia, k, sub_3j, sub_dims, ve):
    n = k ** 5
    T_row = np.zeros(n, dtype=np.float64)
    lA = config_to_labels(ia, k)
    el = np.zeros(30, dtype=np.int32)
    for j in range(5): el[j] = lA[j]

    for p in range(k**5):
        tmp = p; elo = np.empty(5, dtype=np.int32)
        for j in range(5): elo[j] = tmp % k; tmp //= k
        for j in range(5): el[5+j] = elo[j]
        wA = 1.0; ok = True
        for v in range(5):
            s = sub_3j[el[ve[v,0]], el[ve[v,1]], el[ve[v,2]]]
            if s == 0.0: ok = False; break
            wA *= s
        if not ok: continue
        for j in range(5): wA *= sub_dims[elo[j]]

        Tv = np.zeros((5, k, k), dtype=np.float64)
        for vi in range(5):
            for a in range(k):
                for b in range(k):
                    w = sub_3j[elo[vi], a, b]
                    if w > 0: Tv[vi, a, b] = w * sub_dims[a] * sub_dims[b]

        for e10 in range(k):
            for e19 in range(k):
                w5 = Tv[0, e10, e19]
                if w5 == 0.0: continue
                for e11 in range(k):
                    for e12 in range(k):
                        w6 = Tv[1, e11, e12]
                        if w6 == 0.0: continue
                        for e20 in range(k):
                            w10 = sub_3j[e10, e11, e20]
                            if w10 == 0.0: continue
                            for e13 in range(k):
                                for e14 in range(k):
                                    w7 = Tv[2, e13, e14]
                                    if w7 == 0.0: continue
                                    for e21 in range(k):
                                        w11 = sub_3j[e12, e13, e21]
                                        if w11 == 0.0: continue
                                        wf = wA*w5*w6*w7*w10*w11
                                        second_half(k, wf, e14, e19, e20, e21,
                                                   Tv[3], Tv[4], sub_3j, sub_dims, ve, n, T_row)
    return T_row

def build_T_243(sub_3j, sub_dims, ve):
    k = 3; n = 243
    T = np.zeros((n, n), dtype=np.float64)
    t0 = time.time()
    print("  Compiling (row 0)...")
    T[0,:] = build_row_243(0, k, sub_3j, sub_dims, ve)
    t1 = time.time() - t0
    print(f"  Row 0: {t1:.1f}s. ETA: {t1*n/60:.0f} min")
    for ia in range(1, n):
        T[ia,:] = build_row_243(ia, k, sub_3j, sub_dims, ve)
        if (ia+1) % 25 == 0:
            el = time.time()-t0; rate = (ia+1)/el
            print(f"  Row {ia+1}/{n}, {el:.0f}s, ETA {(n-ia-1)/rate:.0f}s")
    print(f"  Total: {time.time()-t0:.0f}s")
    return (T + T.T) / 2

# ================================================================
# ANALYSIS
# ================================================================
def analyse(T, k, irrep_indices, label):
    T = (T + T.T) / 2
    evals, evecs = np.linalg.eigh(T)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]; evecs = evecs[:, idx]
    print(f"\n  {label} top 5: {evals[:5]}")
    chi5_pos = list(irrep_indices).index(4)
    pidx = sum(chi5_pos * k**j for j in range(5))
    c = evecs[pidx, :]
    print(f"  Photon idx={pidx}, ground overlap={c[0]:.6f}")
    # Normalize eigenvalues to avoid overflow
    lam0 = evals[0]
    ratios = evals / lam0  # ratios[0] = 1, all others < 1
    R = {}
    for N in [1,2,3,4,6,8,12,16,24,32]:
        # G(N)/G_free(N) = sum_i c_i^2 * (lambda_i/lambda_0)^N
        # since G_free = lambda_0^N and G = sum c_i^2 lambda_i^N
        R[N] = float(np.sum(c**2 * ratios**N))
    return R

# ================================================================
if __name__ == '__main__':
    print("="*70)
    print("RUNNING OF alpha FROM TRANSFER MATRIX NESTING")
    print("="*70)
    Lambda = 332.0

    # Sector 1: chi3+chi5 (32x32)
    print("\n--- chi3+chi5 (32x32) ---")
    idx1 = [1,4]; k1 = 2
    s3j1 = np.zeros((2,2,2), dtype=np.float64)
    sd1 = np.zeros(2, dtype=np.float64)
    for ia,a in enumerate(idx1):
        sd1[ia] = dims_full[a]
        for ib,b in enumerate(idx1):
            for ic,cc in enumerate(idx1):
                s3j1[ia,ib,ic] = N3j_full[a,b,cc]
    t0 = time.time()
    T1 = build_T_32(s3j1, sd1, VE); T1 = (T1+T1.T)/2
    print(f"  Done in {time.time()-t0:.1f}s")
    R1 = analyse(T1, 2, idx1, "QED (no VP)")

    # Sector 2: chi3+chi3'+chi5 (243x243)
    print("\n--- chi3+chi3p+chi5 (243x243) ---")
    idx2 = [1,2,4]; k2 = 3
    s3j2 = np.zeros((3,3,3), dtype=np.float64)
    sd2 = np.zeros(3, dtype=np.float64)
    for ia,a in enumerate(idx2):
        sd2[ia] = dims_full[a]
        for ib,b in enumerate(idx2):
            for ic,cc in enumerate(idx2):
                s3j2[ia,ib,ic] = N3j_full[a,b,cc]
    T2 = build_T_243(s3j2, sd2, VE)
    R2 = analyse(T2, 3, idx2, "QED+VP")

    # Results
    print("\n" + "="*70)
    print("VACUUM POLARISATION AND RUNNING")
    print("="*70)
    print(f"\n  {'N':>4} {'Q(MeV)':>10} {'QED':>12} {'QED+VP':>12} {'VP/QED':>10} {'1/alpha':>10}")
    print(f"  {'-'*58}")
    for N in [1,2,3,4,6,8,12,16,24,32]:
        Q = Lambda * 60**((N-1)/11.0)
        rq = R1[N]; rv = R2[N]
        vr = rv/rq if rq>0 else 0
        ai = 137.036*rq/rv if rv>0 else 137.036
        print(f"  {N:>4} {Q:>10.1f} {rq:>12.8f} {rv:>12.8f} {vr:>10.6f} {ai:>10.4f}")

    print(f"\n  BETA FUNCTION:")
    prev = None
    for N in [1,2,3,4,6,8,12,16]:
        rq=R1[N]; rv=R2[N]
        if rq>0 and rv>0:
            ai = 137.036*rq/rv
            if prev:
                d = ai-prev[1]
                print(f"    N={prev[0]}->{N}: Delta(1/alpha)={d:+.4f}, per step={d/(N-prev[0]):+.4f}")
            prev = (N, ai)

    out = {'Lambda':Lambda, 'QED':{str(N):float(R1[N]) for N in R1}, 'VP':{str(N):float(R2[N]) for N in R2}}
    with open('alpha_running_results.json','w') as f:
        json.dump(out, f, indent=2)
    print("\nSaved to alpha_running_results.json")
