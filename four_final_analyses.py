#!/usr/bin/env python3
"""
FOUR FINAL ANALYSES
====================
2. Connected correlator (vacuum-subtracted) — verify 1441 MeV
3. Negative eigenvalue sector — unexplored territory
4. Icosahedral Dirac operator (24×24) — boundary leptons
5. Face-pair entanglement → Coulomb potential

Usage: python four_final_analyses.py [path_to_qcd3125_progress_seeded.json]
"""
import numpy as np
import json, sys, os, math
from collections import deque

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

fname = sys.argv[1] if len(sys.argv) > 1 else 'qcd3125_progress_seeded.json'

# ================================================================
# ANALYSIS 2: CONNECTED CORRELATOR
# ================================================================
def analysis_2(evals, evecs, T_max, n):
    print(f"\n{'='*65}")
    print(f"2. CONNECTED CORRELATOR (vacuum-subtracted)")
    print(f"{'='*65}")
    
    k = 5
    n_edges = 5
    all_labels = []
    for c in range(n):
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        all_labels.append(labels)
    
    def make_source(condition):
        src = np.zeros(n)
        for c in range(n):
            if condition(all_labels[c]):
                src[c] = 1.0
        norm = np.linalg.norm(src)
        return src / norm if norm > 0 else src
    
    sources = {
        'glueball': make_source(lambda l: all(x == 4 for x in l)),
        'meson': make_source(lambda l: any(
            (l[i]==1 and l[(i+1)%5]==2) or (l[i]==2 and l[(i+1)%5]==1)
            for i in range(5))),
        'baryon': make_source(lambda l: any(
            l[i]==1 and l[(i+1)%5]==1 and l[(i+2)%5]==1
            for i in range(5))),
        'vector': make_source(lambda l: any(
            (l[i]==1 and l[(i+1)%5]==4) or (l[i]==4 and l[(i+1)%5]==1)
            for i in range(5))),
    }
    
    ratios = evals / T_max
    known = [(135,'pion'),(775,'rho'),(938,'proton'),(1440,'Roper'),(1370,'f0')]
    
    results = {}
    for src_name, src_vec in sources.items():
        # All overlaps
        ov = np.array([np.dot(src_vec, evecs[:, i])**2 for i in range(n)])
        vac_ov = ov[0]  # vacuum overlap
        
        # Connected correlator: subtract vacuum
        N_max = 300
        eff_mass = []
        
        for N in range(1, N_max):
            C_N = 0.0
            C_N1 = 0.0
            for i in range(1, n):  # SKIP vacuum (i=0)
                if ov[i] > 1e-30 and abs(ratios[i]) > 1e-30:
                    log_r = math.log(abs(ratios[i]))
                    sign = 1 if evals[i] > 0 else (-1)**N
                    sign1 = 1 if evals[i] > 0 else (-1)**(N+1)
                    
                    lt_N = N * log_r
                    lt_N1 = (N+1) * log_r
                    
                    if lt_N > -300:
                        C_N += ov[i] * math.exp(lt_N) * sign
                    if lt_N1 > -300:
                        C_N1 += ov[i] * math.exp(lt_N1) * sign1
            
            if C_N > 0 and C_N1 > 0:
                m = -Lambda * math.log(C_N1 / C_N)
                eff_mass.append((N, m))
            elif abs(C_N) > 0 and abs(C_N1) > 0:
                m = -Lambda * math.log(abs(C_N1) / abs(C_N))
                eff_mass.append((N, m))
        
        print(f"\n  --- {src_name} (vacuum overlap = {vac_ov:.6f}) ---")
        print(f"  {'N':>5} {'m_eff MeV':>10}")
        
        for N, m in eff_mass:
            if N in [1,2,3,5,10,20,50,100,200]:
                if 0 < m < 20000:
                    best = min(known, key=lambda h: abs(h[0]-m))
                    err = abs(best[0]-m)/best[0]*100
                    mark = f" → {best[1]} ({err:.1f}%)" if err < 15 else ""
                    print(f"  {N:>5} {m:>10.1f}{mark}")
                else:
                    print(f"  {N:>5} {m:>10.1f} (unphysical)")
        
        # Plateau detection (N=20-100)
        plateau = [m for N, m in eff_mass if 20 <= N <= 100 and 0 < m < 20000]
        if plateau:
            mp = np.mean(plateau)
            ms = np.std(plateau)
            best = min(known, key=lambda h: abs(h[0]-mp))
            err = abs(best[0]-mp)/best[0]*100
            print(f"  PLATEAU (N=20-100): {mp:.1f} ± {ms:.1f} MeV → {best[1]} ({err:.1f}%)")
            results[src_name] = {'plateau_MeV': round(mp,1), 'std': round(ms,1), 
                                  'match': best[1], 'error_pct': round(err,2)}
        else:
            print(f"  NO PLATEAU")
            results[src_name] = None
    
    return results

# ================================================================
# ANALYSIS 3: NEGATIVE EIGENVALUE SECTOR
# ================================================================
def analysis_3(evals, evecs, T_max, n):
    print(f"\n{'='*65}")
    print(f"3. NEGATIVE EIGENVALUE SECTOR (parity-odd / antiparticle)")
    print(f"{'='*65}")
    
    neg_evals = sorted([e for e in evals if e < 0], key=lambda x: -abs(x))
    n_neg = len(neg_evals)
    
    print(f"\n  {n_neg} negative eigenvalues")
    print(f"  Largest |λ_neg| = {abs(neg_evals[0]):.4e}")
    print(f"  T_max (positive) = {T_max:.4e}")
    print(f"  Ratio |λ_neg|/T_max = {abs(neg_evals[0])/T_max:.6f}")
    
    known = [(135,'pion'),(498,'kaon'),(548,'eta'),(775,'rho'),(938,'proton'),
             (958,"eta'"),(1232,'Delta'),(1440,'Roper')]
    
    # Mass spectrum from negative eigenvalues (using |λ|)
    print(f"\n  {'#':>3} {'|λ|':>14} {'Gap':>8} {'Mass MeV':>10} {'Match':>10} {'Err':>6}")
    print(f"  {'-'*55}")
    
    masses_neg = []
    for i, ev in enumerate(neg_evals[:30]):
        ratio = abs(ev) / T_max
        if ratio < 0.9999:
            gap = -math.log(ratio)
            mass = gap * Lambda
            best = min(known, key=lambda h: abs(h[0]-mass))
            err = abs(best[0]-mass)/best[0]*100
            mark = ' *' if err < 5 else ''
            print(f"  {i:>3} {abs(ev):>14.4e} {gap:>8.4f} {mass:>10.1f} {best[1]:>10} {err:>5.1f}%{mark}")
            masses_neg.append({'mass_MeV': round(mass,1), 'match': best[1], 'error': round(err,2)})
    
    # Compare positive vs negative spectrum
    pos_evals = sorted([e for e in evals if e > 0], reverse=True)
    print(f"\n  POSITIVE vs NEGATIVE comparison:")
    print(f"  {'Level':>5} {'Pos gap':>10} {'Neg gap':>10} {'Ratio':>8}")
    for i in range(min(10, len(neg_evals))):
        if i+1 < len(pos_evals) and abs(neg_evals[i]) < T_max * 0.9999:
            pos_gap = -math.log(pos_evals[i+1]/T_max) if pos_evals[i+1] > 0 and pos_evals[i+1] < T_max*0.9999 else 0
            neg_gap = -math.log(abs(neg_evals[i])/T_max)
            r = neg_gap/pos_gap if pos_gap > 0 else 0
            print(f"  {i:>5} {pos_gap:>10.4f} {neg_gap:>10.4f} {r:>8.4f}")
    
    # Eigenvector composition of lightest negative state
    k = 5
    n_edges = 5
    irr_names = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']
    
    # Find the index of the largest negative eigenvalue
    neg_idx = None
    for i in range(n):
        if evals[i] == neg_evals[0]:
            neg_idx = i
            break
    
    if neg_idx is not None:
        vec = evecs[:, neg_idx]
        print(f"\n  Lightest negative state eigenvector composition:")
        fracs = {}
        for r in range(k):
            weight = 0.0
            for c in range(n):
                labels = []
                tmp = c
                for e in range(n_edges):
                    labels.append(tmp % k)
                    tmp //= k
                count = labels.count(r)
                weight += vec[c]**2 * count / n_edges
            fracs[r] = weight
            print(f"    {irr_names[r]:>4}: {weight*100:.2f}%")
    
    return {'n_negative': n_neg, 'lightest_neg_mass': masses_neg[0] if masses_neg else None,
            'masses': masses_neg[:10]}

# ================================================================
# ANALYSIS 4: ICOSAHEDRAL DIRAC OPERATOR (24×24)
# ================================================================
def analysis_4():
    print(f"\n{'='*65}")
    print(f"4. ICOSAHEDRAL DIRAC OPERATOR (12 vertices × 2 spin = 24×24)")
    print(f"{'='*65}")
    
    # Icosahedron vertices
    ico_verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            ico_verts.append([0, s1, s2*phi])
            ico_verts.append([s1, s2*phi, 0])
            ico_verts.append([s1*phi, 0, s2])
    ico_verts = np.array(ico_verts)
    Nv = 12
    
    # Icosahedron edge length
    dists = []
    for i in range(Nv):
        for j in range(i+1, Nv):
            dists.append(np.linalg.norm(ico_verts[i]-ico_verts[j]))
    edge_len = min(dists) + 0.01  # just above minimum
    
    edges = []
    adj = [[] for _ in range(Nv)]
    for i in range(Nv):
        for j in range(i+1, Nv):
            if np.linalg.norm(ico_verts[i]-ico_verts[j]) < edge_len + 0.1:
                edges.append((i,j))
                adj[i].append(j)
                adj[j].append(i)
    Ne = len(edges)
    
    print(f"  Icosahedron: {Nv} vertices, {Ne} edges, degree {len(adj[0])}")
    
    sigma = np.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]],
    ], dtype=np.complex128)
    
    D = np.zeros((2*Nv, 2*Nv), dtype=np.complex128)
    for i, j in edges:
        d_vec = ico_verts[j] - ico_verts[i]
        d_hat = d_vec / np.linalg.norm(d_vec)
        dsig = sum(d_hat[k] * sigma[k] for k in range(3))
        D[2*i:2*i+2, 2*j:2*j+2] += dsig
        D[2*j:2*j+2, 2*i:2*i+2] += -dsig
    
    print(f"  D is anti-Hermitian: {np.allclose(D, -D.conj().T)}")
    
    evals_D = np.linalg.eigvals(D)
    imag_evals = np.sort(np.imag(evals_D))
    
    unique = sorted(set(round(e, 3) for e in imag_evals if abs(e) > 0.01))
    
    print(f"\n  Eigenvalues (±iE):")
    for ev in unique:
        mult = sum(1 for e in imag_evals if abs(e - ev) < 0.02)
        print(f"    ±{abs(ev):.4f}i  (×{mult})")
    
    # Mass gap
    pos_evals_D = [abs(e) for e in unique if e > 0]
    if pos_evals_D:
        gap_D = min(pos_evals_D)
        print(f"\n  Mass gap: {gap_D:.6f}")
        print(f"  In MeV: {gap_D * Lambda:.1f}")
        
        # Match to A₅ quantities  
        cands = {'1/√5': 1/sqrt5, '1/φ': 1/phi, '√(3-√5)': math.sqrt(3-sqrt5),
                 '1': 1.0, '√2': math.sqrt(2), '√3': math.sqrt(3), 
                 '√5': sqrt5, 'φ': phi, '2': 2.0, '√(5-√5)': math.sqrt(5-sqrt5)}
        
        print(f"\n  Eigenvalue matching:")
        for ev in pos_evals_D:
            best = min(cands.items(), key=lambda x: abs(x[1]-ev))
            err = abs(best[1]-ev)/ev*100
            mark = " ← EXACT" if err < 0.1 else (" ← close" if err < 3 else "")
            print(f"    E = {ev:.6f} ≈ {best[0]} = {best[1]:.6f} ({err:.2f}%){mark}")
    
    # Propagator
    if abs(np.linalg.det(D)) > 1e-20:
        S = np.linalg.inv(D)
        print(f"\n  Nearest-neighbor propagator:")
        for i, j in edges[:3]:
            S_block = S[2*i:2*i+2, 2*j:2*j+2]
            d_vec = ico_verts[j] - ico_verts[i]
            d_hat = d_vec / np.linalg.norm(d_vec)
            dsig = sum(d_hat[k] * sigma[k] for k in range(3))
            
            a = np.trace(S_block) / 2
            b = np.trace(S_block @ dsig.conj().T) / 2
            print(f"    Edge ({i},{j}): scalar = {a:.6f}, vector = {b:.6f}")
        
        # Compare with dodecahedral Dirac
        print(f"\n  COMPARISON:")
        print(f"    Dodecahedral Dirac: S(nn) = -1/3 × (d̂·σ)  [1/dim(χ₃)]")
        print(f"    Icosahedral Dirac:  S(nn) = {float(np.real(b)):.6f} × (d̂·σ)")
        print(f"    Ratio icosa/dodec = {abs(float(np.real(b)))/(1/3):.4f}")
    else:
        print(f"\n  D is singular — using pseudoinverse")
        S = np.linalg.pinv(D)
    
    # Compare dodec vs icosa Dirac eigenvalues
    print(f"\n  DODECAHEDRON vs ICOSAHEDRON Dirac:")
    print(f"    Dodec eigenvalues: ±0.457, ±1.071, ±1.732, ±2.189, ±2.803")
    print(f"    Icosa eigenvalues: {[round(abs(e),3) for e in pos_evals_D]}")
    
    return {
        'n_vertices': Nv, 'n_edges': Ne,
        'eigenvalues': [round(abs(e), 6) for e in pos_evals_D],
        'mass_gap': round(gap_D, 6) if pos_evals_D else None,
    }

# ================================================================
# ANALYSIS 5: FACE-PAIR ENTANGLEMENT → COULOMB
# ================================================================
def analysis_5(evals, evecs, T_max, n):
    print(f"\n{'='*65}")
    print(f"5. FACE-PAIR ENTANGLEMENT → COULOMB POTENTIAL")
    print(f"{'='*65}")
    
    psi0 = evecs[:, 0]
    k = 5
    n_edges = 5
    
    # Dodecahedron face structure
    # Each face has 5 edges. Config c labels all 5 boundary edges.
    # For face-pair entanglement, we need to identify which edges
    # belong to which face on the DODECAHEDRON.
    
    # The transfer matrix uses face A (edges 0-4) and face B (opposite).
    # For OTHER face pairs, we need the dodecahedron adjacency.
    
    # Dodecahedron has 12 faces. Distance between faces:
    # d=1: adjacent (share an edge) — 5 neighbors
    # d=2: next-nearest (share a vertex) — 5 neighbors  
    # d=3: opposite (= face B) — 1 face
    
    # For the transfer matrix, the 5 boundary edges of face A are our
    # "system" and the 5 boundary edges of face B are the "environment".
    # The 20 internal edges are traced over.
    
    # But we can do something simpler: use the VACUUM EIGENVECTOR
    # to compute entanglement between EDGE SUBSETS.
    
    # The 5 edges of the observer face = one subsystem
    # Trace out different numbers of edges to get entanglement
    # at different "distances"
    
    # Bipartition: m edges vs (5-m) edges
    # m=1: one edge (nearest probe) → S(1|4) = 2α (already computed)
    # m=2: two adjacent edges → S(2|3)
    # m=3: three edges → S(3|2)
    # etc.
    
    print(f"\n  Using bipartitions of the 5 boundary edges:")
    print(f"  m|5-m means tracing out (5-m) edges\n")
    
    results = {}
    for m in range(1, 5):
        shape_A = k**m
        shape_B = k**(5-m)
        
        M = psi0.reshape(shape_A, shape_B)
        sv = np.linalg.svd(M, compute_uv=False)
        p = sv**2
        p = p[p > 1e-30]
        p = p / p.sum()
        S = -np.sum(p * np.log(p))
        
        S_max = np.log(min(shape_A, shape_B))
        
        print(f"  {m}|{5-m}: S = {S:.8f} nats, S/S_max = {S/S_max:.6f}")
        results[f'{m}|{5-m}'] = round(S, 8)
    
    # Check 1/m scaling (Coulomb-like)
    S_vals = [results[f'{m}|{5-m}'] for m in range(1, 5)]
    
    print(f"\n  SCALING TEST: does S ∝ 1/distance?")
    print(f"  {'m':>3} {'S':>12} {'S/S(1)':>10} {'1/m':>8} {'m×S/S(1)':>10}")
    for m in range(1, 5):
        S = S_vals[m-1]
        ratio = S / S_vals[0]
        inv_m = 1.0 / m
        product = m * ratio
        print(f"  {m:>3} {S:>12.6f} {ratio:>10.4f} {inv_m:>8.4f} {product:>10.4f}")
    
    # Check other scalings
    print(f"\n  ALTERNATIVE SCALINGS:")
    print(f"  {'m':>3} {'S':>12} {'1/m':>8} {'1/m²':>8} {'1/√m':>8} {'ln(m)/m':>8}")
    for m in range(1, 5):
        S = S_vals[m-1]
        print(f"  {m:>3} {S:>12.6f} {S_vals[0]/m:>8.6f} {S_vals[0]/m**2:>8.6f} {S_vals[0]/math.sqrt(m):>8.6f} {S_vals[0]*math.log(m+1)/(m):>8.6f}")
    
    # The key comparison
    print(f"\n  S(1|4) = {S_vals[0]:.8f} ≈ 2α = {2*alpha:.8f} ({abs(S_vals[0]-2*alpha)/(2*alpha)*100:.3f}%)")
    print(f"  S(2|3) = {S_vals[1]:.8f}")
    print(f"  S(3|2) = {S_vals[2]:.8f}")
    print(f"  S(4|1) = {S_vals[3]:.8f}")
    
    # Ratios
    print(f"\n  Ratios:")
    print(f"  S(2|3)/S(1|4) = {S_vals[1]/S_vals[0]:.6f}")
    print(f"  S(3|2)/S(1|4) = {S_vals[2]/S_vals[0]:.6f}")
    print(f"  S(4|1)/S(1|4) = {S_vals[3]/S_vals[0]:.6f}")
    
    # Check against A₅ quantities
    r21 = S_vals[1]/S_vals[0]
    cands = {'2/3': 2/3, '1/φ': 1/phi, 'ln(2)': math.log(2), '√(2/3)': math.sqrt(2/3),
             '3/4': 3/4, 'α/α': 1.0, '2/π': 2/math.pi, '(3-√5)': 3-sqrt5}
    print(f"\n  S(2|3)/S(1|4) = {r21:.6f}")
    for name, val in sorted(cands.items(), key=lambda x: abs(x[1]-r21)):
        err = abs(val-r21)/r21*100
        if err < 10:
            print(f"    ≈ {name} = {val:.6f} ({err:.2f}%)")
    
    return results

# ================================================================
# MAIN
# ================================================================
if __name__ == '__main__':
    print("FOUR FINAL ANALYSES")
    print(f"Loading {fname}...")
    
    with open(fname) as f:
        data = json.load(f)
    
    rows = data['rows']
    n = max(r['row'] for r in rows) + 1
    T = np.zeros((n, n), dtype=np.float64)
    for r in rows:
        T[r['row'], :len(r['data'])] = r['data']
    T = (T + T.T) / 2
    
    print("Diagonalizing 3125×3125...", flush=True)
    evals, evecs = np.linalg.eigh(T)
    idx = np.argsort(np.abs(evals))[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]
    T_max = evals[0]
    print(f"T_max = {T_max:.6e}")
    
    all_results = {}
    all_results['2_connected_correlator'] = analysis_2(evals, evecs, T_max, n)
    all_results['3_negative_eigenvalues'] = analysis_3(evals, evecs, T_max, n)
    all_results['4_icosahedral_dirac'] = analysis_4()
    all_results['5_face_pair_entanglement'] = analysis_5(evals, evecs, T_max, n)
    
    outfile = 'four_final_analyses_results.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")
    
    print(f"\n{'='*65}")
    print(f"DONE")
    print(f"{'='*65}")
