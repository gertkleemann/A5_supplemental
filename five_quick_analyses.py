#!/usr/bin/env python3
"""
FIVE QUICK ANALYSES FROM EXISTING DATA
========================================
1. Proton eigenvector dark matter content
2. Visible-dark vacuum overlap
3. Which mode is the missing 60th? (Theorem 20c)
4. Is the matter-antimatter/QED gap ratio exactly 2?
5. V₄ vacuum — is χ₁ fraction exactly 1/60?

Usage: python five_quick_analyses.py [data_directory]
"""
import numpy as np
import json, sys, os, math

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

basedir = sys.argv[1] if len(sys.argv) > 1 else '.'

irr_names = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']
dims = [1, 3, 3, 4, 5]

# ================================================================
# HELPER: load transfer matrix from progress file
# ================================================================
def load_matrix(fname):
    path = os.path.join(basedir, fname)
    if not os.path.exists(path):
        print(f"  NOT FOUND: {path}")
        return None, None, None
    with open(path) as f:
        data = json.load(f)
    rows = data.get('rows', [])
    n = max(r['row'] for r in rows) + 1
    T = np.zeros((n, n), dtype=np.float64)
    for r in rows:
        T[r['row'], :len(r['data'])] = r['data']
    T = (T + T.T) / 2
    evals, evecs = np.linalg.eigh(T)
    idx = np.argsort(np.abs(evals))[::-1]
    return evals[idx], evecs[:, idx], n

def irrep_composition(eigvec, k, n_edges=5):
    """Compute irrep fractions of an eigenvector in k^5 config space."""
    fracs = {}
    for r in range(k):
        weight = 0.0
        for c in range(len(eigvec)):
            labels = []
            tmp = c
            for i in range(n_edges):
                labels.append(tmp % k)
                tmp //= k
            # Fraction of this config's weight that "belongs" to irrep r
            count = labels.count(r)
            weight += eigvec[c]**2 * count / n_edges
        fracs[r] = weight
    return fracs

# ================================================================
# ANALYSIS 1: PROTON EIGENVECTOR DARK MATTER CONTENT
# ================================================================
def analysis_1():
    print(f"\n{'='*65}")
    print(f"ANALYSIS 1: PROTON DARK MATTER CONTENT")
    print(f"{'='*65}")
    
    evals, evecs, n = load_matrix('qcd3125_progress_seeded.json')
    if evals is None:
        return
    
    T_max = evals[0]
    print(f"  T_max = {T_max:.4e}, n = {n}")
    
    # First ~10 eigenstates
    print(f"\n  {'Level':>5} {'Mass MeV':>10} {'χ₁':>6} {'χ₃':>6} {'χ₃\'':>6} {'χ₄':>6} {'χ₅':>6} {'ID':>10}")
    print(f"  {'-'*65}")
    
    for i in range(min(10, n)):
        if evals[i] <= 0 or evals[i] >= T_max * 0.9999:
            if i == 0:
                fracs = irrep_composition(evecs[:, i], 5)
                print(f"  {i:>5} {'vacuum':>10} {fracs[0]*100:>5.1f}% {fracs[1]*100:>5.1f}% {fracs[2]*100:>5.1f}% {fracs[3]*100:>5.1f}% {fracs[4]*100:>5.1f}% {'vacuum':>10}")
            continue
        
        gap = -math.log(evals[i] / T_max)
        mass = gap * Lambda
        fracs = irrep_composition(evecs[:, i], 5)
        
        # Identify
        if mass < 200: name = "pion?"
        elif mass < 800: name = "eta?"
        elif mass < 1000: name = "proton?"
        elif mass < 1200: name = "N/Δ?"
        elif mass < 1500: name = "N*?"
        else: name = "heavy"
        
        dm_pct = fracs[3] * 100  # χ₄ fraction
        print(f"  {i:>5} {mass:>10.1f} {fracs[0]*100:>5.1f}% {fracs[1]*100:>5.1f}% {fracs[2]*100:>5.1f}% {fracs[3]*100:>5.1f}% {fracs[4]*100:>5.1f}% {name:>10}")
    
    # The proton candidate
    results = {}
    print(f"\n  KEY QUESTION: How much dark matter is in each hadron?")
    for i in range(1, min(8, n)):
        if evals[i] > 0:
            gap = -math.log(evals[i] / T_max)
            mass = gap * Lambda
            fracs = irrep_composition(evecs[:, i], 5)
            print(f"    State {i} ({mass:.0f} MeV): χ₄ = {fracs[3]*100:.2f}%")
            results[f'state_{i}'] = {
                'mass_MeV': round(mass, 1),
                'chi1_pct': round(fracs[0]*100, 3),
                'chi3_pct': round(fracs[1]*100, 3),
                'chi3p_pct': round(fracs[2]*100, 3),
                'chi4_pct': round(fracs[3]*100, 3),
                'chi5_pct': round(fracs[4]*100, 3),
            }
    return results

# ================================================================
# ANALYSIS 2: VISIBLE-DARK VACUUM OVERLAP
# ================================================================
def analysis_2():
    print(f"\n{'='*65}")
    print(f"ANALYSIS 2: VISIBLE-DARK VACUUM OVERLAP")
    print(f"{'='*65}")
    
    # Both are 1024 = 4^5. But they use DIFFERENT irrep sets!
    # Visible: indices [0,1,2,4] in A₅ table
    # Dark: indices [0,1,2,3] in A₅ table
    # The SHARED indices are [0,1,2] = χ₁, χ₃, χ₃'
    # So we can project both vacua onto the shared 3^5 = 243 subspace
    
    evals_v, evecs_v, n_v = load_matrix('qcd1024_progress.json')
    if evals_v is None:
        print("  Need qcd1024_progress.json")
        return
    
    # Load dark - try both possible filenames
    evals_d, evecs_d, n_d = load_matrix('dark1024_progress.json')
    if evals_d is None:
        # Try results file
        rfile = os.path.join(basedir, 'dark1024_results.json')
        if os.path.exists(rfile):
            print(f"  Found results but not progress file — cannot extract eigenvectors")
            print(f"  NEED dark1024_progress.json for this analysis")
            return
        return
    
    psi_v = evecs_v[:, 0]  # visible vacuum
    psi_d = evecs_d[:, 0]  # dark vacuum
    
    # Both live in 4^5 = 1024 space but with different irrep labeling
    # Visible: edge label 0,1,2,3 → χ₁,χ₃,χ₃',χ₅
    # Dark:    edge label 0,1,2,3 → χ₁,χ₃,χ₃',χ₄
    # Labels 0,1,2 are the SAME irreps (χ₁,χ₃,χ₃')
    # Label 3 is χ₅ in visible, χ₄ in dark — DIFFERENT irreps
    
    # Project onto shared subspace: configs where ALL edges are in {0,1,2}
    k = 4
    n_edges = 5
    shared_configs = []
    for c in range(1024):
        labels = []
        tmp = c
        for i in range(n_edges):
            labels.append(tmp % k)
            tmp //= k
        if all(l < 3 for l in labels):  # only χ₁, χ₃, χ₃'
            shared_configs.append(c)
    
    print(f"  Shared configs (χ₁,χ₃,χ₃' only): {len(shared_configs)} out of 1024")
    
    proj_v = np.array([psi_v[c] for c in shared_configs])
    proj_d = np.array([psi_d[c] for c in shared_configs])
    
    # Normalize projections
    norm_v = np.linalg.norm(proj_v)
    norm_d = np.linalg.norm(proj_d)
    
    print(f"  Visible vacuum weight in shared subspace: {norm_v**2*100:.2f}%")
    print(f"  Dark vacuum weight in shared subspace: {norm_d**2*100:.2f}%")
    
    if norm_v > 1e-10 and norm_d > 1e-10:
        overlap = np.dot(proj_v, proj_d) / (norm_v * norm_d)
        print(f"  Overlap = ⟨ψ_vis|ψ_dark⟩ (shared subspace) = {overlap:.6f}")
        print(f"  |Overlap|² = {overlap**2:.6f}")
        if abs(overlap) > 0.99:
            print(f"  → NEAR IDENTICAL in shared subspace")
            print(f"  → The two vacua agree on χ₁,χ₃,χ₃' content")
        elif abs(overlap) > 0.5:
            print(f"  → SIGNIFICANT overlap — sectors share vacuum structure")
        else:
            print(f"  → DIFFERENT vacua — sectors are independent")
        return {
            'visible_weight_in_shared': round(norm_v**2, 6),
            'dark_weight_in_shared': round(norm_d**2, 6),
            'overlap': round(float(overlap), 6),
            'overlap_sq': round(float(overlap**2), 6),
        }
    return None

# ================================================================
# ANALYSIS 3: WHICH MODE IS THE MISSING 60TH?
# ================================================================
def analysis_3():
    print(f"\n{'='*65}")
    print(f"ANALYSIS 3: THE MISSING 60TH MODE (Theorem 20c)")
    print(f"{'='*65}")
    
    eigen_file = os.path.join(basedir, 'icosa_2I_dark_fermionic_eigen.npz')
    bos_file = os.path.join(basedir, 'icosa_2I_bosonic_eigen.npz')
    
    if not os.path.exists(eigen_file):
        print(f"  NOT FOUND: {eigen_file}")
        return
    
    data = np.load(eigen_file, allow_pickle=True)
    evals = data['eigenvalues']
    evecs = data['eigenvectors'] if 'eigenvectors' in data else None
    
    # Sort by eigenvalue (positive ones)
    pos_mask = evals > 0
    pos_evals = evals[pos_mask]
    idx_pos = np.argsort(pos_evals)
    
    print(f"  Dark fermionic: {sum(pos_mask)} positive eigenvalues")
    print(f"  Smallest positive eigenvalue: {pos_evals[idx_pos[0]]:.4e}")
    
    if evecs is not None:
        # The SMALLEST positive eigenvalue's eigenvector is the "dying" mode
        smallest_idx = np.where(pos_mask)[0][idx_pos[0]]
        dying_vec = evecs[:, smallest_idx]
        
        print(f"\n  The 'dying' mode (smallest positive eigenvalue):")
        print(f"  Eigenvalue: {evals[smallest_idx]:.4e}")
        
        # For the 125-dim space with k=5 irreps and 3 edges per icosa vertex
        # The irrep indices for dark fermionic: ρ₂(2), ρ₂'(2), ρ₄'(4), ρ₅(5), ρ₆(6)
        dark_irr_names = ['ρ₂(2)', "ρ₂'(2)", "ρ₄'(4)", 'ρ₅(5)', 'ρ₆(6)']
        dark_dims = [2, 2, 4, 5, 6]
        k = 5
        n_edges_ico = 3  # icosahedron vertex degree is 5 but the boundary has 3 edges per triangular face
        
        # Actually, for icosahedral boundary with 5 irreps, n = 5^3 = 125
        n_edges_ico = 3
        print(f"\n  Eigenvector composition by irrep (125 = 5³ configs, 3 edges per face):")
        fracs = {}
        for r in range(k):
            weight = 0.0
            for c in range(len(dying_vec)):
                labels = []
                tmp = c
                for i in range(n_edges_ico):
                    labels.append(tmp % k)
                    tmp //= k
                count = labels.count(r)
                weight += dying_vec[c]**2 * count / n_edges_ico
            fracs[r] = weight
            print(f"    {dark_irr_names[r]:>10}: {weight*100:.2f}%")
        
        dominant = max(fracs, key=fracs.get)
        print(f"\n  Dominant irrep in dying mode: {dark_irr_names[dominant]} ({fracs[dominant]*100:.1f}%)")
        
        # Also check the LARGEST positive eigenvalue
        largest_idx = np.where(pos_mask)[0][idx_pos[-1]]
        ground_vec = evecs[:, largest_idx]
        
        print(f"\n  Ground state (largest positive eigenvalue) composition:")
        for r in range(k):
            weight = 0.0
            for c in range(len(ground_vec)):
                labels = []
                tmp = c
                for i in range(n_edges_ico):
                    labels.append(tmp % k)
                    tmp //= k
                count = labels.count(r)
                weight += ground_vec[c]**2 * count / n_edges_ico
            print(f"    {dark_irr_names[r]:>10}: {weight*100:.2f}%")
    
    # Compare with bosonic
    if os.path.exists(bos_file):
        bos_data = np.load(bos_file, allow_pickle=True)
        bos_evals = bos_data['eigenvalues']
        bos_pos = sorted([e for e in bos_evals if e > 0], reverse=True)
        dark_pos_sorted = sorted(pos_evals, reverse=True)
        
        print(f"\n  Bosonic has {len(bos_pos)} positive modes")
        print(f"  Dark fermionic has {len(dark_pos_sorted)} positive modes")
        print(f"  The missing mode has eigenvalue between:")
        if len(bos_pos) >= 60 and len(dark_pos_sorted) >= 59:
            print(f"    Bosonic #60 (smallest): {bos_pos[-1]:.4e}")
            print(f"    Dark ferm #59 (smallest): {dark_pos_sorted[-1]:.4e}")
    
    return {
        'n_positive_dark_fermionic': int(sum(pos_mask)),
        'smallest_positive_eigenvalue': float(pos_evals[idx_pos[0]]),
        'dying_mode_dominant_irrep': dark_irr_names[dominant] if evecs is not None else None,
        'dying_mode_composition': {dark_irr_names[r]: round(fracs[r]*100, 2) for r in range(k)} if evecs is not None else None,
    }

# ================================================================
# ANALYSIS 4: IS GAP RATIO EXACTLY 2?
# ================================================================
def analysis_4():
    print(f"\n{'='*65}")
    print(f"ANALYSIS 4: MATTER-ANTIMATTER / QED GAP RATIO")
    print(f"{'='*65}")
    
    # From known results
    gap_mm = 2.207   # χ₃+χ₃' gap
    gap_qed = 4.394  # χ₃+χ₅ gap (QED)
    
    ratio = gap_qed / gap_mm
    print(f"  Gap(χ₃+χ₃') = {gap_mm}")
    print(f"  Gap(χ₃+χ₅)  = {gap_qed}")
    print(f"  Ratio = {ratio:.6f}")
    print(f"  Exactly 2? {abs(ratio - 2.0):.6f} off")
    print(f"  Relative: {abs(ratio-2)/2*100:.4f}%")
    
    # Is the actual ratio 2ln(3)/ln(3) = 2? No, the gaps aren't exactly nln(3)
    print(f"\n  2×ln(3) = {2*math.log(3):.6f}")
    print(f"  4×ln(3) = {4*math.log(3):.6f}")
    print(f"  gap_mm/2ln3 = {gap_mm/(2*math.log(3)):.6f}")
    print(f"  gap_qed/4ln3 = {gap_qed/(4*math.log(3)):.6f}")
    
    # The gap ratio would be exactly 2 if both gaps scale the same way with ln(3)
    print(f"\n  The ratio is {ratio:.4f}")
    print(f"  If exactly 2.000: photon DOUBLES confinement strength")
    print(f"  Physical: adding χ₅ to χ₃+χ₃' requires TWO χ₃ propagators")
    print(f"  Each propagator contributes one factor of the gap")
    
    # Other subsector ratios
    print(f"\n  Other gap ratios:")
    gaps = {
        'χ₃+χ₃\'': 2.207,
        'χ₃+χ₅ (QED)': 4.394,
        'χ₃+χ₃\'+χ₄ (portal)': 2.973,
        'χ₁+χ₄ (V4)': 4.008,
        'χ₄+χ₅ (dark photon)': 4.181,
        'χ₁+χ₄+χ₅ (dark annihil)': 4.100,
    }
    base = gaps['χ₃+χ₃\'']
    ratios = {}
    for name, gap in gaps.items():
        print(f"    {name:>30}: gap = {gap:.3f}, ratio to mm = {gap/base:.4f}")
        ratios[name] = round(gap/base, 6)
    
    return {
        'gap_matter_antimatter': gap_mm,
        'gap_QED': gap_qed,
        'ratio_QED_over_mm': round(ratio, 6),
        'deviation_from_2': round(abs(ratio - 2.0), 6),
        'all_ratios': ratios,
    }

# ================================================================
# ANALYSIS 5: V₄ VACUUM — IS χ₁ = 1/60?
# ================================================================
def analysis_5():
    print(f"\n{'='*65}")
    print(f"ANALYSIS 5: V₄ VACUUM χ₁ FRACTION")
    print(f"{'='*65}")
    
    v4_file = os.path.join(basedir, 'dodec_V4_32_results.json')
    if not os.path.exists(v4_file):
        print(f"  NOT FOUND: {v4_file}")
        return
    
    with open(v4_file) as f:
        d = json.load(f)
    
    vac = d['vacuum_composition']
    chi1 = vac.get('χ₁', vac.get('chi1', 0))
    chi4 = vac.get('χ₄', vac.get('chi4', 0))
    
    print(f"  χ₁ fraction = {chi1:.10f}")
    print(f"  χ₄ fraction = {chi4:.10f}")
    print(f"  Sum = {chi1 + chi4:.10f}")
    
    print(f"\n  Is χ₁ = 1/|A₅| = 1/60 = {1/60:.10f}?")
    print(f"  Difference: {abs(chi1 - 1/60):.6e}")
    print(f"  Relative: {abs(chi1 - 1/60)/(1/60)*100:.4f}%")
    
    print(f"\n  Is χ₁ = dim(χ₁)²/Σdim² = 1/17?")
    print(f"  1/17 = {1/17:.10f}")
    print(f"  Difference: {abs(chi1 - 1/17):.6e}")
    print(f"  Relative: {abs(chi1 - 1/17)/(1/17)*100:.4f}%")
    
    # Other candidates
    cands = {
        '1/|A₅| = 1/60': 1/60,
        'dim²(χ₁)/dim²_sum = 1/17': 1/17,
        'dim²(χ₁)/dim²(χ₄) = 1/16': 1/16,
        '1/|A₅|² = 1/3600': 1/3600,
        'α': alpha,
        'α²': alpha**2,
        '1/200': 1/200,
        'dim(χ₁)/(dim(χ₁)+dim(χ₄))² = 1/25': 1/25,
    }
    
    print(f"\n  Systematic check:")
    best_match = None
    best_err = 999
    for name, val in sorted(cands.items(), key=lambda x: abs(x[1] - chi1)):
        err = abs(val - chi1) / chi1 * 100
        mark = " <---" if err < 2 else ""
        print(f"    {name:>35} = {val:.10f} ({err:.2f}%){mark}")
        if err < best_err:
            best_err = err
            best_match = name
    
    return {
        'chi1_fraction': float(chi1),
        'chi4_fraction': float(chi4),
        'best_match': best_match,
        'best_match_error_pct': round(best_err, 4),
        'is_1_over_60': round(abs(chi1 - 1/60)/(1/60)*100, 4),
        'is_1_over_17': round(abs(chi1 - 1/17)/(1/17)*100, 4),
    }

# ================================================================
# RUN ALL
# ================================================================
if __name__ == '__main__':
    print("FIVE QUICK ANALYSES FROM EXISTING DATA")
    print(f"Looking in: {os.path.abspath(basedir)}")
    
    results = {}
    results['1_proton_dark_content'] = analysis_1()
    results['2_vacuum_overlap'] = analysis_2()
    results['3_missing_60th_mode'] = analysis_3()
    results['4_gap_ratio'] = analysis_4()
    results['5_V4_vacuum'] = analysis_5()
    
    # Save
    outfile = 'five_quick_analyses_results.json'
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {outfile}")
    
    print(f"\n{'='*65}")
    print("DONE")
    print(f"{'='*65}")
