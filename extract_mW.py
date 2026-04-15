#!/usr/bin/env python3
"""
W MASS EXTRACTION FROM 3125×3125 EIGENSYSTEM
=============================================
The W boson mass should emerge from the eigenvalue spectrum of the
dodecahedral transfer matrix. This script loads pre-computed results
and extracts the W mass via eigenstate overlap analysis.

Usage:
  python3 extract_mW.py --progress qcd3125_progress_seeded.json
  python3 extract_mW.py --eigfile qcd3125_eigen.npz
  python3 extract_mW.py --tmatrix qcd3125_T.npy

Requirements: numpy
"""
import numpy as np
import math, json, sys, os, argparse

phi = (1 + np.sqrt(5)) / 2
Lambda = 332.0  # MeV

# A₅ character table
dims = np.array([1, 3, 3, 4, 5], dtype=np.float64)
chars = np.array([
    [1,  1,  1,  1,      1],
    [3, -1,  0,  phi,    1-phi],
    [3, -1,  0,  1-phi,  phi],
    [4,  0,  1, -1,     -1],
    [5,  1, -1,  0,      0],
], dtype=np.float64)
class_sizes = np.array([1, 15, 20, 12, 12], dtype=np.float64)
irr_names = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']

N = 5**5  # 3125

def idx_to_cfg(idx):
    c = []
    for _ in range(5):
        c.append(idx % 5); idx //= 5
    return tuple(reversed(c))

def cfg_to_idx(cfg):
    idx = 0
    for l in cfg:
        idx = idx * 5 + l
    return idx

def irrep_content(vec):
    """Fraction of eigenvector weight in each irrep sector."""
    content = np.zeros(5)
    for i in range(N):
        w = vec[i]**2
        if w < 1e-15: continue
        cfg = idx_to_cfg(i)
        for r in cfg:
            content[r] += w / 5.0
    total = content.sum()
    return content / total if total > 0 else content

def analyse(evals, evecs):
    """Full W mass extraction."""
    order = np.argsort(-np.abs(evals))
    evals = evals[order]; evecs = evecs[:, order]
    T0 = evals[0]
    
    print(f"\n{'='*65}")
    print(f"W MASS EXTRACTION")
    print(f"{'='*65}")
    print(f"  T_max = {T0:.6e}")
    print(f"  Positive eigenvalues: {np.sum(evals > 0)}")
    
    # --- Find eigenstates in the W mass window (70-100 GeV) ---
    print(f"\n--- Eigenstates near m_W = 80 GeV ---")
    w_candidates = []
    for i in range(min(len(evals), 1000)):
        if 0 < evals[i] < T0 * 0.9999:
            mass = -Lambda * math.log(evals[i] / T0)
            if 60000 < mass < 110000:
                content = irrep_content(evecs[:, i])
                w_candidates.append((i, mass, content))
                comp = ', '.join(f'{irr_names[j]}:{content[j]*100:.0f}%' 
                                 for j in range(5) if content[j] > 0.05)
                print(f"  #{i}: {mass/1000:.2f} GeV  [{comp}]")
    
    # --- Source: single χ₄ excitation (the W boson IS χ₄) ---
    print(f"\n--- W source: single χ₄ edge excitation ---")
    s_W = np.zeros(N)
    for edge in range(5):
        cfg = [0]*5; cfg[edge] = 3  # χ₄ on one edge
        s_W[cfg_to_idx(cfg)] += dims[3]
    s_W /= np.linalg.norm(s_W)
    
    overlaps = np.abs(evecs.T @ s_W)**2
    print(f"  {'Rank':>4s} {'Mass(GeV)':>10s} {'|⟨s|n⟩|²':>12s} {'χ₄ content':>10s}")
    hits = []
    for i in range(len(evals)):
        if overlaps[i] > 1e-6 and 0 < evals[i] < T0*0.9999:
            mass = -Lambda * math.log(evals[i] / T0)
            chi4 = irrep_content(evecs[:,i])[3]
            hits.append((i, mass, overlaps[i], chi4))
    hits.sort(key=lambda x: -x[2])
    for i, mass, ov, chi4 in hits[:15]:
        print(f"  {i:>4d} {mass/1000:>10.2f} {ov:>12.4e} {chi4*100:>9.0f}%")
    
    # Weighted mass in W window
    w_hits = [(i,m,ov,c) for i,m,ov,c in hits if 50000<m<200000]
    if w_hits:
        tot = sum(ov for _,_,ov,_ in w_hits)
        avg = sum(ov*m for _,m,ov,_ in w_hits) / tot if tot > 0 else 0
        print(f"\n  Overlap-weighted W mass: {avg/1000:.2f} GeV (obs: 80.38 GeV)")
    
    # --- Source: all-χ₃ face (pattern-match m_W = Λ×3⁵) ---
    print(f"\n--- All-χ₃ face source ---")
    s_all3 = np.zeros(N)
    s_all3[cfg_to_idx([1,1,1,1,1])] = 1.0
    ov3 = np.abs(evecs.T @ s_all3)**2
    print(f"  {'Rank':>4s} {'Mass(GeV)':>10s} {'|⟨s|n⟩|²':>12s}")
    hits3 = []
    for i in range(len(evals)):
        if ov3[i] > 1e-6 and 0 < evals[i] < T0*0.9999:
            mass = -Lambda * math.log(evals[i] / T0)
            hits3.append((i, mass, ov3[i]))
    hits3.sort(key=lambda x: -x[2])
    for i, mass, ov in hits3[:10]:
        print(f"  {i:>4d} {mass/1000:>10.2f} {ov:>12.4e}")
    
    # --- Source: weak current (χ₃→χ₃' vertex) ---
    print(f"\n--- Weak current source (χ₃⊗χ₃' vertex) ---")
    s_weak = np.zeros(N)
    for e1 in range(5):
        for e2 in range(5):
            if e1 != e2:
                cfg = [0]*5; cfg[e1] = 1; cfg[e2] = 2  # χ₃ × χ₃'
                s_weak[cfg_to_idx(cfg)] += dims[1]*dims[2]
    s_weak /= np.linalg.norm(s_weak)
    ov_w = np.abs(evecs.T @ s_weak)**2
    hits_w = []
    for i in range(len(evals)):
        if ov_w[i] > 1e-6 and 0 < evals[i] < T0*0.9999:
            mass = -Lambda * math.log(evals[i] / T0)
            hits_w.append((i, mass, ov_w[i]))
    hits_w.sort(key=lambda x: -x[2])
    print(f"  {'Rank':>4s} {'Mass(GeV)':>10s} {'|⟨s|n⟩|²':>12s}")
    for i, mass, ov in hits_w[:10]:
        print(f"  {i:>4d} {mass/1000:>10.2f} {ov:>12.4e}")
    
    # --- Summary ---
    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"  Pattern-match: Λ × 3⁵ = {Lambda*243:.0f} MeV = {Lambda*243/1000:.2f} GeV")
    print(f"  Observed:      80379 MeV = 80.38 GeV")
    a_inv = 137.036; s2w = 3.0/13.0
    mW_ew = 1000*245.4*math.sqrt(math.pi/a_inv/(math.sqrt(2)*s2w))
    print(f"  Electroweak:   v√(πα/√2sin²θ) = {mW_ew:.0f} MeV")
    print(f"")
    print(f"  NOTE: The W mass at 80 GeV = 243Λ is NOT a transfer matrix")
    print(f"  eigenvalue — it requires λ/λ₀ = exp(-243) ≈ 10⁻¹⁰⁶, far below")
    print(f"  numerical precision. The formula m_W = Λ × dim(χ₃)⁵ is STRUCTURAL")
    print(f"  (the dimension of the all-matter configuration space), not dynamical")
    print(f"  (an eigenvalue gap). The W mass is the face weight, not a mass gap.")
    
    return w_candidates

# ================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eigfile', type=str, default=None)
    parser.add_argument('--tmatrix', type=str, default=None)
    parser.add_argument('--progress', type=str, default=None)
    args = parser.parse_args()
    
    evals = evecs = None
    
    if args.eigfile and os.path.exists(args.eigfile):
        print(f"Loading from {args.eigfile}...")
        d = np.load(args.eigfile)
        evals, evecs = d['eigenvalues'], d['eigenvectors']
    
    elif args.tmatrix and os.path.exists(args.tmatrix):
        print(f"Loading matrix from {args.tmatrix}...")
        T = np.load(args.tmatrix)
        T = (T + T.T) / 2
        print(f"  Diagonalising {T.shape}...")
        evals, evecs = np.linalg.eigh(T)
        np.savez('qcd3125_eigen.npz', eigenvalues=evals, eigenvectors=evecs)
    
    elif args.progress and os.path.exists(args.progress):
        print(f"Loading from {args.progress}...")
        with open(args.progress) as f:
            prog = json.load(f)
        rows = prog.get('rows', [])
        T = np.zeros((3125, 3125))
        for e in rows:
            T[e['row']] = e['data']
        print(f"  Loaded {len(rows)}/3125 rows")
        if len(rows) < 3125:
            print(f"  WARNING: incomplete ({len(rows)/3125*100:.0f}%)")
        T = (T + T.T) / 2
        print(f"  Diagonalising...")
        evals, evecs = np.linalg.eigh(T)
    
    if evals is not None:
        analyse(evals, evecs)
    else:
        print("Usage: python3 extract_mW.py --progress qcd3125_progress_seeded.json")
