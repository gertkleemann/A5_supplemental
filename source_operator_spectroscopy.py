#!/usr/bin/env python3
"""
SOURCE OPERATOR SPECTROSCOPY OF THE 3125 MATRIX
=================================================
Defines meson, baryon, glueball, and dark source operators,
projects the 3125 eigenvectors onto each, and extracts masses
from the leading overlaps.

This is standard lattice QCD methodology applied to the A₅ transfer matrix.
No new computation of T is needed — only projections of known eigenvectors.

Usage: python source_operator_spectroscopy.py [path_to_qcd3125_progress_seeded.json]
"""
import numpy as np
import json, sys, os, math

phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda = 332.0

fname = sys.argv[1] if len(sys.argv) > 1 else 'qcd3125_progress_seeded.json'

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
print(f"Positive eigenvalues: {sum(1 for e in evals if e > 0)}")

# ================================================================
# CONFIG LABELING: config c → 5 edge labels (base-5 digits)
# irreps: 0=χ₁, 1=χ₃, 2=χ₃', 3=χ₄, 4=χ₅
# ================================================================
k = 5
n_edges = 5
irr_names = ['chi1', 'chi3', 'chi3p', 'chi4', 'chi5']
irr_display = ['χ₁', 'χ₃', "χ₃'", 'χ₄', 'χ₅']

def config_labels(c):
    """Return list of 5 irrep indices for config c."""
    labels = []
    tmp = c
    for i in range(n_edges):
        labels.append(tmp % k)
        tmp //= k
    return labels

# Pre-compute all labels
all_labels = [config_labels(c) for c in range(n)]

# ================================================================
# SOURCE OPERATORS
# ================================================================
# Each source is a 3125-dimensional vector
# Convention: edges are numbered 0-4 around the pentagon

def make_source(name, condition):
    """Build source vector from a condition on edge labels."""
    src = np.zeros(n, dtype=np.float64)
    count = 0
    for c in range(n):
        labels = all_labels[c]
        if condition(labels):
            src[c] = 1.0
            count += 1
    if np.linalg.norm(src) > 0:
        src /= np.linalg.norm(src)  # normalize
    print(f"  Source '{name}': {count} matching configs")
    return src

print(f"\n{'='*65}")
print(f"BUILDING SOURCE OPERATORS")
print(f"{'='*65}")

sources = {}

# MESON: quark-antiquark on adjacent edges (any pair)
sources['meson_qq'] = make_source('meson (chi3-chi3p, any pair)',
    lambda l: any(
        (l[i] == 1 and l[(i+1)%5] == 2) or (l[i] == 2 and l[(i+1)%5] == 1)
        for i in range(5)
    ))

# MESON strict: exactly one chi3 and one chi3' on adjacent edges, rest chi5
sources['meson_pure'] = make_source('meson pure (chi3-chi3p adj, rest chi5)',
    lambda l: any(
        ((l[i] == 1 and l[(i+1)%5] == 2) or (l[i] == 2 and l[(i+1)%5] == 1))
        and all(l[j] == 4 for j in range(5) if j != i and j != (i+1)%5)
        for i in range(5)
    ))

# BARYON: three chi3 on three consecutive edges
sources['baryon_3q'] = make_source('baryon (3×chi3 consecutive)',
    lambda l: any(
        l[i] == 1 and l[(i+1)%5] == 1 and l[(i+2)%5] == 1
        for i in range(5)
    ))

# BARYON with gluon: three chi3 + rest chi5
sources['baryon_3q_glue'] = make_source('baryon (3×chi3, rest chi5)',
    lambda l: l.count(1) >= 3 and all(x in [1, 4] for x in l))

# GLUEBALL: all chi5
sources['glueball'] = make_source('glueball (all chi5)',
    lambda l: all(x == 4 for x in l))

# GLUEBALL relaxed: mostly chi5
sources['glueball_4'] = make_source('glueball (4+ chi5)',
    lambda l: l.count(4) >= 4)

# PSEUDOSCALAR: chi3-chi3' on opposite-ish edges (non-adjacent)
sources['pseudoscalar'] = make_source('pseudoscalar (chi3-chi3p non-adj)',
    lambda l: any(
        (l[i] == 1 and l[(i+2)%5] == 2) or (l[i] == 2 and l[(i+2)%5] == 1)
        for i in range(5)
    ))

# VECTOR: chi3-chi5 (quark-gluon)
sources['vector_qg'] = make_source('vector (chi3-chi5 adjacent)',
    lambda l: any(
        (l[i] == 1 and l[(i+1)%5] == 4) or (l[i] == 4 and l[(i+1)%5] == 1)
        for i in range(5)
    ))

# DARK: any chi4 content
sources['dark_any'] = make_source('dark (any chi4)',
    lambda l: 3 in l)

# DARK BARYON: chi4 × chi3 × chi3
sources['dark_baryon'] = make_source('dark baryon (chi4-chi3-chi3)',
    lambda l: any(
        l[i] == 3 and l[(i+1)%5] == 1 and l[(i+2)%5] == 1
        for i in range(5)
    ))

# VACUUM: all chi1
sources['vacuum'] = make_source('vacuum (all chi1)',
    lambda l: all(x == 0 for x in l))

# ================================================================
# PROJECT EIGENVECTORS ONTO SOURCES
# ================================================================
print(f"\n{'='*65}")
print(f"SPECTROSCOPY: PROJECTING EIGENVECTORS ONTO SOURCES")
print(f"{'='*65}")

known_hadrons = [
    (135, 'pion'), (498, 'kaon'), (548, 'eta'), (775, 'rho'),
    (782, 'omega'), (938, 'proton'), (958, "eta'"), (1019, 'phi'),
    (1232, 'Delta'), (1370, 'f0(1370)'), (1440, 'Roper'),
]

results = {}

for src_name, src_vec in sources.items():
    if np.linalg.norm(src_vec) < 1e-10:
        continue
    
    # Compute overlap with all positive eigenvectors
    overlaps = []
    for i in range(n):
        if evals[i] > 0 and evals[i] < T_max * 0.99999:
            ov = np.dot(src_vec, evecs[:, i])**2
            if ov > 1e-15:
                gap = -math.log(evals[i] / T_max)
                mass = gap * Lambda
                overlaps.append((i, ov, gap, mass))
    
    # Sort by overlap (largest first)
    overlaps.sort(key=lambda x: -x[1])
    
    print(f"\n  --- {src_name} ---")
    print(f"  {'Rank':>4} {'Eigvec#':>8} {'Overlap²':>12} {'Gap':>8} {'Mass MeV':>10} {'Match':>12} {'Err':>6}")
    print(f"  {'-'*65}")
    
    src_results = []
    for rank, (i, ov, gap, mass) in enumerate(overlaps[:10]):
        best = min(known_hadrons, key=lambda h: abs(h[0] - mass))
        err = abs(best[0] - mass) / best[0] * 100
        mark = ' *' if err < 5 else ''
        print(f"  {rank:>4} {i:>8} {ov:>12.6f} {gap:>8.4f} {mass:>10.1f} {best[1]:>12} {err:>5.1f}%{mark}")
        
        src_results.append({
            'rank': rank,
            'eigenvector_index': int(i),
            'overlap_sq': round(float(ov), 8),
            'gap': round(gap, 6),
            'mass_MeV': round(mass, 1),
            'closest_hadron': best[1],
            'error_pct': round(err, 2),
        })
    
    results[src_name] = src_results

# ================================================================
# SUMMARY: BEST MATCH PER CHANNEL
# ================================================================
print(f"\n{'='*65}")
print(f"SUMMARY: LEADING STATE PER CHANNEL")
print(f"{'='*65}")

print(f"\n  {'Channel':>20} {'Mass MeV':>10} {'Overlap²':>12} {'Match':>12} {'Err':>6}")
print(f"  {'-'*65}")

for src_name in sources:
    if src_name in results and results[src_name]:
        best = results[src_name][0]
        print(f"  {src_name:>20} {best['mass_MeV']:>10.1f} {best['overlap_sq']:>12.6f} {best['closest_hadron']:>12} {best['error_pct']:>5.1f}%")

# ================================================================
# THE KEY QUESTION: PION AND PROTON
# ================================================================
print(f"\n{'='*65}")
print(f"THE KEY QUESTION: WHERE ARE THE PION AND PROTON?")
print(f"{'='*65}")

# Search all channels for states near 135 MeV and 938 MeV
for target, name in [(135, 'PION'), (938, 'PROTON')]:
    print(f"\n  {name} ({target} MeV):")
    found = False
    for src_name in sources:
        if src_name in results:
            for state in results[src_name]:
                if abs(state['mass_MeV'] - target) / target < 0.15:  # within 15%
                    print(f"    Channel '{src_name}': mass {state['mass_MeV']:.1f} MeV, overlap² {state['overlap_sq']:.6f}, err {state['error_pct']:.1f}%")
                    found = True
    if not found:
        print(f"    NOT FOUND in any channel within 15%")
        print(f"    Closest states per channel:")
        for src_name in sources:
            if src_name in results:
                closest = min(results[src_name], key=lambda s: abs(s['mass_MeV'] - target))
                print(f"      {src_name:>20}: {closest['mass_MeV']:.0f} MeV ({abs(closest['mass_MeV']-target)/target*100:.0f}% off)")

# ================================================================
# SAVE
# ================================================================
outfile = 'source_operator_spectroscopy_results.json'
with open(outfile, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {outfile}")
