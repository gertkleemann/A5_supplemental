#!/usr/bin/env python3
"""
ALL UNCOMPUTED DODECAHEDRAL SUB-SECTORS
========================================
Six fast computations, each under 1 minute at 20 cores.
Run alongside the dark_sector_1024.py (which uses separate files).

Usage: python3 dodec_subsectors.py --cores 10

Output files (all unique, no overlaps):
  dodec_QED_32_results.json          — χ₃+χ₅: pure QED
  dodec_dark_photon_32_results.json  — χ₄+χ₅: dark photon coupling
  dodec_matter_antimatter_32_results.json — χ₃+χ₃': matter-antimatter
  dodec_matter_dark_243_results.json — χ₃+χ₃'+χ₄: matter-dark portal
  dodec_dark_annihil_243_results.json — χ₁+χ₄+χ₅: dark annihilation
  dodec_no_vacuum_1024_results.json  — χ₃+χ₃'+χ₄+χ₅: everything minus vacuum
"""
import numpy as np
import math, time, json, os, sys
from itertools import product as iprod
from collections import deque

# ================================================================
# A₅ DATA
# ================================================================
phi = (1 + math.sqrt(5)) / 2
sqrt5 = math.sqrt(5)
alpha_fw = 1 / (20*phi**4 - (3+5*sqrt5)/308)
Lambda_QCD = 332
N_IRREPS = 5

dims = [1, 3, 3, 4, 5]
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

threej = np.zeros((N_IRREPS, N_IRREPS, N_IRREPS))
for a in range(N_IRREPS):
    for b in range(N_IRREPS):
        for c in range(N_IRREPS):
            threej[a, b, c] = round(
                sum(class_sizes[k]*chars[a][k]*chars[b][k]*chars[c][k]
                    for k in range(5)) / G_ORDER)

# ================================================================
# DODECAHEDRON
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
dv = np.array(dv); Nv = 20

edges = []; adj = [[] for _ in range(Nv)]
for i in range(Nv):
    for j in range(i+1,Nv):
        if abs(np.linalg.norm(dv[i]-dv[j])-2/phi)<0.01:
            eidx=len(edges); edges.append((i,j))
            adj[i].append(eidx); adj[j].append(eidx)
Ne = len(edges)

def _get_edge_idx(v1,v2):
    key=(min(v1,v2),max(v1,v2))
    for i,e in enumerate(edges):
        if e==key: return i
    return None

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

_opp = None
for i in range(len(_faces)):
    for j in range(i+1,len(_faces)):
        if len(set(_faces[i])&set(_faces[j]))==0:
            _opp=(i,j); break
    if _opp: break

face_A = _faces[_opp[0]]; face_B = _faces[_opp[1]]

def _face_edges(fv):
    fv=sorted(fv); r=[]
    for i in range(len(fv)):
        for j in range(i+1,len(fv)):
            e=_get_edge_idx(fv[i],fv[j])
            if e is not None: r.append(e)
    return r

edges_A = _face_edges(face_A)
edges_B = _face_edges(face_B)

visited=[False]*Nv; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

# ================================================================
# TENSOR NETWORK ENGINE
# ================================================================
def compute_T(la, lb):
    fixed = {}
    for idx,eidx in enumerate(edges_A): fixed[eidx]=la[idx]
    for idx,eidx in enumerate(edges_B): fixed[eidx]=lb[idx]
    processed=set(); open_edges=[]; state={():1.0}
    for v in order:
        v_edges=adj[v]; new_e=[]; closing_e=[]
        for eidx in v_edges:
            i,j=edges[eidx]; other=j if i==v else i
            typ='f' if eidx in fixed else 'r'
            if other in processed: closing_e.append((typ,eidx))
            else: new_e.append((typ,eidx))
        edge_pos={e:i for i,e in enumerate(open_edges)}
        cfp=sorted([(eidx,edge_pos[eidx]) for typ,eidx in closing_e if typ=='r'],
                    key=lambda x:-x[1])
        new_state={}; free_new=[eidx for typ,eidx in new_e if typ=='r']
        for fl in iprod(range(N_IRREPS),repeat=len(free_new)):
            for ok,ow in state.items():
                if abs(ow)<1e-15: continue
                el={}; fi=0
                for typ,eidx in new_e:
                    if typ=='f': el[eidx]=fixed[eidx]
                    else: el[eidx]=fl[fi]; fi+=1
                for typ,eidx in closing_e:
                    if typ=='f': el[eidx]=fixed[eidx]
                    else: el[eidx]=ok[edge_pos[eidx]]
                jabc=tuple(el[e] for e in v_edges)
                vw=threej[jabc]
                if vw==0: continue
                ew=1.0
                for eidx in free_new: ew*=dims[el[eidx]]
                nkl=list(ok)
                for _,pos in cfp: nkl.pop(pos)
                for eidx in free_new: nkl.append(el[eidx])
                nk=tuple(nkl)
                new_state[nk]=new_state.get(nk,0.0)+ow*vw*ew
        new_open=[e for e in open_edges
                  if not any(eidx==e for typ,eidx in closing_e if typ=='r')]
        new_open.extend(free_new)
        open_edges=new_open; processed.add(v); state=new_state
    return sum(state.values())

# ================================================================
# WORKER FUNCTION (module-level for Windows pickling)
# ================================================================
def _compute_row(args):
    ia, sector_indices = args
    local_configs = list(iprod(sector_indices, repeat=5))
    ca = local_configs[ia]
    row = [0.0]*len(local_configs)
    for ib in range(len(local_configs)):
        row[ib] = compute_T(ca, local_configs[ib])
    return (ia, row)

# ================================================================
# GENERIC SECTOR COMPUTATION
# ================================================================
def compute_sector(sector_indices, sector_name, result_file, n_cores):
    sector_names = [irr_names[i] for i in sector_indices]
    n_sec = len(sector_indices)
    dims_sq = sum(dims[i]**2 for i in sector_indices)
    configs = list(iprod(sector_indices, repeat=5))
    n_configs = len(configs)
    
    print(f"\n{'='*65}")
    print(f"  {sector_name}")
    print(f"  Sector: {' + '.join(sector_names)}")
    print(f"  Σdim² = {dims_sq}/60 ({dims_sq/60*100:.1f}%)")
    print(f"  Matrix: {n_configs}×{n_configs}, Elements: {n_configs**2:,}")
    print(f"{'='*65}")
    
    args = [(ia, sector_indices) for ia in range(n_configs)]
    
    T_matrix = np.zeros((n_configs, n_configs))
    t_start = time.time()
    
    if n_cores > 1 and n_configs > 4:
        from multiprocessing import Pool
        done = 0
        with Pool(n_cores) as pool:
            for ia, row in pool.imap_unordered(_compute_row, args):
                T_matrix[ia] = row
                done += 1
                if done % max(1, n_configs//10) == 0 or done == n_configs:
                    elapsed = time.time()-t_start
                    print(f"    {done}/{n_configs} ({done/n_configs*100:.0f}%) | {elapsed:.1f}s")
    else:
        for a in args:
            ia, row = _compute_row(a)
            T_matrix[ia] = row
    
    elapsed = time.time()-t_start
    print(f"  Computed in {elapsed:.1f}s")
    
    # Analysis
    T_sym = (T_matrix + T_matrix.T) / 2
    evals_raw, evecs_raw = np.linalg.eigh(T_sym)
    idx = np.argsort(evals_raw)[::-1]
    evals = evals_raw[idx]
    evecs = evecs_raw[:, idx]
    
    T_max = evals[0]
    n_pos = sum(1 for e in evals if e > 0)
    
    masses = []
    for i, ev in enumerate(evals):
        if ev > 0 and ev < T_max * 0.9999:
            gap = -math.log(ev/T_max)
            masses.append({'level':i, 'eigenvalue':float(ev),
                           'gap':float(gap), 'mass_MeV':float(gap*Lambda_QCD)})
    
    print(f"  T_max = {T_max:.6e}")
    print(f"  Positive: {n_pos}, Negative: {n_configs-n_pos}")
    if masses:
        print(f"  First mass gap: {masses[0]['gap']:.4f} = {masses[0]['mass_MeV']:.0f} MeV")
    
    print(f"  First 10 masses (MeV):")
    for i, m in enumerate(masses[:10]):
        print(f"    {i}: {m['mass_MeV']:.0f} MeV (gap={m['gap']:.4f})")
    
    # Vacuum composition
    gs = evecs[:, 0]
    irrep_content = {}
    for ir_idx in sector_indices:
        irrep_content[irr_names[ir_idx]] = 0.0
    for ic, cfg in enumerate(configs):
        w = gs[ic]**2
        for edge_irrep in cfg:
            irrep_content[irr_names[edge_irrep]] += w / 5.0
    
    print(f"  Vacuum composition:")
    for name, frac in irrep_content.items():
        bar = '█' * int(frac * 40)
        print(f"    {name:>4}: {frac*100:5.1f}% {bar}")
    
    # Save
    results = {
        'computation': sector_name,
        'sector_indices': sector_indices,
        'sector_names': sector_names,
        'sector_dims_sq': dims_sq,
        'n_configs': n_configs,
        'T_max': float(T_max),
        'n_positive_eigenvalues': n_pos,
        'eigenvalues_top50': [float(e) for e in evals[:50]],
        'masses_MeV': [m['mass_MeV'] for m in masses[:30]],
        'mass_gaps': [m['gap'] for m in masses[:30]],
        'vacuum_composition': {k: float(v) for k,v in irrep_content.items()},
        'computation_time_sec': elapsed,
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
    
    print(f"DODECAHEDRAL SUB-SECTOR SURVEY")
    print(f"Cores: {N_CORES}")
    print(f"6 computations, all fast (<1 min each at 20 cores)")
    
    sectors = [
        ([1, 4], "QED core: χ₃+χ₅ (matter + photon)",
         "dodec_QED_32_results.json"),
        
        ([3, 4], "Dark photon: χ₄+χ₅ (dark + photon)",
         "dodec_dark_photon_32_results.json"),
        
        ([1, 2], "Matter-antimatter: χ₃+χ₃' (no forces)",
         "dodec_matter_antimatter_32_results.json"),
        
        ([1, 2, 3], "Matter-dark portal: χ₃+χ₃'+χ₄",
         "dodec_matter_dark_243_results.json"),
        
        ([0, 3, 4], "Dark annihilation: χ₁+χ₄+χ₅",
         "dodec_dark_annihil_243_results.json"),
        
        ([1, 2, 3, 4], "No-vacuum: χ₃+χ₃'+χ₄+χ₅",
         "dodec_no_vacuum_1024_results.json"),
    ]
    
    all_results = {}
    t_total = time.time()
    
    for sector_indices, name, result_file in sectors:
        if os.path.exists(result_file):
            print(f"\n  {result_file} exists, skipping. Delete to recompute.")
            continue
        r = compute_sector(sector_indices, name, result_file, N_CORES)
        all_results[name] = r
    
    total_time = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"ALL DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*65}")
    
    # Cross-comparison
    print(f"\nCROSS-COMPARISON:")
    print(f"{'Sector':45} {'T_max':>12} {'1st gap':>8} {'1st mass':>10}")
    print(f"{'-'*80}")
    for name, r in all_results.items():
        gap = r['mass_gaps'][0] if r['mass_gaps'] else 0
        mass = r['masses_MeV'][0] if r['masses_MeV'] else 0
        print(f"{name:45} {r['T_max']:>12.4e} {gap:>8.4f} {mass:>8.0f} MeV")
    
    # Compare with existing results if available
    for fname, desc in [('qcd1024_results.json', 'Visible sector (χ₁+χ₃+χ₃\'+χ₅)'),
                         ('dark1024_results.json', 'Dark sector (χ₁+χ₃+χ₃\'+χ₄)')]:
        if os.path.exists(fname):
            with open(fname) as f:
                r = json.load(f)
            gap = r.get('mass_gaps', r.get('dark_masses_MeV', [0]))[0] if 'mass_gaps' in r else 0
            print(f"{'[existing] '+desc:45} {r.get('T_max',r.get('eigenvalues_top100',[0])[0]):>12.4e} {gap:>8.4f}")
