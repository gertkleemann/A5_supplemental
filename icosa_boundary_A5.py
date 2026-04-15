#!/usr/bin/env python3
"""
ICOSAHEDRAL BOUNDARY TRANSFER MATRIX (NUMBA-OPTIMIZED)
=======================================================
Leptons are boundary modes of the A₅ cell. Their masses come from 
the icosahedral transmission coefficient, not the dodecahedral bulk.

Usage:
  python3 icosa_boundary.py --sector 0,1 --cores 4        # χ₁+χ₃ (8×8, seconds)
  python3 icosa_boundary.py --sector 0,1,2 --cores 8      # +χ₃' (27×27, minutes)
  python3 icosa_boundary.py --sector 0,1,2,4 --cores 16   # +χ₅ (64×64, ~hour)
  python3 icosa_boundary.py --sector 0,1,2,3,4 --cores 16 # all (125×125, hours)

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
# ICOSAHEDRON GEOMETRY
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

visited=[False]*12; order=[]; q=deque()
for v in sorted(face_A): q.append(v); visited[v]=True
while q:
    v=q.popleft(); order.append(v)
    for eidx in adj[v]:
        i,j=edges[eidx]; nb=j if i==v else i
        if not visited[nb]: visited[nb]=True; q.append(nb)

# ================================================================
# CONTRACTION PLAN
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
# NUMBA KERNEL
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

def compute_row(args):
    ia, sector_range = args
    cfgs = list(iprod(sector_range, repeat=3))
    n = len(cfgs); la = np.array(cfgs[ia], dtype=np.int64)
    row = np.zeros(n, dtype=np.float64)
    for ib in range(n):
        lb = np.array(cfgs[ib], dtype=np.int64)
        row[ib] = compute_T_numba(la, lb, N_IR, dims_sec, fivej_sec,
                                   plan_edge_types, plan_edge_indices, plan_n_edges,
                                   plan_n_new, plan_n_open_before, plan_n_open_after,
                                   plan_closing, plan_n_closing,
                                   plan_surviving, plan_n_surviving)
    return (ia, row.tolist())

# ================================================================
# SECTOR SETUP (module level for multiprocessing visibility)
# ================================================================
SECTOR = [0, 1]; N_CORES = 1
for i,arg in enumerate(sys.argv):
    if arg=='--sector' and i+1<len(sys.argv):
        SECTOR=[int(x) for x in sys.argv[i+1].split(',')]
    if arg=='--cores' and i+1<len(sys.argv):
        N_CORES=int(sys.argv[i+1])

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

configs = list(iprod(range(int(N_IR)), repeat=3))
n_configs = len(configs)

# ================================================================
if __name__ == '__main__':
    print(f"\nICOSAHEDRAL BOUNDARY TRANSFER MATRIX")
    print(f"Sector: {SECTOR} ({N_IR} irreps), Cores: {N_CORES}")
    print(f"Peak state: {int(N_IR)}^{max_open} = {int(N_IR)**max_open}")
    print(f"Matrix: {n_configs}×{n_configs}")
    print(f"5j nonzero: {np.count_nonzero(fivej_sec)}/{int(N_IR)**5}")

    print("JIT compiling...", end=' ', flush=True)
    t0=time.time()
    la_t=np.zeros(3,dtype=np.int64)
    _=compute_T_numba(la_t,la_t,N_IR,dims_sec,fivej_sec,
                       plan_edge_types,plan_edge_indices,plan_n_edges,
                       plan_n_new,plan_n_open_before,plan_n_open_after,
                       plan_closing,plan_n_closing,plan_surviving,plan_n_surviving)
    print(f"done [{time.time()-t0:.1f}s]")

    # Benchmark
    t0=time.time()
    la=np.zeros(3,dtype=np.int64); lb=np.array([int(N_IR)-1]*3,dtype=np.int64)
    v=compute_T_numba(la,lb,N_IR,dims_sec,fivej_sec,
                       plan_edge_types,plan_edge_indices,plan_n_edges,
                       plan_n_new,plan_n_open_before,plan_n_open_after,
                       plan_closing,plan_n_closing,plan_surviving,plan_n_surviving)
    dt=time.time()-t0
    print(f"Benchmark: {dt:.3f}s/element → est {dt*n_configs**2/max(N_CORES,1)/60:.1f} min total")

    SAVE_FILE = f'icosa_{"-".join(str(s) for s in SECTOR)}_progress.json'
    T_matrix = np.zeros((n_configs, n_configs))
    completed_rows = set()
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f: prog=json.load(f)
        for entry in prog.get('rows',[]):
            T_matrix[entry['row']]=entry['data']; completed_rows.add(entry['row'])
        print(f"Loaded {len(completed_rows)} rows from {SAVE_FILE}")

    remaining=[ia for ia in range(n_configs) if ia not in completed_rows]
    print(f"Done: {len(completed_rows)}, Remaining: {len(remaining)}")

    if remaining:
        t_start=time.time(); done_session=0
        def save_prog():
            rows=[{'row':int(ia),'data':T_matrix[ia].tolist()} for ia in sorted(completed_rows)]
            with open(SAVE_FILE,'w') as f: json.dump({'rows':rows},f)

        if N_CORES>1:
            from multiprocessing import Pool
            args=[(ia,list(range(int(N_IR)))) for ia in remaining]
            with Pool(N_CORES) as pool:
                for ia,row in pool.imap_unordered(compute_row,args):
                    T_matrix[ia]=row; completed_rows.add(ia); done_session+=1
                    el=time.time()-t_start; rate=done_session/el
                    left=len(remaining)-done_session; eta=left/rate if rate>0 else 0
                    pct=len(completed_rows)/n_configs*100
                    print(f"  Row {ia:>4} | {len(completed_rows):>4}/{n_configs} ({pct:.1f}%) | "
                          f"{el/60:.1f}min | ~{eta/60:.1f}min left",flush=True)
                    if done_session%5==0: save_prog()
        else:
            for ia in remaining:
                _,row=compute_row((ia,list(range(int(N_IR)))))
                T_matrix[ia]=row; completed_rows.add(ia); done_session+=1
                el=time.time()-t_start; rate=done_session/el
                left=len(remaining)-done_session; eta=left/rate if rate>0 else 0
                print(f"  Row {ia:>4}/{n_configs} | {el/60:.2f}min | ~{eta/60:.1f}min left")
                if done_session%5==0: save_prog()
        save_prog()
        print(f"Computation: {(time.time()-t_start)/60:.2f} min")

    T_sym=(T_matrix+T_matrix.T)/2
    evals,evecs=np.linalg.eigh(T_sym)
    idx=np.argsort(evals)[::-1]; evals=evals[idx]; evecs=evecs[:,idx]
    T_max=evals[0]

    print(f"\n{'='*60}")
    print(f"ICOSAHEDRAL BOUNDARY ANALYSIS")
    print(f"{'='*60}")
    print(f"T_max: {T_max:.6e}, Positive: {sum(1 for e in evals if e>0)}")

    print(f"\nMass spectrum:")
    known=[(0.511,'e'),(105.7,'μ'),(135,'π'),(332,'Λ'),(775,'ρ'),(938,'p'),(1777,'τ')]
    for i in range(min(20,len(evals))):
        if 0<evals[i]<T_max*0.9999:
            gap=-math.log(evals[i]/T_max); mass=gap*Lambda_QCD
            best=min(known,key=lambda h:abs(h[0]-mass))
            err=abs(best[0]-mass)/best[0]*100
            mark='★' if err<5 else '●' if err<10 else ''
            print(f"  {i:>3}: {mass:>8.1f} MeV → {best[1]:>3} ({best[0]:>7}) {err:5.1f}% {mark}")

    chi3_local=SECTOR.index(1) if 1 in SECTOR else -1
    if chi3_local>=0:
        print(f"\nLepton source (χ₃={chi3_local}):")
        for k in range(3):
            cfg=[0]*3; cfg[k]=chi3_local; cfg_idx=configs.index(tuple(cfg))
            ov=evecs[cfg_idx,:]
            C1=sum(ov[n]**2*(evals[n]/T_max) for n in range(len(evals)) if evals[n]>0)
            meff=-math.log(C1)*Lambda_QCD if C1>0 else 0
            print(f"  Edge {k}: m_eff={meff:.1f} MeV, C(1)={C1:.4e}")

    results={'sector':SECTOR,'n_configs':n_configs,'T_max':float(T_max),
             'eigenvalues':[float(e) for e in evals[:50]],
             'mass_gap':float(-math.log(evals[1]/T_max)) if 0<evals[1]<T_max else 0}
    with open('icosa_boundary_results.json','w') as f: json.dump(results,f,indent=2)
    print(f"\nSaved to icosa_boundary_results.json")
