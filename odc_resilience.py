import numpy as np, csv, os
RATE_KW, DT_H, H, DEADLINE = 7.0, 1.0, 24, 19
def hour_to_slot(h): return int(round((h - 12) % 24))
clock=[(s+12)%24 for s in range(H)]; shape=[]
for h in clock:
    v=(0.45+0.30*np.exp(-((h-19)**2)/(2*2.2**2))+0.12*np.exp(-((h-8)**2)/(2*1.6**2))-0.18*np.exp(-((h-3)**2)/(2*2.5**2)))
    shape.append(max(v,0.18))
D=np.array(shape)/max(shape)*43.0; PEAK_D=D.max()

def load_fleet():
    if os.path.exists("NHTS_Fleet_Data.csv"):
        fl=[]
        for r in csv.DictReader(open("NHTS_Fleet_Data.csv")):
            arr=min(hour_to_slot(float(r["Arrival_Hour"])),DEADLINE-1); win=list(range(arr,DEADLINE))
            fl.append({"arr":arr,"E":min(float(r["Energy_Used_kWh"]),0.95*len(win)*RATE_KW*DT_H),"win":win})
        return fl,"original NHTS_Fleet_Data.csv"

    if os.environ.get("ODC_ALLOW_SYNTHETIC_FLEET") != "1":
        raise SystemExit(
            f"\nNHTS_Fleet_Data.csv not found in {os.getcwd()}.\n"
            "  Table 8, Fig. 14 and Fig. 15 are computed on the REAL 50-vehicle NHTS\n"
            "  fleet. A synthetic reconstruction does not reproduce them, and writing\n"
            "  its output to odc_penetration_sweep.csv would put numbers in the\n"
            "  repository that the manuscript contradicts.\n"
            "  Supply the fleet CSV, or set ODC_ALLOW_SYNTHETIC_FLEET=1 to explore\n"
            "  with a synthetic fleet (the Table 8 check will then fail, by design).\n")
    print("  WARNING: ODC_ALLOW_SYNTHETIC_FLEET=1 -- using a SYNTHETIC fleet (seed 6). "
          "These numbers do NOT reproduce the manuscript. Do not ship the CSV.")
    rng=np.random.default_rng(6); vmt=rng.lognormal(np.log(37.5),0.55,50); E=np.clip(vmt/3.5,2.0,45.0)
    arrh=np.clip(rng.normal(18.0,1.6,50),13.0,21.5); fl=[]
    for i in range(50):
        arr=min(hour_to_slot(arrh[i]),DEADLINE-1); win=list(range(arr,DEADLINE))
        fl.append({"arr":arr,"E":min(E[i],0.95*len(win)*RATE_KW*DT_H),"win":win})
    return fl,"SYNTHETIC reconstruction (seed 6) -- NOT the manuscript fleet"

def masks(fleet):
    N=len(fleet); M=np.zeros((N,H),bool); E=np.zeros(N); arr=np.zeros(N,int)
    for n,ev in enumerate(fleet):
        M[n,ev["win"]]=True; E[n]=ev["E"]; arr[n]=ev["arr"]
    return M,E,arr

def project_batch(Y,M,E,n_iter=45):
    if Y.shape[0]==0: return Y
    lo=np.full(Y.shape[0],-1e4); hi=np.full(Y.shape[0],1e4)
    for _ in range(n_iter):
        lam=0.5*(lo+hi); s=(np.clip(Y+lam[:,None],0,RATE_KW)*M).sum(1)*DT_H
        over=s>E; hi=np.where(over,lam,hi); lo=np.where(over,lo,lam)
    return np.clip(Y+0.5*(lo+hi)[:,None],0,RATE_KW)*M

def run_odc(M,E,base,gamma=None,tol=1e-3,max_iter=1500,return_rounds=False):
    N=M.shape[0]
    if N==0: return (np.zeros((0,H)),0) if return_rounds else np.zeros((0,H))
    gamma=0.9/N if gamma is None else gamma; R=np.zeros((N,H)); p_prev=None; rounds=0
    for k in range(max_iter):
        rounds=k+1; p=base+R.sum(0); R=project_batch(R-gamma*p[None,:],M,E)
        if p_prev is not None and np.max(np.abs(p-p_prev))<tol: break
        p_prev=p
    return (R,rounds) if return_rounds else R
def vdc(n,b=2):
    q,d=0.0,1.0
    while n>0: d*=b; q+=(n%b)/d; n//=b
    return q
def stagger_load(fleet):
    R=np.zeros((len(fleet),H))
    for n,ev in enumerate(fleet):
        win,need=ev["win"],ev["E"]; slack=len(win)-int(np.ceil(need/(RATE_KW*DT_H)))
        t=win[0]+int(round(vdc(n+1)*max(slack,0))); e=need
        while e>1e-9 and t<DEADLINE:
            put=min(RATE_KW,e/DT_H); R[n,t]=put; e-=put*DT_H; t+=1
        if e>1e-9:
            for t in win:
                if e<=1e-9: break
                room=RATE_KW-R[n,t]
                if room>0: put=min(room,e/DT_H); R[n,t]+=put; e-=put*DT_H
    return R.sum(0)
def unc_load_idx(M,E,arr,idx):
    R=np.zeros(H)
    for n in idx:
        e,t=E[n],arr[n]
        while e>1e-9 and t<DEADLINE:
            put=min(RATE_KW,e/DT_H); R[t]+=put; e-=put*DT_H; t+=1
    return R

fleet,src=load_fleet(); N=len(fleet); M,E,arr=masks(fleet)
peak_unc=(D+unc_load_idx(M,E,arr,range(N))).max()
peak_stg=(D+stagger_load(fleet)).max()
peak_odc0=(D+run_odc(M,E,D).sum(0)).max()
print(f"fleet: {src} (N={N})")
print(f"clean: unc={peak_unc:.1f}  stagger={peak_stg:.1f}  ODC={peak_odc0:.1f} kW")


# PUBLISHED REFERENCE (manuscript Table 8) 

TABLE8 = {            # eta: (stagger_kW, stagger_red%, odc_kW, odc_red%, rounds)
    0.25: (54.8, 29.8, 43.0, 44.9, 55),
    0.50: (74.1, 30.3, 47.2, 55.6, 67),
    0.75: (102.8, 33.3, 59.7, 61.3, 90),
    1.00: (118.2, 41.5, 71.4, 64.7, 58),
}
TOL_KW, TOL_PCT = 0.15, 0.15   # a 0.1 kW / 0.1 pp print tolerance


def check_against_table8(rows, source):
    """rows: list of (eta, n, stg_kW, stg_red, odc_kW, odc_red, rounds)."""
    bad = []
    for eta, _n, ps, rs, po, ro, rd in rows:
        ref = TABLE8.get(round(eta, 2))
        if ref is None:
            continue
        if abs(ps - ref[0]) > TOL_KW:  bad.append(f"eta={eta}: stagger {ps:.1f} vs Table 8 {ref[0]}")
        if abs(rs - ref[1]) > TOL_PCT: bad.append(f"eta={eta}: stagger red {rs:.1f}% vs {ref[1]}%")
        if abs(po - ref[2]) > TOL_KW:  bad.append(f"eta={eta}: ODC {po:.1f} vs Table 8 {ref[2]}")
        if abs(ro - ref[3]) > TOL_PCT: bad.append(f"eta={eta}: ODC red {ro:.1f}% vs {ref[3]}%")
        if int(rd) != int(ref[4]):     bad.append(f"eta={eta}: rounds {rd} vs Table 8 {ref[4]}")
    if bad:
        raise SystemExit(
            f"\n{source} DISAGREES WITH MANUSCRIPT TABLE 8:\n"
            + "".join(f"    {b}\n" for b in bad)
            + "  Either the manuscript table or this code is wrong. Do not ship the\n"
              "  CSV until they agree; a referee recomputing your recovery fraction\n"
              "  from the CSV gets a different number from the one you report.\n")

    rec_full = [100 * r[3] / r[5] for r in rows]
    rec_tab  = [100 * round(r[3], 1) / round(r[5], 1) for r in rows]
    print(f"  [Table 8 check] {source}: all {len(rows)} rows match the manuscript.")
    print(f"  [Table 8 check] recovery from Table 8 as printed: "
          f"{', '.join(f'{x:.1f}%' for x in rec_tab)}  (mean {sum(rec_tab)/len(rec_tab):.1f}%)")
    print(f"  [Table 8 check] recovery at full precision      : "
          f"{', '.join(f'{x:.1f}%' for x in rec_full)}  (mean {sum(rec_full)/len(rec_full):.1f}%)")
    print(f"  [Table 8 check] manuscript states: mean 59.8%, range 54.3%-66.4%, "
          f"64.1% at eta=1.00  -> matches the rounded-table path.")


# EV-penetration sweep: reproduces the manuscript ODC-vs-stagger table
# Active fleet = first int(eta*N + 0.5) vehicles (13/25/38/50 at eta=0.25..1.00)
print("\n-- EV-penetration sweep (ODC vs communication-free stagger) --")
print(f"{'eta':>5} {'Nact':>5} | {'stg kW':>7} {'red%':>6} | {'ODC kW':>7} {'red%':>6} {'rounds':>7}")
pen_rows=[]
for _eta in (0.25,0.50,0.75,1.00):
    _n=int(_eta*N+0.5); _idx=list(range(_n)); _sub=[fleet[i] for i in _idx]
    _pu=(D+unc_load_idx(M,E,arr,_idx)).max()
    _ps=(D+stagger_load(_sub)).max()
    _Ro,_rr=run_odc(M[_idx],E[_idx],D,return_rounds=True); _po=(D+_Ro.sum(0)).max()
    _rs=(_pu-_ps)/_pu*100.0; _ro=(_pu-_po)/_pu*100.0
    pen_rows.append((_eta,_n,_ps,_rs,_po,_ro,_rr))
    print(f"{_eta:>5.2f} {_n:>5} | {_ps:>7.1f} {_rs:>6.1f} | {_po:>7.1f} {_ro:>6.1f} {_rr:>7}")
check_against_table8(pen_rows, "odc_resilience.py")
with open("odc_penetration_sweep.csv","w",newline="") as _f:
    _w=csv.writer(_f); _w.writerow(["eta","n_vehicles","stagger_peak_kw","stagger_reduction_pct",
                                    "odc_peak_kw","odc_reduction_pct","odc_rounds"])
    for _row in pen_rows: _w.writerow([f"{_row[0]:.2f}",_row[1],f"{_row[2]:.1f}",f"{_row[3]:.1f}",
                                       f"{_row[4]:.1f}",f"{_row[5]:.1f}",_row[6]])
print("wrote odc_penetration_sweep.csv")

def smooth_noise(rng):
    w=rng.standard_normal(H+6); k=np.exp(-0.5*(np.arange(-3,4)/1.5)**2); k/=k.sum()
    e=np.convolve(w,k,'same')[3:-3]; return e/(e.std()+1e-9)

SEEDS=12
fracs=np.round(np.arange(0,0.62,0.06),3); drop=[]
for f in fracs:
    v=[]
    for s in range(SEEDS):
        rng=np.random.default_rng(1000+s); nd=int(round(f*N)); ix=rng.permutation(N)
        dl=unc_load_idx(M,E,arr,ix[:nd]); ci=ix[nd:]
        Rc=run_odc(M[ci],E[ci],D+dl); peak=(D+dl+(Rc.sum(0) if len(ci) else 0)).max(); v.append(peak)
    v=np.array(v); drop.append((f,v.mean(),np.percentile(v,2.5),np.percentile(v,97.5)))
errs=np.round(np.arange(0,0.40,0.04),3); fc=[]
for eps in errs:
    v=[]
    for s in range(SEEDS):
        rng=np.random.default_rng(2000+s); Dfc=np.clip(D+eps*PEAK_D*smooth_noise(rng),0,None)
        R=run_odc(M,E,Dfc); v.append((D+R.sum(0)).max())
    v=np.array(v); fc.append((eps,v.mean(),np.percentile(v,2.5),np.percentile(v,97.5)))

def crossover(rows,thr):
    x=[r[0] for r in rows]; y=[r[1] for r in rows]
    for i in range(1,len(y)):
        if (y[i-1]-thr)*(y[i]-thr)<=0 and y[i]!=y[i-1]:
            return x[i-1]+(thr-y[i-1])*(x[i]-x[i-1])/(y[i]-y[i-1])
    return None
fstar=crossover(drop,peak_stg); estar=crossover(fc,peak_stg)
print("\n-- dropout sweep (ODC realized peak) --")
for f,m,lo,hi in drop: print(f"  f={f:.2f}: {m:6.1f} kW [{lo:.0f},{hi:.0f}]")
print(f"  CROSSOVER f* = {fstar*100:.1f}% unreachable" if fstar else "  no crossover")
print("-- forecast-error sweep (ODC realized peak) --")
for e,m,lo,hi in fc: print(f"  RMSE={e*100:4.1f}%: {m:6.1f} kW [{lo:.0f},{hi:.0f}]")
print(f"  CROSSOVER RMSE* = {estar*100:.1f}% of peak" if estar else "  no crossover")
np.savez("odc_resilience.npz",
  fracs=np.array([r[0] for r in drop]),drop_mean=np.array([r[1] for r in drop]),
  drop_lo=np.array([r[2] for r in drop]),drop_hi=np.array([r[3] for r in drop]),
  errs=np.array([r[0] for r in fc]),fc_mean=np.array([r[1] for r in fc]),
  fc_lo=np.array([r[2] for r in fc]),fc_hi=np.array([r[3] for r in fc]),
  peak_unc=peak_unc,peak_stg=peak_stg,peak_odc0=peak_odc0,fstar=fstar or np.nan,estar=estar or np.nan)
print("saved odc_resilience.npz")
