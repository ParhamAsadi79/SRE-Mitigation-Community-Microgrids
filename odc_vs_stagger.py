import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# configuration 
RATE_KW  = 7.0           # Level-2 charger
DT_H     = 1.0           # hourly slots
H        = 24            # horizon: slot 0 = 12:00 ... slot 19 = 07:00 (deadline) ... slot 23 = 11:00
DEADLINE = 19            # 07:00 next morning
FLEET_CSV = "NHTS_Fleet_Data.csv"

def hour_to_slot(h):     # map clock hour (0-24) to horizon slot (12:00 origin)
    return int(round((h - 12) % 24))

# Representative community base load D(t): 50-home residential, SF/TMY3-like,
# evening peak ~19-21h, overnight trough, morning bump; scaled to ~43 kW peak. 
clock = [(s + 12) % 24 for s in range(H)]
shape = []
for h in clock:
    v = (0.45
         + 0.30*np.exp(-((h-19)**2)/(2*2.2**2))   # evening peak ~19-21h
         + 0.12*np.exp(-((h-8)**2)/(2*1.6**2))     # morning bump ~8h
         - 0.18*np.exp(-((h-3)**2)/(2*2.5**2)))    # overnight trough ~3h
    shape.append(max(v, 0.18))
shape = np.array(shape)
D = shape / shape.max() * 43.0                      # peak base load ~43 kW (CSV P_peak_building_kw)

# Real EV fleet from NHTS 
fleet = []
for r in csv.DictReader(open(FLEET_CSV)):
    arr = hour_to_slot(float(r["Arrival_Hour"]))
    E   = float(r["Energy_Used_kWh"])
    if arr >= DEADLINE:                             # arrived after deadline window: clamp into window
        arr = DEADLINE - 1
    window = list(range(arr, DEADLINE))             # slots in which EV may charge
    E = min(E, 0.95 * len(window) * RATE_KW * DT_H) # ensure feasible within window
    fleet.append({"arr": arr, "E": E, "win": window})
N = len(fleet)

# box+sum projection: project y onto {0<=r<=RATE on window, sum r*dt = E} 
def project(y, window, E):
    r = np.zeros(H)
    yv = y[window]
    lo, hi = -1e4, 1e4
    for _ in range(80):                             # bisection on the dual multiplier
        lam = 0.5 * (lo + hi)
        rv = np.clip(yv + lam, 0.0, RATE_KW)
        if rv.sum() * DT_H > E:
            hi = lam
        else:
            lo = lam
    r[window] = np.clip(yv + 0.5 * (lo + hi), 0.0, RATE_KW)
    return r

# ODC (Gan et al.)
def run_odc(gamma, tol=1e-3, max_iter=5000):
    """Returns (charging matrix, rounds-to-converge, per-round peak history)."""
    R = np.zeros((N, H))                            # init charging profiles = 0
    p_prev, hist = None, []
    for k in range(1, max_iter + 1):
        total = D + R.sum(axis=0)
        p = total.copy()                            # U(x)=x^2/2 -> U'(x)=x : marginal price (Eq. 7)
        hist.append(total.max())
        for n, ev in enumerate(fleet):              # broadcast p; each EV solves prox step (Eq. 8)
            R[n] = project(R[n] - gamma * p, ev["win"], ev["E"])
        if p_prev is not None and np.max(np.abs(p - p_prev)) < tol:
            return R, k, hist                       # converged; k = communication rounds
        p_prev = p
    return R, max_iter, hist

# van der Corput stagger (Alg. 1, comm-free)
def vdc(n, base=2):
    q, d = 0.0, 1.0
    while n > 0:
        d *= base; q += (n % base) / d; n //= base
    return q

def run_stagger():
    R = np.zeros((N, H))
    for n, ev in enumerate(fleet):
        win, need = ev["win"], ev["E"]
        slack = len(win) - int(np.ceil(need / (RATE_KW * DT_H)))   # spare slots in window
        t = win[0] + int(round(vdc(n + 1) * max(slack, 0)))        # deterministic low-discrepancy delay
        e = need
        while e > 1e-9 and t < DEADLINE:
            put = min(RATE_KW, e / DT_H); R[n, t] = put; e -= put * DT_H; t += 1
        if e > 1e-9:                                                 # backfill if offset overran deadline
            for t in win:
                if e <= 1e-9: break
                room = RATE_KW - R[n, t]
                if room > 0:
                    put = min(room, e / DT_H); R[n, t] += put; e -= put * DT_H
    return R

def run_uncoordinated():
    R = np.zeros((N, H))
    for n, ev in enumerate(fleet):
        e, t = ev["E"], ev["arr"]
        while e > 1e-9 and t < DEADLINE:
            put = min(RATE_KW, e / DT_H); R[n, t] = put; e -= put * DT_H; t += 1
    return R

# run 
gamma = 0.9 / N                                     # gamma < 1/(N*B), B = 1
R_odc, rounds, hist = run_odc(gamma)
R_stg = run_stagger()
R_unc = run_uncoordinated()

tot_unc = D + R_unc.sum(0)
tot_stg = D + R_stg.sum(0)
tot_odc = D + R_odc.sum(0)
peak_unc, peak_stg, peak_odc = tot_unc.max(), tot_stg.max(), tot_odc.max()
red_stg = (peak_unc - peak_stg) / peak_unc * 100
red_odc = (peak_unc - peak_odc) / peak_unc * 100
gap     = (peak_stg - peak_odc) / peak_odc * 100
capture = (peak_unc - peak_stg) / (peak_unc - peak_odc) * 100

# figure
C = {"unc": "#D55E00", "stg": "#0072B2", "odc": "#009E73", "base": "#999999"}  # Wong palette
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})
xt = np.arange(H)
xl = [f"{(s + 12) % 24:02d}" for s in range(H)]

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))

ax[0].plot(xt, tot_unc, color=C["unc"], lw=1.8, label=f"Uncoordinated (rebound), peak {peak_unc:.0f} kW")
ax[0].plot(xt, tot_stg, color=C["stg"], lw=1.8, label=f"vdC stagger (comm-free), peak {peak_stg:.0f} kW")
ax[0].plot(xt, tot_odc, color=C["odc"], lw=1.8, label=f"ODC (Gan et al.), peak {peak_odc:.0f} kW")
ax[0].fill_between(xt, 0, D, color=C["base"], alpha=0.25, label="Base (non-EV) load")
ax[0].set_xticks(xt[::3]); ax[0].set_xticklabels(xl[::3]); ax[0].set_xlabel("Hour of day")
ax[0].set_ylabel("Community load (kW)"); ax[0].set_ylim(0, None)
ax[0].legend(fontsize=6.3, loc="upper right", framealpha=0.9)
ax[0].set_title("(a) Total-load profiles", fontsize=9); ax[0].grid(alpha=0.25, lw=0.4)

ax[1].plot(range(1, len(hist) + 1), hist, color=C["odc"], lw=1.6, label="ODC peak per round")
ax[1].axhline(peak_stg, color=C["stg"], lw=1.6, ls="--", label="vdC stagger (0 rounds)")
ax[1].axhline(peak_odc, color=C["odc"], lw=0.8, ls=":", alpha=0.7)
ax[1].annotate(f"converged: {rounds} rounds", xy=(rounds, peak_odc),
               xytext=(rounds * 0.30, peak_odc + 22), fontsize=6.6,
               arrowprops=dict(arrowstyle="->", lw=0.7, color="k"))
ax[1].set_xlabel("Communication round $k$"); ax[1].set_ylabel("Community peak (kW)")
ax[1].legend(fontsize=6.6, loc="upper right", framealpha=0.9)
ax[1].set_title("(b) ODC convergence cost", fontsize=9); ax[1].grid(alpha=0.25, lw=0.4)
ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))

plt.tight_layout()
fig.savefig("F11_odc_comparison.pdf", bbox_inches="tight")
fig.savefig("F11_odc_comparison.png", dpi=300, bbox_inches="tight")
plt.close(fig)

np.savez("odc_results.npz", D=D, unc=tot_unc, stg=tot_stg, odc=tot_odc, hist=np.array(hist),
         rounds=rounds, peak_unc=peak_unc, peak_stg=peak_stg, peak_odc=peak_odc,
         red_stg=red_stg, red_odc=red_odc, gap=gap, capture=capture, N=N)

# manuscript reference
# Section 5.6 and Fig. 14 report these. They are asserted rather than trusted:
# this script and odc_resilience.py both compute the eta=1.00 point, and for four
# audit rounds the shipped odc_penetration_sweep.csv disagreed with Table 8
# because nothing compared the two paths.
SEC56 = {"peak_unc": 201.9, "peak_stg": 118.2, "peak_odc": 71.4, "rounds": 58,
         "red_stg": 41.5, "red_odc": 64.7, "gap": 65.6, "capture": 64.1}
_bad = []
for _k, _v in [("peak_unc", peak_unc), ("peak_stg", peak_stg), ("peak_odc", peak_odc),
               ("red_stg", red_stg), ("red_odc", red_odc), ("gap", gap), ("capture", capture)]:
    if abs(_v - SEC56[_k]) > 0.15:
        _bad.append(f"{_k}: {_v:.1f} vs manuscript {SEC56[_k]}")
if int(rounds) != SEC56["rounds"]:
    _bad.append(f"rounds: {rounds} vs manuscript {SEC56['rounds']}")
if _bad:
    raise SystemExit("\nodc_vs_stagger.py DISAGREES WITH SECTION 5.6 / Fig. 14:\n"
                     + "".join(f"    {b}\n" for b in _bad)
                     + "  Fix the code or the manuscript before shipping either.\n")
print("  [Sec 5.6 check] all 8 reported values match the manuscript.")

# report
print("=" * 60)
print(f"{'method':<26}{'peak kW':>10}{'reduction':>12}{'comm rounds':>12}")
print("-" * 60)
print(f"{'Uncoordinated (rebound)':<26}{peak_unc:>10.1f}{'--':>12}{0:>12}")
print(f"{'vdC stagger (comm-free)':<26}{peak_stg:>10.1f}{red_stg:>11.1f}%{0:>12}")
print(f"{'ODC (Gan et al.)':<26}{peak_odc:>10.1f}{red_odc:>11.1f}%{rounds:>12}")
print("=" * 60)
print(f"Stagger captures {capture:.1f}% of ODC's peak reduction at zero communication.")
print(f"Stagger peak is {gap:.1f}% above the ODC optimum (price of communication-freeness).")
print(f"ODC step size gamma = {gamma:.4f} < 1/N = {1/N:.4f}; converged in {rounds} rounds.")
print("Saved: F11_odc_comparison.pdf, F11_odc_comparison.png, odc_results.npz")
