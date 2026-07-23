import numpy as np, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RATE_KW  = 7.0
DT_H     = 1.0
H        = 24
DEADLINE = 19
FLEET_CSV = "NHTS_Fleet_Data.csv"

def hour_to_slot(h):
    return int(round((h - 12) % 24))

# common representative community base load
clock = [(s + 12) % 24 for s in range(H)]
shape = []
for h in clock:
    v = (0.45
         + 0.30*np.exp(-((h-19)**2)/(2*2.2**2))
         + 0.12*np.exp(-((h-8)**2)/(2*1.6**2))
         - 0.18*np.exp(-((h-3)**2)/(2*2.5**2)))
    shape.append(max(v, 0.18))
D = np.array(shape) / max(shape) * 43.0

# full real NHTS fleet 
allfleet = []
for r in csv.DictReader(open(FLEET_CSV)):
    arr = hour_to_slot(float(r["Arrival_Hour"]))
    E   = float(r["Energy_Used_kWh"])
    if arr >= DEADLINE:
        arr = DEADLINE - 1
    win = list(range(arr, DEADLINE))
    E = min(E, 0.95 * len(win) * RATE_KW * DT_H)
    allfleet.append({"arr": arr, "E": E, "win": win})

def project(y, window, E):
    r = np.zeros(H); yv = y[window]; lo, hi = -1e4, 1e4
    for _ in range(80):
        lam = 0.5 * (lo + hi); rv = np.clip(yv + lam, 0.0, RATE_KW)
        if rv.sum() * DT_H > E: hi = lam
        else: lo = lam
    r[window] = np.clip(yv + 0.5 * (lo + hi), 0.0, RATE_KW); return r

def run_odc(fleet, gamma, tol=1e-3, max_iter=8000):
    Nl = len(fleet); R = np.zeros((Nl, H)); p_prev = None
    for k in range(1, max_iter + 1):
        p = (D + R.sum(0)).copy()
        for n, ev in enumerate(fleet):
            R[n] = project(R[n] - gamma * p, ev["win"], ev["E"])
        if p_prev is not None and np.max(np.abs(p - p_prev)) < tol:
            return R, k
        p_prev = p
    return R, max_iter

def vdc(n, base=2):
    q, d = 0.0, 1.0
    while n > 0:
        d *= base; q += (n % base) / d; n //= base
    return q

def run_stagger(fleet):
    Nl = len(fleet); R = np.zeros((Nl, H))
    for n, ev in enumerate(fleet):
        win, need = ev["win"], ev["E"]
        slack = len(win) - int(np.ceil(need / (RATE_KW * DT_H)))
        t = win[0] + int(round(vdc(n + 1) * max(slack, 0))); e = need
        while e > 1e-9 and t < DEADLINE:
            put = min(RATE_KW, e / DT_H); R[n, t] = put; e -= put * DT_H; t += 1
        if e > 1e-9:
            for t in win:
                if e <= 1e-9: break
                room = RATE_KW - R[n, t]
                if room > 0:
                    put = min(room, e / DT_H); R[n, t] += put; e -= put * DT_H
    return R

def run_unc(fleet):
    Nl = len(fleet); R = np.zeros((Nl, H))
    for n, ev in enumerate(fleet):
        e, t = ev["E"], ev["arr"]
        while e > 1e-9 and t < DEADLINE:
            put = min(RATE_KW, e / DT_H); R[n, t] = put; e -= put * DT_H; t += 1
    return R


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
    # Recovery fraction = stagger reduction / ODC reduction
    rec_full = [100 * r[3] / r[5] for r in rows]
    rec_tab  = [100 * round(r[3], 1) / round(r[5], 1) for r in rows]
    print(f"  [Table 8 check] {source}: all {len(rows)} rows match the manuscript.")
    print(f"  [Table 8 check] recovery from Table 8 as printed: "
          f"{', '.join(f'{x:.1f}%' for x in rec_tab)}  (mean {sum(rec_tab)/len(rec_tab):.1f}%)")
    print(f"  [Table 8 check] recovery at full precision      : "
          f"{', '.join(f'{x:.1f}%' for x in rec_full)}  (mean {sum(rec_full)/len(rec_full):.1f}%)")
    print(f"  [Table 8 check] manuscript states: mean 59.8%, range 54.3%-66.4%, "
          f"64.1% at eta=1.00  -> matches the rounded-table path.")


# sweep 
levels = [(0.25, 13), (0.50, 25), (0.75, 38), (1.00, 50)]
rows = []
print(f"{'eta':>5}{'EVs':>5}{'unc kW':>9}{'stg kW':>9}{'odc kW':>9}"
      f"{'stg red%':>10}{'odc red%':>10}{'gap%':>8}{'rounds':>8}")
print("-" * 73)
for eta, nev in levels:
    fleet = allfleet[:nev]; gamma = 0.9 / nev
    pu = (D + run_unc(fleet).sum(0)).max()
    ps = (D + run_stagger(fleet).sum(0)).max()
    Ro, rd = run_odc(fleet, gamma); po = (D + Ro.sum(0)).max()
    rs = (pu - ps) / pu * 100; ro = (pu - po) / pu * 100; gp = (ps - po) / po * 100
    print(f"{eta:>5.2f}{nev:>5}{pu:>9.1f}{ps:>9.1f}{po:>9.1f}{rs:>10.1f}{ro:>10.1f}{gp:>8.1f}{rd:>8}")
    rows.append((eta, nev, pu, ps, po, rs, ro, gp, rd))
check_against_table8([(r[0], r[1], r[3], r[5], r[4], r[6], r[8]) for r in rows],
                     "odc_penetration_sweep.py")
np.save("odc_sweep_rows.npy", np.array(rows))

# This path now writes the CSV too. Previously only odc_resilience.py did, so
# running this script left the CSV untouched and silently stale.
with open("odc_penetration_sweep.csv", "w", newline="") as _f:
    _w = csv.writer(_f)
    _w.writerow(["eta", "n_vehicles", "stagger_peak_kw", "stagger_reduction_pct",
                 "odc_peak_kw", "odc_reduction_pct", "odc_rounds"])
    for r in rows:
        _w.writerow([f"{r[0]:.2f}", r[1], f"{r[3]:.1f}", f"{r[5]:.1f}",
                     f"{r[4]:.1f}", f"{r[6]:.1f}", r[8]])
print("wrote odc_penetration_sweep.csv")

# figure 
etas   = [r[0] for r in rows]
stg_rd = [r[5] for r in rows]
odc_rd = [r[6] for r in rows]
rounds = [r[8] for r in rows]
C = {"stg": "#0072B2", "odc": "#009E73"}
plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.linewidth": 0.8})

fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
x = np.arange(len(etas)); w = 0.38

ax[0].bar(x - w/2, stg_rd, w, color=C["stg"], label="vdC stagger (0 comms)")
ax[0].bar(x + w/2, odc_rd, w, color=C["odc"], label="ODC (Gan et al.)")
for xi, v in zip(x - w/2, stg_rd): ax[0].text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
for xi, v in zip(x + w/2, odc_rd): ax[0].text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=6.5)
ax[0].set_xticks(x); ax[0].set_xticklabels([f"{e:.2f}" for e in etas])
ax[0].set_xlabel(r"EV penetration $\eta$"); ax[0].set_ylabel("Peak reduction vs rebound (%)")
ax[0].set_ylim(0, max(odc_rd) + 12); ax[0].legend(fontsize=6.6, loc="upper left", framealpha=0.9)
ax[0].set_title("(a) Peak reduction across penetration", fontsize=9); ax[0].grid(axis="y", alpha=0.25, lw=0.4)

ax[1].bar(x, rounds, 0.5, color=C["odc"], label="ODC rounds to converge")
ax[1].axhline(0, color=C["stg"], lw=1.8, ls="--", label="vdC stagger (0 rounds)")
for xi, v in zip(x, rounds): ax[1].text(xi, v + 1.5, f"{v}", ha="center", fontsize=6.8)
ax[1].set_xticks(x); ax[1].set_xticklabels([f"{e:.2f}" for e in etas])
ax[1].set_xlabel(r"EV penetration $\eta$"); ax[1].set_ylabel("Bidirectional comm. rounds")
ax[1].set_ylim(0, max(rounds) + 16); ax[1].legend(fontsize=6.6, loc="upper right", framealpha=0.9)
ax[1].set_title("(b) Communication cost across penetration", fontsize=9); ax[1].grid(axis="y", alpha=0.25, lw=0.4)

plt.tight_layout()
fig.savefig("F12_odc_penetration_sweep.pdf", bbox_inches="tight")
fig.savefig("F12_odc_penetration_sweep.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nKey: gap = stagger peak above ODC optimum; stagger uses 0 rounds at every eta.")
print("Saved: F12_odc_penetration_sweep.pdf, F12_odc_penetration_sweep.png, odc_sweep_rows.npy")
