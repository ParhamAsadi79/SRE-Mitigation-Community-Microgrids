#!/usr/bin/env python3
"""
deadline_audit.py
=============================================================================
Reconstruct the EV deadline-miss rate across ALL houses, scenarios, and
conditions, from the per-house telemetry (ev_soc_kwh).

MISS DEFINITION (matches OccupancyPlugin.py lines ~2126-2128)
    target_kwh = EV_DEADLINE_TARGET_FRAC * ev_capacity_kwh
    miss  <=>  SoC_at_departure < target_kwh - 1e-6
With the code default EV_DEADLINE_TARGET_FRAC = 1.0 and capacity 50 kWh, this
means: a departure is a MISS if the EV was NOT at a full 100% charge (< 50 kWh)
when it left. NOTE this is a strict "expected full at departure" criterion, not
"ran empty". Pass --target-frac / --capacity to match your code exactly.

Also reported (diagnostic only): "depleted" = post-departure SoC ~ 0, i.e. the
EV actually ran to empty -- a strict operational failure, a subset of misses.

DETECTION: the EV's SoC only steps down at the morning departure (trip energy
deducted), so a DEPARTURE is a downward SoC step in the morning window; the
SoC just before the step is SoC_at_departure.

CAVEATS (per honesty requirements)
  * The miss VALUE depends on --capacity matching your code's ev_capacity_kwh.
    The script warns if any house's peak SoC exceeds the assumed capacity
    (which would mean the capacity is wrong). Confirm it against the code.
  * This reads telemetry only; it is a RECONSTRUCTION of the code's counter,
    not the counter itself. It does not validate the EV physics.
  * Verify detection with --debug on a real EV house before trusting the rate.

Usage
  python deadline_audit.py --results DIR [--capacity 50.0] [--target-frac 1.0]
                           [--debug house_03]
=============================================================================
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

MIN_DROP_KWH   = 0.5    # a downward SoC step this large registers a departure
SOC_FLOOR_KWH  = 0.10   # post-departure SoC <= this counts as "depleted" (ran empty)
DEFAULT_CAPACITY_KWH = 50.0
DEFAULT_TARGET_FRAC  = 1.0


def detect_departures(soc, hour=None, dep_window=(5, 12)):
    """
    Return [(idx, pre, post, drop), ...] for downward SoC steps.
    If `hour` is given, only steps whose hour is in dep_window (the morning
    departure window) count -- excludes spurious drops (evening arrival-SoC
    reset, day-boundary discontinuity) that are not trips.
    """
    soc = np.asarray(soc, dtype=float)
    if len(soc) < 2:
        return []
    d = np.diff(soc)
    out = []
    for i in np.where(d < -MIN_DROP_KWH)[0]:
        if hour is not None and not (dep_window[0] <= int(hour[i + 1]) < dep_window[1]):
            continue
        pre, post = float(soc[i]), float(soc[i + 1])
        out.append((int(i + 1), pre, post, pre - post))
    return out


def audit_house(path, target_kwh, soc_floor):
    """Per-house stats, or None if no EV in this file."""
    try:
        df = pd.read_csv(path, usecols=lambda c: c in ("ev_soc_kwh", "hour"))
    except Exception:
        try:
            df = pd.read_csv(path)
        except Exception:
            return None
    if "ev_soc_kwh" not in df.columns:
        return None
    soc = df["ev_soc_kwh"].to_numpy(dtype=float)
    if np.allclose(soc, 0.0):
        return dict(has_ev=False)
    hr = df["hour"].to_numpy() if "hour" in df.columns else None
    deps = detect_departures(soc, hour=hr)
    peak_soc = float(soc.max())
    if not deps:
        return dict(has_ev=True, departures=0, below_target=0, empty_throttled=0,
                    empty_infeasible=0, min_soc_at_dep=np.nan, peak_soc=peak_soc)
    below_target = empty_throttled = empty_infeasible = 0
    soc_at_dep = []
    for _, pre, post, drop in deps:
        soc_at_dep.append(pre)
        is_miss = pre < target_kwh - 1e-6         # code's test: SoC_at_departure < target
        is_empty = post <= soc_floor              # ran to empty after the trip
        if is_miss:
            below_target += 1
        if is_empty:
            if is_miss:
                empty_throttled += 1              # below target AND empty -> subset of misses
            else:
                empty_infeasible += 1             # (near) full yet empty -> trip exceeds capacity
    return dict(has_ev=True, departures=len(deps), below_target=below_target,
                empty_throttled=empty_throttled, empty_infeasible=empty_infeasible,
                min_soc_at_dep=float(np.min(soc_at_dep)), peak_soc=peak_soc)


def parse_cond(path):
    m = re.search(r"Rho(\d+)_Pen(\d+)_Inel(\d+)[/\\]Scenario_(\d+)", path.replace("\\", "/"))
    if not m:
        return None
    rho, pen, inel, scen = (int(x) for x in m.groups())
    return dict(rho=rho, pen=pen, inel=inel, scen=scen)


def debug_one(results_dir, house_tag, target_kwh, soc_floor):
    pat = os.path.join(results_dir, "Rho*_Pen*_Inel*", "Scenario_*", house_tag, "house_telemetry.csv")
    files = glob.glob(pat)
    files = [f for f in files if not np.allclose(
        pd.read_csv(f, usecols=["ev_soc_kwh"])["ev_soc_kwh"].to_numpy(float), 0.0)] if files else []
    if not files:
        print(f"[debug] no EV-carrying telemetry found for '{house_tag}'.")
        return
    f = files[0]
    print(f"[debug] {f}")
    df = pd.read_csv(f)
    soc = df["ev_soc_kwh"].to_numpy(float)
    hr = df["hour"].to_numpy() if "hour" in df.columns else None
    deps = detect_departures(soc, hour=hr)
    typ = float(np.median([d[3] for d in deps])) if deps else float("nan")
    print(f"[debug] SoC range [{soc.min():.2f}, {soc.max():.2f}] kWh; peak SoC (capacity?) = {soc.max():.2f}; "
          f"target for miss = {target_kwh:.2f} kWh")
    print(f"[debug] {len(deps)} departures; typical drop (consumption) = {typ:.2f} kWh")
    print(f"[debug] first 6 departures (idx, pre_SoC, post_SoC, drop, hour):")
    for idx, pre, post, drop in deps[:6]:
        hh = int(df['hour'].iloc[idx]) if 'hour' in df else -1
        tag = "MISS(<target)" if pre < target_kwh - 1e-6 else "ok(full)"
        tag += " +EMPTY" if post <= soc_floor else ""
        print(f"          idx={idx:6d}  pre={pre:6.2f}  post={post:6.2f}  drop={drop:6.2f}  hour={hh:2d}  {tag}")
    print("[debug] CONFIRM: departures once/day near hour 7 or 9; a departure is a MISS if pre_SoC < target.")


def main():
    ap = argparse.ArgumentParser(description="EV deadline-miss audit across all houses/scenarios.")
    ap.add_argument("--results", required=True, help="Results_Sensitivity dir")
    ap.add_argument("--capacity", type=float, default=DEFAULT_CAPACITY_KWH,
                    help="EV usable capacity kWh -- MUST match ev_capacity_kwh in the code (default 50)")
    ap.add_argument("--target-frac", type=float, default=DEFAULT_TARGET_FRAC,
                    help="EV_DEADLINE_TARGET_FRAC from the code (default 1.0 = full)")
    ap.add_argument("--soc-floor", type=float, default=SOC_FLOOR_KWH,
                    help="post-departure SoC (kWh) at/below which the EV is 'depleted' (default 0.10)")
    ap.add_argument("--debug", metavar="HOUSE", default=None,
                    help="print SoC trace + departures for one house dir and exit")
    args = ap.parse_args()
    target_kwh = args.target_frac * args.capacity

    if args.debug:
        debug_one(args.results, args.debug, target_kwh, args.soc_floor)
        return

    files = glob.glob(os.path.join(args.results, "Rho*_Pen*_Inel*", "Scenario_*",
                                   "*", "house_telemetry.csv"))
    if not files:
        print("No telemetry files found under --results.")
        return

    keys = ('departures', 'below_target', 'empty_throttled', 'empty_infeasible')
    tot = dict(files=0, ev_files=0, **{k: 0 for k in keys})
    by_scen = defaultdict(lambda: dict(ev_files=0, **{k: 0 for k in keys}))
    by_pen = defaultdict(lambda: dict(ev_files=0, **{k: 0 for k in keys}))
    min_soc_seen = np.inf
    max_peak_soc = 0.0

    print("=" * 78)
    print("EV DEADLINE-MISS AUDIT  (reconstruction of the code's counter; see header)")
    print(f"MISS = SoC_at_departure < target = {args.target_frac} x {args.capacity} = {target_kwh:.2f} kWh")
    print("=" * 78)
    for f in files:
        tot["files"] += 1
        cond = parse_cond(f)
        r = audit_house(f, target_kwh, args.soc_floor)
        if r is None or not r.get("has_ev"):
            continue
        tot["ev_files"] += 1
        max_peak_soc = max(max_peak_soc, r.get("peak_soc", 0.0))
        for k in keys:
            tot[k] += r[k]
            if cond:
                by_scen[cond["scen"]][k] += r[k]
                by_pen[cond["pen"]][k] += r[k]
        if cond:
            by_scen[cond["scen"]]["ev_files"] += 1
            by_pen[cond["pen"]]["ev_files"] += 1
        if not np.isnan(r.get("min_soc_at_dep", np.nan)):
            min_soc_seen = min(min_soc_seen, r["min_soc_at_dep"])

    print(f"\nscanned {tot['files']} telemetry files; {tot['ev_files']} carried an EV.")
    if max_peak_soc > args.capacity + 1e-3:
        print(f"  ** WARNING: peak SoC seen ({max_peak_soc:.2f} kWh) EXCEEDS assumed capacity "
              f"({args.capacity} kWh). Your --capacity is likely wrong; misses under-counted.")
    if tot["departures"] == 0:
        print("No departures detected -- run --debug on a known EV house.")
        return
    dep = tot["departures"]
    pc = lambda x: f"{x:7d}  = {x/dep*100:.3f}%"
    print(f"total departure events (EV house-days): {dep}")
    print(f"  MISSES (SoC < target)          : {pc(tot['below_target'])}   <-- reproduces the code's counter")
    print(f"     of those, ran empty         : {pc(tot['empty_throttled'])}   (throttled below trip need; subset of misses)")
    print(f"  CAPACITY-INFEASIBLE (separate) : {pc(tot['empty_infeasible'])}   (departed full yet ran empty; trip > {args.capacity:.0f} kWh)")
    print(f"     -> NOT a control failure and NOT counted as a miss: the EV's daily trip physically")
    print(f"        exceeds its battery, so it runs empty in EVERY scenario incl. S0. This is the")
    print(f"        source of the earlier 'empty > misses' anomaly, now separated out.")
    print(f"  lowest SoC-at-departure seen anywhere: {min_soc_seen:.2f} kWh")

    print("\nby scenario (miss rate = SoC<target; S1 is Algorithm-1):")
    for s in sorted(by_scen):
        b = by_scen[s]; d = b["departures"] or 1
        print(f"  S{s}: {b['below_target']:7d}/{b['departures']:7d} = {b['below_target']/d*100:7.3f}%  "
              f"({b['ev_files']} EV files)")

    print("\nby EV penetration (miss rate = SoC<target):")
    for p in sorted(by_pen):
        b = by_pen[p]; d = b["departures"] or 1
        print(f"  {p:3d}%: {b['below_target']:7d}/{b['departures']:7d} = {b['below_target']/d*100:7.3f}%")

    print("\n" + "=" * 78)
    print("MISSES reproduces OccupancyPlugin.py's test (SoC_at_departure < target). 'ran empty' is")
    print("split into throttled (a subset of misses) and capacity-infeasible (disjoint: trip > battery,")
    print("a data/modelling issue -- NOTE at least one NHTS house has a daily trip above the EV capacity).")
    print("If MISSES disagrees with the paper's figure, the paper likely cites a different quantity")
    print("(e.g. an override rate). This is a reconstruction from telemetry, not the in-code counter.")


if __name__ == "__main__":
    main()
