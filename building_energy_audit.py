import re
import sys
from pathlib import Path

import pandas as pd

MANUSCRIPT = {"mean per house (kWh)": 4123, "archetype means (kWh)": "3,450-4,693",
              "individual houses (kWh)": "3,146-5,747", "fleet-mean spread across scenarios (kWh)": 1.7,
              "median per-house spread across scenarios (kWh)": 74}


def main(root: str) -> None:
    base = Path(root)
    rows = []
    files = sorted(base.glob("Rho*_Pen*_Inel*/Scenario_*/**/house_telemetry.csv"))
    print(f"found {len(files)} house_telemetry.csv files under {base}")
    for k, f in enumerate(files, 1):
        cond = re.search(r"Rho(\d+)_Pen(\d+)_Inel(\d+)", str(f))
        scen = re.search(r"Scenario_(\d+)", str(f))
        house = re.findall(r"house[_-]?(\d+)", str(f.relative_to(base)), flags=re.I)
        try:
            df = pd.read_csv(f, usecols=lambda c: c.strip().lower() == "building_w")
        except ValueError:
            continue
        w = df.iloc[:, 0].astype(float)
        step_h = 8760.0 / len(w)                      # full-year series; 10-min steps give 1/6 h
        rows.append({"rho": int(cond.group(1)), "pen": int(cond.group(2)), "inel": int(cond.group(3)),
                     "scenario": int(scen.group(1)), "house": int(house[-1]) if house else -1,
                     "building_kwh": w.sum() * step_h / 1000.0})
        if k % 500 == 0:
            print(f"  ...{k}/{len(files)}")
    d = pd.DataFrame(rows)
    d["archetype"] = d["house"] % 5
    d.to_csv("building_energy_per_house.csv", index=False)
    main5 = d[d.scenario <= 4]
    by_s = d.groupby("scenario").building_kwh.mean()
    arch = main5[main5.house >= 0].groupby("archetype").building_kwh.mean()
    key = ["rho", "pen", "inel", "house"]
    spread = main5.groupby(key).building_kwh.agg(lambda x: x.max() - x.min())
    s5 = by_s.drop(5, errors="ignore")
    print("\nfleet mean per house by scenario (kWh):", {int(s): round(v, 1) for s, v in by_s.items()})
    stats = {"mean per house (kWh)": round(main5.building_kwh.mean(), 0),
             "archetype means (kWh)": f"{arch.min():,.0f}-{arch.max():,.0f}  ({', '.join(f'A{a}={v:,.0f}' for a, v in arch.items())})",
             "individual houses (kWh)": f"{main5.building_kwh.min():,.0f}-{main5.building_kwh.max():,.0f}",
             "fleet-mean spread across scenarios (kWh)": round(s5.max() - s5.min(), 1),
             "median per-house spread across scenarios (kWh)": round(spread.median(), 0)}
    print(f"\n{'statistic (S0-S4, ' + str(len(main5)) + ' house-runs)':52s} {'data':>28s}   manuscript")
    for k in stats:
        print(f"  {k:50s} {str(stats[k]):>28s}   {MANUSCRIPT[k]}")
    print("\nwrote building_energy_per_house.csv")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")