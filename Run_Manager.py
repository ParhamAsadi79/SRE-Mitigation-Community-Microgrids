from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import norm as _scipy_norm  # BCa bootstrap (B-02 fix)
    _SCIPY_AVAILABLE = True
except ImportError:
    _scipy_norm = None
    _SCIPY_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:
    print("Please install tqdm:  pip install tqdm")
    sys.exit(1)

try:
    from SALib.analyze import sobol as sobol_analyze
    
    try:
        from SALib.sample.sobol import sample as _sobol_sample
    except ImportError:
        from SALib.sample import saltelli as _saltelli_mod
        _sobol_sample = _saltelli_mod.sample
    _SALIB_AVAILABLE = True
except ImportError:
    _SALIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _PLOTLY_AVAILABLE = True
except ImportError:
    _PLOTLY_AVAILABLE = False


# 1.  GLOBAL CONSTANTS

GLOBAL_RANDOM_SEED: int = 42

RATE_OFF_PEAK: float = 0.39940  # real E-TOU-C summer off-peak ($/kWh), LP representative
RATE_ON_PEAK:  float = 0.52240  # real E-TOU-C summer peak 16:00-21:00 ($/kWh)

EXPORT_RATE:   float = 0.038   # NEM 3.0 net billing avoided-cost

# NEM 3.0 export tiering
NEM3_ANNUAL_EXPORT_CAP_KWH: float = 3_000.0   # Full credit up to this
NEM3_AVOIDED_RATE:          float = 0.022       # Rate above the cap

# CAISO grid average emissions intensity (gCO2/kWh) - 2023 annual hourly avg
CO2_BY_HOUR: Dict[int, float] = {
     0: 296,  1: 287,  2: 280,  3: 276,  4: 276,  5: 289,
     6: 303,  7: 282,  8: 235,  9: 179, 10: 143, 11: 125,
    12: 116, 13: 113, 14: 120, 15: 150, 16: 205, 17: 277,
    18: 339, 19: 376, 20: 362, 21: 341, 22: 323, 23: 307,
}

BOOTSTRAP_RESAMPLES: int   = 10_000
BOOTSTRAP_CI_ALPHA:  float = 0.05
RAMP_WINDOW_HOURS:   Tuple[int, int] = (20, 23)

DEMAND_CHARGE_RATE_USD_PER_KW_MONTH: float = 37.37   # $/kW/month, PG&E B-19 2024

# EnergyPlus day-of-yearcalendar month 
_MONTH_END_DOY: List[int] = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

# Building archetypes 
ARCHETYPES: List[Dict[str, Any]] = [
    {   # A0 - modern south-facing (best solar, tightest construction)
        "label":       "A0_Modern_South",
        "description": "Post-2010, south-facing, 2,050 sqft equiv. (Title 24 2022)",
        "north_axis":  0,
        "pv_frac":     0.90,
        "in02_thick":  0.1397,   # R-20 wall batt
        "in46_thick":  0.1524,   # R-38 attic
        "in05_thick":  0.2876,   # attic IN05 (cond 0.043) matched to same R as in46
        "infil_scale": 0.70,     # tight envelope
        "load_scale":  1.00,     # baseline (850 W lights, 850 W equip)
    },
    {   # A1 - pre-code east-facing (worst envelope, east solar)
        "label":       "A1_PreCode_East",
        "description": "Pre-1978, east-facing, 1,750 sqft equiv. (pre-code construction)",
        "north_axis":  90,
        "pv_frac":     0.55,     # reduced - partial roof shading typical of 1970s ranch
        "in02_thick":  0.0508,   # R-11 wall batt (minimum 1975 MEC)
        "in46_thick":  0.0508,   # R-11 attic (pre-code)
        "in05_thick":  0.0832,   # attic IN05 (cond 0.043) matched to same R as in46
        "infil_scale": 1.50,     # leaky old construction
        "load_scale":  0.85,     # smaller floor area
    },
    {   # A2 - modern north-facing (worst PV, large house)
        "label":       "A2_Modern_North",
        "description": "Post-2010, north-facing, 2,350 sqft equiv. (low PV yield)",
        "north_axis":  180,
        "pv_frac":     0.65,     # low effective yield on north-facing roof
        "in02_thick":  0.1397,   # R-20
        "in46_thick":  0.1524,   # R-38
        "in05_thick":  0.2876,   # attic IN05 (cond 0.043) matched to same R as in46
        "infil_scale": 0.70,
        "load_scale":  1.15,     # larger house
    },
    {   # A3 - mid-code west-facing (afternoon peak solar, moderate insulation)
        "label":       "A3_MidCode_West",
        "description": "1990s code, west-facing, 1,450 sqft equiv. (afternoon peak)",
        "north_axis":  270,
        "pv_frac":     0.70,
        "in02_thick":  0.0762,   # R-13 (Title 24 1992)
        "in46_thick":  0.0762,   # R-19 attic
        "in05_thick":  0.1438,   # attic IN05 (cond 0.043) matched to same R as in46
        "infil_scale": 1.10,     # moderate leakage
        "load_scale":  0.70,     # smaller house
    },
    {   # A4 — large modern SE-facing (best compromise, largest house)
        "label":       "A4_Large_ModernSE",
        "description": "Post-2010, SE-facing, 2,650 sqft equiv. (largest archetype)",
        "north_axis":  45,
        "pv_frac":     0.85,
        "in02_thick":  0.1397,   # R-20
        "in46_thick":  0.1524,   # R-38
        "in05_thick":  0.2876,   # attic IN05 (cond 0.043) matched to same R as in46
        "infil_scale": 0.65,     # tightest envelope
        "load_scale":  1.30,     # largest house
    },
]


# 2.  CONFIGURATION

class SwarmConfig:
    
    ENERGYPLUS_EXE: Optional[Path] = None

    SOURCE_IDF      = Path("HVACTemplate-5ZonePTHP.idf")
    FINAL_IDF       = Path("HVACTemplate-5ZonePTHP_Run.idf")    

    ARCHETYPE_IDF_PATHS: List[Path] = []
    WEATHER_FILE    = Path("USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw").resolve()
    FLEET_DATA_FILE = Path("NHTS_Fleet_Data.csv")
    PLUGIN_FILE     = Path("OccupancyPlugin.py").resolve()

    # i.i.d. STAGGER ABLATION  

    ABLATION_ARM:           str       = "vdc"      # "off" | "vdc" | "iid"
    ABLATION_SCENARIOS:     List[int] = [1]
    PYTHONHASHSEED_VALUE:   int       = 42

    BASE_OUT_DIR    = Path("Results_Sensitivity" if ABLATION_ARM == "off"
                           else f"Results_Ablation_{ABLATION_ARM.upper()}").resolve()

    EV_PENETRATION_RATES: List[float] = [0.25, 0.50, 0.75, 1.00]
    INELASTICITY_RATIOS:  List[float] = [0.10, 0.30, 0.50]

    # BATTERY-PENETRATION SENSITIVITY
    BATTERY_PEN_SWEEP:          bool        = False
    BATTERY_PENETRATION_RATES:  List[float] = [0.25, 0.50, 0.75, 1.00]
    BATTERY_PEN_SWEEP_EV_PEN:   float       = 0.25 

    MARKOV_PERSISTENCE_LEVELS: List[float] = [0.40, 0.60, 0.80]

    SCENARIOS:            List[int]   = [0, 1, 2, 3, 4]

    ENABLE_S5:            bool        = True
    SCENARIOS_EXT:        List[int]   = [0, 1, 2, 3, 4, 5]

    ENABLE_FAIR_BENCHMARKS: bool      = True

    @staticmethod
    def active_scenarios() -> "List[int]":

        if SwarmConfig.ABLATION_ARM != "off":
            return SwarmConfig.ABLATION_SCENARIOS
        return (SwarmConfig.SCENARIOS_EXT
                if SwarmConfig.ENABLE_S5
                else SwarmConfig.SCENARIOS)

    @staticmethod
    def swept_penetration_rates() -> "List[float]":

        return (SwarmConfig.BATTERY_PENETRATION_RATES
                if SwarmConfig.BATTERY_PEN_SWEEP
                else SwarmConfig.EV_PENETRATION_RATES)

    @staticmethod
    def plugin_stagger_mode() -> str:
 
        try:
            tree = ast.parse(SwarmConfig.PLUGIN_FILE.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return "unknown"
        for node in tree.body:
            name = None
            if isinstance(node, ast.AnnAssign):
                name = getattr(node.target, "id", None)
            elif isinstance(node, ast.Assign) and node.targets:
                name = getattr(node.targets[0], "id", None)
            if name == "ALGO1_STAGGER_MODE":
                try:
                    return str(ast.literal_eval(node.value))
                except (ValueError, SyntaxError):
                    return "unknown"
        return "unknown"

    @staticmethod
    def plugin_sim_seed() -> "Optional[int]":

        try:
            tree = ast.parse(SwarmConfig.PLUGIN_FILE.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            return None
        for node in tree.body:
            name = None
            if isinstance(node, ast.AnnAssign):
                name = getattr(node.target, "id", None)
            elif isinstance(node, ast.Assign) and node.targets:
                name = getattr(node.targets[0], "id", None)
            if name == "SIM_SEED":
                try:
                    return int(ast.literal_eval(node.value))
                except (ValueError, SyntaxError):
                    return None
        return None

    @staticmethod
    def run_mode_signature() -> "Dict[str, Any]":

        return {
            "stagger_mode":      SwarmConfig.plugin_stagger_mode(),
            "ablation_arm":      str(SwarmConfig.ABLATION_ARM),
            "sim_seed":          SwarmConfig.plugin_sim_seed(),
            "pythonhashseed":    int(SwarmConfig.PYTHONHASHSEED_VALUE),
            "battery_pen_sweep": bool(SwarmConfig.BATTERY_PEN_SWEEP),
            "enable_s5":         bool(SwarmConfig.ENABLE_S5),
            "scenarios":         list(SwarmConfig.active_scenarios()),
        }

    @staticmethod
    def assert_run_mode_consistent(logger=None) -> "Dict[str, Any]":

        sig  = SwarmConfig.run_mode_signature()
        mode = sig["stagger_mode"]
        want = "vdc" if SwarmConfig.ABLATION_ARM == "off" else SwarmConfig.ABLATION_ARM
        if SwarmConfig.ABLATION_ARM not in ("off", "vdc", "iid"):
            raise RuntimeError(
                f"ABLATION_ARM must be 'off', 'vdc', or 'iid'; got "
                f"{SwarmConfig.ABLATION_ARM!r}. A typo here would otherwise pick an arm.")

        # 1. manager vs plugin
        if mode == "unknown":
            raise RuntimeError(
                f"Cannot read ALGO1_STAGGER_MODE from {SwarmConfig.PLUGIN_FILE}.\n"
                f"  Expected the v14.1 switch. Refusing to guess which arm this is.")
        if mode != want:
            raise RuntimeError(
                f"STAGGER-MODE MISMATCH - refusing to run.\n"
                f"  Run_Manager.py     : ABLATION_ARM = '{SwarmConfig.ABLATION_ARM}' "
                f"(expects '{want}')\n"
                f"  {SwarmConfig.PLUGIN_FILE.name:<19}: ALGO1_STAGGER_MODE = '{mode}'\n"
                f"  Set BOTH to the same arm. Out of step, this run would complete\n"
                f"  normally and produce a plausible, wrong result.")

        # 2. tree vs config
        marker = SwarmConfig.BASE_OUT_DIR / SwarmConfig.RUN_MODE_FILENAME
        if marker.exists():
            try:
                prev = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                prev = {}
            clash = {k: (prev.get(k), sig[k])
                     for k in ("stagger_mode", "battery_pen_sweep")
                     if k in prev and prev.get(k) != sig[k]}
            if clash:
                detail = "\n".join(
                    f"    {k}: tree was built as {old!r}, this run is {new!r}"
                    for k, (old, new) in clash.items())
                raise RuntimeError(
                    f"OUTPUT-TREE ARM MISMATCH - refusing to run.\n"
                    f"  Tree : {SwarmConfig.BASE_OUT_DIR}\n"
                    f"{detail}\n"
                    f"  Completed houses are skipped without checking which arm\n"
                    f"  produced them, so this run would silently aggregate the\n"
                    f"  OLD data under the new label. Use a fresh output tree.")

        # 3. S5 vs ablation
        if SwarmConfig.ABLATION_ARM != "off" and SwarmConfig.ENABLE_S5:
            msg = ("ENABLE_S5 = True is IGNORED while ABLATION_ARM != 'off'; S5 is "
                   "excluded from the ablation by design (see ABLATION_ARM).")
            (logger.warning if logger else print)(f"[v14.1] {msg}")

        SwarmConfig.BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(sig, indent=2), encoding="utf-8")
        return sig

    @staticmethod
    def build_broadcast_profiles(base_dir) -> "List[str]":

        from pathlib import Path as _Path
        base = _Path(base_dir)
        ready, missing = [], []
        if base.is_dir():
            for cond_dir in sorted(p for p in base.glob("Rho*_Pen*_Inel*")
                                   if p.is_dir()):
                if any(cond_dir.glob("community_ts_*_S1.csv")):
                    ready.append(cond_dir.name)
                else:
                    missing.append(cond_dir.name)
        print(f"[S5 broadcast] {len(ready)} condition(s) have an S1 broadcast ready; "
              f"{len(missing)} missing.")
        if missing:
            print("[S5 broadcast] Missing S1 community series for: "
                  + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else ""))
            print("[S5 broadcast] Run the Phase-1 (S0-S4) sweep first so the aggregator "
                  "writes community_ts_..._S1.csv; otherwise S5 falls back to S1.")
        return ready

    PROCESS_TIMEOUT_S:  int = 3_600
    MAX_RETRIES:        int = 1

    SIDECAR_FILENAME:   str = "house_telemetry.csv"
    MANIFEST_FILENAME:  str = "run_manifest.jsonl"

    METRICS_FILENAME:   str = ("SRE_metrics_summary.csv" if ABLATION_ARM == "off"
                               else f"SRE_metrics_summary_ablation_{ABLATION_ARM}.csv")

    RUN_MODE_FILENAME:  str = "run_mode.json"

    RANDOM_STATE:       int = 42

    @classmethod
    def resolve(cls) -> None:
        """Auto-detect EnergyPlus executable on all major platforms."""
        if cls.ENERGYPLUS_EXE is not None and cls.ENERGYPLUS_EXE.exists():
            return

        system = platform.system()
        candidates: List[Path] = []

        if system == "Windows":
            for root in [Path("C:/"), Path("D:/")]:
                if root.exists():
                    candidates += sorted(root.glob("EnergyPlusV*"), reverse=True)
            exe_name = "energyplus.exe"
        elif system == "Darwin":
            candidates += sorted(
                Path("/Applications").glob("EnergyPlus*"), reverse=True)
            exe_name = "energyplus"
        else:   
            for prefix in [Path("/usr/local"), Path("/opt"), Path.home()]:
                candidates += sorted(prefix.glob("EnergyPlus*"), reverse=True)
            exe_name = "energyplus"

        for cand in candidates:
            exe = (cand / exe_name) if cand.is_dir() else cand
            if exe.exists():
                cls.ENERGYPLUS_EXE = exe.resolve()
                return

        # PATH fallback
        found = shutil.which("energyplus") or shutil.which("energyplus.exe")
        if found:
            cls.ENERGYPLUS_EXE = Path(found).resolve()
            return

        raise FileNotFoundError(
            "EnergyPlus executable not found. "
            "Set SwarmConfig.ENERGYPLUS_EXE or ensure it is on PATH.")


# 3.  PRE-FLIGHT VALIDATOR

class ConfigValidator:


    REQUIRED_KEYS        = {"house_id", "max_capacity", "probs", "profiles"}
    REQUIRED_MONTHS      = set(range(1, 13))
    REQUIRED_PROFILE_LEN = 24

    @classmethod
    def validate_all(cls, config_files: List[Path],
                     logger: logging.Logger) -> bool:
        all_ok = True
        for cfg in config_files:
            ok, errors = cls._validate_one(cfg)
            if not ok:
                for err in errors:
                    logger.error(f"[VALIDATE] {cfg.name}: {err}")
                all_ok = False
        if all_ok:
            logger.info(
                f"[VALIDATE] All {len(config_files)} configs passed.")
        return all_ok

    @classmethod
    def _validate_one(cls, path: Path) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            return False, [str(exc)]

        missing = cls.REQUIRED_KEYS - set(data.keys())
        if missing:
            return False, [f"Missing keys: {missing}"]

        probs = data["probs"]
        for month in cls.REQUIRED_MONTHS:
            for wkd in ("False", "True"):
                key = f"{month}_{wkd}"
                if key not in probs:
                    errors.append(f"Missing prob key: '{key}'")
                    continue
                
                vec = [float(x) for x in probs[key]]
                
                expected_k = max(2, len(data.get("profiles", {})))
                if len(vec) != expected_k:
                    errors.append(f"'{key}' length={len(vec)}, expected {expected_k}")
                if abs(sum(vec) - 1.0) > 1e-4:
                    errors.append(f"'{key}' sums to {sum(vec):.6f}")
                if any(v < 0 for v in vec):
                    errors.append(f"'{key}' contains negatives")

        profiles = data["profiles"]
        if len(profiles) < 2:
            errors.append(f"Expected at least 2 profiles, found {len(profiles)}")
            
        for pid, vals in profiles.items():
            v = [float(x) for x in vals]
            if len(v) != cls.REQUIRED_PROFILE_LEN:
                errors.append(f"Profile '{pid}' length={len(v)}")
            if any(not (0.0 <= x <= 1.0) for x in v):
                errors.append(f"Profile '{pid}' values outside [0,1]")

        mc = float(data.get("max_capacity", 0))
        if not (0 < mc <= 100):
            errors.append(f"max_capacity={mc} out of range (0,100]")

        return len(errors) == 0, errors

    @classmethod
    def validate_idf(cls, idf_path: Path, zones: List[str],
                     logger: logging.Logger) -> bool:

        try:
            raw = idf_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            logger.error(f"[IDF-VALIDATE] Cannot read {idf_path}: {exc}")
            return False

        stripped = [re.sub(r"!.*", "", line) for line in raw.splitlines()]
        tokens   = re.sub(r"\s+", " ", " ".join(stripped))
        ok = True

        if not re.search(r"PythonPlugin\s*:\s*Instance", tokens, re.I):
            logger.error(
                "[IDF-VALIDATE] PythonPlugin:Instance not found. "
                "OccupancyPlugin will NOT execute.")
            ok = False

        loc_m = re.search(r"Site\s*:\s*Location\s*,\s*([^,;]+)", tokens, re.I)
        if loc_m:
            loc_name = loc_m.group(1).strip()
            if not re.search(r"SAN[_\s]FRANCISCO", loc_name, re.I):
                logger.error(
                    f"[IDF-VALIDATE] Geographic mismatch (F-02): "
                    f"Site:Location='{loc_name}' is not San Francisco. "
                    "Both IDF files must use SAN_FRANCISCO_CA_USA TMY3-724940 "
                    "to match the weather file, PG&E tariff, and CAISO CO2 data. "
                    "See F-02 fix notes in module docstrings.")
                ok = False
            else:
                logger.info(f"[IDF-VALIDATE] Geographic check OK: {loc_name}")
        else:
            logger.warning(
                "[IDF-VALIDATE] Site:Location not found - "
                "cannot verify geographic consistency (F-02).")

        ts_m = re.search(r"Timestep\s*,\s*(\d+)\s*;", tokens, re.I)
        if ts_m and int(ts_m.group(1)) != 6:
            logger.warning(
                f"[IDF-VALIDATE] Timestep={ts_m.group(1)}/hr (expected 6).")
        elif not ts_m:
            logger.warning("[IDF-VALIDATE] Timestep object not found.")

        if ok:
            logger.info("[IDF-VALIDATE] IDF integrity check passed.")
            
        return ok


# 4.  FREE-FUNCTION WORKER  (module-level for ProcessPoolExecutor pickling)

def _worker_task(
    config_path_str:  str,
    scenario:         int,
    pen_rate:         float,
    inel_rate:        float,
    has_ev:           bool,
    zones:            List[str],
    fleet_entry:      Optional[Dict],
    house_dir_str:    str,
    retry_number:     int,
    has_battery:      bool  = True,  # battery-penetration sensitivity (default: has one)
    archetype_idf_str: str   = "",   
    markov_rho:        float = 0.60, # MARKOV-SENS 

    ep_exe_str:   str = "",   # str(SwarmConfig.ENERGYPLUS_EXE)
    weather_str:  str = "",   # str(SwarmConfig.WEATHER_FILE)
    plugin_str:   str = "",   # str(SwarmConfig.PLUGIN_FILE)
    timeout_s:    int = 3600, # SwarmConfig.PROCESS_TIMEOUT_S
    sidecar_name: str = "house_telemetry.csv",  # SwarmConfig.SIDECAR_FILENAME
) -> Dict[str, Any]:

    config_path = Path(config_path_str)
    house_dir   = Path(house_dir_str)

    house_id = config_path.stem.replace("config_", "").lower().strip()
    pen_str  = int(pen_rate  * 100)
    inel_str = int(inel_rate * 100)
    rho_str  = int(round(markov_rho * 100))  
    label    = (f"[Rho:{rho_str}%|Pen:{pen_str:>3}%|Inel:{inel_str:>2}%|S{scenario}"
                f"|r{retry_number}] {house_id}")

    sidecar  = house_dir / (sidecar_name or SwarmConfig.SIDECAR_FILENAME)
    end_file = house_dir / "eplusout.end"

    # Resume check (only skip on first attempt)
    if retry_number == 0 and end_file.exists() and sidecar.exists():
        return {"status": "skipped", "label": label, "duration": 0.0}

    try:
        house_dir.mkdir(parents=True, exist_ok=True)

        with open(config_path, encoding="utf-8") as fh:
            house_data = json.load(fh)

        house_data.update({
            "zones":              zones,
            "scenario":           scenario,
            "has_ev":             has_ev,
            "has_battery":        has_battery,   # battery-penetration sensitivity
            "inelasticity_ratio": inel_rate,
            "markov_rho":         markov_rho,  # MARKOV-SENS
        })
        # S5 (one-way broadcast)
        if scenario == 5:
            _bcast_path = (house_dir.parents[1]
                           / f"community_ts_Rho{rho_str}_Pen{pen_str}"
                             f"_Inel{inel_str}_S1.csv")
            house_data["broadcast_csv"] = str(_bcast_path)
        if has_ev and fleet_entry:

            house_data.update({
                "ev_arrival_hour": float(fleet_entry.get("Arrival_Hour", -1.0)),
                "ev_distance":     float(fleet_entry.get("Distance_Miles", 37.5)),
            })
        elif has_ev and not fleet_entry:
            pass   # config JSON already contains RUNTIME_DEFAULTS from ExtractBrainsIntoJSON
        elif not has_ev:
            house_data["ev_arrival_hour"] = -1.0

        (house_dir / "occupancy_config.json").write_text(
            json.dumps(house_data, indent=2), encoding="utf-8")
            
        # copy per-house archetype IDF
        idf_src = Path(archetype_idf_str) if archetype_idf_str else Path(plugin_str).parent / "HVACTemplate-5ZonePTHP_Run.idf"
        if not idf_src.exists():
            idf_src = Path(plugin_str).parent / "HVACTemplate-5ZonePTHP_Run.idf"  # graceful fallback
            
        shutil.copy2(str(idf_src), str(house_dir / "Local_Run.idf"))
        shutil.copy2(plugin_str or str(SwarmConfig.PLUGIN_FILE),
                     str(house_dir / "OccupancyPlugin.py"))

        proc_env = os.environ.copy()
        proc_env["ENERGYPLUS_OUTPUT_DIR"] = str(house_dir)
        proc_env["PYTHONHASHSEED"] = str(SwarmConfig.PYTHONHASHSEED_VALUE)
        proc_env["PYTHONPATH"] = (
            str(house_dir) + os.pathsep + proc_env.get("PYTHONPATH", ""))

        cmd = [
            ep_exe_str or str(SwarmConfig.ENERGYPLUS_EXE),
            "-x",                              # run ExpandObjects preprocessor
            "-w", weather_str or str(SwarmConfig.WEATHER_FILE),  # weather file
            "-d", str(house_dir),              # output directory
            "Local_Run.idf",                   # IDF positional arg (no -r flag)
        ]
        
        timeout = (timeout_s or SwarmConfig.PROCESS_TIMEOUT_S) * (2 ** retry_number)
        t0      = time.monotonic()

        result  = subprocess.run(
            cmd, capture_output=True, text=True,
            cwd=str(house_dir), env=proc_env,
            timeout=timeout)

        dur = time.monotonic() - t0

        if result.returncode == 0 and sidecar.exists():
            return {"status": "success", "label": label, "duration": round(dur, 2)}

        # Write crash log
        (house_dir / "CRASH_LOG.txt").write_text(
            f"=== stdout ===\n{result.stdout}\n"
            f"=== stderr ===\n{result.stderr}\n"
            f"=== RC={result.returncode} | sidecar={sidecar.exists()} ===\n",
            encoding="utf-8")
            
        reason = (f"EPLUS_RC={result.returncode}"
                  if result.returncode != 0
                  else "SIDECAR_MISSING")
                  
        return {"status": "error", "label": label,
                "reason": reason, "duration": round(dur, 2)}

    except subprocess.TimeoutExpired:
        return {"status": "error", "label": label,
                "reason": f"TIMEOUT>{timeout}s",
                "duration": float(timeout)}
    except Exception as exc:
        return {"status": "error", "label": label,
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[:500],
                "duration": 0.0}


# 5.  AGGREGATION HELPERS  (module-level for ProcessPool pickling)

def _infer_dt_hr(df: pd.DataFrame) -> float:

    try:
        ts  = df["ts_key"].sort_values().diff().dropna()
        pos = ts[(ts > 0) & (ts < 1000)]
        if len(pos) == 0:
            return 10.0 / 60.0
        dt_native = float(pos.mode().iloc[0])
        dt_min = max(0.5, dt_native / 10.0)
        return dt_min / 60.0
    except Exception:
        return 10.0 / 60.0


def _bootstrap_ci_bca(
    values:      np.ndarray,
    stat_fn,
    n_resamples: int   = BOOTSTRAP_RESAMPLES,
    alpha:       float = BOOTSTRAP_CI_ALPHA,
) -> Tuple[float, float]:

    n = len(values)
    if n == 0:
        return float("nan"), float("nan")

    if n < 2:
        theta = stat_fn(values)
        return theta, theta

    if not _SCIPY_AVAILABLE:
        rng  = np.random.default_rng(GLOBAL_RANDOM_SEED)
        boot = np.array([
            stat_fn(rng.choice(values, size=n, replace=True))
            for _ in range(n_resamples)
        ])
        return (
            float(np.nanpercentile(boot, 100.0 * alpha / 2.0)),
            float(np.nanpercentile(boot, 100.0 * (1.0 - alpha / 2.0))),
        )

    rng  = np.random.default_rng(GLOBAL_RANDOM_SEED)
    boot = np.array([
        stat_fn(rng.choice(values, size=n, replace=True))
        for _ in range(n_resamples)
    ])

    theta = stat_fn(values)

    prop_below = float(np.mean(boot < theta))
    # Clamp strictly inside (0,1) to keep ppf finite; 0.5/n is the Laplace prior
    prop_below = np.clip(prop_below, 0.5 / n_resamples, 1.0 - 0.5 / n_resamples)
    z0 = float(_scipy_norm.ppf(prop_below))

    # Acceleration a via jackknife influence values
    jack = np.array([stat_fn(np.delete(values, i)) for i in range(n)])
    jack_mean = float(np.mean(jack))
    diff      = jack_mean - jack              # influence scores
    num       = float(np.sum(diff ** 3))
    den       = 6.0 * float(np.sum(diff ** 2) ** 1.5)
    a         = num / den if abs(den) > 1e-15 else 0.0

    # Adjusted quantile levels
    za = float(_scipy_norm.ppf(alpha / 2.0))
    zb = float(_scipy_norm.ppf(1.0 - alpha / 2.0))

    def _adjusted_p(z_alpha: float) -> float:
        numer = z0 + z_alpha
        denom = 1.0 - a * numer
        # Guard: if denom <= 0 the BCa correction diverges; fall back to z_alpha
        if abs(denom) < 1e-12:
            return float(_scipy_norm.cdf(z_alpha))
        return float(_scipy_norm.cdf(z0 + numer / denom))

    p1 = _adjusted_p(za)
    p2 = _adjusted_p(zb)

    # Clamp to valid percentile range to guard against extreme skew
    p1 = float(np.clip(p1, 0.0, 1.0))
    p2 = float(np.clip(p2, 0.0, 1.0))
    if p1 > p2:
        p1, p2 = p2, p1

    return (
        float(np.nanpercentile(boot, 100.0 * p1)),
        float(np.nanpercentile(boot, 100.0 * p2)),
    )


_bootstrap_ci = _bootstrap_ci_bca


def _compute_cost_vectorised(
    net_w:             np.ndarray,
    prices:            np.ndarray,
    dt_hr:             float,
    annual_export_kwh: float = 0.0,
) -> float:

    import_kwh  = np.maximum(net_w / 1000.0 * dt_hr, 0.0)
    export_kwh  = np.maximum(-net_w / 1000.0 * dt_hr, 0.0)

    import_cost = float((import_kwh * prices).sum())

    total_export    = float(export_kwh.sum())
    cap_remaining   = max(0.0, NEM3_ANNUAL_EXPORT_CAP_KWH - annual_export_kwh)
    within_cap      = min(total_export, cap_remaining)
    above_cap       = max(0.0, total_export - within_cap)
    export_credit   = within_cap * EXPORT_RATE + above_cap * NEM3_AVOIDED_RATE

    return import_cost - export_credit


def _compute_demand_charge_annual(
    comm_net_w: np.ndarray,
    day_arr:    np.ndarray,
    dt_hr:      float,
) -> float:

    import_w  = np.maximum(comm_net_w, 0.0)           # W - converted to kW below
    month_end = np.array(_MONTH_END_DOY)               # [31, 59, …, 365]

    # Map day-of-year (1-365) to month (1-12) using right-side searchsorted.
    # day_arr - 1 converts to 0-indexed day before applying cumulative bounds.
    month_arr = np.searchsorted(month_end, day_arr - 1, side="right") + 1
    month_arr = np.clip(month_arr, 1, 12).astype(int)

    total_demand_usd = 0.0
    for m in range(1, 13):
        mask = month_arr == m
        if mask.sum() == 0:
            continue
        peak_kw = float(import_w[mask].max()) / 1000.0
        total_demand_usd += peak_kw * DEMAND_CHARGE_RATE_USD_PER_KW_MONTH

    return total_demand_usd


def _aggregate_one_condition(
    base_dir_str: str,
    pen_rate:     float,
    inel_rate:    float,
    scenario:     int,
    markov_rho:   float = 0.60,  # MARKOV-SENS
) -> Optional[Dict]:

    base_dir = Path(base_dir_str)
    pen_str  = int(pen_rate * 100)
    inel_str = int(inel_rate * 100)
    rho_pct  = int(round(markov_rho * 100))
    scen_dir = (base_dir
                / f"Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}"
                / f"Scenario_{scenario}")

    sidecar_paths = list(scen_dir.rglob(SwarmConfig.SIDECAR_FILENAME))
    if not sidecar_paths:
        return None

    try:
        dfs: List[pd.DataFrame] = []
        for sp in sidecar_paths:
            try:
                df = pd.read_csv(sp, dtype={"day": int, "hour": int})
                if "net_grid_w" not in df.columns:
                    continue

                df["ts_key"] = (
                    df["day"].astype(np.int64) * 1_000_000
                    + df["hour"].astype(np.int64) * 10_000
                    + (df["minute"].astype(float).round(1) * 10.0).round().astype(np.int64)
                )
                dfs.append(df.set_index("ts_key"))
            except Exception:
                continue

        if not dfs:
            return None

        # infer dt_hr from first available sidecar
        dt_hr = _infer_dt_hr(dfs[0].reset_index())

        common_idx = dfs[0].index
        for _df in dfs[1:]:
            common_idx = common_idx.intersection(_df.index)
        common_idx = common_idx.sort_values()

        max_len = max(len(df) for df in dfs)

        _MIN_COMMON_FRAC = 0.95
        if len(common_idx) < _MIN_COMMON_FRAC * max_len:
            common_frac = len(common_idx) / max(1, max_len)

            _OLD_TSKEY_RATIO = 29.0 / 144.0   # ≈ 0.2014
            _OLD_TSKEY_TOL   = 0.005
            looks_like_old_tskey_bug = (
                abs(common_frac - _OLD_TSKEY_RATIO) < _OLD_TSKEY_TOL
                and abs(len(common_idx) - 365 * 29) < 30
                and abs(max_len - 365 * 144) < 100
            )
            if looks_like_old_tskey_bug:
                warnings.warn(
                    f"[VM-06/DIAG-OLD-TSKEY] "
                    f"Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}_S{scenario}: "
                    f"common_frac={common_frac:.4f} matches the v9.5 ts_key "
                    f"collision signature (29/144 ≈ 0.2014).  This means "
                    f"sidecars in this scen_dir were written by a pre-v9.7 "
                    f"plugin or contain stale data with the OLD ts_key "
                    f"formula.  Recommended action: delete the per-condition "
                    f"output dirs under {scen_dir.parent} and re-run Stage 3 "
                    f"with the current OccupancyPlugin.",
                    RuntimeWarning, stacklevel=2)

            warnings.warn(
                f"[VM-06] Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}_S{scenario}: "
                f"only {len(common_idx)}/{max_len} common ts_key rows "
                f"({100*common_frac:.1f}%) — "
                f"condition discarded to avoid zero-fill bias.",
                RuntimeWarning, stacklevel=2)
            return None

        ref_df  = dfs[0].reindex(common_idx)   # reference for price/co2/hour/day
        ref_idx = common_idx
        aligned = [df.reindex(common_idx) for df in dfs]

        # Stack all houses into 2-D numpy arrays (n_houses × T)
        net_w_all  = np.stack([df["net_grid_w"].values for df in aligned])
        pv_w_all   = np.stack([df["pv_w"].values       for df in aligned])
        ev_w_all   = np.stack([df["ev_w"].values        for df in aligned])
        batt_w_all = np.stack([df["battery_w"].values  for df in aligned])

        # read building_w from sidecar (HVAC + lights + equipment).
        if all("building_w" in df.columns for df in aligned):
            building_w_all = np.stack(
                [df["building_w"].values for df in aligned])
        else:
            # Backward-compat fallback (algebraic recovery)
            building_w_all = net_w_all - ev_w_all - batt_w_all + pv_w_all
            warnings.warn(
                f"[v11.0] sidecar lacks building_w; recovered algebraically.",
                RuntimeWarning, stacklevel=2)

        if (np.any(np.isnan(net_w_all)) or np.any(np.isnan(pv_w_all))
                or np.any(np.isnan(ev_w_all)) or np.any(np.isnan(batt_w_all))
                or np.any(np.isnan(building_w_all))):
            nan_counts = {
                "net_grid_w": int(np.isnan(net_w_all).sum()),
                "pv_w":       int(np.isnan(pv_w_all).sum()),
                "ev_w":       int(np.isnan(ev_w_all).sum()),
                "battery_w":  int(np.isnan(batt_w_all).sum()),
                "building_w": int(np.isnan(building_w_all).sum()),
            }
            warnings.warn(
                f"[NaN-GUARD] Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}_S{scenario}: "
                f"NaN values detected in stacked arrays after reindex: {nan_counts}. "
                f"Condition discarded to prevent silent metric corruption.",
                RuntimeWarning, stacklevel=2)
            return None

        comm_net_w      = net_w_all.sum(axis=0)
        comm_pv_w       = pv_w_all.sum(axis=0)
        comm_ev_w       = ev_w_all.sum(axis=0)
        comm_batt_w     = batt_w_all.sum(axis=0)
        comm_building_w = building_w_all.sum(axis=0)   

        prices_arr = ref_df["price_usd_kwh"].values.astype(float)
        co2_arr    = ref_df["co2_g_kwh"].values.astype(float)

        if "hour" in ref_df.columns:
            hour_arr = ref_df["hour"].values.astype(int)
        else:
            hour_arr = np.array(
                [(int(k) // 10_000) % 100 for k in ref_idx], dtype=int)

        if "day" in ref_df.columns:
            day_arr = ref_df["day"].values.astype(int)
        else:
            day_arr = np.array(
                [int(k) // 1_000_000 for k in ref_idx], dtype=int)
        day_arr = np.clip(day_arr, 1, 365)

        n_houses = len(aligned)

        # Coincidence Factor
        import_w   = np.maximum(comm_net_w, 0.0)
        t_star     = int(import_w.argmax())                    # community peak timestep
        P_peak     = float(import_w[t_star])

        per_house_peak_w = np.maximum(net_w_all, 0.0).max(axis=1)   # shape (n_houses,)
        sum_indiv_peaks  = float(per_house_peak_w.sum())
        CF               = P_peak / sum_indiv_peaks if sum_indiv_peaks > 0 else 1.0

        # Bootstrap CI on CF - resample houses (community composition uncertainty).

        def _cf_stat(idx_sample):
            sub_net   = net_w_all[idx_sample, :]                 # (k, T)
            sub_comm  = sub_net.sum(axis=0)
            sub_peak  = float(np.maximum(sub_comm, 0.0).max())
            sub_peaks = float(np.maximum(sub_net, 0.0).max(axis=1).sum())
            return sub_peak / sub_peaks if sub_peaks > 0 else 1.0

        rng_arr = np.arange(n_houses)
        cf_lo, cf_hi = _bootstrap_ci(rng_arr, _cf_stat)

        # ON-PEAK-WINDOW rebound metrics (16:00–21:00 TOU peak block)
        onpeak_mask = np.isin(hour_arr, range(16, 21))
        if onpeak_mask.sum() > 0:
            _imp_op   = np.maximum(comm_net_w[onpeak_mask], 0.0)
            P_peak_op = float(_imp_op.max())
            _hp_op    = np.maximum(net_w_all[:, onpeak_mask], 0.0).max(axis=1)
            _sig_op   = float(_hp_op.sum())
            CF_onpeak = P_peak_op / _sig_op if _sig_op > 0 else 1.0
        else:
            P_peak_op = float("nan")
            CF_onpeak = float("nan")

        # Ramp rate
        ramp_mask = np.isin(hour_arr,
                            range(RAMP_WINDOW_HOURS[0], RAMP_WINDOW_HOURS[1] + 1))
        if ramp_mask.sum() > 1:
            delta_w  = float(np.abs(np.diff(comm_net_w[ramp_mask])).max())
            dt_min   = max(1.0, dt_hr * 60.0)   # Guard div-by-zero
            ramp_w_min = delta_w / dt_min
        else:
            ramp_w_min = 0.0

        # Duck-curve metrics
        midday_mask = np.isin(hour_arr, range(10, 16))
        trough_kw   = (float(import_w[midday_mask].min()) / 1000.0
                       if midday_mask.sum() > 0 else float("nan"))
        E_annual_kwh = float(import_w.sum() * dt_hr / 1000.0)
        load_factor  = (E_annual_kwh / max(1e-9, P_peak / 1000.0 * 8_760.0))

        P50 = float(np.percentile(import_w, 50)) / 1000.0
        P90 = float(np.percentile(import_w, 90)) / 1000.0
        P99 = float(np.percentile(import_w, 99)) / 1000.0

        # Cost: TOU energy + NEM export credit
        E_cost_usd = _compute_cost_vectorised(comm_net_w, prices_arr, dt_hr)

        # Demand charge: PG&E B-1 monthly peak
        demand_charge_usd = _compute_demand_charge_annual(
            comm_net_w, day_arr, dt_hr)
        total_bill_usd    = E_cost_usd + demand_charge_usd

        # IMPLEMENT-BILL-DECOMP: per-component bill attribution

        _import_w  = np.maximum(comm_net_w, 0.0)
        _export_w  = np.maximum(-comm_net_w, 0.0)
        _is_import = comm_net_w > 0.0

        _bld_pos  = np.maximum(comm_building_w, 0.0)
        _ev_pos   = np.maximum(comm_ev_w,       0.0)
        _batt_pos = np.maximum(comm_batt_w,     0.0)

        _demand_pool = _bld_pos + _ev_pos + _batt_pos
        _demand_pool = np.where(_demand_pool > 0.0, _demand_pool, 1.0)

        _import_kwh = _import_w / 1000.0 * dt_hr
        _import_per_ts_cost = _import_kwh * prices_arr   # $/timestep

        _share_bld  = np.where(_is_import, _bld_pos  / _demand_pool, 0.0)
        _share_ev   = np.where(_is_import, _ev_pos   / _demand_pool, 0.0)
        _share_batt = np.where(_is_import, _batt_pos / _demand_pool, 0.0)

        _bld_cost  = float(np.sum(_share_bld  * _import_per_ts_cost))
        _ev_cost   = float(np.sum(_share_ev   * _import_per_ts_cost))
        _batt_cost = float(np.sum(_share_batt * _import_per_ts_cost))

        # PV export credit (matches _compute_cost_vectorised exactly).
        _total_export_kwh = float((_export_w / 1000.0 * dt_hr).sum())
        _within_cap = min(_total_export_kwh, NEM3_ANNUAL_EXPORT_CAP_KWH)
        _above_cap  = max(0.0, _total_export_kwh - _within_cap)
        _pv_credit = (_within_cap * EXPORT_RATE
                      + _above_cap * NEM3_AVOIDED_RATE)

        # By construction:  E_baseline + E_ev + E_batt - E_pv_credit = E_cost_usd
        _component_sum = _bld_cost + _ev_cost + _batt_cost - _pv_credit
        _decomp_err_pct = (abs(_component_sum - E_cost_usd)
                           / max(abs(E_cost_usd), 1.0)) * 100.0

        # ADD-PEAK-COMPONENT-ATTRIB: per-component contribution at t*
        P_peak_building_kw = float(comm_building_w[t_star]) / 1000.0
        P_peak_ev_kw       = float(comm_ev_w[t_star])       / 1000.0
        P_peak_batt_kw     = float(comm_batt_w[t_star])     / 1000.0
        P_peak_pv_kw       = float(comm_pv_w[t_star])       / 1000.0

        # IMPLEMENT-OPTIMAL-LB: lower-bound benchmarks on demand charge 
        E_total_kwh_year   = float(_import_w.sum() * dt_hr / 1000.0)
        P_uniform_kw       = E_total_kwh_year / 8_760.0
        # PG&E B-1: $19.71 / kW / month × 12 months
        demand_charge_lb_uniform_usd = (
            P_uniform_kw * DEMAND_CHARGE_RATE_USD_PER_KW_MONTH * 12.0)


        lp_opf_status = "ok"     # populated below; emitted into CSV

        lp_opf_peak_kw   = float("nan")
        lp_opf_peak_hour = -1
        try:
            from scipy.optimize import linprog as _linprog

            # Aggregate to hourly resolution (in kW, NOT W)
            n_steps = len(comm_net_w)
            n_hours = max(1, int(round(n_steps * dt_hr)))
            hours_per_step = max(1, n_steps // n_hours)
            n_hours = n_steps // hours_per_step

            # Convert W → kW during aggregation
            comm_bld_hourly_kw = (comm_building_w[:n_hours * hours_per_step]
                                  .reshape(n_hours, hours_per_step).mean(axis=1)) / 1000.0
            comm_ev_hourly_kw  = (comm_ev_w[:n_hours * hours_per_step]
                                  .reshape(n_hours, hours_per_step).mean(axis=1)) / 1000.0
            comm_pv_hourly_kw  = (comm_pv_w[:n_hours * hours_per_step]
                                  .reshape(n_hours, hours_per_step).mean(axis=1)) / 1000.0
            # OPF-BATTERY: aggregate battery for the LP-OPF epigraph
            comm_batt_hourly_kw = (comm_batt_w[:n_hours * hours_per_step]
                                   .reshape(n_hours, hours_per_step).mean(axis=1)) / 1000.0

            # Reduce to 24-hour average representative day (in kW)
            n_days = max(1, n_hours // 24)
            n_use  = n_days * 24
            bld_24_kw  = comm_bld_hourly_kw[:n_use].reshape(n_days, 24).mean(axis=0)
            ev_24_kw   = comm_ev_hourly_kw[:n_use].reshape(n_days, 24).mean(axis=0)
            pv_24_kw   = comm_pv_hourly_kw[:n_use].reshape(n_days, 24).mean(axis=0)
            # OPF-BATTERY: 24-hr battery profile for LP epigraph
            batt_24_kw = comm_batt_hourly_kw[:n_use].reshape(n_days, 24).mean(axis=0)

            # G2V / V2G decomposition
            g2v_24_kw = np.maximum(ev_24_kw,  0.0)   # charging draw, >= 0
            v2g_24_kw = np.maximum(-ev_24_kw, 0.0)   # discharge magnitude, >= 0
            # Daily totals (kWh, since we hourly-aggregated and dt = 1 h here)
            E_g2v_day_kwh = float(np.sum(g2v_24_kw))
            E_v2g_day_kwh = float(np.sum(v2g_24_kw))

            # Hourly TOU price (real PG&E E-TOU-C, two-tier, peak 16:00-21:00):
            _hourly_prices = np.full(24, RATE_OFF_PEAK)
            for h_ in range(24):
                if 16 <= h_ <= 20: _hourly_prices[h_] = RATE_ON_PEAK

            # LP variables: x_0..x_23 (hourly community G2V charging in kW)
            #               + p (annual peak in kW)
            # Energy budget: Σ x_h × 1h = E_g2v_day_kwh   (>= 0 by construction)
            # Per-hour bound: 50 EVs × 7 kW = 350 kW absolute community max
            P_ev_max_kw  = max(50.0, n_houses * 7.0)

            # Objective: 365 × Σ price_h × x_h + DEMAND × 12 × p
            c = np.concatenate([_hourly_prices * 365.0,
                                [DEMAND_CHARGE_RATE_USD_PER_KW_MONTH * 12.0]])

            LP_TIE_EPS = 1e-6
            c[:24] = c[:24] + LP_TIE_EPS * np.arange(24, 0, -1, dtype=float)

            A_eq = np.concatenate([np.ones(24), [0.0]]).reshape(1, -1)
            b_eq = np.array([E_g2v_day_kwh])

            # Inequality (epigraph): for each h,
            #   bld_h + x_h + batt_h - pv_h - v2g_h <= p
            # ⇔ x_h - p <= pv_h + v2g_h - bld_h - batt_h
            A_ub = np.zeros((24, 25))
            for h_ in range(24):
                A_ub[h_, h_] = 1.0
                A_ub[h_, 24] = -1.0
            b_ub = pv_24_kw + v2g_24_kw - bld_24_kw - batt_24_kw

            bounds = [(0.0, P_ev_max_kw)] * 24 + [(0.0, None)]

            _res = _linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )

            if _res.success:
                _peak_lp_kw = float(_res.x[24])
                demand_charge_lb_opf_usd = (
                    _peak_lp_kw * DEMAND_CHARGE_RATE_USD_PER_KW_MONTH * 12.0)
                lp_opf_status = "ok"
                lp_opf_peak_kw = _peak_lp_kw

                try:
                    x_sol = np.asarray(_res.x[:24], dtype=float)
                    load_h = bld_24_kw + x_sol + batt_24_kw - pv_24_kw - v2g_24_kw

                    lp_opf_peak_hour = int(np.argmax(load_h))
                except Exception:
                    lp_opf_peak_hour = -1
            else:
                demand_charge_lb_opf_usd = float("nan")

                _msg = (getattr(_res, "message", "") or "").strip()
                if "infeasible" in _msg.lower():
                    lp_opf_status = "infeasible"
                elif "unbounded" in _msg.lower():
                    lp_opf_status = "unbounded"
                elif _msg:
                    lp_opf_status = f"solver_fail:{_msg[:40]}"
                else:
                    lp_opf_status = (
                        f"solver_fail:status={getattr(_res,'status','?')}")
                sys.stderr.write(
                    f"[LP-OPF v12.2] solver returned not-success: "
                    f"status={getattr(_res, 'status', '?')} "
                    f"msg={_msg!r}  "
                    f"(E_g2v_day={E_g2v_day_kwh:.2f} kWh, "
                    f"E_v2g_day={E_v2g_day_kwh:.2f} kWh)\n"
                )
        except Exception as _exc:
            demand_charge_lb_opf_usd = float("nan")
            E_g2v_day_kwh = float("nan")
            E_v2g_day_kwh = float("nan")
            lp_opf_status = f"exc:{type(_exc).__name__}"

            sys.stderr.write(
                f"[LP-OPF v12.2] exception: "
                f"{type(_exc).__name__}: {_exc}\n"
            )

        # Optimality gap of actual schedule vs LP-OPF
        if (demand_charge_lb_opf_usd == demand_charge_lb_opf_usd
                and demand_charge_lb_opf_usd > 1e-3):
            optimality_gap_pct = round(
                (demand_charge_usd - demand_charge_lb_opf_usd)
                / demand_charge_lb_opf_usd * 100.0, 2)
        else:
            optimality_gap_pct = float("nan")

        comm_free_gap_pct = float("nan")
        active_set_rho_cert = float("nan")
        if SwarmConfig.ENABLE_FAIR_BENCHMARKS:
            try:
                _T = net_w_all.shape[1]
                _grid_kw = comm_net_w / 1000.0                      # community net (kW)
                _actual_peak_kw = float(np.max(_grid_kw))
                # Flexible (EV) power per house, per step, in kW.
                _ev_kw_all = ev_w_all / 1000.0                      # (n_houses, T) kW
                # Inflexible base = community net minus EV draw.
                _base_kw = _grid_kw - _ev_kw_all.sum(axis=0)
                _flex_energy_kwh = float(_ev_kw_all.sum() * dt_hr)

                import fair_benchmarks as _fb
                _win_mask = (hour_arr >= 21) | (hour_arr < 7)
                _achievable_peak_kw = _fb.min_peak_waterfill(
                    _base_kw, _win_mask, _flex_energy_kwh, dt_hr)
                if _achievable_peak_kw > 1e-6:
                    comm_free_gap_pct = round(
                        (_actual_peak_kw - _achievable_peak_kw)
                        / _achievable_peak_kw * 100.0, 2)

                # (ii) corrected active-set certificate from EV-active intervals.
                _active = (_ev_kw_all > 1e-3)                    # (n_houses, T) bool
                _count = _active.sum(axis=0).astype(float)       # A(t)
                if _count.max() > 0:
                        _peak_a = float(_count.max())
                        _mean_a = float(_count[_count > 0].mean())
                        # normalized start times: first active step per house.
                        _starts = []
                        for _h in range(_active.shape[0]):
                            _idx = np.argmax(_active[_h]) if _active[_h].any() else -1
                            if _idx >= 0:
                                _starts.append(_idx / _T)
                        _starts = np.sort(np.array(_starts))
                        _nN = _starts.size
                        if _nN > 1:
                            _w = float(_count.sum() / (_nN * _T))    # mean active fraction
                            # sliding-window discrepancy on a coarse grid.
                            _edges = np.linspace(0, 1, 1024)
                            _dw = 0.0
                            for _tt in _edges:
                                _c = np.count_nonzero((_starts > _tt - _w) & (_starts <= _tt))
                                _dw = max(_dw, abs(_c / _nN - _w))
                            active_set_rho_cert = round(
                                (_peak_a - _mean_a) / max(_nN * _dw, 1e-9), 3)

                print(f"[FAIR-BENCH] actual_peak={_actual_peak_kw:.1f}kW "
                      f"achievable={_achievable_peak_kw:.1f}kW "
                      f"gap={comm_free_gap_pct}% rho_cert={active_set_rho_cert}")
            except Exception as _fb_exc:
                print(f"[FAIR-BENCH] disabled: {type(_fb_exc).__name__}: {_fb_exc}")

        # CO2 - ELECTRIC-ONLY grid carbon

        E_carbon_kg = float(
            (np.maximum(comm_net_w, 0.0) / 1000.0 * dt_hr
             * co2_arr / 1000.0).sum())

        # E_ev_kwh historically = Σ max(comm_ev_w, 0) × dt — i.e. GROSS grid
        E_ev_kwh = float(np.maximum(comm_ev_w, 0.0).sum() * dt_hr / 1000.0)

        E_ev_grid_import_kwh = E_ev_kwh   # by definition; same column, named honestly
        E_ev_grid_export_kwh = float(
            np.maximum(-comm_ev_w, 0.0).sum() * dt_hr / 1000.0)

        E_pv_kwh = float(comm_pv_w.sum() * dt_hr / 1000.0)

        # LYAPUNOV COMMUNITY-AGGREGATION
        L_per_house_max  = float("nan")
        L_n_elastic      = 0
        L_n_total_parsed = 0
        L_p_baseline_kw  = float("nan")
        L_community      = float("nan")
        try:
            import re, json
            elastic_lyap_values = []
            n_elastic_seen = 0
            n_total_seen   = 0

            summary_paths = list(scen_dir.rglob("annual_summary.json"))
            if summary_paths:
                for sp in summary_paths:
                    try:
                        with open(sp, "r", encoding="utf-8") as fh:
                            data = json.load(fh)
                        n_total_seen += 1
                        if bool(data.get("elastic", False)):
                            n_elastic_seen += 1
                            lh = data.get("lyap_h", None)
                            if lh is not None:
                                try:
                                    v = float(lh)
                                    if v == v and v >= 0:
                                        elastic_lyap_values.append(v)
                                except Exception:
                                    pass
                    except Exception:
                        continue
            else:

                err_paths = list(scen_dir.rglob("eplusout.err"))
                for ep in err_paths:
                    try:
                        with open(ep, "r", encoding="utf-8",
                                  errors="replace") as fh:
                            text = fh.read()
                    except Exception:
                        continue
                    for m in re.finditer(
                        r"elastic=(True|False).*?Lyap_h=([0-9.eE+\-nan]+)",
                        text, flags=re.DOTALL,
                    ):
                        n_total_seen += 1
                        is_elastic = (m.group(1) == "True")
                        if is_elastic:
                            n_elastic_seen += 1
                            try:
                                v = float(m.group(2))
                                if v == v and v >= 0:
                                    elastic_lyap_values.append(v)
                            except Exception:
                                pass

            L_n_elastic      = n_elastic_seen
            L_n_total_parsed = n_total_seen

            if elastic_lyap_values:
                L_per_house_max = float(max(elastic_lyap_values))

            try:
                if n_houses > 0 and len(comm_building_w) > 0:
                    p95_comm_w = float(np.quantile(
                        np.maximum(comm_building_w, 0.0), 0.95))
                    L_p_baseline_kw = (p95_comm_w / 1000.0) / float(n_houses)
            except Exception:
                L_p_baseline_kw = float("nan")

            P_CHARGER_KW = 7.0   # matches OccupancyPlugin.EV_CHARGER_POWER_W
            if (scenario == 1
                    and L_per_house_max == L_per_house_max          # not NaN
                    and L_n_elastic > 0
                    and L_n_total_parsed > 0
                    and L_p_baseline_kw == L_p_baseline_kw           # not NaN
                    and L_p_baseline_kw > 1e-6):                     # avoid 1/0
                f_elastic = L_n_elastic / max(1, L_n_total_parsed)
                f_power   = P_CHARGER_KW / L_p_baseline_kw
                L_community = float(L_per_house_max * f_elastic * f_power)
            else:
                L_community = float("nan")
        except Exception:

            L_per_house_max  = float("nan")
            L_n_elastic      = 0
            L_p_baseline_kw  = float("nan")
            L_community      = float("nan")

        # FAIRNESS-INDEX: Jain's fairness index of per-house bills
        # J = (Σ x_i)² / (n × Σ x_i²) ∈ [1/n, 1]
        try:
            per_house_import_w   = np.maximum(net_w_all, 0.0)              # (n_houses, T)
            per_house_import_kwh = per_house_import_w / 1000.0 * dt_hr     # (n_houses, T)
            per_house_bills_usd  = (per_house_import_kwh * prices_arr[None, :]).sum(axis=1)
            _bsum  = float(per_house_bills_usd.sum())
            _bsq   = float((per_house_bills_usd ** 2).sum())
            if _bsq > 1e-9 and n_houses > 0:
                jain_fairness = (_bsum * _bsum) / (n_houses * _bsq)
            else:
                jain_fairness = float("nan")
        except Exception:
            jain_fairness = float("nan")

        # Save community time series
        ts_df = pd.DataFrame({
            "ts_key":             list(ref_idx),
            "hour":               hour_arr,
            "price_usd_kwh":      prices_arr,
            "co2_g_kwh":          co2_arr,
            "community_net_kw":   comm_net_w  / 1000.0,
            "community_pv_kw":    comm_pv_w   / 1000.0,
            "community_ev_kw":    comm_ev_w   / 1000.0,
            "community_batt_kw":  comm_batt_w / 1000.0,
        })
        ts_dir  = base_dir / f"Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}"
        ts_dir.mkdir(parents=True, exist_ok=True)
        ts_path = (ts_dir
                   / f"community_ts_Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}_S{scenario}.csv")
        ts_df.to_csv(ts_path, index=False, float_format="%.4f")

        return {
            "pen_rate":              pen_rate,
            "inel_rate":             inel_rate,
            "scenario":              scenario,
            "markov_rho":            markov_rho, 
            "n_houses":              n_houses,
            "dt_hr":                 round(dt_hr, 5),
            "P_community_peak_kw":   round(P_peak / 1000.0, 4),
            "P_onpeak_peak_kw":      round(P_peak_op / 1000.0, 4) if P_peak_op == P_peak_op else float("nan"),
            "CF_onpeak":             round(CF_onpeak, 6) if CF_onpeak == CF_onpeak else float("nan"),
            "P50_kw":                round(P50,  4),
            "P90_kw":                round(P90,  4),
            "P99_kw":                round(P99,  4),
            "CF":                    round(CF,   6),
            "CF_ci_lo":              round(cf_lo, 6) if cf_lo == cf_lo else float("nan"),
            "CF_ci_hi":              round(cf_hi, 6) if cf_hi == cf_hi else float("nan"),
            "ramp_rate_kw_per_min":  round(ramp_w_min / 1000.0, 6),
            "trough_kw":             round(trough_kw,   4) if trough_kw == trough_kw else float("nan"),
            "load_factor":           round(load_factor, 6),
            "E_annual_kwh":          round(E_annual_kwh, 2),
            "E_pv_kwh":              round(E_pv_kwh,     2),
            "E_ev_kwh":              round(E_ev_kwh,     2),
            # EV-ENERGY-SPLIT - unambiguous EV grid energy 
            "E_ev_grid_import_kwh":  round(E_ev_grid_import_kwh, 2),
            "E_ev_grid_export_kwh":  round(E_ev_grid_export_kwh, 2),
            "E_cost_usd":            round(E_cost_usd,        2),
            "demand_charge_usd":     round(demand_charge_usd, 2),
            "total_bill_usd":        round(total_bill_usd,    2),
            "E_carbon_kg":           round(E_carbon_kg,       2),
            # IMPLEMENT-BILL-DECOMP - four-component bill attribution
            "E_baseline_cost_usd":   round(_bld_cost,  2),
            "E_ev_cost_usd":         round(_ev_cost,   2),
            "E_batt_cost_usd":       round(_batt_cost, 2),
            "E_pv_credit_usd":       round(_pv_credit, 2),
            "decomp_err_pct":        round(_decomp_err_pct, 4),
            # ADD-PEAK-COMPONENT-ATTRIB - components at t*
            "P_peak_building_kw":    round(P_peak_building_kw, 4),
            "P_peak_ev_kw":          round(P_peak_ev_kw,       4),
            "P_peak_batt_kw":        round(P_peak_batt_kw,     4),
            "P_peak_pv_kw":          round(P_peak_pv_kw,       4),
            # IMPLEMENT-OPTIMAL-LB - centralized OPF benchmarks
            "demand_charge_lb_uniform_usd": round(demand_charge_lb_uniform_usd, 2),
            "demand_charge_lb_opf_usd":     round(demand_charge_lb_opf_usd, 2)
                if demand_charge_lb_opf_usd == demand_charge_lb_opf_usd
                else float("nan"),
            "optimality_gap_pct":           optimality_gap_pct,
            # communication-free frontier gap
            "comm_free_oracle_gap_pct":     comm_free_gap_pct,

            "active_set_rho_cert":          active_set_rho_cert,

            "lp_opf_status":         lp_opf_status,
            "lp_opf_g2v_day_kwh":    round(E_g2v_day_kwh, 2)
                if E_g2v_day_kwh == E_g2v_day_kwh else float("nan"),
            "lp_opf_v2g_day_kwh":    round(E_v2g_day_kwh, 2)
                if E_v2g_day_kwh == E_v2g_day_kwh else float("nan"),
            # LP-OPF-DISCLOSURE - benchmark transparency
            "lp_opf_peak_kw":        round(lp_opf_peak_kw, 4)
                if lp_opf_peak_kw == lp_opf_peak_kw else float("nan"),
            "lp_opf_peak_hour":      int(lp_opf_peak_hour),
            # LYAPUNOV COMMUNITY-AGGREGATION 

            "L_community":          round(L_community, 6)
                if L_community == L_community else float("nan"),
            "L_per_house_max":      round(L_per_house_max, 6)
                if L_per_house_max == L_per_house_max else float("nan"),
            "L_n_elastic":          int(L_n_elastic),
            "L_p_baseline_kw":      round(L_p_baseline_kw, 4)
                if L_p_baseline_kw == L_p_baseline_kw else float("nan"),
            # ADD-FAIRNESS-INDEX - Jain's fairness over per-house bills
            "jain_fairness":        round(jain_fairness, 6)
                if jain_fairness == jain_fairness else float("nan"),
        }

    except Exception:
        sys.stderr.write(
            f"[Aggregator] Exception Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}_S{scenario}:\n"
            + traceback.format_exc() + "\n")
        return None


def _compute_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
      SRE_ratio = CF(S2) / CF(S0)   — TOU rebound vs uncontrolled baseline.
      MIT_ratio = CF(S1) / CF(S2)   — Algorithm 1 mitigation of TOU rebound.
      VF_ratio  = CF(S4) / CF(S1)   — Valley-filling vs Algorithm 1.
                  VF_ratio > 1 → Algorithm 1 attains the lower CF.
                  VF_ratio < 1 → valley-filling attains the lower CF. 
    """
    df = df.copy()
    for col in ("SRE_ratio", "SRE_ci_lo", "SRE_ci_hi",
                "MIT_ratio", "MIT_ci_lo", "MIT_ci_hi",
                "VF_ratio",  "VF_ci_lo",  "VF_ci_hi",
                "HYB_ratio", "HYB_ci_lo", "HYB_ci_hi",
                "SRE_onpeak", "MIT_onpeak", "VF_onpeak"):
        df[col] = float("nan")

    _groupby_cols = (
        ["pen_rate", "inel_rate", "markov_rho"]
        if "markov_rho" in df.columns
        else ["pen_rate", "inel_rate"]
    )
    for _group_key, grp in df.groupby(_groupby_cols):
        idx   = grp.index
        cf    = grp.set_index("scenario")["CF"].to_dict()
        ci_lo = grp.set_index("scenario")["CF_ci_lo"].to_dict()
        ci_hi = grp.set_index("scenario")["CF_ci_hi"].to_dict()

        cf0 = cf.get(0, float("nan"))
        cf1 = cf.get(1, float("nan"))
        cf2 = cf.get(2, float("nan"))
        cf4 = cf.get(4, float("nan"))

        # on-peak CF: SRE=CF(S2)/CF(S0), MIT=CF(S1)/CF(S2), VF=CF(S4)/CF(S1).
        if "CF_onpeak" in grp.columns:
            cf_op = grp.set_index("scenario")["CF_onpeak"].to_dict()
            op0, op1 = cf_op.get(0, float("nan")), cf_op.get(1, float("nan"))
            op2, op4 = cf_op.get(2, float("nan")), cf_op.get(4, float("nan"))
            if op0 == op0 and op2 == op2 and op0 > 0:
                df.loc[idx, "SRE_onpeak"] = op2 / op0
            if op1 == op1 and op2 == op2 and op2 > 0:
                df.loc[idx, "MIT_onpeak"] = op1 / op2
            if op4 == op4 and op1 == op1 and op1 > 0:
                df.loc[idx, "VF_onpeak"] = op4 / op1

        if cf0 == cf0 and cf2 == cf2 and cf0 > 0:
            sre = cf2 / cf0
            s2_lo, s2_hi = ci_lo.get(2, float("nan")), ci_hi.get(2, float("nan"))
            s0_lo, s0_hi = ci_lo.get(0, float("nan")), ci_hi.get(0, float("nan"))
            df.loc[idx, "SRE_ratio"] = sre
            if all(x == x for x in (s2_lo, s2_hi, s0_lo, s0_hi)) and s0_hi > 0 and s0_lo > 0:
                df.loc[idx, "SRE_ci_lo"] = s2_lo / s0_hi
                df.loc[idx, "SRE_ci_hi"] = s2_hi / s0_lo

        # MIT_ratio: Algorithm 1 (S1) vs TOU rebound (S2)
        if cf1 == cf1 and cf2 == cf2 and cf2 > 0:
            mit = cf1 / cf2
            s1_lo, s1_hi = ci_lo.get(1, float("nan")), ci_hi.get(1, float("nan"))
            s2_lo, s2_hi = ci_lo.get(2, float("nan")), ci_hi.get(2, float("nan"))
            df.loc[idx, "MIT_ratio"] = mit
            if all(x == x for x in (s1_lo, s1_hi, s2_lo, s2_hi)) and s2_hi > 0 and s2_lo > 0:
                df.loc[idx, "MIT_ci_lo"] = s1_lo / s2_hi
                df.loc[idx, "MIT_ci_hi"] = s1_hi / s2_lo

        # VF_ratio: valley-filling (S4) relative to Algorithm 1 (S1)
        # VF_ratio > 1 means Algorithm 1 achieves lower CF (better) than valley-fill
        if cf4 == cf4 and cf1 == cf1 and cf1 > 0:
            vf = cf4 / cf1
            s4_lo, s4_hi = ci_lo.get(4, float("nan")), ci_hi.get(4, float("nan"))
            s1_lo, s1_hi = ci_lo.get(1, float("nan")), ci_hi.get(1, float("nan"))
            df.loc[idx, "VF_ratio"] = vf
            if all(x == x for x in (s4_lo, s4_hi, s1_lo, s1_hi)) and s1_hi > 0 and s1_lo > 0:
                df.loc[idx, "VF_ci_lo"] = s4_lo / s1_hi
                df.loc[idx, "VF_ci_hi"] = s4_hi / s1_lo

        # HYB_ratio: hybrid (S5) relative to Algorithm 1 (S1) 
        cf5 = cf.get(5, float("nan"))
        if cf5 == cf5 and cf1 == cf1 and cf1 > 0:
            df.loc[idx, "HYB_ratio"] = cf5 / cf1
            s5_lo, s5_hi = ci_lo.get(5, float("nan")), ci_hi.get(5, float("nan"))
            s1_lo, s1_hi = ci_lo.get(1, float("nan")), ci_hi.get(1, float("nan"))
            if all(x == x for x in (s5_lo, s5_hi, s1_lo, s1_hi)) and s1_hi > 0 and s1_lo > 0:
                df.loc[idx, "HYB_ci_lo"] = s5_lo / s1_hi
                df.loc[idx, "HYB_ci_hi"] = s5_hi / s1_lo

    return df


# 6.  COMMUNITY AGGREGATOR

class CommunityAggregator:

    def __init__(self, base_dir: Path, logger: logging.Logger):
        self.base_dir = base_dir
        self.logger   = logger

    def run(self) -> Optional[pd.DataFrame]:
        self.logger.info("[Aggregator] Starting community-level SRE analysis...")
        t0 = time.monotonic()

        conditions: List[Tuple[float, float, int, float]] = []
        for rho in SwarmConfig.MARKOV_PERSISTENCE_LEVELS:
            rho_pct = int(round(rho * 100))

            for pen in SwarmConfig.swept_penetration_rates():
                pen_str = int(pen * 100)
                for inel in SwarmConfig.INELASTICITY_RATIOS:
                    inel_str = int(inel * 100)
                    for scen in SwarmConfig.active_scenarios():
                        d = (self.base_dir
                             / f"Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}"
                             / f"Scenario_{scen}")
                        if d.exists() and list(d.rglob(SwarmConfig.SIDECAR_FILENAME)):
                            conditions.append((pen, inel, scen, rho))

        if not conditions:
            self.logger.warning("[Aggregator] No completed conditions found.")
            return None

        self.logger.info(
            f"[Aggregator] Aggregating {len(conditions)} conditions in parallel...")

        ctx     = get_context("spawn")
        summary: List[Dict] = []

        with ProcessPoolExecutor(
                max_workers=min(8, len(conditions)),
                mp_context=ctx) as pool:
            future_map = {
                pool.submit(
                    _aggregate_one_condition,
                    str(self.base_dir), pen, inel, scen, rho): (pen, inel, scen, rho)
                for pen, inel, scen, rho in conditions
            }
            with tqdm(total=len(conditions), desc="Aggregating",
                      unit="cond", dynamic_ncols=True) as pbar:
                for fut in as_completed(future_map):
                    res  = fut.result()
                    cond = future_map[fut]
                    if res is not None:
                        summary.append(res)
                    else:
                        pen, inel, scen, rho = cond
                        self.logger.warning(
                            f"[Aggregator] Failed: "
                            f"Rho{int(rho*100)}_Pen{int(pen*100)}"
                            f"_Inel{int(inel*100)}_S{scen}")
                    pbar.update(1)

        if not summary:
            self.logger.error("[Aggregator] All conditions failed.")
            return None

        df = (pd.DataFrame(summary)
              .sort_values(["markov_rho", "pen_rate", "inel_rate", "scenario"])
              .reset_index(drop=True))
        df = _compute_ratios(df)

        out = self.base_dir / SwarmConfig.METRICS_FILENAME
        df.to_csv(out, index=False, float_format="%.6f")
        self.logger.info(f"[Aggregator] Summary -> {out}")

        try:
            if "lp_opf_status" in df.columns and "scenario" in df.columns:
                self.logger.info("[LP-OPF v12.2] success / fail by scenario:")
                for scen, sub in df.groupby("scenario", sort=True):
                    total = len(sub)
                    n_ok  = int((sub["lp_opf_status"] == "ok").sum())
                    fail_modes = sub.loc[sub["lp_opf_status"] != "ok",
                                         "lp_opf_status"].value_counts()
                    if total == n_ok:
                        self.logger.info(
                            f"  S{int(scen)}: {n_ok}/{total} ok")
                    else:
                        self.logger.warning(
                            f"  S{int(scen)}: {n_ok}/{total} ok — "
                            f"failures: {fail_modes.to_dict()}")
        except Exception as _exc:
            self.logger.warning(
                f"[LP-OPF v12.2] could not summarize success rate: {_exc}")

        try:
            if "L_community" in df.columns and "scenario" in df.columns:
                s1_df = df[df["scenario"] == 1]
                if len(s1_df) > 0:
                    L_vals = s1_df["L_community"].dropna()
                    if len(L_vals) > 0:
                        L_max  = float(L_vals.max())
                        L_med  = float(L_vals.median())
                        n_violate = int((L_vals >= 1.0).sum())
                        self.logger.info(
                            f"[Lyapunov v12.3] S1 community certificate "
                            f"L_community: median={L_med:.4f}, "
                            f"max={L_max:.4f} over {len(L_vals)} cells")
                        if n_violate > 0:
                            self.logger.warning(
                                f"[Lyapunov v12.3] {n_violate}/{len(L_vals)} "
                                f"S1 cells have L_community ≥ 1 - Algorithm-1 "
                                f"mitigation NOT guaranteed at those operating "
                                f"points.  Paper Section VI-E should disclose.")
                        else:
                            self.logger.info(
                                f"[Lyapunov v12.3] All S1 cells have "
                                f"L_community < 1 — mitigation guarantee "
                                f"empirically corroborated.")
                    else:
                        self.logger.warning(
                            "[Lyapunov v12.3] No L_community values for S1 - "
                            "check eplusout.err parsing (regex may have failed).")
        except Exception as _exc:
            self.logger.warning(
                f"[Lyapunov v12.3] could not summarize certificate: {_exc}")

        elapsed = time.monotonic() - t0
        self.logger.info(f"[Aggregator] Done in {elapsed:.1f}s.")

        self._log_highlights(df)

        if _SALIB_AVAILABLE:
            self._run_sobol(df)
        else:
            self.logger.info(
                "[Sobol] SALib not available. Install with: pip install SALib")

        self._export_latex(df)

        if _PLOTLY_AVAILABLE:
            self._export_html_report(df)
        else:
            self.logger.info(
                "[HTML] Plotly not available. Install with: pip install plotly")

        return df

    # Sobol 

    def _run_sobol(self, df: pd.DataFrame) -> None:
        
        self.logger.info("[Sobol] v10.0 SOTA - 3-factor (pen, inel, rho); "
                         "discrete-grid + GP-surrogate dual analysis.")

        sre_valid = df.dropna(subset=["SRE_ratio"]).copy()
        if "markov_rho" not in sre_valid.columns:
            self.logger.warning("[Sobol] markov_rho column missing - skipping.")
            return

        # Collapse to 36 unique condition cells (SRE_ratio is invariant to scenario)
        cond_df = (
            sre_valid.drop_duplicates(subset=["pen_rate", "inel_rate", "markov_rho"])
                     [["pen_rate", "inel_rate", "markov_rho", "SRE_ratio"]]
                     .reset_index(drop=True)
        )
        n_cells = len(cond_df)
        self.logger.info(f"[Sobol] Unique (pen, inel, rho) cells: {n_cells}")

        problem = {
            "num_vars": 3,
            "names":  ["pen_rate", "inel_rate", "markov_rho"],
            "bounds": [
                [SwarmConfig.EV_PENETRATION_RATES[0], SwarmConfig.EV_PENETRATION_RATES[-1]],
                [SwarmConfig.INELASTICITY_RATIOS[0],  SwarmConfig.INELASTICITY_RATIOS[-1]],
                [SwarmConfig.MARKOV_PERSISTENCE_LEVELS[0],
                 SwarmConfig.MARKOV_PERSISTENCE_LEVELS[-1]],
            ],
        }

        # Pass 1: discrete-grid Sobol via nearest-neighbour lookup
        pen_grid  = np.array(SwarmConfig.EV_PENETRATION_RATES)
        inel_grid = np.array(SwarmConfig.INELASTICITY_RATIOS)
        rho_grid  = np.array(SwarmConfig.MARKOV_PERSISTENCE_LEVELS)

        def _lookup_grid(pen: float, inel: float, rho: float) -> float:
            p = pen_grid[np.argmin(np.abs(pen_grid - pen))]
            i = inel_grid[np.argmin(np.abs(inel_grid - inel))]
            r = rho_grid[np.argmin(np.abs(rho_grid - rho))]
            mask = (
                (cond_df["pen_rate"]   == p)
                & (cond_df["inel_rate"]  == i)
                & (cond_df["markov_rho"] == r)
            )
            row = cond_df[mask]
            return float(row["SRE_ratio"].iloc[0]) if len(row) else float("nan")

        N1 = 512
        try:
            X1 = _sobol_sample(problem, N1, calc_second_order=False)
        except Exception as exc:
            self.logger.warning(f"[Sobol] Pass-1 sampling failed: {exc}")
            return
        Y1 = np.array([_lookup_grid(x[0], x[1], x[2]) for x in X1])
        valid1 = ~np.isnan(Y1)
        if valid1.sum() < 64:
            self.logger.warning(
                f"[Sobol] Pass-1 has only {int(valid1.sum())} valid samples — "
                "skipping discrete-grid analysis.")
        else:
            try:
                si1 = sobol_analyze.analyze(
                    problem, Y1, calc_second_order=False, print_to_console=False)
                rows1 = [
                    {"factor":  name,
                     "S1":      round(float(si1["S1"][j]),      4),
                     "S1_conf": round(float(si1["S1_conf"][j]), 4),
                     "ST":      round(float(si1["ST"][j]),      4),
                     "ST_conf": round(float(si1["ST_conf"][j]), 4)}
                    for j, name in enumerate(problem["names"])
                ]
                pass1_path = self.base_dir / "Sobol_Sensitivity_Indices.csv"
                pd.DataFrame(rows1).to_csv(pass1_path, index=False)
                self.logger.info(
                    f"[Sobol/Pass-1 discrete-grid] saved -> {pass1_path}\n"
                    + pd.DataFrame(rows1).to_string(index=False))
            except Exception as exc:
                self.logger.warning(f"[Sobol] Pass-1 analyze failed: {exc}")

        # Pass 2: GP-surrogate Sobol
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                RBF, ConstantKernel as _C_kernel, WhiteKernel,
            )
            from sklearn.model_selection import KFold
        except Exception:
            self.logger.info(
                "[Sobol/Pass-2] scikit-learn GP unavailable — surrogate skipped.")
            return

        X_train = cond_df[["pen_rate", "inel_rate", "markov_rho"]].values.astype(float)
        y_train = cond_df["SRE_ratio"].values.astype(float)

        kernel = (
            _C_kernel(1.0, (1e-3, 1e3))
            * RBF(length_scale=[0.3, 0.3, 0.3],
                  length_scale_bounds=(1e-3, 1e2))      
            + WhiteKernel(noise_level=1e-3,
                          noise_level_bounds=(1e-7, 1e0))  
        )
        gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            normalize_y=True,
            random_state=SwarmConfig.RANDOM_STATE,
        )

        import warnings as _warnings
        from sklearn.exceptions import ConvergenceWarning as _ConvWarn

        _conv_warn_count = {"n": 0}

        def _fit_with_quiet(estimator, X, y):
            with _warnings.catch_warnings(record=True) as wlist:
                _warnings.simplefilter("always", _ConvWarn)
                estimator.fit(X, y)
                _conv_warn_count["n"] += sum(
                    1 for w in wlist if issubclass(w.category, _ConvWarn))
            return estimator

        try:
            _fit_with_quiet(gp, X_train, y_train)
        except Exception as exc:
            self.logger.warning(f"[Sobol/Pass-2] GP fit failed: {exc}")
            return

        try:
            kf = KFold(n_splits=5, shuffle=True,
                       random_state=SwarmConfig.RANDOM_STATE)
            r2_folds = []
            for tr, te in kf.split(X_train):
                gcv = GaussianProcessRegressor(
                    kernel=kernel, n_restarts_optimizer=5,
                    normalize_y=True, random_state=SwarmConfig.RANDOM_STATE,
                )
                _fit_with_quiet(gcv, X_train[tr], y_train[tr])
                yp = gcv.predict(X_train[te])
                ss_res = float(np.sum((y_train[te] - yp) ** 2))
                ss_tot = float(np.sum((y_train[te] - y_train[te].mean()) ** 2))
                r2_folds.append(1.0 - ss_res / max(ss_tot, 1e-12))
            r2_mean, r2_std = float(np.mean(r2_folds)), float(np.std(r2_folds))
        except Exception:
            r2_mean = r2_std = float("nan")

        in_sample_r2 = float(gp.score(X_train, y_train))

        if _conv_warn_count["n"] > 0:
            self.logger.info(
                f"[Sobol/Pass-2] GP convergence: {_conv_warn_count['n']} fits "
                f"saturated a kernel bound (across main fit + 5 CV folds).  "
                f"Indices remain valid; the surrogate's flexibility is "
                f"intentionally bounded.")
        self.logger.info(
            f"[Sobol/Pass-2 GP surrogate] kernel={kernel} | "
            f"5-fold CV R² = {r2_mean:.3f} ± {r2_std:.3f} | "
            f"in-sample R² = {in_sample_r2:.3f}")

        if r2_mean < 0.30:
            self.logger.warning(
                f"[Sobol/Pass-2] CV R² = {r2_mean:.3f} < 0.30 — surrogate is "
                "untrustworthy.  Reporting indices but flagging as low-confidence.")

        N2 = 4096
        try:
            X2 = _sobol_sample(problem, N2, calc_second_order=True)
            y_surr = gp.predict(X2)
            si2 = sobol_analyze.analyze(
                problem, y_surr, calc_second_order=True, print_to_console=False)
        except Exception as exc:
            self.logger.warning(f"[Sobol/Pass-2] Saltelli/analyze failed: {exc}")
            return

        rows2 = [
            {"factor":  name,
             "S1":      round(float(si2["S1"][j]),      4),
             "S1_conf": round(float(si2["S1_conf"][j]), 4),
             "ST":      round(float(si2["ST"][j]),      4),
             "ST_conf": round(float(si2["ST_conf"][j]), 4)}
            for j, name in enumerate(problem["names"])
        ]
        rows2_df = pd.DataFrame(rows2)
        pass2_path = self.base_dir / "Sobol_Surrogate_Indices.csv"
        rows2_df.to_csv(pass2_path, index=False)

        # Second-order S2 matrix 
        s2_records = []
        names_ = problem["names"]
        for i, ni in enumerate(names_):
            for j_ in range(i + 1, len(names_)):
                nj = names_[j_]
                s2_records.append({
                    "factor_i": ni,
                    "factor_j": nj,
                    "S2":      round(float(si2["S2"][i][j_]),      4),
                    "S2_conf": round(float(si2["S2_conf"][i][j_]), 4),
                })
        s2_path = self.base_dir / "Sobol_Surrogate_S2_Interactions.csv"
        pd.DataFrame(s2_records).to_csv(s2_path, index=False)

        self.logger.info(
            f"[Sobol/Pass-2 surrogate] saved -> {pass2_path}\n"
            + rows2_df.to_string(index=False))
        self.logger.info(
            f"[Sobol/Pass-2 S2 interactions] saved -> {s2_path}\n"
            + pd.DataFrame(s2_records).to_string(index=False))

    # LaTeX export

    def _export_latex(self, df: pd.DataFrame) -> None:

        scen_labels = {
            0: "S0",
            1: r"\textbf{S1}",
            2: "S2",
            3: "S3",
            4: "S4",
        }
        lines = [
            r"% Auto-generated by Run_Manager.py v12.1",
            r"% IEEE Transactions on Smart Grid",
            r"% CF definition: simultaneous-snapshot (IEEE Std 141-1993, Red Book)",
            r"% SRE = CF(S2)/CF(S0); MIT = CF(S1)/CF(S2); VF = CF(S4)/CF(S1)",
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\caption{Community Coincidence Factor, Synchronisation Rebound Effect (SRE), and Annual Bill Components "
            r"across the sensitivity matrix. "
            r"CF computed as simultaneous-snapshot per IEEE Std 141-1993. "
            r"S0: uncontrolled; S1: Algorithm 1 (proposed); "
            r"S2: TOU rebound; S3: flat baseline; S4: valley-filling benchmark. "
            r"95\,\% BCa bootstrap confidence intervals in parentheses "
            r"(bias-corrected \& accelerated; Efron \& Tibshirani 1993). "
            r"SRE\,=\,CF(S2)/CF(S0); MIT\,=\,CF(S1)/CF(S2); VF\,=\,CF(S4)/CF(S1).}",
            r"\label{tab:sre_metrics}",
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{cccrrrrrrrr}",
            r"\toprule",
            r"$\rho_\mathrm{EV}$ & $\phi_\mathrm{inel}$ & S & "
            r"CF & SRE ratio & MIT ratio & VF ratio & "
            r"Ramp [kW/min] & $E_\mathrm{TOU}$ [\$] & "
            r"$D_\mathrm{demand}$ [\$] & $B_\mathrm{total}$ [\$] \\",
            r"\midrule",
        ]
        prev_pen = None
        _latex_sort = (
            ["markov_rho", "pen_rate", "inel_rate", "scenario"]
            if "markov_rho" in df.columns
            else ["pen_rate", "inel_rate", "scenario"]
        )
        for _, row in df.sort_values(_latex_sort).iterrows():
            pen  = int(row["pen_rate"]  * 100)
            inel = int(row["inel_rate"] * 100)
            scen = int(row["scenario"])
            if prev_pen is not None and pen != prev_pen:
                lines.append(r"\midrule")
            prev_pen = pen

            def _fmt(val, lo=None, hi=None, digits=3):
                s = f"{val:.{digits}f}" if val == val else "---"
                if lo is not None and hi is not None and lo == lo and hi == hi:
                    s += f" ({lo:.{digits}f}--{hi:.{digits}f})"
                return s

            cf_s  = _fmt(row["CF"], row.get("CF_ci_lo"), row.get("CF_ci_hi"))
            sre_s = _fmt(row.get("SRE_ratio", float("nan")), row.get("SRE_ci_lo"), row.get("SRE_ci_hi"))
            mit_s = _fmt(row.get("MIT_ratio", float("nan")), row.get("MIT_ci_lo"), row.get("MIT_ci_hi"))
            vf_s  = _fmt(row.get("VF_ratio",  float("nan")), row.get("VF_ci_lo"),  row.get("VF_ci_hi"))
            
            rmp_s = (f"{row['ramp_rate_kw_per_min']:.3f}" if row.get("ramp_rate_kw_per_min") == row.get("ramp_rate_kw_per_min") else "---")
            cst_s = (f"{row['E_cost_usd']:.0f}" if row.get("E_cost_usd") == row.get("E_cost_usd") else "---")
            dmd_s = (f"{row['demand_charge_usd']:.0f}" if row.get("demand_charge_usd") == row.get("demand_charge_usd") else "---")
            tot_s = (f"{row['total_bill_usd']:.0f}" if row.get("total_bill_usd") == row.get("total_bill_usd") else "---")

            s_lbl = scen_labels.get(scen, str(scen))
            cells = [f"{pen}\\%", f"{inel}\\%", s_lbl, cf_s, sre_s, mit_s, vf_s, rmp_s, cst_s, dmd_s, tot_s]
            if scen == 1:
                cells = [r"\textbf{" + c + r"}" for c in cells]
            lines.append(" & ".join(cells) + r" \\")

        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        tex_path = self.base_dir / "SRE_metrics_table.tex"
        tex_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info(f"[LaTeX] Table -> {tex_path}")

    #  HTML interactive report 

    def _export_html_report(self, df: pd.DataFrame) -> None:
        """4-panel Plotly report."""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "Coincidence Factor (CF) by Scenario — IEEE Std 141",
                    "SRE Ratio = CF(S2)/CF(S0)",
                    "MIT Ratio = CF(S1)/CF(S2)",
                    "VF Ratio = CF(S4)/CF(S1)",
                ),
                horizontal_spacing=0.12,
                vertical_spacing=0.16,
            )
            pen_labels  = [f"{int(p*100)}%" for p in SwarmConfig.EV_PENETRATION_RATES]
            scen_colors = {0: "#607D8B", 1: "#2196F3", 2: "#F44336", 3: "#9E9E9E", 4: "#4CAF50", 5: "#9C27B0"}
            scen_names  = {0: "S0 Uncontrolled", 1: "S1 Smart (Algo 1)", 2: "S2 TOU Rebound", 3: "S3 Flat Baseline", 4: "S4 Valley-Filling", 5: "S5 Hybrid (VF+LD+SP)"}
            inel_colors = ["#E91E63", "#FF9800", "#4CAF50"]

            for scen in SwarmConfig.active_scenarios():
                sub = df[df["scenario"] == scen].sort_values("pen_rate")
                y   = sub.groupby("pen_rate")["CF"].mean().reindex(SwarmConfig.EV_PENETRATION_RATES, fill_value=float("nan")).values
                fig.add_trace(go.Bar(name=scen_names[scen], x=pen_labels, y=y, marker_color=scen_colors[scen], legendgroup=f"s{scen}", showlegend=True), row=1, col=1)

            for j, inel in enumerate(SwarmConfig.INELASTICITY_RATIOS):
                sub = df[(df["inel_rate"] == inel) & df["SRE_ratio"].notna()].sort_values("pen_rate")
                y   = sub.groupby("pen_rate")["SRE_ratio"].mean().reindex(SwarmConfig.EV_PENETRATION_RATES, fill_value=float("nan")).values
                fig.add_trace(go.Scatter(name=f"inel={int(inel*100)}%", x=pen_labels, y=y, mode="lines+markers", line_color=inel_colors[j], legendgroup=f"inel{j}", showlegend=True), row=1, col=2)

            for j, inel in enumerate(SwarmConfig.INELASTICITY_RATIOS):
                sub = df[(df["inel_rate"] == inel) & df["MIT_ratio"].notna()].sort_values("pen_rate")
                y   = sub.groupby("pen_rate")["MIT_ratio"].mean().reindex(SwarmConfig.EV_PENETRATION_RATES, fill_value=float("nan")).values
                fig.add_trace(go.Scatter(name=f"inel={int(inel*100)}%", x=pen_labels, y=y, mode="lines+markers", line_color=inel_colors[j], legendgroup=f"inel{j}", showlegend=False), row=2, col=1)

            for j, inel in enumerate(SwarmConfig.INELASTICITY_RATIOS):
                sub = df[(df["inel_rate"] == inel) & df["VF_ratio"].notna()].sort_values("pen_rate")
                y   = sub.groupby("pen_rate")["VF_ratio"].mean().reindex(SwarmConfig.EV_PENETRATION_RATES, fill_value=float("nan")).values
                fig.add_trace(go.Scatter(name=f"inel={int(inel*100)}%", x=pen_labels, y=y, mode="lines+markers", line_color=inel_colors[j], legendgroup=f"inel{j}", showlegend=False), row=2, col=2)

            fig.add_hline(y=1.0, line_dash="dash", line_color="black", annotation_text="SRE=1 (no rebound)", row=1, col=2)
            fig.add_hline(y=1.0, line_dash="dot", line_color="green", annotation_text="VF=1 (Algo 1 ≡ Valley-Fill)", row=2, col=2)

            fig.update_layout(
                title=dict(text="IEEE Smart Grid — Community SRE Analysis<br><sup>Run_Manager v12.1 — SF Marine CZ3C, NEM 3.0, PG&E B-1, E+ 25.2/26.1</sup>", font=dict(size=15)),
                barmode="group", height=760, template="plotly_white", legend=dict(orientation="h", y=-0.1, xanchor="center", x=0.5),
            )
            html_path = self.base_dir / "SRE_interactive_report.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn")
            self.logger.info(f"[HTML] Report -> {html_path}")
        except Exception as exc:
            self.logger.warning(f"[HTML] Failed: {exc}")

    #  Console highlights

    def _log_highlights(self, df: pd.DataFrame) -> None:
        self.logger.info("=" * 70)
        self.logger.info("  COMMUNITY SRE ANALYSIS — KEY FINDINGS")
        self.logger.info("=" * 70)
        _log_cols = (["pen_rate", "inel_rate", "markov_rho"] if "markov_rho" in df.columns else ["pen_rate", "inel_rate"])
        for _log_key, grp in df.groupby(_log_cols):
            sre = grp["SRE_ratio"].dropna().mean()
            mit = grp["MIT_ratio"].dropna().mean()
            vf  = grp["VF_ratio"].dropna().mean()
            cfs = {s: grp.loc[grp["scenario"] == s, "CF"].mean() for s in SwarmConfig.active_scenarios()}
            dmd = grp["demand_charge_usd"].dropna().mean()
            tot = grp["total_bill_usd"].dropna().mean()
            pen, inel = _log_key[:2]
            rho_s = f"{_log_key[2]:.2f}" if len(_log_key) > 2 else "n/a"
            self.logger.info(
                f"  Rho={rho_s} | Pen={int(pen*100):>3}% | Inel={int(inel*100):>2}% | "
                f"CF(S1={cfs.get(1,0):.3f} S2={cfs.get(2,0):.3f}) | "
                f"SRE={sre:.3f} | MIT={mit:.3f} | VF={vf:.3f} | TotalBill=${tot:.0f}/yr")
        self.logger.info("=" * 70)


# 7.  SWARM MANAGER

class SensitivitySwarmManager:

    def __init__(self, max_threads: int, resume: bool = False):
        self.max_workers = max_threads
        self.resume      = resume
        self.fleet_map:    Dict[str, Any] = {}
        self.zones:        List[str]      = []
        self.config_files: List[Path]     = []

        SwarmConfig.BASE_OUT_DIR.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

        self.run_mode = SwarmConfig.assert_run_mode_consistent(self.logger)
        self.logger.info(
            f"[v14.2] arm='{SwarmConfig.ABLATION_ARM}' stagger='{self.run_mode['stagger_mode']}' "
            f"sim_seed={self.run_mode['sim_seed']} | "
            f"scenarios={self.run_mode['scenarios']} | "
            f"out={SwarmConfig.BASE_OUT_DIR.name} | "
            f"metrics={SwarmConfig.METRICS_FILENAME}")

    def _setup_logging(self) -> None:
        self.logger = logging.getLogger("SwarmManager")
        self.logger.setLevel(logging.DEBUG)
        if self.logger.handlers:
            return
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(
            SwarmConfig.BASE_OUT_DIR / "Swarm_Execution.log",
            mode="w", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self.logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        self.logger.addHandler(ch)

    def load_data(self) -> None:
        SwarmConfig.resolve()
        self.logger.info(f"EnergyPlus: {SwarmConfig.ENERGYPLUS_EXE}")

        for path, label in [
            (SwarmConfig.PLUGIN_FILE,  "OccupancyPlugin.py"),
            (SwarmConfig.WEATHER_FILE, "EPW weather file"),
        ]:
            if not path.exists():
                raise FileNotFoundError(f"{label} not found: {path}")

        if SwarmConfig.FLEET_DATA_FILE.exists():
            fleet_df = pd.read_csv(SwarmConfig.FLEET_DATA_FILE)
            self.fleet_map = fleet_df.set_index(
                fleet_df["House_ID"].str.lower().str.strip()).to_dict("index")
            self.logger.info(f"Loaded {len(self.fleet_map)} NHTS profiles.")
        else:
            self.logger.warning("NHTS_Fleet_Data.csv not found.")

        self.config_files = sorted(
            Path(".").glob("config_house_*.json"),
            key=lambda p: int(re.search(r"\d+", p.stem).group())
                          if re.search(r"\d+", p.stem) else 0)
        if not self.config_files:
            raise ValueError("No config_house_*.json files found.")
        self.logger.info(f"Discovered {len(self.config_files)} config files.")

        if not ConfigValidator.validate_all(self.config_files, self.logger):
            raise ValueError("Config validation failed.")

        self._build_idf()
        ConfigValidator.validate_idf(SwarmConfig.FINAL_IDF, self.zones, self.logger)

        total = (len(self.config_files)
                 * len(SwarmConfig.active_scenarios())
                 * len(SwarmConfig.swept_penetration_rates())
                 * len(SwarmConfig.INELASTICITY_RATIOS)
                 * len(SwarmConfig.MARKOV_PERSISTENCE_LEVELS))
        self.logger.info(f"Sensitivity matrix: {total} total runs.")

    # Archetype IDF modification helper

    @staticmethod
    def _apply_archetype(base_content: str, arch: Dict[str, Any]) -> str:

        c = base_content

        # 1. North Axis - replace numeric field in Building object
        c = re.sub(
            r'(\s+)([\d.]+)(,\s*!-\s*North Axis \{deg\})',
            lambda m: f"{m.group(1)}{arch['north_axis']}.{m.group(3)}",
            c, count=1)

        # 2. PV active fraction
        c = re.sub(
            r'(\s+)([\d.]+)(,\s*!-\s*Fraction of Surface Area with Active Solar Cells)',
            lambda m: f"{m.group(1)}{arch['pv_frac']}{m.group(3)}",
            c, count=1)

        # 3. Wall insulation - IN02 thickness (3rd field after object name)
        c = re.sub(
            r'(IN02,[^\n]*\n\s*\w+,[^\n]*\n\s*)([-+0-9.Ee]+)(,)',
            lambda m: f"{m.group(1)}{arch['in02_thick']:.4f}{m.group(3)}",
            c, count=1)

        # 4. Attic/ceiling insulation --- IN05 thickness (SingleFamilyHouse CEILING:LIVING)
        c = re.sub(
            r'(IN05,[^\n]*\n\s*\w+,[^\n]*\n\s*)([-+0-9.Ee]+)(,)',
            lambda m: f"{m.group(1)}{arch['in05_thick']:.4f}{m.group(3)}",
            c, count=1)

        # 5. Infiltration flow/zone scaling
        scale_i = arch['infil_scale']
        c = re.sub(
            r'(ZoneInfiltration:DesignFlowRate,[\s\S]*?flow/zone,[^\n]*\n\s*)([-+0-9.Ee]+)(,)',
            lambda m: f"{m.group(1)}{float(m.group(2)) * scale_i:.6f}{m.group(3)}",
            c)

        # 6. Lights design level
        scale_l = arch['load_scale']
        c = re.sub(
            r'(LightingLevel,[^\n]*\n\s*)([-+0-9.Ee]+)(,)',
            lambda m: f"{m.group(1)}{round(float(m.group(2)) * scale_l)}{m.group(3)}",
            c)

        # 6b. Equipment design level
        c = re.sub(
            r'(EquipmentLevel,[^\n]*\n\s*)([-+0-9.Ee]+)(,)',
            lambda m: f"{m.group(1)}{round(float(m.group(2)) * scale_l)}{m.group(3)}",
            c)

        return c

    def _build_idf(self) -> None:
        """
        Compile 5 archetype IDF files from the source template.
        """
        if not SwarmConfig.SOURCE_IDF.exists():
            raise FileNotFoundError(f"Source IDF missing: {SwarmConfig.SOURCE_IDF}")

        base_content = SwarmConfig.SOURCE_IDF.read_text(
            encoding="utf-8", errors="replace")

        # Zone name extraction
        stripped = [re.sub(r"!.*", "", line) for line in base_content.splitlines()]
        tokens   = re.sub(r"\s+", " ", " ".join(stripped))

        zone_matches = re.findall(
            r"People\s*,\s*[^,]+\s*,\s*([^,;]+)", tokens, re.I)
        self.zones = sorted({z.strip() for z in zone_matches if z.strip()})
        if not self.zones:
            self.zones = [
                "SPACE1-1", "SPACE2-1", "SPACE3-1", "SPACE4-1", "SPACE5-1"]
            self.logger.warning(
                f"Zone autodetection failed; using fallback: {self.zones}")
        else:
            self.logger.info(f"Detected zones: {self.zones}")

        # Strip duplicate control objects
        stripped_content = re.sub(r"(?is)SimulationControl\s*,[^;]*;", "", base_content)
        stripped_content = re.sub(r"(?is)RunPeriod\s*,[^;]*;", "", stripped_content)

        # Common injected block
        injected = (
            "\n! --- INJECTED BY Run_Manager.py v9.5 ---\n"
            "SimulationControl, Yes, Yes, No, No, Yes, No, 1;\n"
            "RunPeriod, Annual_Run, 1,1,,12,31,,Tuesday,Yes,Yes,No,Yes,Yes;\n\n"
            "Output:Variable,*,Site Outdoor Air Drybulb Temperature,Timestep;\n"
            "Output:Variable,*,Site Diffuse Solar Radiation Rate per Area,Timestep;\n"
            "Output:Variable,*,Site Direct Solar Radiation Rate per Area,Timestep;\n"
            "Output:Variable,*,Site Solar Altitude Angle,Timestep;\n"
            "Output:Variable,*,Zone Air Temperature,Timestep;\n"
            "Output:Variable,*,Zone People Occupant Count,Timestep;\n"
            "Output:Variable,*,Zone Electric Equipment Electricity Rate,Timestep;\n"
            "Output:Variable,*,Facility Net Purchased Electricity Rate,Timestep;\n"
            "Output:Variable,*,Facility Total Produced Electricity Rate,Timestep;\n"
            "Output:Variable,*,Generator Produced DC Electricity Rate,Timestep;\n"
            "Output:Variable,*,Electric Storage Battery Charge State,Timestep;\n"
            "Output:Variable,*,Electric Storage Charge Power,Timestep;\n"
            "Output:Variable,*,Electric Storage Discharge Power,Timestep;\n"
            "! -------------------------------------------\n"
        )

        # Build 5 archetype IDF files
        SwarmConfig.ARCHETYPE_IDF_PATHS = []
        for i, arch in enumerate(ARCHETYPES):
            arch_content = self._apply_archetype(stripped_content, arch)
            arch_header = (
                f"!  Archetype {arch['label']} (Run_Manager v6.0 FIX-ARCHETYPE) \n"
                f"! {arch['description']}\n"
                f"! N-axis={arch['north_axis']}°  PV={arch['pv_frac']}  "
                f"IN02={arch['in02_thick']}m  IN05={arch['in05_thick']}m  "
                f"infil×{arch['infil_scale']}  loads×{arch['load_scale']}\n"
                "! Zone names and actuator handles identical to source IDF.\n"
            )
            final = arch_header + arch_content + "\n" + injected
            arch_idf = SwarmConfig.SOURCE_IDF.parent / f"HVACTemplate-5ZonePTHP_A{i}.idf"
            arch_idf.write_text(final, encoding="utf-8")
            SwarmConfig.ARCHETYPE_IDF_PATHS.append(arch_idf)
            kb = arch_idf.stat().st_size // 1024
            self.logger.info(
                f"  [Archetype A{i}] {arch['label']:28s} -> {arch_idf.name} ({kb} KB)")

        # FINAL_IDF = A0 (backward-compatibility alias)
        SwarmConfig.FINAL_IDF = SwarmConfig.ARCHETYPE_IDF_PATHS[0]
        self.logger.info(
            f"[IDF] {len(ARCHETYPES)} archetype IDFs compiled. "
            f"FINAL_IDF alias -> A0 ({SwarmConfig.FINAL_IDF.name})")

    def execute_swarm(self) -> None:

        total = (len(self.config_files)
                 * len(SwarmConfig.active_scenarios())
                 * len(SwarmConfig.swept_penetration_rates())
                 * len(SwarmConfig.INELASTICITY_RATIOS)
                 * len(SwarmConfig.MARKOV_PERSISTENCE_LEVELS))

        self.logger.info(
            f"Dispatching {total} runs "
            f"({len(SwarmConfig.MARKOV_PERSISTENCE_LEVELS)} ρ-levels × "
            f"{len(SwarmConfig.swept_penetration_rates())} pen × "
            f"{len(SwarmConfig.INELASTICITY_RATIOS)} inel × "
            f"{len(SwarmConfig.active_scenarios())} scenarios × "
            f"{len(self.config_files)} houses) | "
            f"{self.max_workers} workers | "
            f"stagger='{self.run_mode['stagger_mode']}' | "
            f"completed runs are always skipped")

        t0     = time.monotonic()
        counts = defaultdict(int)

        manifest_path = SwarmConfig.BASE_OUT_DIR / SwarmConfig.MANIFEST_FILENAME
        manifest_fh   = manifest_path.open("w", encoding="utf-8")

        # Build job list
        jobs: List[Dict] = []
        for markov_rho in SwarmConfig.MARKOV_PERSISTENCE_LEVELS:
            rho_pct = int(round(markov_rho * 100))
            for pen_idx, sweep_pen in enumerate(SwarmConfig.swept_penetration_rates()):

                if SwarmConfig.BATTERY_PEN_SWEEP:
                    ev_pen, batt_pen = SwarmConfig.BATTERY_PEN_SWEEP_EV_PEN, sweep_pen

                    ev_seed_idx = 900
                else:
                    ev_pen, batt_pen = sweep_pen, 1.0
                    ev_seed_idx = pen_idx
                # EV assignment (dedicated stream, as before).
                rng   = random.Random(GLOBAL_RANDOM_SEED + ev_seed_idx)
                n_evs = int(round(len(self.config_files) * ev_pen))
                ev_set: Set[Path] = set(rng.sample(self.config_files, n_evs))

                batt_rng  = random.Random(GLOBAL_RANDOM_SEED + 7919 + pen_idx)
                n_batt    = int(round(len(self.config_files) * batt_pen))
                batt_set: Set[Path] = set(batt_rng.sample(self.config_files, n_batt))

                for inel_rate in SwarmConfig.INELASTICITY_RATIOS:
                    for scenario in SwarmConfig.active_scenarios():
                        for cfg in self.config_files:
                            house_id  = cfg.stem.replace("config_", "").lower()

                            pen_str   = int(sweep_pen * 100)
                            inel_str  = int(inel_rate * 100)
                            
                            house_dir = (SwarmConfig.BASE_OUT_DIR
                                         / f"Rho{rho_pct}_Pen{pen_str}_Inel{inel_str}"
                                         / f"Scenario_{scenario}"
                                         / house_id)
                                         
                            m = re.search(r"(\d+)", house_id)
                            fkey = (f"house_{int(m.group(1)) - 1}"
                                    if m else house_id)
                                    
                            # assign archetype by house index
                            house_num   = int(m.group(1)) if m else 0
                            arch_idx    = house_num % len(ARCHETYPES)
                            arch_idf    = (str(SwarmConfig.ARCHETYPE_IDF_PATHS[arch_idx])
                                           if SwarmConfig.ARCHETYPE_IDF_PATHS
                                           else str(SwarmConfig.FINAL_IDF))
                                           
                            jobs.append({
                                "config_path_str":  str(cfg),
                                "scenario":         scenario,
                                "pen_rate":         sweep_pen,
                                "inel_rate":        inel_rate,
                                "has_ev":           cfg in ev_set,
                                "has_battery":      cfg in batt_set,
                                "zones":            self.zones,
                                "fleet_entry":      self.fleet_map.get(fkey),
                                "house_dir_str":    str(house_dir),
                                "retry_number":     0,
                                "archetype_idf_str": arch_idf,
                                "markov_rho":        markov_rho,
                                "ep_exe_str":   str(SwarmConfig.ENERGYPLUS_EXE),
                                "weather_str":  str(SwarmConfig.WEATHER_FILE),
                                "plugin_str":   str(SwarmConfig.PLUGIN_FILE),
                                "timeout_s":    SwarmConfig.PROCESS_TIMEOUT_S,
                                "sidecar_name": SwarmConfig.SIDECAR_FILENAME,
                            })

        ctx     = get_context("spawn")
        retry_q: List[Dict] = []

        try:
            with ProcessPoolExecutor(
                    max_workers=self.max_workers,
                    mp_context=ctx) as pool:

                with tqdm(total=total, desc="Sensitivity Matrix",
                          unit="run", dynamic_ncols=True, colour="green") as pbar:

                    futures = {pool.submit(_worker_task, **j): j for j in jobs}

                    for fut in as_completed(futures):
                        res    = fut.result()
                        status = res["status"]
                        counts[status] += 1
                        manifest_fh.write(json.dumps(res) + "\n")
                        manifest_fh.flush()

                        if status == "error":
                            self.logger.debug(
                                f"FAIL [{res.get('reason','?')}] {res['label']}")
                            orig = futures[fut]
                            if orig["retry_number"] < SwarmConfig.MAX_RETRIES:
                                rj = dict(orig)
                                rj["retry_number"] += 1
                                retry_q.append(rj)

                        pbar.set_postfix(
                            ok=counts["success"],
                            skip=counts["skipped"],
                            fail=counts["error"],
                            refresh=False)
                        pbar.update(1)

            if retry_q:
                self.logger.info(
                    f"Retrying {len(retry_q)} failed runs "
                    f"(timeout={SwarmConfig.PROCESS_TIMEOUT_S * 2}s)...")
                with ProcessPoolExecutor(
                        max_workers=max(1, self.max_workers // 2),
                        mp_context=ctx) as pool:
                    with tqdm(total=len(retry_q), desc="Retry Pass",
                              unit="run", dynamic_ncols=True, colour="yellow") as pbar:
                        for fut in as_completed(
                                {pool.submit(_worker_task, **j): j
                                 for j in retry_q}):
                            res = fut.result()
                            counts["error"] -= 1
                            counts[res["status"]] += 1
                            manifest_fh.write(
                                json.dumps({**res, "is_retry": True}) + "\n")
                            manifest_fh.flush()
                            pbar.update(1)

        finally:
            manifest_fh.close()
            
        elapsed_min = (time.monotonic() - t0) / 60.0

        print("\n" + "=" * 65)
        print("  SWARM EXECUTION COMPLETE")
        print("=" * 65)
        print(f"  Wall-clock time   : {elapsed_min:.2f} minutes")
        print(f"  Successful runs   : {counts['success']}")
        print(f"  Skipped (cached)  : {counts['skipped']}")
        print(f"  Failed runs       : {counts['error']}")
        print(f"  Manifest (JSONL)  : {manifest_path}")
        if counts["error"]:
            print("  See Swarm_Execution.log for failure details.")
        else:
            print("  Zero failures. Perfect run.")
        print("=" * 65)

    def run_community_aggregation(self) -> None:
        CommunityAggregator(SwarmConfig.BASE_OUT_DIR, self.logger).run()


# 8.  ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IEEE Smart Grid — Sensitivity Swarm Manager v12.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Run_Manager.py                          # Full run, all CPUs-1
  python Run_Manager.py --resume                 # (no-op: see below)
  python Run_Manager.py --aggregate-only         # Aggregate existing results
  python Run_Manager.py --threads 8              # Fixed parallelism
  python Run_Manager.py --ep-exe /path/to/ep     # Override EP path
  python Run_Manager.py --timeout 7200           # 2-hr per-run timeout

i.i.d. stagger ablation (Supplementary Material S10):
  The ablation is PAIRED: run both arms, identical seeds, one difference.
    Arm A (control) : plugin ALGO1_STAGGER_MODE="vdc", ABLATION_ARM="vdc"
    Arm B (test)    : plugin ALGO1_STAGGER_MODE="iid", ABLATION_ARM="iid"
    -> python Run_Manager.py     # 1,800 runs each -> Results_Ablation_{VDC,IID}/
    -> python analyze_iid_ablation.py    # paired stats + CF decomposition
  Both switches must agree; the manager aborts at startup if they do not.
  Reset BOTH to the canonical arm ("off" / "vdc") before rebuilding the artifact.
""")

    parser.add_argument(
        "--threads", type=int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="Parallel workers (default: CPU count - 1)")
    parser.add_argument(
        "--resume", action="store_true",
        help="Deprecated no-op: runs with a complete eplusout.end + sidecar are "
             "ALWAYS skipped, with or without this flag (the two-phase S5 "
             "procedure depends on that). Retained for backward compatibility.")
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Skip simulations; aggregate existing results only")
    parser.add_argument(
        "--ep-exe", type=str, default=None,
        help="Explicit EnergyPlus executable path (overrides auto-detect)")
    parser.add_argument(
        "--no-retry", action="store_true",
        help="Disable automatic retry of failed runs")
    parser.add_argument(
        "--timeout", type=int, default=SwarmConfig.PROCESS_TIMEOUT_S,
        help=f"Per-run timeout seconds (default: {SwarmConfig.PROCESS_TIMEOUT_S})")

    args, _ = parser.parse_known_args()

    if args.ep_exe:
        SwarmConfig.ENERGYPLUS_EXE = Path(args.ep_exe).resolve()
    if args.no_retry:
        SwarmConfig.MAX_RETRIES = 0
    SwarmConfig.PROCESS_TIMEOUT_S = args.timeout

    try:
        manager = SensitivitySwarmManager(
            max_threads=args.threads,
            resume=args.resume)

        if not args.aggregate_only:
            manager.load_data()
            print(f"\n--> Stage 3: Executing {args.threads}-process swarm...")
            manager.execute_swarm()
        else:
            manager._setup_logging()
            manager.logger.info("--aggregate-only: skipping simulation stage.")

        print("\n--> Stage 4: Running community SRE aggregation...")
        manager.run_community_aggregation()

        print("\nPipeline complete.")
        print(f"  Results : {SwarmConfig.BASE_OUT_DIR / SwarmConfig.METRICS_FILENAME}")
        print(f"  LaTeX   : {SwarmConfig.BASE_OUT_DIR / 'SRE_metrics_table.tex'}")
        if _PLOTLY_AVAILABLE:
            print(f"  Report  : {SwarmConfig.BASE_OUT_DIR / 'SRE_interactive_report.html'}")
        print("  Next    : Run IEEE_Sensitivity_Analysis.py for publication figures.")

    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results in {SwarmConfig.BASE_OUT_DIR}.")
        sys.exit(130)
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        traceback.print_exc()
        sys.exit(1)