from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

from pyenergyplus.plugin import EnergyPlusPlugin


# 1. MODULE-LEVEL CONSTANTS  (unchanged from v9.5)

# PG&E E-TOU-C residential tariff
RATE_PEAK_SUMMER:    float = 0.52240   # Jun 1-Sep 30, peak 16:00-21:00 (4-9 PM)
RATE_OFFPEAK_SUMMER: float = 0.39940   # Jun 1-Sep 30, all other hours
RATE_PEAK_WINTER:    float = 0.39757   # Oct 1-May 31, peak 16:00-21:00 (4-9 PM)
RATE_OFFPEAK_WINTER: float = 0.36757   # Oct 1-May 31, all other hours
PEAK_START_H:        int   = 16        # 4 PM, first peak hour
PEAK_END_H:          int   = 20        # last peak hour (21:00 reverts to off-peak)
# Reference off-peak rate (annual mean of the seasonal off-peak rates)
RATE_OFF_PEAK:       float = (RATE_OFFPEAK_SUMMER + RATE_OFFPEAK_WINTER) / 2.0
EXPORT_RATE_NEM:     float = 0.038      # NEM 3.0 net billing avoided-cost export

# S3 (Flat-tariff control) energy rate = time-weighted average of the seasonal
FLAT_TARIFF_SUMMER:  float = (5.0 * RATE_PEAK_SUMMER + 19.0 * RATE_OFFPEAK_SUMMER) / 24.0
FLAT_TARIFF_WINTER:  float = (5.0 * RATE_PEAK_WINTER + 19.0 * RATE_OFFPEAK_WINTER) / 24.0
FLAT_TARIFF_RATE:    float = (FLAT_TARIFF_SUMMER + FLAT_TARIFF_WINTER) / 2.0  # annual mean (compat)

# CAISO grid-average emissions intensity (gCO2/kWh), annual representative day.
CO2_BY_HOUR: Dict[int, float] = {
     0: 296,  1: 287,  2: 280,  3: 276,  4: 276,  5: 289,
     6: 303,  7: 282,  8: 235,  9: 179, 10: 143, 11: 125,
    12: 116, 13: 113, 14: 120, 15: 150, 16: 205, 17: 277,
    18: 339, 19: 376, 20: 362, 21: 341, 22: 323, 23: 307,
}
CO2_CHARGE_GATE: float = 300.0

# Markov Chain
MARKOV_PERSISTENCE: float = 0.60
ZONE_DEVIATE_P:     float = 0.25

# Battery (Powerwall 2 nominal spec)
BATTERY_CAPACITY_WH:      float = 13_500.0
BATTERY_POWER_W:          float = 5_000.0
BATTERY_INIT_SOC_FRAC:    float = 0.50
BATTERY_MIN_SOC_FRAC:     float = 0.10
BATTERY_MAX_SOC_FRAC:     float = 0.95
BATTERY_ETA_CHARGE:       float = 0.96
BATTERY_ETA_DISCHARGE:    float = 0.96

# GLOBAL STORAGE MASTER SWITCH. 
ENABLE_BATTERY: bool = True

# Battery DoD-weighted Wohler degradation
BATT_WOHLER_A:     float = 7.543e-4
BATT_WOHLER_GAMMA: float = 1.671
BATT_WOHLER_N_REF: float = 3_500.0

# PV (IEC 61215 monocrystalline silicon)
PV_AREA_M2:             float = 20.0
PV_ETA_STC:             float = 0.18
PV_TEMP_COEFF:          float = 0.004
PV_NOCT_C:              float = 45.0
PV_SYSTEM_DERATE:       float = 0.88  
SOLAR_THRESHOLD_W_M2:   float = 25.0

# EV hardware (Level-2 SAE J1772 + bidirectional V2G)
EV_CHARGE_POWER_KW: float = 7
EV_CHARGER_POWER_W: float = EV_CHARGE_POWER_KW * 1000.0
EV_CAPACITY_KWH:    float = 50.0
EV_ETA_CHARGE:      float = 0.95
EV_ETA_V2G:         float = 0.92
EV_MIN_V2G_SOC_KWH: float = 10.0
EV_CHARGER_KW_NET:  float = (EV_CHARGER_POWER_W / 1000.0) * EV_ETA_CHARGE

# Disabled to match the stated charge-only design.
ENABLE_V2G: bool = False

# NHTS-derived EV SoC model
EV_EPA_CONSUMPTION_MI_PER_KWH: float = 3.5
EV_SOC_MIN_ARRIVAL:            float = 0.10
EV_NHTS_MEDIAN_VMT_MILES:      float = 37.5

# Algorithm 1 - anchor times for the decentralized stagger
ALGO1_PRICE_DROP_START_H: float = 21.0   # 21:00 - on-peak to mid-peak edge,
                                          # the largest single price drop
                                          # of the day; Algorithm-1 stagger
                                          # window anchor.
TARIFF_OFF_PEAK_START_H:  float = 22.0   # 22:00 - mid-peak to off-peak edge
                                          # per PG&E E-TOU-C 2024 (matches
                                          # RATE_OFF_PEAK schedule above).
ALGO1_OFF_PEAK_START_H:  float = ALGO1_PRICE_DROP_START_H

ALGO1_MORNING_ROLLOVER_H: float = 12.0

ALGO1_MAX_DELAY_MIN:    int   = 60          
EMPIRICAL_DEPARTURE_H:         float = 7.0
EMPIRICAL_DEPARTURE_H_WEEKEND: float = 9.5

# At-home state machine timing parameters
EV_CALENDAR_JITTER_H:        float = 0.25  # ± uniform jitter on arrival/departure
EV_DEADLINE_TARGET_FRAC:     float = 1.0   # SoC fraction expected at departure

# window-aware stagger and decentralized rate modulation
ALGO1_TAU_SAFETY_H:   float = 0.25     # safety margin against SoC/weather noise
ALGO1_TAU_MAX_HARD_H: float = 8.0      # hard cap τ_max 

ALGO1_ADAPTIVE_WINDOW: bool  = True
ALGO1_TAU_MAX_ADAPT_H: float = 11.0    # upper bound on the adaptive slack (h);

ALGO1_DRM_ENABLE:     bool  = True     # enable continuous rate modulation
ALGO1_DRM_MIN_RATE_FRAC: float = 0.10  # minimum charger rate (avoids stalling)
ALGO1_PRICE_GATE_HI:  float = 1.4      # local shadow-price throttle threshold
ALGO1_PRICE_GATE_LO:  float = 1.0      # full-rate restore threshold
ALGO1_PRICE_ALPHA:    float = 0.6      # weight on price in shadow signal
ALGO1_PRICE_BETA:     float = 0.4      # weight on CO2 in shadow signal

ALGO1_BROADCAST_GAMMA: float = 0.70    # throttle strength applied to the load EXCESS
#                                        over a house's threshold; tunable.
ALGO1_BROADCAST_FLOOR: float = 0.30    # floor on the S5 battery-recharge rate under the
#                                        broadcast throttle (matches the EV arbiter floor).
ALGO1_BROADCAST_THETA_MIN: float = 0.15  # lowest per-house back-off threshold (fraction
#                                          of the community peak at which this home starts
#                                          throttling). Homes here back off aggressively.
ALGO1_BROADCAST_THETA_MAX: float = 0.85  # highest per-house threshold; these homes throttle
#                                          only near the community peak, and only weakly.
ALGO1_BROADCAST_THETA_BASE: int   = 3    # van der Corput base for the threshold spread;
#                                          base 3 is orthogonal to the base-2 stagger.

# LAYER-1 CONSTRUCTION SWITCH (i.i.d. ABLATION ARM)
ALGO1_STAGGER_MODE: str = "vdc"   # "vdc" (deployed default) , "iid" 
ALGO1_IID_SEED:     int = 42      # follows the RANDOM_STATE = 42 sweep convention

SIM_SEED: int = 42

ALGO1_V2G_RESERVE_KWH: float = 5.0     # extra SoC kept beyond MIN for departure rescue

# Valley-Filling (Scenario 4)
VALLEY_START_H:   float = 0.0
VALLEY_END_H:     float = 6.0

ENABLE_SOLAR_CHARGE:  bool  = False
SOLAR_CHARGE_START_H: float = 10.0   # PV meaningfully available; concentrates draw at high irradiance

# V2G off-site cost 
V2G_OFFSITE_RATE_USD_KWH: float = 0.25
V2G_OFFSITE_ETA:          float = EV_ETA_CHARGE

# Occupancy & lighting
WATTS_PER_PERSON:    float = 135.0
BASE_EQUIP_LOAD_W:   float = 45.0
MAX_LIGHTS_W:        float = 200.0
LIGHTS_MIN_FRAC:     float = 0.12
SMOOTHING_ALPHA:     float = 0.35

# HVAC fixed-schedule pre-cooling
PRECOOL_HOUR_START: int   = 14
PRECOOL_HOUR_END:   int   = 16
PRECOOL_TEMP_C:     float = 24.0
PRECOOL_COOL_SP_C:  float = 21.5
PRECOOL_HEAT_SP_C:  float = 19.0

# Telemetry — v9.6 adds building_w column
SIDECAR_FILENAME: str  = "house_telemetry.csv"
SIDECAR_FLUSH_N:  int  = 24
SIDECAR_HEADER: List[str] = [
    "day", "hour", "minute",
    "price_usd_kwh", "co2_g_kwh", "is_peak",
    "outdoor_temp_c", "irradiance_w_m2",
    "occupancy_people",
    "building_w",            # HVAC+lights+equip from E+ meter
    "pv_w", "ev_w", "battery_w", "net_grid_w",
    "ev_soc_kwh", "battery_soc_wh", "battery_soh_frac",
    "hvac_precool", "scenario",
]


# 2. PURE-FUNCTION HELPERS

def _nhts_soc_from_distance(distance_miles: float, capacity_kwh: float) -> float:
    """NHTS trip distance to EV arrival SoC."""
    if capacity_kwh <= 0.0:
        return EV_SOC_MIN_ARRIVAL
    energy_used = distance_miles / EV_EPA_CONSUMPTION_MI_PER_KWH
    soc = 1.0 - energy_used / capacity_kwh
    return max(EV_SOC_MIN_ARRIVAL, min(1.0, soc))


def _get_tariff(hour: int, month: int) -> Tuple[float, bool]:
    """PG&E E-TOU-C 2024: on-peak 17–20, mid-peak 16 & 21, else off-peak."""
    is_summer = 6 <= month <= 9
    is_peak   = PEAK_START_H <= hour <= PEAK_END_H
    if is_summer:
        return (RATE_PEAK_SUMMER if is_peak else RATE_OFFPEAK_SUMMER), is_peak
    return (RATE_PEAK_WINTER if is_peak else RATE_OFFPEAK_WINTER), is_peak


def _get_tariff_for_scenario(hour: int, scenario_mode: int, month: int) -> Tuple[float, bool]:

    if scenario_mode == 3:
        # Flat tariff: no within-day price variation, no peak window.
        is_summer = 6 <= month <= 9
        return (FLAT_TARIFF_SUMMER if is_summer else FLAT_TARIFF_WINTER), False
    return _get_tariff(hour, month)


def pv_power_w(irradiance_w_m2: float, outdoor_temp_c: float) -> float:
    """IEC 61215 / Faiman cell-temperature + linear α_Pmax derating."""
    if irradiance_w_m2 < SOLAR_THRESHOLD_W_M2:
        return 0.0
    t_cell   = outdoor_temp_c + irradiance_w_m2 * (PV_NOCT_C - 20.0) / 800.0
    derating = 1.0 - PV_TEMP_COEFF * max(0.0, t_cell - 25.0)
    return max(0.0, PV_AREA_M2 * PV_ETA_STC * derating * PV_SYSTEM_DERATE * irradiance_w_m2)


def van_der_corput(n: int, base: int = 2) -> float:

    q, denom = 0.0, 1.0
    while n > 0:
        denom *= base
        q += (n % base) / denom
        n //= base
    return q


def _iid_uniform_frac(house_slot_idx: int,
                      day_count:      int,
                      seed:           int = ALGO1_IID_SEED) -> float:

    MASK64 = 0xFFFFFFFFFFFFFFFF
    z = (seed           * 0x9E3779B97F4A7C15
         + house_slot_idx * 0xBF58476D1CE4E5B9
         + day_count      * 0x94D049BB133111EB) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z = z ^ (z >> 31)
    return z / 2.0 ** 64


def _stable_house_seed(house_id: str, sim_seed: int = SIM_SEED) -> int:

    payload = f"{int(sim_seed)}:{house_id}".encode("utf-8")
    digest  = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2 ** 31)


def build_transition_matrix(
    pi: List[float], rho: float = MARKOV_PERSISTENCE
) -> List[List[float]]:
    """Persistence-mixture: T[i][j] = ρ·δ(i,j) + (1−ρ)·π[j]."""
    n = len(pi)
    T = []
    for i in range(n):
        row = [(rho + (1.0 - rho) * pi[j]) if i == j
               else ((1.0 - rho) * pi[j])
               for j in range(n)]
        s = sum(row)
        T.append([v / s for v in row])
    return T


def markov_sample(rng: random.Random, T: List[List[float]], prev: int) -> int:
    """Sample next Markov state from transition row T[prev]."""
    return rng.choices(range(len(T[prev])), weights=T[prev], k=1)[0]


# 3. MAIN PLUGIN CLASS

class OccupancyPlugin(EnergyPlusPlugin):

    _NAN = float("nan")

    def __init__(self) -> None:
        super().__init__()
        self._initialised: bool = False
        self.do_setup:     bool = True

        # Scenario flags
        self.scenario_mode:      int   = 1

        self._broadcast_profile: Dict[int, float] = {}
        self._broadcast_peak_kw: float = 0.0

        self._broadcast_theta: float = 0.5
        self.has_ev:             bool  = True

        self.has_battery:        bool  = True
        self.is_elastic:         bool  = True
        self.inelasticity_ratio: float = 0.30
        self.smart_delay_min:    int   = 0

        self._house_slot_idx:    int   = 0

        self._day_count_for_vdc: int   = 0

        # NHTS EV profile
        self.forced_arrival_h:  float = -1.0
        self.forced_soc_frac:   float = 0.786
        self.ev_distance_miles: float = EV_NHTS_MEDIAN_VMT_MILES
        self.ev_capacity_kwh:   float = EV_CAPACITY_KWH

        # House-level state
        self.markov_rho:             float = MARKOV_PERSISTENCE
        self.ev_soc_kwh:             float = 0.0
        self.ev_arrival_h:           float = -1.0

        self.ev_at_home:             bool  = True
        self._dep_fired_today:       bool  = False
        self._arr_fired_today:       bool  = False
        self.ev_departure_h_today:   float = EMPIRICAL_DEPARTURE_H
        self.ev_arrival_h_today:     float = EMPIRICAL_DEPARTURE_H + 24.0  
        self.ev_daily_kwh_consumed:  float = 0.0
        self.battery_soc_wh:         float = (
            BATTERY_CAPACITY_WH * BATTERY_INIT_SOC_FRAC
        )
        self.battery_soh_frac:       float = 1.0
        self.battery_cum_kwh:        float = 0.0
        self._v2g_kwh_today:         float = 0.0
        self._valley_start_offset_h: float = 0.0

        # Zone-level state
        self.zones:           List[str]              = []
        self.zone_capacities: Dict[str, float]       = {}
        self.zone_smooth:     Dict[str, float]       = {}
        self.zone_profiles:   Dict[str, List[float]] = {}
        self.zone_cluster:    Dict[str, int]         = {}
        self.master_cluster:  int                    = -1

        # Behavioural model
        self.probs:          Dict[Tuple[int, bool], List[float]]       = {}
        self.profiles:       Dict[int, List[float]]                    = {}
        self.trans_matrices: Dict[Tuple[int, bool], List[List[float]]] = {}
        self.rng:            random.Random                             = random.Random()
        self.max_capacity:   float                                     = 5.0
        self.last_day:       int                                       = -1

        # EnergyPlus API handles
        self.h_sun_diffuse:   int = -1
        self.h_sun_direct:    int = -1
        self.h_sun_altitude:  int = -1          # solar altitude for beam projection
        self.h_outdoor_temp:  int = -1
        self.h_building_elec: int = -1          # Electricity:Facility meter (bldg total incl HVAC)
        self.h_people:  Dict[str, int] = {}
        self.h_lights:  Dict[str, int] = {}
        self.h_equip:   Dict[str, int] = {}
        self.h_cool_sp: Dict[str, int] = {}
        self.h_heat_sp: Dict[str, int] = {}

        # Phase-1 to Phase-2 handoff state (single timestep scope)
        self._ts_valid:        bool  = False
        self._ts_day:          int   = -1
        self._ts_hour:         int   = -1
        self._ts_minute:       float = self._NAN
        self._ts_price:        float = self._NAN
        self._ts_co2_g_kwh:    float = self._NAN
        self._ts_is_peak:      bool  = False
        self._ts_outdoor_temp: float = self._NAN
        self._ts_irradiance:   float = self._NAN
        self._ts_total_occ:    float = self._NAN
        self._ts_pv_w:         float = self._NAN
        self._ts_ev_w:         float = self._NAN
        self._ts_batt_w:       float = self._NAN
        self._ts_hvac_precool: bool  = False
        self._ts_dt_hr:        float = self._NAN

        # Telemetry
        self._csv_file:   Optional[object] = None
        self._csv_writer: Optional[object] = None
        self._csv_buffer: List             = []
        self._is_warmup:  bool             = True

        # Annual statistics
        self._stat_pv_kwh:           float = 0.0
        self._stat_ev_kwh:           float = 0.0
        self._stat_v2g_kwh:          float = 0.0
        self._stat_batt_kwh:         float = 0.0
        self._stat_building_kwh:     float = 0.0   
        self._stat_co2_saved_kg:     float = 0.0
        self._stat_co2_peak_kg:      float = 0.0
        self._stat_co2_offpeak_kg:   float = 0.0
        self._stat_offsite_cost_usd: float = 0.0
        #  departure deadline audit
        self._stat_dep_events:           int   = 0
        self._stat_dep_deadline_misses:  int   = 0
        self._stat_dep_soc_at_dep_kwh:   float = 0.0   # cumulative for averaging
        self._stat_drive_kwh_consumed:   float = 0.0   # cumulative driving energy
        self._run_wall_t0:           float = 0.0
        self._warned_handles:        bool  = False
        self._warned_no_meter:       bool  = False    # one-shot meter warning
        self._meter_ok_logged:       bool  = False    # one-shot re-acquire-success log
        self.h_building_var:         int   = -1       # facility-elec VARIABLE fallback handle
        self._var_probe_done:        bool  = False    # probe-candidate-keys once latch
        self._summary_written:       bool  = False    # emit-once latch for annual summary

    # 4. ALGORITHM 1 TAU-MAX AND CONFIG LOAD

    def _algorithm1_tau_max(self, is_weekend: bool = False) -> int:

        if (self.forced_arrival_h < 0
                or not self.has_ev
                or self.ev_capacity_kwh == 0):
            return 0
        e_deficit   = self.ev_capacity_kwh * max(0.0, 1.0 - self.forced_soc_frac)
        h_to_charge = e_deficit / EV_CHARGER_KW_NET
        base_dep = (EMPIRICAL_DEPARTURE_H_WEEKEND
                    if is_weekend else EMPIRICAL_DEPARTURE_H)
        dep_h = (base_dep + 24.0
                 if base_dep <= ALGO1_PRICE_DROP_START_H
                 else base_dep)
        window_h    = dep_h - ALGO1_PRICE_DROP_START_H
        tau_safe_h  = max(0.0, window_h - h_to_charge - ALGO1_TAU_SAFETY_H)

        clip_h = (ALGO1_TAU_MAX_ADAPT_H
                  if ALGO1_ADAPTIVE_WINDOW
                  else ALGO1_TAU_MAX_HARD_H)
        tau_safe_h  = min(clip_h, tau_safe_h)
        return int(tau_safe_h * 60.0)

    def _load_config(self, state) -> bool:
        """Load occupancy_config.json; build Markov matrices; open sidecar."""
        try:
            cfg_path = "occupancy_config.json"
            if not os.path.exists(cfg_path):
                self.api.runtime.issue_severe(
                    state, f"[OccupancyPlugin] Config not found: {cfg_path}"
                )
                return False

            with open(cfg_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            # CONFIG_REQUIRED_FIELDS check
            _REQUIRED: List[tuple] = [
                ("scenario",           1,    "scenario mode (1=S1)"),
                ("has_ev",             True, "EV present flag"),
                ("inelasticity_ratio", 0.30, "fraction of inelastic occupants"),
                ("ev_arrival_hour",    -1.0, "EV arrival hour (-1 = no arrival)"),
                ("ev_distance",        37.5, "daily EV distance in miles"),
                ("zones",              [],   "EnergyPlus zone name list"),
                ("markov_rho",         0.60, "Markov persistence rho"),
            ]
            _missing = [
                f"  '{k}' missing — using default {d!r}  ({desc})"
                for k, d, desc in _REQUIRED if k not in data
            ]
            if _missing:
                self.api.runtime.issue_warning(state,
                    "[OccupancyPlugin][M-02] Config is missing runtime fields:\n"
                    + "\n".join(_missing))

            self.scenario_mode      = int(data.get("scenario", 1))
            self.has_ev             = bool(data.get("has_ev", True))
            # Battery-penetration sensitivity
            self.has_battery        = bool(data.get("has_battery", True))
            self.inelasticity_ratio = float(data.get("inelasticity_ratio", 0.30))

            if self.scenario_mode == 5:
                self._load_broadcast_profile(str(data.get("broadcast_csv", "")).strip())

            house_seed = _stable_house_seed(str(data.get("house_id", "house")))
            self.rng.seed(house_seed)
            self.is_elastic = self.rng.random() > self.inelasticity_ratio

            try:
                hid_str = str(data.get("house_id", ""))
                # House IDs follow "House_01", "House_02", ..., "House_50"
                _digits = "".join(ch for ch in hid_str if ch.isdigit())
                self._house_slot_idx = int(_digits) if _digits else 0
            except Exception:
                self._house_slot_idx = 0

            # S5 heterogeneous broadcast
            if self.scenario_mode == 5:
                if self._house_slot_idx > 0:
                    _theta_frac = van_der_corput(
                        self._house_slot_idx, base=ALGO1_BROADCAST_THETA_BASE)
                else:
                    _theta_frac = 0.5  # unknown slot -> neutral midpoint
                self._broadcast_theta = (
                    ALGO1_BROADCAST_THETA_MIN
                    + (ALGO1_BROADCAST_THETA_MAX - ALGO1_BROADCAST_THETA_MIN)
                    * _theta_frac)

            arr_h = float(data.get("ev_arrival_hour", -1.0))
            if self.has_ev and arr_h > 0:
                self.forced_arrival_h  = arr_h
                dist_mi = float(data.get("ev_distance", EV_NHTS_MEDIAN_VMT_MILES))
                self.ev_distance_miles = dist_mi
                self.forced_soc_frac   = _nhts_soc_from_distance(
                    dist_mi, self.ev_capacity_kwh
                )
            else:
                self.forced_arrival_h = -1.0
                if not self.has_ev:
                    self.ev_capacity_kwh = 0.0

            if self.scenario_mode in (1, 5):

                tau_max_min = self._algorithm1_tau_max(is_weekend=False)
                if tau_max_min > 0 and self._house_slot_idx > 0:
                    if ALGO1_STAGGER_MODE == "iid":
                        frac = _iid_uniform_frac(self._house_slot_idx, 0)
                    else:
                        frac = van_der_corput(self._house_slot_idx, base=2)
                    self.smart_delay_min = int(round(frac * tau_max_min))
                else:
                    self.smart_delay_min = self.rng.randint(0, tau_max_min)
            else:
                self.smart_delay_min = 0

            if (self.scenario_mode == 4
                    and self.has_ev
                    and self.ev_capacity_kwh > 0):
                e_deficit    = self.ev_capacity_kwh * max(
                    0.0, 1.0 - self.forced_soc_frac)
                min_chg_h    = e_deficit / EV_CHARGER_KW_NET
                valley_win_h = max(
                    0.0, VALLEY_END_H - VALLEY_START_H - min_chg_h)
                self._valley_start_offset_h = self.rng.uniform(0.0, valley_win_h)

            self.max_capacity = float(data.get("max_capacity", 5.0))
            self.zones        = [
                z.strip() for z in data.get("zones", []) if z.strip()
            ]
            for zone in self.zones:
                self.zone_smooth[zone]  = 0.0
                self.zone_cluster[zone] = -1

            self.profiles = {int(k): v for k, v in data["profiles"].items()}

            raw: Dict[Tuple[int, bool], List[float]] = {}
            for key, vec in data["probs"].items():
                parts  = key.split("_")
                month  = int(parts[0])
                is_wkd = parts[1].lower() == "true"
                raw[(month, is_wkd)] = [float(x) for x in vec]
            self.probs = raw

            self.markov_rho = float(data.get("markov_rho", MARKOV_PERSISTENCE))
            self.trans_matrices = {
                ctx: build_transition_matrix(pi, rho=self.markov_rho)
                for ctx, pi in self.probs.items()
            }

            # Reset battery + annual stats
            self.battery_soc_wh   = BATTERY_CAPACITY_WH * BATTERY_INIT_SOC_FRAC
            self.battery_soh_frac = 1.0
            self.battery_cum_kwh  = 0.0
            self._stat_pv_kwh = self._stat_ev_kwh = 0.0
            self._stat_v2g_kwh = self._stat_batt_kwh = 0.0
            self._prev_residual_w = 0.0   # building+EV-PV cache for S0/S3 self-consumption
            self._stat_building_kwh = 0.0
            self._stat_co2_saved_kg = 0.0
            self._stat_co2_peak_kg  = self._stat_co2_offpeak_kg = 0.0
            # initialize at-home state machine.
            if self.has_ev and self.ev_capacity_kwh > 0:
                self.ev_at_home  = True
                self.ev_soc_kwh  = self.forced_soc_frac * self.ev_capacity_kwh
            else:
                self.ev_at_home  = False
                self.ev_soc_kwh  = 0.0

            self._dep_fired_today           = False
            self._arr_fired_today           = False
            self._stat_dep_events           = 0
            self._stat_dep_deadline_misses  = 0
            self._stat_dep_soc_at_dep_kwh   = 0.0
            self._stat_drive_kwh_consumed   = 0.0
            self._run_wall_t0 = time.monotonic()
            self._summary_written = False

            self._open_sidecar()
            self._initialised = True
            return True

        except Exception as exc:
            self.api.runtime.issue_severe(
                state, f"[OccupancyPlugin] Config load failed: {exc}"
            )
            return False

    def _setup_handles(self, state) -> bool:

        try:
            exch = self.api.exchange
            if not exch.api_data_fully_ready(state) or not self.zones:
                return False

            self.h_sun_diffuse  = exch.get_variable_handle(
                state, "Site Diffuse Solar Radiation Rate per Area", "Environment"
            )
            self.h_sun_direct   = exch.get_variable_handle(
                state, "Site Direct Solar Radiation Rate per Area", "Environment"
            )

            self.h_sun_altitude = exch.get_variable_handle(
                state, "Site Solar Altitude Angle", "Environment"
            )
            self.h_outdoor_temp = exch.get_variable_handle(
                state, "Site Outdoor Air Drybulb Temperature", "Environment"
            )

            self.h_building_elec = exch.get_meter_handle(
                state, "Electricity:Facility"
            )
            if self.h_building_elec <= -1 and not self._warned_no_meter:
                self.api.runtime.issue_warning(state,
                    "[OccupancyPlugin][v13.8] Electricity:Facility meter handle "
                    "is -1 at setup (expected when first requested during "
                    "sizing/warmup, before EnergyPlus finalises meter handles). "
                    "The meter IS declared; the plugin re-acquires the handle "
                    "lazily in the meter-read path during the run period.  If "
                    "building_kwh stays 0, the handle never resolved and "
                    "net_grid_w falls back to ev + batt - pv (INCOMPLETE)."
                )
                self._warned_no_meter = True

            any_bad = False
            for zone in self.zones:
                self.h_people[zone]  = exch.get_actuator_handle(
                    state, "People", "Number of People", f"{zone} People 1")
                self.h_lights[zone]  = exch.get_actuator_handle(
                    state, "Lights", "Electricity Rate", f"{zone} Lights 1")
                self.h_equip[zone]   = exch.get_actuator_handle(
                    state, "ElectricEquipment", "Electricity Rate",
                    f"{zone} ElecEq 1")
                self.h_cool_sp[zone] = exch.get_actuator_handle(
                    state, "Zone Temperature Control", "Cooling Setpoint", zone)
                self.h_heat_sp[zone] = exch.get_actuator_handle(
                    state, "Zone Temperature Control", "Heating Setpoint", zone)

                h_cap = exch.get_internal_variable_handle(
                    state, "People Number of People", f"{zone} People 1")
                self.zone_capacities[zone] = (
                    exch.get_internal_variable_value(state, h_cap)
                    if h_cap > -1
                    else self.max_capacity / max(1, len(self.zones))
                )

                if (self.h_people.get(zone, -1) < 0
                        or self.h_equip.get(zone, -1) < 0):
                    any_bad = True

            if any_bad and not self._warned_handles:
                self.api.runtime.issue_warning(state,
                    "[OccupancyPlugin] One or more actuator handles returned "
                    "-1.  Verify IDF object names follow "
                    "'<ZONE> People 1', '<ZONE> Lights 1', "
                    "'<ZONE> ElecEq 1'."
                )
                self._warned_handles = True

            return True

        except Exception as exc:
            self.api.runtime.issue_severe(
                state, f"[OccupancyPlugin] Handle setup failed: {exc}"
            )
            return False

    # 5. LIFECYCLE CALLBACKS

    def on_begin_new_environment(self, state) -> int:

        self._close_sidecar()  
        self.do_setup   = True
        self.last_day   = -1
        self.zones      = []
        self._is_warmup = True
        self._ts_valid  = False
        return 0

    # 5b. S5 ONE-WAY COMMUNITY-LOAD BROADCAST  (opt-in hybrid; no-op for S0-S4)

    def _load_broadcast_profile(self, path: str) -> None:

        self._broadcast_profile = {}
        self._broadcast_peak_kw = 0.0
        try:
            import csv as _csv
            from pathlib import Path as _Path

            src = _Path(path) if path else None

            if src is None or src.is_dir() or (not src.exists()):
                search_dir = src if (src is not None and src.is_dir()) else (
                    src.parent if src is not None else None)
                if search_dir is not None and search_dir.is_dir():
                    hits = sorted(search_dir.glob("community_ts_*_S1.csv"))
                    src = hits[0] if hits else None
            if src is None or not src.exists():
                return  

            sums: Dict[int, float] = {}
            counts: Dict[int, int] = {}
            with open(src, "r", encoding="utf-8") as fh:
                reader = _csv.DictReader(fh)
                if (reader.fieldnames is None
                        or "hour" not in reader.fieldnames
                        or "community_net_kw" not in reader.fieldnames):
                    return
                for row in reader:
                    try:
                        h = int(float(row["hour"])) % 24
                        kw = float(row["community_net_kw"])
                    except (TypeError, ValueError):
                        continue
                    sums[h] = sums.get(h, 0.0) + kw
                    counts[h] = counts.get(h, 0) + 1

            profile = {h: sums[h] / counts[h] for h in sums if counts[h] > 0}
            if profile:
                self._broadcast_profile = profile
                self._broadcast_peak_kw = max(max(profile.values()), 1e-6)
        except Exception:
            self._broadcast_profile = {}
            self._broadcast_peak_kw = 0.0

    def _broadcast_load_frac(self, hour: int) -> float:
        """Return the normalized broadcast community load at hour, in [0, 1].
        """
        if not self._broadcast_profile or self._broadcast_peak_kw <= 0.0:
            return 0.0
        net_kw = self._broadcast_profile.get(int(hour) % 24, 0.0)
        return max(0.0, min(1.0, net_kw / self._broadcast_peak_kw))

    # 6. PHYSICS ENGINES

    def _compute_ev_dispatch(
        self,
        hour: int,
        minute: float,
        is_peak: bool,
        dt_hr: float,
        co2_g_kwh: float = 200.0,
        price: float = RATE_OFF_PEAK,
        is_weekend: bool = False,
    ) -> float:
        """
        Returns grid_draw_w (> 0 = charge, < 0 = V2G export, 0 = idle).
        """
        if not self.has_ev or self.ev_capacity_kwh == 0:
            return 0.0
        if not self.ev_at_home:
            return 0.0

        # Departure-aware V2G discharge
        if (ENABLE_V2G
                and self.scenario_mode == 1
                and self.is_elastic
                and is_peak
                and self.ev_soc_kwh > EV_MIN_V2G_SOC_KWH + ALGO1_V2G_RESERVE_KWH):

            base_dep = (EMPIRICAL_DEPARTURE_H_WEEKEND
                        if is_weekend else EMPIRICAL_DEPARTURE_H)
            dep_h    = base_dep + 24.0
            now_h    = hour + minute / 60.0
            hours_to_dep = max(0.5, dep_h - now_h)

            # Energy needed to reach max acceptable SoC by departure
            target_soc_kwh = self.ev_capacity_kwh * 0.90
            energy_needed_after_v2g = max(
                0.0, target_soc_kwh - (self.ev_soc_kwh - 1.0)  
            )
            time_needed_after_v2g = (
                energy_needed_after_v2g / EV_CHARGER_KW_NET
            )
            # Stagger window will be hours [21h, dep_h]
            offpeak_window_h = max(0.0, dep_h - ALGO1_PRICE_DROP_START_H)
            v2g_safe = (time_needed_after_v2g
                        + ALGO1_TAU_SAFETY_H < offpeak_window_h)

            if v2g_safe:
                avail_kwh = (self.ev_soc_kwh
                             - EV_MIN_V2G_SOC_KWH
                             - ALGO1_V2G_RESERVE_KWH)
                step_kwh  = (EV_CHARGER_POWER_W / 1000.0) * dt_hr
                dchg_kwh  = min(step_kwh, avail_kwh)
                if dchg_kwh > 1e-6:
                    self.ev_soc_kwh     -= dchg_kwh
                    self._stat_v2g_kwh  += dchg_kwh
                    self._v2g_kwh_today += dchg_kwh
                    return -(dchg_kwh * EV_ETA_V2G / dt_hr) * 1000.0
            # else fall through to G2V branch (also returns 0 if not is_peak window)

        # G2V charging 
        if self.ev_soc_kwh >= self.ev_capacity_kwh:
            return 0.0

        # Default rate fraction = 1.0 (full charger power)
        rate_frac = 1.0
        can_charge = False

        if self.scenario_mode == 0:
            # S0: uncontrolled - charge on arrival at full rate.  Price
            # signal: full PG&E E-TOU-C schedule (set in Phase 1 via
            # _get_tariff_for_scenario).  Battery does passive PV self-consumption.
            can_charge = True

        elif self.scenario_mode == 3:
            can_charge = True

        elif self.scenario_mode == 2:
            # S2: TOU rebound - elastic households wait for off-peak (no stagger)
            can_charge = (not self.is_elastic) or (not is_peak)

        elif self.scenario_mode in (1, 5):
            # S1 (and S5): identical EV dispatch. S5 is S1 PLUS the one-way
            # community-load broadcast, which enters only as an extra term in
            # the shadow price below (guarded by scenario_mode == 5). With the
            # broadcast absent or gamma = 0, S5 reduces exactly to S1.
            if not self.is_elastic:
                # Inelastic users: charge at full rate, no stagger
                can_charge = True
            else:
                # Elastic users: stagger + rate modulation across off-peak
                base_dep = (EMPIRICAL_DEPARTURE_H_WEEKEND
                            if is_weekend else EMPIRICAL_DEPARTURE_H)

                dep_h    = base_dep + 24.0
                now_h    = hour + minute / 60.0

                eff_now_h = (now_h + 24.0
                             if hour < ALGO1_MORNING_ROLLOVER_H
                             else now_h)
                t_since_offpeak_open_h = eff_now_h - ALGO1_PRICE_DROP_START_H
                stagger_h = self.smart_delay_min / 60.0

                # Open the window only after this house's stagger delay
                if (not is_peak) and (t_since_offpeak_open_h >= stagger_h):
                    can_charge = True

                    # DRM - rate scales by remaining-time / energy-needed
                    if ALGO1_DRM_ENABLE:
                        deficit_kwh   = (self.ev_capacity_kwh
                                         - self.ev_soc_kwh)
                        time_remain_h = max(
                            0.1, dep_h - eff_now_h - ALGO1_TAU_SAFETY_H)
                        # Required avg power to finish (DC side, then AC inflate)
                        p_req_w = (deficit_kwh * 1000.0
                                   / time_remain_h / EV_ETA_CHARGE)
                        rate_frac = max(
                            ALGO1_DRM_MIN_RATE_FRAC,
                            min(1.0, p_req_w / EV_CHARGER_POWER_W)
                        )

                    # Local shadow-price arbiter
                    pi_local = (
                        ALGO1_PRICE_ALPHA * (price / RATE_OFF_PEAK)
                        + ALGO1_PRICE_BETA * (co2_g_kwh / CO2_CHARGE_GATE)
                    )
                    # S5: one-way community-load broadcast, HETEROGENEOUS threshold.
   
                    if self.scenario_mode == 5:
                        _excess = max(0.0,
                                      self._broadcast_load_frac(hour)
                                      - self._broadcast_theta)
                        pi_local += ALGO1_BROADCAST_GAMMA * _excess
                    if pi_local >= ALGO1_PRICE_GATE_HI:
                        # Heavy throttle when grid is stressed
                        rate_frac *= 0.30
                    elif pi_local >= ALGO1_PRICE_GATE_LO:
                        # Linear taper between LO and HI
                        scale = 1.0 - 0.7 * (
                            (pi_local - ALGO1_PRICE_GATE_LO)
                            / max(1e-6, (ALGO1_PRICE_GATE_HI - ALGO1_PRICE_GATE_LO))
                        )
                        rate_frac *= max(0.30, scale)
                    # else: full rate_frac retained

        elif self.scenario_mode == 4:
            # S4: valley-fill - randomized within 0-6 AM window
            valley_open_h = VALLEY_START_H + self._valley_start_offset_h
            t = hour + minute / 60.0
            can_charge = (valley_open_h <= t <= VALLEY_END_H)

        if self.scenario_mode in (0, 1, 2, 3, 5):
            base_dep_o = (EMPIRICAL_DEPARTURE_H_WEEKEND
                          if is_weekend else EMPIRICAL_DEPARTURE_H)
            dep_h_abs   = base_dep_o + 24.0
            now_h_abs   = hour + minute / 60.0
            eff_now     = (now_h_abs + 24.0
                           if hour < ALGO1_MORNING_ROLLOVER_H
                           else now_h_abs)
            time_remain = dep_h_abs - eff_now
            deficit_kwh = self.ev_capacity_kwh - self.ev_soc_kwh
            needed_h    = deficit_kwh / max(1e-6, EV_CHARGER_KW_NET)
            if time_remain < needed_h + ALGO1_TAU_SAFETY_H:
                can_charge = True
                if self.scenario_mode in (1, 5) and self.is_elastic:

                    rate_frac = 1.0

        if not can_charge:
            return 0.0

        # Final dispatch with rate modulation
        rate_frac   = max(0.0, min(1.0, rate_frac))
        eff_charger_w = EV_CHARGER_POWER_W * rate_frac
        deficit_kwh = self.ev_capacity_kwh - self.ev_soc_kwh
        step_kwh    = (eff_charger_w / 1000.0) * dt_hr * EV_ETA_CHARGE
        actual_kwh  = min(step_kwh, deficit_kwh)
        if actual_kwh <= 0.0:
            return 0.0
        self.ev_soc_kwh   += actual_kwh
        self._stat_ev_kwh += actual_kwh
        return (actual_kwh / EV_ETA_CHARGE / dt_hr) * 1000.0

    # At-home state machine event handler
    def _check_ev_calendar_events(self, now_h: float) -> None:
        
        if not self.has_ev or self.ev_capacity_kwh == 0:
            return

        # DEPARTURE - fires once per day in the morning window
        if (not self._dep_fired_today
                and self.ev_at_home
                and now_h >= self.ev_departure_h_today):
            soc_pre_drive = self.ev_soc_kwh
            self._stat_dep_soc_at_dep_kwh += soc_pre_drive
            self._stat_dep_events         += 1

            target_kwh = EV_DEADLINE_TARGET_FRAC * self.ev_capacity_kwh
            if soc_pre_drive < target_kwh - 1e-6:
                self._stat_dep_deadline_misses += 1

            consumed = min(self.ev_daily_kwh_consumed, self.ev_soc_kwh)
            self.ev_soc_kwh = max(0.0, self.ev_soc_kwh - consumed)
            self._stat_drive_kwh_consumed += consumed

            self.ev_at_home       = False
            self._dep_fired_today = True
            return

        # ARRIVAL - fires once per day in the evening window
        if (not self._arr_fired_today
                and (not self.ev_at_home)
                and now_h >= self.ev_arrival_h_today):
            self.ev_at_home       = True
            self._arr_fired_today = True
            return

    def _compute_battery_dispatch(
        self,
        hour: int, minute: float,
        is_peak: bool,
        dt_hr: float,
        co2_g_kwh: float,
        price: float,
    ) -> float:
        """Home battery price-arbitrage + carbon-aware dispatch"""
        if not ENABLE_BATTERY:
            return 0.0
        # PER-HOUSE STORAGE PRESENCE (battery-penetration sensitivity)
        if not self.has_battery:
            return 0.0

        eff_cap_wh = BATTERY_CAPACITY_WH * self.battery_soh_frac
        min_wh     = eff_cap_wh * BATTERY_MIN_SOC_FRAC
        max_wh     = eff_cap_wh * BATTERY_MAX_SOC_FRAC

        self.battery_soc_wh = max(min_wh, min(max_wh, self.battery_soc_wh))
        soc_wh  = self.battery_soc_wh
        step_wh = BATTERY_POWER_W * dt_hr

        # S0 / S3: PASSIVE SELF-CONSUMPTION (MSC) battery 
        if self.scenario_mode in (0, 3):
            residual_w = self._prev_residual_w
            if residual_w < 0.0 and soc_wh < max_wh:
                # PV surplus -> store it instead of exporting
                charge_ac_w = min(-residual_w, BATTERY_POWER_W)
                dc_stored   = min(charge_ac_w * BATTERY_ETA_CHARGE * dt_hr,
                                  max_wh - soc_wh)
                self.battery_soc_wh += dc_stored
                self._update_battery_soh(dc_stored)
                return dc_stored / BATTERY_ETA_CHARGE / dt_hr
            if residual_w > 0.0 and soc_wh > min_wh:
                # deficit -> discharge to cover load, reducing grid import
                discharge_ac_w = min(residual_w, BATTERY_POWER_W)
                dc_removed     = min(
                    discharge_ac_w / BATTERY_ETA_DISCHARGE * dt_hr,
                    soc_wh - min_wh)
                self.battery_soc_wh -= dc_removed
                self._update_battery_soh(dc_removed)
                self._stat_batt_kwh += dc_removed / 1000.0
                return -(dc_removed * BATTERY_ETA_DISCHARGE / dt_hr)
            return 0.0

        # Discharge: peak shaving
        if is_peak and soc_wh > min_wh:
            dc_max_wh   = (BATTERY_POWER_W / BATTERY_ETA_DISCHARGE) * dt_hr
            dc_removed  = min(dc_max_wh, soc_wh - min_wh)
            ac_export_w = -(dc_removed * BATTERY_ETA_DISCHARGE / dt_hr)
            self.battery_soc_wh -= dc_removed
            self._update_battery_soh(dc_removed)
            self._stat_batt_kwh += dc_removed / 1000.0
            return ac_export_w

        # Charge: off-peak + carbon gate
        if (not is_peak
                and soc_wh < max_wh
                and co2_g_kwh < CO2_CHARGE_GATE):
            # BATTERY CHARGE POLICY for S1/S4/S5 (rate-limited, communication-free)
            if self.scenario_mode in (1, 4, 5):
                eff_now_h = hour + minute / 60.0
                if ENABLE_SOLAR_CHARGE:
                    if eff_now_h < SOLAR_CHARGE_START_H:
                        return 0.0            # wait for the solar window (no overnight grid charge)
                    remaining_h = max(dt_hr, float(PEAK_START_H) - eff_now_h)
                else:
                    remaining_h = max(dt_hr, VALLEY_END_H - eff_now_h)
                deficit_wh = max_wh - soc_wh
                dc_rate_w  = min(BATTERY_POWER_W * BATTERY_ETA_CHARGE,
                                 deficit_wh / remaining_h)
                # S5 ONLY: throttle the recharge by the same one-way broadcast used for EV charging
                if self.scenario_mode == 5:
                    _excess = max(0.0,
                                  self._broadcast_load_frac(hour)
                                  - self._broadcast_theta)
                    dc_rate_w *= max(ALGO1_BROADCAST_FLOOR,
                                     1.0 - ALGO1_BROADCAST_GAMMA * _excess)
                dc_stored  = min(dc_rate_w * dt_hr, max_wh - soc_wh)
            else:
                dc_stored = min(step_wh * BATTERY_ETA_CHARGE, max_wh - soc_wh)
            self.battery_soc_wh += dc_stored
            self._update_battery_soh(dc_stored)
            return dc_stored / BATTERY_ETA_CHARGE / dt_hr

        return 0.0

    def _update_battery_soh(self, delta_wh: float) -> None:
        """DoD-weighted Wohler."""
        if delta_wh <= 0.0:
            return
        self.battery_cum_kwh += delta_wh / 1000.0
        eff_cap_wh = max(1.0, BATTERY_CAPACITY_WH * self.battery_soh_frac)
        delta_soc  = min(1.0, delta_wh / eff_cap_wh)
        k_cycle    = BATT_WOHLER_A * (delta_soc ** BATT_WOHLER_GAMMA)
        d_soh      = k_cycle / (2.0 * BATT_WOHLER_N_REF)
        self.battery_soh_frac = max(0.70, self.battery_soh_frac - d_soh)

    def _compute_zone_loads(
        self, zone: str, hour: int, irradiance_w_m2: float,
    ) -> Tuple[float, float, float]:
        """IIR-smoothed occupancy + equip + LED lighting."""
        profile  = self.zone_profiles.get(zone, [0.0] * 24)
        cap      = self.zone_capacities.get(
            zone, self.max_capacity / max(1, len(self.zones)))
        target   = profile[hour] * cap
        prev     = self.zone_smooth.get(zone, 0.0)
        smoothed = prev + SMOOTHING_ALPHA * (target - prev)
        self.zone_smooth[zone] = smoothed

        equip_w = smoothed * WATTS_PER_PERSON + BASE_EQUIP_LOAD_W
        dl_frac = min(
            1.0, irradiance_w_m2 / max(1e-9, 3.0 * SOLAR_THRESHOLD_W_M2))
        dim      = 1.0 - (1.0 - LIGHTS_MIN_FRAC) * dl_frac
        occ_frac = min(1.0, smoothed / max(0.01, cap))
        lights_w = MAX_LIGHTS_W * dim * max(LIGHTS_MIN_FRAC, occ_frac)
        return smoothed, equip_w, lights_w

    # 7. DAILY MARKOV / DIVERSITY REFRESH

    def _daily_refresh(self, month: int, is_weekend: bool) -> None:
        """Midnight: Markov step, zone diversity, EV state, V2G off-site cost."""
        ctx  = (month, is_weekend)
        _n_k = max(1, len(self.profiles))
        pi   = self.probs.get(ctx, [1.0 / _n_k] * _n_k)
        T    = self.trans_matrices.get(ctx)

        if self.master_cluster < 0 or T is None:
            self.master_cluster = self.rng.choices(
                range(len(pi)), weights=pi, k=1)[0]
        else:
            self.master_cluster = markov_sample(self.rng, T, self.master_cluster)

        for zone in self.zones:
            z_cluster = (
                self.rng.choices(range(len(pi)), weights=pi, k=1)[0]
                if self.rng.random() < ZONE_DEVIATE_P
                else self.master_cluster
            )
            self.zone_cluster[zone]  = z_cluster
            self.zone_profiles[zone] = self.profiles.get(z_cluster, [0.0] * 24)

        # V2G off-site restoration cost
        if self._v2g_kwh_today > 0.0 and self.scenario_mode == 1:
            offsite_ac_kwh = self._v2g_kwh_today / V2G_OFFSITE_ETA
            self._stat_offsite_cost_usd += (
                offsite_ac_kwh * V2G_OFFSITE_RATE_USD_KWH
            )
        self._v2g_kwh_today = 0.0

        # sample TODAY's calendar events (departure, arrival, VMT)

        self._dep_fired_today = False
        self._arr_fired_today = False
        if self.has_ev and self.ev_capacity_kwh > 0:
            if self.forced_arrival_h > 0:
                # Empirical (NHTS-driven) calendar
                self.ev_arrival_h_today = (
                    self.forced_arrival_h
                    + self.rng.uniform(-EV_CALENDAR_JITTER_H,
                                       +EV_CALENDAR_JITTER_H)
                )
                base_dep = (EMPIRICAL_DEPARTURE_H_WEEKEND
                            if is_weekend else EMPIRICAL_DEPARTURE_H)
                self.ev_departure_h_today = (
                    base_dep
                    + self.rng.uniform(-EV_CALENDAR_JITTER_H,
                                       +EV_CALENDAR_JITTER_H)
                )
                # Driving energy expected to be deducted at today's departure.
                self.ev_daily_kwh_consumed = (
                    self.ev_distance_miles
                    / max(1e-6, EV_EPA_CONSUMPTION_MI_PER_KWH)
                )
                # Backward-compatibility alias
                self.ev_arrival_h = self.ev_arrival_h_today
            else:
                # Synthetic / fallback calendar. Roll a presence coin
                if self.rng.random() > 0.5:
                    arr = float(self.rng.randint(17, 20))
                    self.ev_arrival_h_today = arr
                    base_dep = (EMPIRICAL_DEPARTURE_H_WEEKEND
                                if is_weekend else EMPIRICAL_DEPARTURE_H)
                    self.ev_departure_h_today = (
                        base_dep
                        + self.rng.uniform(-EV_CALENDAR_JITTER_H,
                                           +EV_CALENDAR_JITTER_H)
                    )
                    vmt = max(1.0, self.rng.lognormvariate(3.63, 0.75))
                    self.ev_distance_miles      = vmt
                    self.ev_daily_kwh_consumed  = (
                        vmt / max(1e-6, EV_EPA_CONSUMPTION_MI_PER_KWH)
                    )
                    self.ev_arrival_h           = arr
                else:
                    # No driving today 
                    self.ev_arrival_h_today    = 99.0
                    self.ev_departure_h_today  = 99.0
                    self.ev_daily_kwh_consumed = 0.0
                    self.ev_arrival_h          = -1.0

        if self.scenario_mode in (1, 5):

            tau_max_min_today = self._algorithm1_tau_max(is_weekend=is_weekend)
            if tau_max_min_today > 0 and self._house_slot_idx > 0:
                if ALGO1_STAGGER_MODE == "iid":

                    frac = _iid_uniform_frac(
                        self._house_slot_idx, self._day_count_for_vdc
                    )
                else:
                    day_offset_idx = (
                        (self._house_slot_idx + self._day_count_for_vdc * 7919) % 1024
                    ) or 1   # avoid slot 0 , vdc(0) = 0
                    frac = van_der_corput(day_offset_idx, base=2)
                self.smart_delay_min = int(round(frac * tau_max_min_today))
            else:
                # Fallback only if vdc slot couldn't be parsed
                self.smart_delay_min = self.rng.randint(
                    0, tau_max_min_today
                )
            self._day_count_for_vdc += 1

    # 8. TELEMETRY (sidecar CSV)
    def _open_sidecar(self) -> None:
        """Open sidecar CSV in ENERGYPLUS_OUTPUT_DIR"""
        out_dir = os.environ.get("ENERGYPLUS_OUTPUT_DIR", ".")
        path    = os.path.join(out_dir, SIDECAR_FILENAME)
        try:
            self._csv_file   = open(path, "w", newline="", encoding="utf-8")
            self._csv_writer = csv.writer(self._csv_file)
            self._csv_writer.writerow(SIDECAR_HEADER)
            self._csv_buffer = []
        except OSError as err:
            self._csv_writer = None
            print(f"[OccupancyPlugin] WARNING: Cannot open sidecar: {err}")

    def _write_telemetry_row(
        self,
        day: int, hour: int, minute: float,
        price: float, co2: float, is_peak: bool,
        t_out: float, irr: float,
        occ: float,
        building_w: float,                
        pv_w: float, ev_w: float, batt_w: float, net_w: float,
        hvac: bool,
    ) -> None:
        """Buffer one row; flush every SIDECAR_FLUSH_N rows."""
        if self._csv_writer is None or self._is_warmup:
            return
        self._csv_buffer.append([
            day, hour, f"{minute:.1f}",
            f"{price:.4f}", f"{co2:.0f}", int(is_peak),
            f"{t_out:.2f}", f"{irr:.1f}",
            f"{occ:.3f}",
            f"{building_w:.1f}",          
            f"{pv_w:.1f}", f"{ev_w:.1f}", f"{batt_w:.1f}", f"{net_w:.1f}",
            f"{self.ev_soc_kwh:.3f}", f"{self.battery_soc_wh:.1f}",
            f"{self.battery_soh_frac:.4f}",
            int(hvac), self.scenario_mode,
        ])
        if len(self._csv_buffer) >= SIDECAR_FLUSH_N:
            self._csv_writer.writerows(self._csv_buffer)
            self._csv_file.flush()
            self._csv_buffer = []

    def _close_sidecar(self) -> None:
        """Flush and close sidecar"""
        if self._csv_writer and self._csv_buffer:
            self._csv_writer.writerows(self._csv_buffer)
            self._csv_buffer = []
        if self._csv_file:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except OSError:
                pass
            self._csv_file   = None
            self._csv_writer = None

    # 9. PHASE 1 CALLBACK - dispatch + actuate + state handoff

    def on_begin_timestep_before_predictor(self, state) -> int:
        try:

            self._ts_valid = False

            if self.do_setup:
                if not self.zones:
                    self._load_config(state)
                elif self._setup_handles(state):
                    self.do_setup = False
                return 0

            if not self.profiles or not self._initialised:
                return 0

            exch = self.api.exchange
            if self._is_warmup and not exch.warmup_flag(state):
                self._is_warmup = False

            hour         = exch.hour(state)
            current_time = exch.current_time(state)
            minute       = (current_time - int(current_time)) * 60.0
            day          = exch.day_of_year(state)
            month        = exch.month(state)
            dt_hr        = exch.zone_time_step(state)

            price, is_peak = _get_tariff_for_scenario(hour, self.scenario_mode, month)
            co2_g_kwh      = CO2_BY_HOUR.get(hour, 250.0)

            outdoor_temp = (
                exch.get_variable_value(state, self.h_outdoor_temp)
                if self.h_outdoor_temp > -1 else 15.0
            )

            # GHI = DHI + DNI*cos(zenith) = DHI + DNI*sin(altitude)
            _dhi = (exch.get_variable_value(state, self.h_sun_diffuse)
                    if self.h_sun_diffuse > -1 else 0.0)
            _dni = (exch.get_variable_value(state, self.h_sun_direct)
                    if self.h_sun_direct > -1 else 0.0)
            # Fallback 90 deg => cos(zenith)=1 
            _alt_deg = (exch.get_variable_value(state, self.h_sun_altitude)
                        if self.h_sun_altitude > -1 else 90.0)
            _cos_zenith = max(0.0, math.sin(math.radians(_alt_deg)))
            irradiance = max(0.0, _dhi + _dni * _cos_zenith)

            # is_weekend now needed every timestep 
            is_weekend_now = exch.day_of_week(state) in (1, 7)
            if day != self.last_day:
                month = exch.month(state)
                self._daily_refresh(month, is_weekend_now)
                self.last_day = day

            now_h_for_events = float(hour) + float(minute) / 60.0
            self._check_ev_calendar_events(now_h_for_events)

            pv_w   = pv_power_w(irradiance, outdoor_temp)
            ev_w   = self._compute_ev_dispatch(
                hour, minute, is_peak, dt_hr,
                co2_g_kwh=co2_g_kwh,
                price=price,
                is_weekend=is_weekend_now,
            )
            batt_w = self._compute_battery_dispatch(
                hour, minute, is_peak, dt_hr, co2_g_kwh, price
            )

            if not self._is_warmup:
                pv_kwh = pv_w / 1000.0 * dt_hr
                self._stat_pv_kwh += pv_kwh
                co2_credit_kg = pv_kwh * co2_g_kwh / 1000.0
                self._stat_co2_saved_kg += co2_credit_kg
                if is_peak:
                    self._stat_co2_peak_kg    += co2_credit_kg
                else:
                    self._stat_co2_offpeak_kg += co2_credit_kg

            hvac_precool = (
                self.scenario_mode in (1, 2, 5)
                and PRECOOL_HOUR_START <= hour <= PRECOOL_HOUR_END
                and outdoor_temp > PRECOOL_TEMP_C
            )

            total_occ = 0.0
            for zone in self.zones:
                people, equip_w, lights_w = self._compute_zone_loads(
                    zone, hour, irradiance
                )
                total_occ += people

                cs = self.h_cool_sp.get(zone, -1)
                hs = self.h_heat_sp.get(zone, -1)
                if hvac_precool:
                    if cs > -1:
                        exch.set_actuator_value(state, cs, PRECOOL_COOL_SP_C)
                    if hs > -1:
                        exch.set_actuator_value(state, hs, PRECOOL_HEAT_SP_C)
                else:
                    if cs > -1:
                        exch.reset_actuator(state, cs)
                    if hs > -1:
                        exch.reset_actuator(state, hs)

                hp = self.h_people.get(zone, -1)
                he = self.h_equip.get(zone,  -1)
                hl = self.h_lights.get(zone, -1)
                if hp > -1:
                    exch.set_actuator_value(state, hp, people)
                if he > -1:
                    exch.set_actuator_value(state, he, equip_w)
                if hl > -1:
                    exch.set_actuator_value(state, hl, lights_w)

            # Net grid power is NOT computed here - Phase 2 adds building_w.
            self._ts_day          = day
            self._ts_hour         = hour
            self._ts_minute       = minute
            self._ts_price        = price
            self._ts_co2_g_kwh    = co2_g_kwh
            self._ts_is_peak      = is_peak
            self._ts_outdoor_temp = outdoor_temp
            self._ts_irradiance   = irradiance
            self._ts_total_occ    = total_occ
            self._ts_pv_w         = pv_w
            self._ts_ev_w         = ev_w
            self._ts_batt_w       = batt_w
            self._ts_hvac_precool = hvac_precool
            self._ts_dt_hr        = dt_hr
            self._ts_valid        = True
            return 0

        except Exception as exc:
            self.api.runtime.issue_severe(
                state, f"[OccupancyPlugin] Phase-1 crash: {exc}"
            )
            return 1

    # 10. PHASE 2 CALLBACK - read meter, compute true net, write CSV

    def on_end_of_zone_timestep_before_zone_reporting(self, state) -> int:

        try:
            if not self._ts_valid:
                return 0
            if not self.profiles or not self._initialised:
                return 0

            exch = self.api.exchange

            if self.h_building_elec <= -1:
                self.h_building_elec = exch.get_meter_handle(
                    state, "Electricity:Facility")
                if self.h_building_elec > -1 and not self._meter_ok_logged:
                    self.api.runtime.issue_warning(state,
                        "[OccupancyPlugin][v13.8] Electricity:Facility handle "
                        "resolved via deferred re-acquire; building_kwh is now live.")
                    self._meter_ok_logged = True

            if self.h_building_elec <= -1 and self.h_building_var <= -1 and not self._var_probe_done:
                self._var_probe_done = True
                for _k in ("Whole Building", "Facility", "Environment", ""):
                    _h = exch.get_variable_handle(
                        state, "Facility Net Purchased Electricity Rate", _k)
                    if _h > -1:
                        self.h_building_var = _h
                        self.api.runtime.issue_warning(state,
                            "[OccupancyPlugin][v13.9] building electricity read via "
                            "VARIABLE Facility Net Purchased Electricity Rate, key='"
                            + _k + "' (meter handle was -1).")
                        break
                if self.h_building_var <= -1:
                    self.api.runtime.issue_warning(state,
                        "[OccupancyPlugin][v13.9] variable fallback failed for all "
                        "probed keys; grep eplusout.rdd for the exact "
                        "facility-electricity variable name and key.")
            if self.h_building_elec > -1:
                building_j = exch.get_meter_value(state, self.h_building_elec)
                dt_sec     = self._ts_dt_hr * 3600.0
                building_w = (building_j / dt_sec) if dt_sec > 0.0 else 0.0
            elif self.h_building_var > -1:
                building_w = exch.get_variable_value(state, self.h_building_var)  
            else:
                building_w = 0.0   # both meter and variable unresolved this step

            if not self._is_warmup:
                self._stat_building_kwh += (
                    building_w / 1000.0 * self._ts_dt_hr
                )

            # net grid draw:
            #   + building_w (HVAC + lights + equip, always >= 0)
            #   + ev_w       (+ charge, - V2G export)
            #   + batt_w     (+ charge, - peak-shave export)
            #   - pv_w       (generation exports to meter)
            net_grid_w = (
                building_w
                + self._ts_ev_w
                + self._ts_batt_w
                - self._ts_pv_w
            )

            #  residual net load (building + EV - PV, battery excluded)
            self._prev_residual_w = (
                building_w + self._ts_ev_w - self._ts_pv_w
            )

            self._write_telemetry_row(
                self._ts_day, self._ts_hour, self._ts_minute,
                self._ts_price, self._ts_co2_g_kwh, self._ts_is_peak,
                self._ts_outdoor_temp, self._ts_irradiance,
                self._ts_total_occ,
                building_w,
                self._ts_pv_w, self._ts_ev_w, self._ts_batt_w, net_grid_w,
                self._ts_hvac_precool,
            )

            self._ts_valid = False
            return 0

        except Exception as exc:
            self.api.runtime.issue_severe(
                state, f"[OccupancyPlugin] Phase-2 crash: {exc}"
            )
            return 1

    # 11. END-OF-RUN REPORTING
    def on_end_of_zone_timestep_after_zone_reporting(self, state) -> int:

        if getattr(self, "_summary_written", False):
            return 0
        try:
            exch = self.api.exchange
            if exch.warmup_flag(state):
                return 0

            day  = exch.day_of_year(state)
            hour = exch.hour(state)
            now  = exch.current_time(state)
            minute = (now - int(now)) * 60.0
            is_last_timestep = (
                day == 365
                and hour >= 23
                and minute >= 49.99   
            )
            if not is_last_timestep:
                return 0
        except Exception:
            return 0

        self._summary_written = True
        return self._emit_annual_summary(state)

    # legacy alias
    def on_end_of_run_period(self, state) -> int:
 
        return self._emit_annual_summary(state)

    def _emit_annual_summary(self, state) -> int:

        elapsed = time.monotonic() - self._run_wall_t0
        self._close_sidecar()

        try:
            T_charge_h = (
                (self.ev_capacity_kwh * (1.0 - self.forced_soc_frac))
                / max(0.1, EV_CHARGER_KW_NET)
            ) if self.has_ev else 0.0
            tau_max_h = self._algorithm1_tau_max(is_weekend=False) / 60.0
            tau_max_safe = max(0.1, tau_max_h)
            simultaneity = T_charge_h / tau_max_safe
            elastic_factor = 1.0 if self.is_elastic else 0.0
            lyap_h = (simultaneity ** 2) * elastic_factor
        except Exception:
            lyap_h = float("nan")

        # average SoC at departure (across all observed departures)
        if self._stat_dep_events > 0:
            avg_soc_at_dep_kwh = (
                self._stat_dep_soc_at_dep_kwh / self._stat_dep_events
            )
        else:
            avg_soc_at_dep_kwh = float("nan")

        # write annual_summary.json - the PRIMARY emission channel.
        try:
            import json
            from pathlib import Path
            summary = {
                "plugin_version": "12.5",
                "sim_seed": int(SIM_SEED),
                "wall_seconds": float(elapsed),
                "scenario": int(self.scenario_mode),
                "elastic": bool(self.is_elastic),
                "tau_min": int(self.smart_delay_min),
                "stagger_mode": str(ALGO1_STAGGER_MODE),
                "nhts_dist_mi": float(self.ev_distance_miles),
                "soc_arrival": float(self.forced_soc_frac),
                "building_kwh": float(self._stat_building_kwh),
                "pv_kwh": float(self._stat_pv_kwh),
                "ev_charged_kwh": float(self._stat_ev_kwh),
                "v2g_kwh": float(self._stat_v2g_kwh),
                "v2g_offsite_cost_usd": float(self._stat_offsite_cost_usd),
                "batt_kwh": float(self._stat_batt_kwh),
                "co2_avoided_kg": float(self._stat_co2_saved_kg),
                "co2_peak_kg": float(self._stat_co2_peak_kg),
                "co2_offpeak_kg": float(self._stat_co2_offpeak_kg),
                "markov_rho": float(self.markov_rho),
                "battery_soh_frac": float(self.battery_soh_frac),
                "lyap_h": float(lyap_h)
                          if lyap_h == lyap_h else None,    
                "dep_events": int(self._stat_dep_events),
                "dep_misses": int(self._stat_dep_deadline_misses),
                "avg_soc_at_dep_kwh": float(avg_soc_at_dep_kwh)
                                      if avg_soc_at_dep_kwh == avg_soc_at_dep_kwh
                                      else None,
                "drive_consumed_kwh": float(self._stat_drive_kwh_consumed),
            }
            import os
            out_dir = os.environ.get("ENERGYPLUS_OUTPUT_DIR", ".")
            out_path = Path(out_dir) / "annual_summary.json"
            out_path.write_text(json.dumps(summary, indent=2))
        except Exception:
            pass

        try:
            self.api.runtime.issue_warning(state,
                f"[OccupancyPlugin v12.5] , ANNUAL SUMMARY ,  "
                f"Wall={elapsed:.1f}s | "
                f"S={self.scenario_mode} | "
                f"elastic={self.is_elastic} | "
                f"tau={self.smart_delay_min}min | "
                f"NHTS_dist={self.ev_distance_miles:.1f}mi | "
                f"SoC_arrival={self.forced_soc_frac:.3f} | "
                f"Building={self._stat_building_kwh:.1f}kWh | "
                f"PV={self._stat_pv_kwh:.1f}kWh | "
                f"EV_charged={self._stat_ev_kwh:.1f}kWh | "
                f"V2G={self._stat_v2g_kwh:.1f}kWh | "
                f"V2G_offsite_cost=${self._stat_offsite_cost_usd:.2f} | "
                f"Batt={self._stat_batt_kwh:.1f}kWh | "
                f"CO2_avoided={self._stat_co2_saved_kg:.1f}kg "
                f"(peak={self._stat_co2_peak_kg:.1f}kg "
                f"offpeak={self._stat_co2_offpeak_kg:.1f}kg) | "
                f"markov_rho={self.markov_rho:.2f} | "
                f"SoH={self.battery_soh_frac:.3f} | "
                f"Lyap_h={lyap_h:.4f} | "
                f"dep_events={self._stat_dep_events} | "
                f"dep_misses={self._stat_dep_deadline_misses} | "
                f"avg_soc_at_dep={avg_soc_at_dep_kwh:.2f}kWh | "
                f"drive_consumed={self._stat_drive_kwh_consumed:.1f}kWh"
            )
        except Exception:
            pass
        return 0