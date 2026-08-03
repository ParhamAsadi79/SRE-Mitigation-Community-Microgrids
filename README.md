# SRE-Mitigation-Community-Microgrids

Replication package for:

> **Mitigating the Synchronization Rebound Effect in Community Microgrids: A
> Communication-Free Stagger with Star-Discrepancy Guarantees**
> Parham Asadi, Navid Shirzadi, Fuzhan Nasiri
> *Applied Energy* (under review).
> Archived release: https://doi.org/10.5281/zenodo.21520013

This repository contains the dispatcher, the EnergyPlus co-simulation harness, the
analysis pipeline, the validation suite, and the datasets underlying every figure,
table, and reported number in the manuscript and its Supplementary Material.

---

## 1. What the study does

Time-of-use (TOU) tariffs intended to flatten residential demand instead steer every
price-responsive device toward the same low-price hour, resynchronizing a
neighbourhood into a coincident peak above the pre-control baseline. We call this the
**synchronization rebound effect (SRE)** and show, across a 9,000-run EnergyPlus 26.1
sweep (36 conditions x 5 scenarios x 50 houses), that it more than doubles the
community coincidence factor (SRE = 2.035) and is driven by synchronized
behind-the-meter **battery** recharge (250 kW), not by vehicle charging (43 kW even at
full electrification).

The mitigation is a three-layer, **communication-free** dispatcher:

1. a van der Corput base-2 stagger computed locally from an immutable house identifier
   (star discrepancy `O(log N / N)`, tighter than the `O(N^-1/2)` of i.i.d. delay);
2. a deadline-feasibility rate gate; and
3. a price-and-carbon shadow-price arbiter.

**Scenarios.** `S0` uncontrolled (random arrival, no DR) · `S1` Algorithm 1 (the
method) · `S2` TOU rebound (the operative counterfactual) · `S3` flat tariff ·
`S4` fixed valley-fill · `S5` one-way broadcast (Section 6.3 experiment).

---

## 2. Requirements

| Component | Version used |
|---|---|
| Python | 3.11 (3.10+ supported) |
| EnergyPlus | **26.1**, built with the Python Plugin interface enabled |
| OS | Windows 10/11 or Linux (commands below give both forms) |

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins the interpreter-level dependencies (numpy, pandas, scipy,
matplotlib, SALib, scikit-learn, tslearn, statsmodels). The LP-OPF bound is solved with
the HiGHS dual-revised simplex through `scipy.optimize.linprog`; the exact solver
version is recorded in `requirements.txt`.

**EnergyPlus.** `Run_Manager.py` invokes the EnergyPlus executable directly. Set the
install path at the top of `Run_Manager.py` (`EPLUS_DIR`) before the first run.

**Weather.** The sweep uses the TMY3 file for San Francisco Intl. (WMO 724940). It is
redistributed under the EnergyPlus weather-data terms as
`weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw`; if your copy is absent,
download it from the EnergyPlus weather repository and place it at that path.

**Reproducibility.** The entire sweep descends from a single integer seed
(`SIM_SEED = 42` in `Run_Manager.py`); every per-house draw is derived through a
platform-independent BLAKE2b digest of the house identifier, so results are identical
across processes and machines. The analysis seed (`RANDOM_STATE = 42` in
`IEEE_Sensitivity_Analysis.py`) governs the bootstrap and the Gaussian-process
surrogate.

---

## 3. Repository contents

### Simulation core
| File | Role |
|---|---|
| `Run_Manager.py` | Sweep orchestrator: builds the 36-cell factorial, launches EnergyPlus per house/scenario, aggregates `SRE_metrics_summary.csv`. |
| `OccupancyPlugin.py` | The EnergyPlus Python Plugin implementing **Algorithm 1** (vdC stagger, rate gate, shadow-price arbiter), the battery policy, the PV submodel, and the occupancy Markov chain. |
| `Generate_NHTS_Data.py` | Builds `NHTS_Fleet_Data.csv` (50-vehicle fleet) from the 2017 NHTS daily-VMT distribution. |
| `ShopWithPVandBattery.idf`, `HVACTemplate-5ZonePTHP.idf` | Base EnergyPlus models. |
| `config_house_21.json` | Example per-house configuration emitted by the manager. |

### Analysis and figures
| File | Role |
|---|---|
| `IEEE_Sensitivity_Analysis.py` | Master analysis pipeline: all data figures (F01–F16) and tables (T01–T08), BCa bootstrap, Cliff's delta, Sobol decomposition, k-Shape archetypes. |
| `make_schematic_figures1.py` | Schematic figures (Fig. 1–6, Fig. 11, the introduction SRE mechanism) and the **graphical abstract**. |
| `active_set_apriori.py`, `active_set_theory.py` | Numerical verification of **Proposition 1** (a-priori active-set overshoot bound) and the realized discrepancy certificate. |
| `odc_vs_stagger.py` | Controlled head-to-head against the communicated decentralized optimum of Gan et al. (2013). |
| `odc_penetration_sweep.py` | The same comparison swept across EV penetration (Table 8, F12). |
| `odc_resilience.py`, `plot_resilience.py` | Communication-degradation and forecast-error resilience (Section 5.7, F13). |
| `iid_stagger_ablation.py`, `analyze_iid_ablation.py` | The i.i.d.-randomized stagger ablation (Supplementary S10, F16/T08). |
| `charge_power_analysis.py` | Charge-power sensitivity support (F11/T05). |
| `fair_benchmarks.py`, `yu2024_comparison.py` | Auxiliary comparisons against published baselines. |

### Validation suite
| File | Checks |
|---|---|
| `validate_pipeline.py` | Metric recomputation, per-house energy balance, EV deadline audit, LP-OPF sanity, Sobol sanity, discrepancy certificate. |
| `validate_physical_fidelity.py` | PV yield against PVWatts and the building energy benchmark. |
| `validate_load_intensity.py` | Annual electricity intensity against the ResStock CZ-3C benchmark. |
| `deadline_audit.py` | Reconstructs the realized EV deadline-miss rate from per-house telemetry (cited in Section 4.1). |
| `aggregate_override_rate.py` | Layer-2 safety-override rate, overall and per scenario. |
| `mtr_annual_totals.py` | Annual `Electricity:Facility` / `:Building` / `:HVAC` totals from the `.mtr` files. |
| `compare_heating.py` | Heating energy by building archetype. |
| `check_seasonal_tariff.py` | Confirms the E-TOU-C seasonal price surface used by the plugin. |
| `verify_seed_fix.py`, `verify_stagger_mode_patch.py` | Regression checks on the seeding discipline and the stagger-mode flag. |

### Datasets
| File | Contents |
|---|---|
| `SRE_metrics_summary.csv` | **Primary dataset.** 216 rows = 36 conditions x 6 scenarios (180 rows S0–S4 for the main sweep, 36 broadcast rows). |
| `SRE_metrics_summary_BCAST_HET_csv.csv` | Heterogeneous-threshold one-way broadcast (S5), Section 6.3. |
| `SRE_metrics_summary_BCAST_UNIFORM.csv` | Uniform-response one-way broadcast (S5), Section 6.3. |
| `SRE_metrics_summary_ablation_vdc.csv` / `..._iid.csv` | Paired vdC vs i.i.d. ablation arms (Supplementary S10). |
| `results_7kw.csv`, `results_11kw.csv`, `results_19kw.csv` | Battery-free charge-power sweep at 7.0 / 11.5 / 19.2 kW (F11/T05, Supplementary S4). |
| `Sobol_Sensitivity_Indices.csv`, `Sobol_Surrogate_Indices.csv`, `Sobol_Surrogate_S2_Interactions.csv` | Sobol first-/total-order indices (Pass-1 Saltelli and Pass-2 GP surrogate). |
| `NHTS_Fleet_Data.csv` | The 50-vehicle fleet (distance, energy, arrival SoC, arrival hour). |
| `odc_penetration_sweep.csv` | ODC vs stagger across penetration (Table 8). |
| `iid_ablation_*.csv` | Scaling, deployment, overshoot, and paired-result summaries for the ablation. |
| `meter_peaks.csv` | Per-run meter peaks (9,000 rows) used by the validation suite. |
| `T01…T08_*.csv` / `.tex` | Generated manuscript tables. |

---

## 4. Reproducing the study

> **Path convention.** Commands are shown with a Windows line-continuation `^`.
> On Linux/macOS replace `^` with `\` and `\` in paths with `/`.
> All commands assume you are in the repository root.

### 4.0 Preserve any previous archive

Re-running writes into the same tree. Skipping this step silently returns the **old**
data:

```bat
ren Results_Sensitivity Results_Sensitivity_OLDSEED
```

### 4.1 Phase 1 — canonical sweep S0–S4 (9,000 runs, ~2.5 h)

Settings: `OccupancyPlugin.py` → `ALGO1_STAGGER_MODE = "vdc"`;
`Run_Manager.py` → `ABLATION_ARM = "off"`, `ENABLE_S5 = False`.

```bash
python Run_Manager.py
```

### 4.2 Phase 2 — add the broadcast scenario S5 (+1,800 runs, ~30 min)

Settings: `Run_Manager.py` → `ENABLE_S5 = True`. S0–S4 are skipped by design in this
pass; that is correct, the existing S1 profile feeds the broadcast.

```bash
python Run_Manager.py
copy Results_Sensitivity\SRE_metrics_summary.csv SRE_metrics_summary_BCAST_HET.csv
```

### 4.3 Uniform-response broadcast variant

```bat
REM 1. Plugin: make every house respond identically (the naive policy)
REM    ALGO1_BROADCAST_THETA_MIN = 0.0
REM    ALGO1_BROADCAST_THETA_MAX = 0.0
REM 2. Delete ONLY the S5 runs; S0-S4 must stay.
for /d %d in (Results_Sensitivity\Rho*_Pen*_Inel*) do @if exist "%d\Scenario_5" rmdir /s /q "%d\Scenario_5"
REM 3. ENABLE_S5 = True, ABLATION_ARM = "off"   (~30 min)
python Run_Manager.py
copy Results_Sensitivity\SRE_metrics_summary.csv SRE_metrics_summary_BCAST_UNIFORM.csv
REM 4. RESTORE the canonical tree: theta back to 0.15 / 0.85, delete the S5 dirs again, re-run 4.2.
```

### 4.4 Battery-penetration sweep (F14 / T06)

```
# OccupancyPlugin.py
ALGO1_STAGGER_MODE = "vdc";  ENABLE_BATTERY = True;  EV_CHARGE_POWER_KW = 7.0
# Run_Manager.py
ABLATION_ARM = "off";  BATTERY_PEN_SWEEP = True;  ENABLE_S5 = False
BASE_OUT_DIR = Path("Results_BattPen").resolve()
```

```bash
python Run_Manager.py
```

### 4.5 Battery-free charge-power sweep (F11 / T05)

Run three times, changing only the charge power:

```
# OccupancyPlugin.py — ENABLE_BATTERY = False for all three
ENABLE_BATTERY = False
EV_CHARGE_POWER_KW = 7.0      # then 11.5, then 19.2
# Run_Manager.py
ABLATION_ARM = "off";  BATTERY_PEN_SWEEP = False;  ENABLE_S5 = False
BASE_OUT_DIR = Path("Results_Power7").resolve()    # then _Power11, _Power19
```

```bash
python Run_Manager.py
copy Results_Power7\SRE_metrics_summary.csv  results_7kw.csv
copy Results_Power11\SRE_metrics_summary.csv results_11kw.csv
copy Results_Power19\SRE_metrics_summary.csv results_19kw.csv
```

### 4.6 i.i.d.-stagger ablation (Supplementary S10, F16 / T08)

Both arms sweep the same 36 conditions under the same per-house seeds; only the
Layer-1 construction differs (`ALGO1_STAGGER_MODE = "vdc"` vs `"iid"`, with
`ABLATION_ARM` set accordingly). Then:

```bash
python analyze_iid_ablation.py
```

which writes `iid_ablation_paired_result.csv` and `iid_ablation_per_condition.csv`
(36 rows: matched per-cell peaks, CF, the peak difference, and the conservation
invariants).

### 4.7 Regenerate every publication asset

```bat
python IEEE_Sensitivity_Analysis.py --results-dir Results_Sensitivity ^
  --power-csvs 7.0:results_7kw.csv 11.5:results_11kw.csv 19.2:results_19kw.csv ^
  --battpen-csv Results_BattPen\SRE_metrics_summary.csv ^
  --broadcast-csv SRE_metrics_summary_BCAST_HET.csv ^
  --broadcast-naive-csv SRE_metrics_summary_BCAST_UNIFORM.csv ^
  --ablation-vdc-csv SRE_metrics_summary_ablation_vdc.csv ^
  --ablation-iid-csv SRE_metrics_summary_ablation_iid.csv
```

Outputs land in `Results_Sensitivity/Publication/` (override with `--out-subdir`).
Useful switches: `--skip-figures`, `--skip-tables`, `--skip-html`.

Then the schematic figures and the graphical abstract:

```bash
python make_schematic_figures1.py
```

And the standalone experiments:

```bash
python odc_vs_stagger.py            # Fig. 14, the head-to-head
python odc_penetration_sweep.py     # Table 8, F12
python odc_resilience.py            # Section 5.7
python plot_resilience.py           # Fig. 15 / F13
python active_set_apriori.py        # Proposition 1 verification (Fig. 4)
```

---

## 5. Validation and audit suite

Every check below was run before submission. Replace `<RESULTS>` with your results
directory (e.g. `Results_Sensitivity`).

**Full pipeline check** — metric recomputation, per-house energy balance, EV deadline
audit, LP-OPF bound sanity, Sobol sanity, discrepancy certificate:

```bash
python validate_pipeline.py --results <RESULTS> --csv SRE_metrics_summary.csv --sobol-dir . --log Swarm_Execution.log --sample 0
```

**PV yield and building benchmark** (PVWatts comparison):

```bash
python validate_physical_fidelity.py <RESULTS> --scenario 0
```

**EV deadline audit** (per-house SoC traces and realized miss rate):

```bash
python deadline_audit.py --results <RESULTS>
python deadline_audit.py --results <RESULTS> --capacity 50 --target-frac 1.0
```

**Layer-2 safety-override rate**, overall and per scenario:

```bash
python aggregate_override_rate.py --results <RESULTS>
```

**Load intensity** against the ResStock CZ-3C benchmark:

```bash
python validate_load_intensity.py --resstock-hint
python validate_load_intensity.py --results <RESULTS> --benchmark-mean 6500 --benchmark-lo 4000 --benchmark-hi 9500 --floor-area-m2 140
```

**Annual meter totals** (`Electricity:Facility`, `:Building`, `:HVAC`):

```bash
python mtr_annual_totals.py --results <RESULTS> --sample-per-cell 2
python mtr_annual_totals.py eplusout.mtr
```

**Heating by archetype**:

```bash
python compare_heating.py --results <RESULTS> --sample-per-arch 3
```

**Tariff surface and seeding regressions**:

```bash
python check_seasonal_tariff.py
python verify_seed_fix.py
python verify_stagger_mode_patch.py
```

---

## 6. Figure and table provenance

| Asset | Produced by | Primary input |
|---|---|---|
| Fig. 1–6, 11, graphical abstract | `make_schematic_figures1.py` | schematic / manuscript constants |
| Fig. 4 (Proposition 1) | `active_set_apriori.py` | analytic |
| F01–F10 | `IEEE_Sensitivity_Analysis.py` | `SRE_metrics_summary.csv` |
| F11 / T05 | `IEEE_Sensitivity_Analysis.py` | `results_{7,11,19}kw.csv` |
| F12, Table 8 | `odc_penetration_sweep.py` | `NHTS_Fleet_Data.csv` |
| F13 | `plot_resilience.py` | `odc_resilience.py` output |
| F14 / T06 | `IEEE_Sensitivity_Analysis.py` | `--battpen-csv` |
| F15 / T07 (Table 9) | `IEEE_Sensitivity_Analysis.py` | broadcast CSVs |
| F16 / T08 | `IEEE_Sensitivity_Analysis.py` | ablation arm CSVs |
| T01–T04 | `IEEE_Sensitivity_Analysis.py` | `SRE_metrics_summary.csv` |

---

## 7. Headline results (for cross-checking a fresh run)

| Quantity | Value |
|---|---|
| Rebound ratio SRE = CF(S2)/CF(S0) | 2.035, BCa 95% CI [1.931, 2.160] |
| Mitigation index MIT = CF(S1)/CF(S2) | 0.859, BCa 95% CI [0.848, 0.871]; below 1.0 in 36/36 cells |
| Community peak, S1 vs S2 | 224.5 kW vs 310.2 kW |
| Battery contribution at the peak instant (S2) | 250 kW (50 packs x 5 kW) |
| Coincident EV draw (S2) | 7 kW at 25% penetration → 43 kW at 100% |
| Storage scaling of the S2 peak | R² = 0.9998 |
| Battery-free rebound (7 kW) | SRE = 1.143 |
| Annual bill reduction vs rebound | $38,625 (12.8%) |
| Modelled grid CO₂ reduction vs rebound | 3.7% |
| Jain fairness (S1) | 0.861, level with the rebound (0.860) |
| Recovery of the communicated optimum | 59.8% mean (54.3–66.4%), 64.1% at full electrification |
| Connectivity crossover | 48.2% of the fleet unreachable |

---

## 8. Citation

```bibtex
@article{Asadi2026SRE,
  author  = {Asadi, Parham and Shirzadi, Navid and Nasiri, Fuzhan},
  title   = {Mitigating the Synchronization Rebound Effect in Community
             Microgrids: A Communication-Free Stagger with Star-Discrepancy
             Guarantees},
  journal = {Applied Energy},
  year    = {2026},
  note    = {Replication package: https://doi.org/10.5281/zenodo.21520013}
}
```

## 9. License

Released under the MIT License — see `LICENSE`. The EnergyPlus weather file and the
2017 NHTS source data are redistributed under their own terms.

## 10. Contact

Parham Asadi — `as_parha@live.concordia.ca`
Department of Building, Civil and Environmental Engineering, Concordia University,
1455 De Maisonneuve Blvd. W., Montréal, H3G 1M8, QC, Canada.
