from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import re
import numpy as np
import pandas as pd
import scipy.stats as stats

import matplotlib
matplotlib.use("Agg")          # no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap, Normalize
import seaborn as sns

# Interactive HTML reporting
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    warnings.warn("plotly not installed - interactive HTML report will be skipped.")

# IEEE PUBLICATION STYLE
IEEE_STYLE: Dict[str, object] = {
    "font.family":         ["serif"],
    "font.serif":          ["Times New Roman", "Nimbus Roman",
                            "Liberation Serif", "DejaVu Serif"],
    "font.size":            8.5,
    "axes.titlesize":       9.5,
    "axes.labelsize":       8.5,
    "xtick.labelsize":      7.5,
    "ytick.labelsize":      7.5,
    "legend.fontsize":      7.5,
    "figure.titlesize":     10.0,
    "axes.linewidth":       0.8,
    "axes.spines.top":      False,
    "axes.spines.right":    False,
    "xtick.major.width":    0.8,
    "ytick.major.width":    0.8,
    "xtick.major.size":     3.0,
    "ytick.major.size":     3.0,
    "lines.linewidth":      1.2,
    "lines.markersize":     4.0,
    "savefig.bbox":         "tight",
    "savefig.pad_inches":   0.05,
    "savefig.dpi":          300,
    "figure.dpi":           120,
    "pdf.fonttype":         42,        # embed TrueType 
    "ps.fonttype":          42,
    "axes.grid":            True,
    "grid.alpha":           0.25,
    "grid.linestyle":       ":",
    "grid.linewidth":       0.5,
}

# Wong 2011 colorblind-safe palette (Nature Methods, 2011, "Points of view: Color blindness")
COLORS_WONG: Dict[str, str] = {
    "black":      "#000000",
    "orange":     "#E69F00",
    "skyblue":    "#56B4E9",
    "green":      "#009E73",
    "yellow":     "#F0E442",
    "blue":       "#0072B2",
    "vermillion": "#D55E00",
    "purple":     "#CC79A7",
}

# Per-scenario colour mapping 
SCENARIO_COLORS: Dict[int, str] = {
    0: COLORS_WONG["black"],       # Baseline (no DR)
    1: COLORS_WONG["green"],       # Algorithm 1 - the mitigation hero
    2: COLORS_WONG["vermillion"],  # TOU rebound - the worst case
    3: COLORS_WONG["skyblue"],     # Flat-tariff baseline
    4: COLORS_WONG["purple"],      # Valley-fill
}
SCENARIO_NAMES: Dict[int, str] = {
    0: "S0: Random Arrival (No DR)",
    1: "S1: Algorithm 1 (Mitigation)",
    2: "S2: TOU Rebound",
    3: "S3: Flat Tariff",
    4: "S4: Valley-Fill",
}
SCENARIO_NAMES_SHORT: Dict[int, str] = {
    0: "S0",
    1: "S1",
    2: "S2",
    3: "S3",
    4: "S4",
}

# Penetration to marker / hatch (for consistency across grouped figures)
PEN_MARKERS: Dict[float, str] = {0.25: "o", 0.50: "s", 0.75: "D", 1.00: "^"}
PEN_LABELS:  Dict[float, str] = {0.25: "25%", 0.50: "50%", 0.75: "75%", 1.00: "100%"}

# Single-column / double-column widths (inches)
WIDTH_SINGLE = 3.5
WIDTH_DOUBLE = 7.16

# LOGGING
LOG_FMT = "%(asctime)s | %(levelname)-7s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("IEEE_Analysis")

# Silence noisy upstream warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="seaborn")

# Silence the very chatty fontTools subset logger that fires whenever
logging.getLogger("fontTools").setLevel(logging.WARNING)
logging.getLogger("fontTools.subset").setLevel(logging.WARNING)
logging.getLogger("fontTools.ttLib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)


# DATA LAYER
@dataclass
class AnalysisData:
    """Container holding the loaded SRE_metrics_summary.csv and derived views."""
    df:           pd.DataFrame                # full 180-row table (all 5 scenarios)
    df_s0:        pd.DataFrame                # scenario 0 (no DR baseline)
    df_s1:        pd.DataFrame                # scenario 1 (Algorithm 1)
    df_s2:        pd.DataFrame                # scenario 2 (TOU rebound)
    df_s3:        pd.DataFrame                # scenario 3 (inelastic-only)
    df_s4:        pd.DataFrame                # scenario 4 (valley-fill)
    sobol_p1:     Optional[pd.DataFrame] = None   # Sobol_Sensitivity_Indices.csv
    sobol_p2:     Optional[pd.DataFrame] = None   # Sobol_Surrogate_Indices.csv
    sobol_s2:     Optional[pd.DataFrame] = None   # Sobol_Surrogate_S2_Interactions.csv
    n_conditions: int = 0
    n_scenarios:  int = 0
    pens:         Sequence[float] = field(default_factory=list)
    inels:        Sequence[float] = field(default_factory=list)
    rhos:         Sequence[float] = field(default_factory=list)


def load_data(results_dir: Path) -> AnalysisData:

    csv_path = results_dir / "SRE_metrics_summary.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"SRE_metrics_summary.csv not found at {csv_path}. "
            f"Run Run_Manager.py Stage 4 first."
        )

    df = pd.read_csv(csv_path)
    log.info(f"Loaded {len(df):,} rows × {len(df.columns)} columns from {csv_path.name}")

    numeric_cols = [c for c in df.columns if c not in ("scenario",)]
    for c in numeric_cols:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    _KNOWN_SCENARIOS = [0, 1, 2, 3, 4]
    _extra = sorted(int(s) for s in df["scenario"].unique() if s not in _KNOWN_SCENARIOS)
    if _extra:
        log.warning(f"Dropping unlabeled scenario(s) {_extra} not in the S0-S4 set; "
                    f"the published analysis covers S0-S4.")
        df = df[df["scenario"].isin(_KNOWN_SCENARIOS)].copy()

    sobol_p1 = _read_optional_csv(results_dir / "Sobol_Sensitivity_Indices.csv")
    sobol_p2 = _read_optional_csv(results_dir / "Sobol_Surrogate_Indices.csv")
    sobol_s2 = _read_optional_csv(results_dir / "Sobol_Surrogate_S2_Interactions.csv")

    pens  = sorted(df["pen_rate"].unique())
    inels = sorted(df["inel_rate"].unique())
    rhos  = sorted(df["markov_rho"].unique())

    n_cond = len(df.drop_duplicates(subset=["pen_rate", "inel_rate", "markov_rho"]))
    n_scen = df["scenario"].nunique()

    log.info(
        f"Design: {n_cond} conditions × {n_scen} scenarios | "
        f"pen={pens} | inel={inels} | rho={rhos}"
    )

    return AnalysisData(
        df=df,
        df_s0=df[df["scenario"] == 0].copy(),
        df_s1=df[df["scenario"] == 1].copy(),
        df_s2=df[df["scenario"] == 2].copy(),
        df_s3=df[df["scenario"] == 3].copy(),
        df_s4=df[df["scenario"] == 4].copy(),
        sobol_p1=sobol_p1,
        sobol_p2=sobol_p2,
        sobol_s2=sobol_s2,
        n_conditions=n_cond,
        n_scenarios=n_scen,
        pens=pens,
        inels=inels,
        rhos=rhos,
    )


def _read_optional_csv(p: Path) -> Optional[pd.DataFrame]:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            log.warning(f"Failed to read {p.name}: {e}")
    else:
        log.info(f"Optional file not found (skipping): {p.name}")
    return None


# STATISTICAL UTILITIES

def bootstrap_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot:    int   = 5_000,
    alpha:     float = 0.05,
    seed:      int   = 42,
    rng:       Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:

    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    point = float(statistic(x))
    if x.size < 2:
        return (point, point, point)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    boots = np.array([statistic(x[i]) for i in idx])
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (point, float(lo), float(hi))


def bca_ci(
    x: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot:    int   = 10_000,
    alpha:     float = 0.05,
    seed:      int   = 42,
    rng:       Optional[np.random.Generator] = None,
) -> Tuple[float, float, float]:

    if rng is None:
        rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return (np.nan, np.nan, np.nan)
    point = float(statistic(x))
    if n < 2:
        return (point, point, point)
    # Bootstrap replicates of the statistic.
    idx   = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([statistic(x[i]) for i in idx])
    # Bias-correction z0 from the share of replicates below the estimate.
    prop = np.clip(np.mean(boots < point), 1.0 / n_boot, 1.0 - 1.0 / n_boot)
    z0   = float(stats.norm.ppf(prop))
    # Acceleration a from the jackknife (skewness of leave-one-out estimates).
    jack = np.array([statistic(np.delete(x, i)) for i in range(n)])
    jbar = jack.mean()
    num  = np.sum((jbar - jack) ** 3)
    den  = 6.0 * (np.sum((jbar - jack) ** 2)) ** 1.5
    a    = float(num / den) if den != 0 else 0.0
    # BCa-adjusted percentile endpoints.
    zl, zh = float(stats.norm.ppf(alpha / 2)), float(stats.norm.ppf(1 - alpha / 2))
    a1 = float(stats.norm.cdf(z0 + (z0 + zl) / (1 - a * (z0 + zl))))
    a2 = float(stats.norm.cdf(z0 + (z0 + zh) / (1 - a * (z0 + zh))))
    lo, hi = np.percentile(boots, [100 * a1, 100 * a2])
    return (point, float(lo), float(hi))


def cliff_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta non-parametric effect size on [-1, +1].
      |δ| < 0.147 - negligible
      |δ| < 0.33  - small
      |δ| < 0.474 - medium
      otherwise   - large
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")

    diff = x[:, None] - y[None, :]
    return float(np.sign(diff).mean())


def cliff_delta_label(d: float) -> str:
    """Map Cliff's delta magnitude to a qualitative label."""
    if np.isnan(d):
        return "n/a"
    a = abs(d)
    if a < 0.147: return "negligible"
    if a < 0.33:  return "small"
    if a < 0.474: return "medium"
    return "large"


def wilcoxon_test(
    x: np.ndarray,
    y: np.ndarray,
    alternative: str = "less",
) -> Dict[str, float]:

    x, y = np.asarray(x), np.asarray(y)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return dict(stat=np.nan, p=np.nan, n=len(x), delta=np.nan, test="too_few")
    try:
        if len(x) == len(y):
            res = stats.wilcoxon(x, y, alternative=alternative, zero_method="wilcox")
            test_name = "wilcoxon_signed_rank"
        else:
            res = stats.mannwhitneyu(x, y, alternative=alternative)
            test_name = "mann_whitney_u"
        delta = cliff_delta(x, y)
        return dict(
            stat=float(res.statistic),
            p=float(res.pvalue),
            n=int(len(x)),
            delta=float(delta),
            test=test_name,
        )
    except Exception as e:
        log.warning(f"Wilcoxon failed: {e}")
        return dict(stat=np.nan, p=np.nan, n=len(x), delta=np.nan, test="failed")


def fmt_p(p: float) -> str:

    if np.isnan(p):
        return "n/a"
    if p < 1e-3:
        return r"$< 0.001$"
    if p < 1e-2:
        return f"${p:.3f}$"
    return f"${p:.2f}$"


def fmt_value_ci(point: float, lo: float, hi: float, decimals: int = 3) -> str:
    """Format point [lo, hi] for tables/captions."""
    if np.isnan(point):
        return "n/a"
    fmt = f"{{:.{decimals}f}}"
    return f"{fmt.format(point)} [{fmt.format(lo)}, {fmt.format(hi)}]"


# FIGURE GENERATION HELPERS

def setup_style() -> None:
    """Apply IEEE TSG matplotlib style globally."""
    plt.rcParams.update(IEEE_STYLE)
    sns.set_style("ticks", rc={"axes.spines.top": False, "axes.spines.right": False})



DRAW_ON_ARTWORK_TITLES: bool = False


def figure_title(fig: plt.Figure, text: str, **kw) -> None:
    """Draw an on-artwork figure title only in draft mode; no-op for submission."""
    if DRAW_ON_ARTWORK_TITLES:
        fig.suptitle(text, **kw)


def axes_title(ax: plt.Axes, text: str, **kw) -> None:

    if DRAW_ON_ARTWORK_TITLES:
        ax.set_title(text, **kw)



import json as _json, datetime as _dt

PROVENANCE_OWNER = "IEEE_Sensitivity_Analysis.py"


def _claim_output(out_dir: Path, base_name: str) -> None:
    """Refuse to overwrite a figure owned by a different generator."""
    side = Path(out_dir) / f"{base_name}.provenance.json"
    if side.exists():
        try:
            prev = _json.loads(side.read_text()).get("owner")
        except Exception:
            prev = None
        if prev and prev != PROVENANCE_OWNER:
            raise SystemExit(
                f"\nREFUSING TO OVERWRITE: '{base_name}' is owned by {prev}, "
                f"not by {PROVENANCE_OWNER}.\n"
                f"Two generators are competing for one base name. Rename one of them; "
                f"a silent overwrite here produces a figure that contradicts its caption.\n"
                f"If this is deliberate, delete {side} and re-run.\n")
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(_json.dumps(
        {"owner": PROVENANCE_OWNER,
         "written": _dt.datetime.now().isoformat(timespec="seconds")}, indent=2))


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    name: str,
    *,
    formats: Tuple[str, ...] = ("png", "pdf", "svg"),
) -> List[Path]:
    """Save figure in multiple formats (PNG@300 for review, PDF/SVG for LaTeX)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _claim_output(out_dir, name)
    paths: List[Path] = []
    for ext in formats:
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    log.info(f"  saved {name}.{{{','.join(formats)}}}  ({len(paths)} files)")
    return paths


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.18, y: float = 1.05) -> None:
    """Add an IEEE-style lowercase panel label like '(a)' to an Axes."""
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10.0, fontweight="bold",
    )


# FIGURE 1 - HEADLINE MITIGATION RESULT
#   "Algorithm 1 mitigates the SRE rebound across the entire 36-condition
#    experimental design.  Mitigation strengthens monotonically with EV
#    penetration."
# Two-panel layout:
#   (a) Violin + per-condition strip of MIT_ratio vs penetration
#   (b) MIT_ratio vs penetration with bootstrap 95% CIs for the conditional mean
def fig01_mitigation_headline(data: AnalysisData, out_dir: Path) -> None:
    setup_style()
    df = data.df_s1.copy()  # MIT_ratio is a condition-level metric, same across S
    df = df.sort_values("pen_rate")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.2))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel (a): violin + jittered points
    pens = sorted(df["pen_rate"].unique())
    parts = ax_a.violinplot(
        [df[df["pen_rate"] == p]["MIT_ratio"].dropna().values for p in pens],
        positions=range(len(pens)),
        widths=0.7, showmeans=False, showmedians=False, showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(COLORS_WONG["green"])
        pc.set_alpha(0.30)
        pc.set_edgecolor(COLORS_WONG["green"])
        pc.set_linewidth(0.8)

    # Jittered strip
    rng = np.random.default_rng(42)
    for i, p in enumerate(pens):
        y = df[df["pen_rate"] == p]["MIT_ratio"].dropna().values
        x = i + rng.uniform(-0.10, 0.10, size=len(y))
        ax_a.scatter(x, y, s=12, c=COLORS_WONG["green"], alpha=0.85,
                     edgecolors="white", linewidths=0.4, zorder=3)

    # MIT = 1 reference line (above which there is NO mitigation)
    ax_a.axhline(1.0, color=COLORS_WONG["vermillion"], lw=1.2, ls="--", zorder=2)
    ax_a.text(len(pens) - 0.15, 1.005, "no mitigation",
              fontsize=7, ha="right", va="bottom",
              color=COLORS_WONG["vermillion"])

    ax_a.set_xticks(range(len(pens)))
    ax_a.set_xticklabels([PEN_LABELS[p] for p in pens])
    ax_a.set_xlabel("EV penetration")
    ax_a.set_ylabel(r"$MIT = CF(\mathrm{S1}) / CF(\mathrm{S2})$")
    ax_a.set_ylim(0.40, 1.05)
    add_panel_label(ax_a, "(a)")

    # Panel (b): conditional mean + bootstrap 95% CI
    # Use a single RNG so the four per-penetration bootstrap calls draw
    # independent resampling sequences instead of identical ones.
    rng_b = np.random.default_rng(seed=42)
    means_lo, means, means_hi = [], [], []
    for p in pens:
        y = df[df["pen_rate"] == p]["MIT_ratio"].dropna().values
        m, lo, hi = bootstrap_ci(y, statistic=np.mean, rng=rng_b)
        means.append(m); means_lo.append(lo); means_hi.append(hi)

    means    = np.array(means)
    means_lo = np.array(means_lo)
    means_hi = np.array(means_hi)
    err = np.vstack([means - means_lo, means_hi - means])

    ax_b.errorbar(
        pens, means, yerr=err,
        fmt="o-", color=COLORS_WONG["green"],
        ecolor=COLORS_WONG["green"], elinewidth=1.0, capsize=3.5,
        markersize=6, markerfacecolor="white", markeredgewidth=1.4,
        label=r"$\mathrm{mean}\,MIT \pm 95\%\,\mathrm{CI}$",
    )
    ax_b.axhline(1.0, color=COLORS_WONG["vermillion"], lw=1.2, ls="--",
                 label="no mitigation threshold")
    ax_b.set_xticks(pens)
    ax_b.set_xticklabels([PEN_LABELS[p] for p in pens])
    ax_b.set_xlabel("EV penetration")
    ax_b.set_ylabel("Mitigation Index (MIT)")
    ax_b.set_ylim(0.40, 1.05)
    ax_b.legend(loc="upper right", framealpha=0.9, fontsize=7.0)
    add_panel_label(ax_b, "(b)")

    n_total = int(df["MIT_ratio"].notna().sum())
    n_below_1 = int((df["MIT_ratio"] < 1.0).sum())
 
    pen_means = (df.groupby("pen_rate")["MIT_ratio"].mean()
                 .reindex(sorted(df["pen_rate"].unique())))
    sharpest_pen = float(pen_means.idxmin())
    sharpest_val = float(pen_means.min())
    weakest_pen  = float(pen_means.idxmax())
    weakest_val  = float(pen_means.max())
    figure_title(fig, 
        f"Mitigation across {n_total} conditions: $MIT < 1$ in "
        f"{n_below_1}/{n_total} cases.  Sharpest at "
        f"$\\eta={sharpest_pen:.2f}$ "
        f"($MIT={sharpest_val:.2f}$); weakest at "
        f"$\\eta={weakest_pen:.2f}$ "
        f"($MIT={weakest_val:.2f}$).",
        y=1.02, fontsize=9.5,
    )
    save_figure(fig, out_dir, "F01_mitigation_headline")


# FIGURE 2 - CF HEATMAP (THE "MITIGATION MAP")
# A 5*4 heatmap shows CF for each (scenario * penetration) pair.
# This is the figure they scan first to see whether the mitigation
# claim has a clean experimental signature.
def fig02_cf_heatmap(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    # Average CF over (inel * rho) within each (scenario, pen)
    pivot = (data.df
             .groupby(["scenario", "pen_rate"])["CF"]
             .mean()
             .unstack("pen_rate"))
    pivot = pivot.loc[sorted(pivot.index)]
    pivot = pivot[sorted(pivot.columns)]

    cmap = plt.get_cmap("magma_r")

    fig, ax = plt.subplots(figsize=(WIDTH_SINGLE * 1.25, 3.0))

    ny, nx = pivot.values.shape
    im = ax.pcolormesh(
        np.arange(nx + 1) - 0.5,
        np.arange(ny + 1) - 0.5,
        pivot.values,
        cmap=cmap, vmin=0.40, vmax=0.95,
        shading="flat", rasterized=False, linewidth=0, antialiased=False,
    )
    ax.set_xlim(-0.5, nx - 0.5)
    ax.set_ylim(ny - 0.5, -0.5)   # keep S0 at the top, matching imshow

    # Annotate each cell with its CF value
    for i, scen in enumerate(pivot.index):
        for j, pen in enumerate(pivot.columns):
            v = pivot.iloc[i, j]
            text_color = "white" if v > 0.65 else "black"
            ax.text(j, i, f"{v:.2f}",
                    ha="center", va="center",
                    color=text_color, fontsize=8.5, fontweight="bold")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([PEN_LABELS[p] for p in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([SCENARIO_NAMES_SHORT[s] for s in pivot.index])
    ax.set_xlabel("EV penetration")
    ax.set_ylabel("Scenario")

    # Custom colorbar with discrete steps
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Coincidence Factor (CF)", rotation=270, labelpad=12)
    cbar.ax.tick_params(labelsize=7)

    ax.grid(False)  # heatmap doesn't need grid

    figure_title(fig, 
        "CF averaged over inelasticity and Markov persistence",
        y=1.01, fontsize=9.0,
    )
    save_figure(fig, out_dir, "F02_cf_heatmap")


# FIGURE 3 - SOBOL FACTOR DECOMPOSITION
# which experimental factor (penetration, inelasticity, persistence)
# explains the variance of SRE_ratio?  Compare Pass-1 (real data) and Pass-2
# (GP surrogate) side by side.
# If neither Sobol CSV is present, fall back to a discrete-grid Sobol on
# SRE_ratio computed directly from the loaded data.
def fig03_sobol_indices(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    # Load Sobol indices (with fallback)
    p1 = data.sobol_p1
    p2 = data.sobol_p2

    if p1 is None:
        p1 = _fallback_factor_decomposition(data, target="SRE_ratio")
        log.warning("Sobol Pass-1 CSV missing — using ANOVA-based fallback.")

    factors_p1 = list(p1["factor"]) if "factor" in p1.columns else list(p1.index)

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE, 3.0))

    x = np.arange(len(factors_p1))
    width = 0.18

    # Bars for Pass-1
    s1_p1 = p1.set_index("factor")["S1"].reindex(factors_p1).values \
            if "factor" in p1.columns else p1["S1"].values
    st_p1 = p1.set_index("factor")["ST"].reindex(factors_p1).values \
            if "factor" in p1.columns else p1["ST"].values
    s1c_p1 = p1.set_index("factor").get("S1_conf", pd.Series(np.zeros(len(factors_p1)), index=factors_p1)).values
    stc_p1 = p1.set_index("factor").get("ST_conf", pd.Series(np.zeros(len(factors_p1)), index=factors_p1)).values

    bars_a = ax.bar(x - 1.5*width, s1_p1, width, yerr=s1c_p1, capsize=2.5,
                    color=COLORS_WONG["blue"], alpha=0.85,
                    edgecolor="white", linewidth=0.5,
                    label=r"$S_1$ (Pass-1, real data)")

    # Only plot ST bars if values are finite (fallback decomposition skips ST)
    if np.all(np.isfinite(st_p1)):
        bars_b = ax.bar(x - 0.5*width, st_p1, width, yerr=stc_p1, capsize=2.5,
                        color=COLORS_WONG["skyblue"], alpha=0.85,
                        edgecolor="white", linewidth=0.5,
                        label=r"$S_T$ (Pass-1, real data)")

    # Pass-2 GP surrogate
    if p2 is not None:
        s1_p2 = p2.set_index("factor")["S1"].reindex(factors_p1).values
        st_p2 = p2.set_index("factor")["ST"].reindex(factors_p1).values
        s1c_p2 = p2.set_index("factor").get("S1_conf", pd.Series(np.zeros(len(factors_p1)), index=factors_p1)).values
        stc_p2 = p2.set_index("factor").get("ST_conf", pd.Series(np.zeros(len(factors_p1)), index=factors_p1)).values
        ax.bar(x + 0.5*width, s1_p2, width, yerr=s1c_p2, capsize=2.5,
               color=COLORS_WONG["green"], alpha=0.85,
               edgecolor="white", linewidth=0.5,
               label=r"$S_1$ (Pass-2, GP surrogate)")
        ax.bar(x + 1.5*width, st_p2, width, yerr=stc_p2, capsize=2.5,
               color=COLORS_WONG["yellow"], alpha=0.85, edgecolor="black",
               linewidth=0.4,
               label=r"$S_T$ (Pass-2, GP surrogate)")

    factor_labels = {
        "pen_rate":   "EV penetration",
        "inel_rate":  "Inelasticity",
        "markov_rho": "Markov $\\rho$",
    }
    ax.set_xticks(x)
    ax.set_xticklabels([factor_labels.get(f, f) for f in factors_p1])
    ax.set_ylabel("Sobol sensitivity index")
    ax.set_ylim(-0.05, 1.10)
    ax.axhline(0, color="black", lw=0.5, zorder=0)
    ax.legend(loc="upper right", framealpha=0.92, ncol=2, fontsize=7.0)

    figure_title(fig, 
        "Variance decomposition of $SRE$ across the experimental design",
        y=1.01, fontsize=9.5,
    )
    save_figure(fig, out_dir, "F03_sobol_indices")


def _fallback_factor_decomposition(
    data: AnalysisData,
    target: str = "SRE_ratio",
) -> pd.DataFrame:
    
    df = (data.df
          .drop_duplicates(subset=["pen_rate", "inel_rate", "markov_rho"])
          .copy())
    df = df[["pen_rate", "inel_rate", "markov_rho", target]].dropna()
    factors = ["pen_rate", "inel_rate", "markov_rho"]
    var_total = float(df[target].var(ddof=0))   # population variance (ddof=0); numerator below uses the same convention so the ratio is consistent
    rows = []
    for f in factors:
        # First-order S1 = Var[E[Y|X_f]] / Var[Y]
        gm = df.groupby(f)[target].mean()
        # E[Y|X_f] takes one value per group; weight by group size
        sizes = df.groupby(f)[target].size().values
        weights = sizes / sizes.sum()
        mean_total = float(df[target].mean())
        var_first = float(np.sum(weights * (gm.values - mean_total) ** 2))
        s1 = var_first / max(var_total, 1e-12)
        # Clip to [0, 1] for safety — any small overshoot is finite-sample noise
        s1 = float(np.clip(s1, 0.0, 1.0))
        rows.append({
            "factor":  f,
            "S1":      s1,
            "ST":      np.nan,   
            "S1_conf": 0.0,
            "ST_conf": 0.0,
        })
    return pd.DataFrame(rows)


# FIGURE 4 - BILL DECOMPOSITION
# a stacked-bar breakdown of who pays what in each scenario.
# PV credit appears as a NEGATIVE bar (revenue, not expense).
def fig04_bill_decomposition(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    # Average bill components per (scenario, pen_rate); inel and rho averaged
    df = (data.df.groupby(["scenario", "pen_rate"])[
              ["E_baseline_cost_usd", "E_ev_cost_usd",
               "E_batt_cost_usd", "E_pv_credit_usd",
               "demand_charge_usd", "total_bill_usd"]
          ].mean().reset_index())

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.5))
    gs  = GridSpec(1, 5, figure=fig, wspace=0.10)

    component_colors = {
        "E_baseline_cost_usd":   COLORS_WONG["skyblue"],
        "E_ev_cost_usd":         COLORS_WONG["orange"],
        "E_batt_cost_usd":       COLORS_WONG["yellow"],
        "demand_charge_usd":     COLORS_WONG["vermillion"],
        "E_pv_credit_usd":       COLORS_WONG["green"],
    }
    component_labels = {
        "E_baseline_cost_usd":   "Baseline (HVAC + lights + equip)",
        "E_ev_cost_usd":         "EV charging (G2V net)",
        "E_batt_cost_usd":       "Battery charging",
        "demand_charge_usd":     "Demand charge",
        "E_pv_credit_usd":       "PV credit (NBT)",
    }

    pens = sorted(df["pen_rate"].unique())
    scenarios = sorted(df["scenario"].unique())

    # Compute a global y-range so that all five panels share the same scale.
    pos_components = [c for c in ["E_baseline_cost_usd", "E_ev_cost_usd",
                                  "E_batt_cost_usd", "demand_charge_usd"]
                      if df[c].abs().max() > 1e-9]
    pos_total = df[pos_components].sum(axis=1).max()
    pv_max = df["E_pv_credit_usd"].max()
    y_top = float(pos_total) * 1.10
    y_bot = -float(pv_max) * 1.30

    handles_for_legend = []
    labels_for_legend  = []
    sharey_ax: Optional[plt.Axes] = None

    for ax_i, scen in enumerate(scenarios):
        if sharey_ax is None:
            ax = fig.add_subplot(gs[0, ax_i])
            sharey_ax = ax
        else:
            ax = fig.add_subplot(gs[0, ax_i], sharey=sharey_ax)
        sub = df[df["scenario"] == scen].sort_values("pen_rate")

        x = np.arange(len(pens))
        width = 0.65

        # Stack positive components
        bottom_pos = np.zeros(len(pens))
        for comp in pos_components:
            vals = sub[comp].values
            b = ax.bar(x, vals, width, bottom=bottom_pos,
                       color=component_colors[comp],
                       edgecolor="white", linewidth=0.4)
            bottom_pos += vals
            if ax_i == 0:
                handles_for_legend.append(b)
                labels_for_legend.append(component_labels[comp])

        # PV credit goes BELOW zero (it's a negative number for the bill)
        pv_neg = -sub["E_pv_credit_usd"].values
        b_pv = ax.bar(x, pv_neg, width,
                      color=component_colors["E_pv_credit_usd"],
                      edgecolor="white", linewidth=0.4, alpha=0.85)
        if ax_i == 0:
            handles_for_legend.append(b_pv)
            labels_for_legend.append(component_labels["E_pv_credit_usd"])

        # Net total bill marker
        net = sub["total_bill_usd"].values
        ax.scatter(x, net, marker="D", s=24, color="black", zorder=5,
                   edgecolors="white", linewidths=0.6,
                   label="Net total" if ax_i == 0 else None)

        ax.set_xticks(x)
        ax.set_xticklabels([PEN_LABELS[p] for p in pens], rotation=0, fontsize=7)
        ax.set_title(SCENARIO_NAMES_SHORT[scen], fontsize=9, pad=4)
        ax.axhline(0, color="black", lw=0.6)
        # Pin the global y-range so every panel uses the same vertical scale
        ax.set_ylim(y_bot, y_top)
        if ax_i == 0:
            ax.set_ylabel("Annual bill component (USD)")
        else:
            # Hide y-tick labels on inner panels but keep the gridlines -
            # the shared scale is the point of the figure.
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

        # Tight kUSD axis formatting (applied to all, only visible on leftmost)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}")
        )

    # Legend below the figure (saves space)
    fig.legend(
        handles_for_legend, labels_for_legend,
        loc="lower center", ncol=3, fontsize=7.0, frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    figure_title(fig, "Annual bill decomposition",
                 y=1.01, fontsize=9.5)
    save_figure(fig, out_dir, "F04_bill_decomposition")


# FIGURE 5 - PEAK-COMPONENT ATTRIBUTION
# at the community peak timestep t*, who is responsible for the load?
# This is the plot for the SRE narrative.
def fig05_peak_attribution(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    df = (data.df.groupby(["scenario", "pen_rate"])[
              ["P_peak_building_kw", "P_peak_ev_kw",
               "P_peak_batt_kw", "P_peak_pv_kw",
               "P_community_peak_kw"]
          ].mean().reset_index())

    pens = sorted(df["pen_rate"].unique())
    scens = sorted(df["scenario"].unique())

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.5))
    gs = GridSpec(1, 4, figure=fig, wspace=0.10)

    component_colors = {
        "P_peak_building_kw":  COLORS_WONG["skyblue"],
        "P_peak_ev_kw":        COLORS_WONG["orange"],
        "P_peak_batt_kw":      COLORS_WONG["yellow"],
        "P_peak_pv_kw":        COLORS_WONG["green"],   # PV displaces (negative)
    }
    component_labels = {
        "P_peak_building_kw":  "Building (HVAC + lights + equip)",
        "P_peak_ev_kw":        "EV charging",
        "P_peak_batt_kw":      "Battery (charging)",
        "P_peak_pv_kw":        "PV (offset, negative)",
    }


    pos_components = [c for c in ["P_peak_building_kw", "P_peak_ev_kw",
                                  "P_peak_batt_kw"]
                      if df[c].abs().max() > 1e-9]
    pos_total = df[pos_components].sum(axis=1).max()
    pv_max = df["P_peak_pv_kw"].max()
    y_top = float(pos_total) * 1.08
    y_bot = -float(pv_max) * 1.20 if pv_max > 0 else -10.0

    handles_for_legend = []
    labels_for_legend  = []
    sharey_ax: Optional[plt.Axes] = None

    for ax_i, pen in enumerate(pens):
        if sharey_ax is None:
            ax = fig.add_subplot(gs[0, ax_i])
            sharey_ax = ax
        else:
            ax = fig.add_subplot(gs[0, ax_i], sharey=sharey_ax)
        sub = df[df["pen_rate"] == pen].sort_values("scenario")

        x = np.arange(len(scens))
        width = 0.7

        # Positive stack: building + ev + batt
        bottom = np.zeros(len(scens))
        for comp in pos_components:
            vals = sub[comp].values
            b = ax.bar(x, vals, width, bottom=bottom,
                       color=component_colors[comp],
                       edgecolor="white", linewidth=0.4)
            bottom += vals
            if ax_i == 0:
                handles_for_legend.append(b)
                labels_for_legend.append(component_labels[comp])

        # PV displaces - show as negative
        pv_neg = -sub["P_peak_pv_kw"].values
        b_pv = ax.bar(x, pv_neg, width,
                      color=component_colors["P_peak_pv_kw"],
                      edgecolor="white", linewidth=0.4, alpha=0.85)
        if ax_i == 0:
            handles_for_legend.append(b_pv)
            labels_for_legend.append(component_labels["P_peak_pv_kw"])

        # Total community peak as black diamond
        ax.scatter(x, sub["P_community_peak_kw"].values, marker="D", s=24,
                   color="black", zorder=5, edgecolors="white", linewidths=0.6)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_NAMES_SHORT[s] for s in scens], fontsize=7.0)
        ax.set_title(f"Pen = {PEN_LABELS[pen]}", fontsize=9, pad=4)
        ax.axhline(0, color="black", lw=0.6)
        # Pin the shared y-range
        ax.set_ylim(y_bot, y_top)
        if ax_i == 0:
            ax.set_ylabel("Component contribution at $t^*$ (kW)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)

    fig.legend(
        handles_for_legend, labels_for_legend,
        loc="lower center", ncol=4, fontsize=7.0, frameon=False,
        bbox_to_anchor=(0.5, -0.05),
    )

    figure_title(fig, 
        "Peak-component attribution at the community peak timestep $t^*$",
        y=1.01, fontsize=9.5,
    )
    save_figure(fig, out_dir, "F05_peak_attribution")


# FIGURE 6 - PARETO FRONT: CF vs ANNUAL BILL
# a 2-D scatter of all 180 conditions, x = total_bill, y = CF,
# coloured by scenario, sized by penetration.
def fig06_pareto_cf_bill(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.7, 3.6))

    # Marker size scaled by penetration
    pen_to_size: Dict[float, float] = {
        0.25: 24,
        0.50: 50,
        0.75: 90,
        1.00: 140,
    }

    for scen in sorted(data.df["scenario"].unique()):
        sub = data.df[data.df["scenario"] == scen]
        ax.scatter(
            sub["total_bill_usd"] / 1000.0,
            sub["CF"],
            s=[pen_to_size[p] for p in sub["pen_rate"]],
            c=SCENARIO_COLORS[scen],
            alpha=0.65,
            edgecolors="white",
            linewidths=0.7,
            label=SCENARIO_NAMES[scen],
        )

    ax.set_xlabel("Annual community bill (kUSD)")
    ax.set_ylabel("Coincidence Factor (CF)")
    axes_title(ax, "Cost versus grid stress",
                 fontsize=9.5)

    # Two-part legend: scenarios (colours) and sizes (penetration)
    leg1 = ax.legend(loc="lower right", framealpha=0.92,
                     fontsize=7.0, title="Scenario",
                     title_fontsize=7.5)
    ax.add_artist(leg1)
    size_handles = [
        plt.scatter([], [], s=pen_to_size[p], c="gray", alpha=0.7,
                    edgecolors="white", linewidths=0.7,
                    label=PEN_LABELS[p])
        for p in sorted(pen_to_size.keys())
    ]
    ax.legend(handles=size_handles, loc="upper left", framealpha=0.92,
              fontsize=7.0, title="Penetration", title_fontsize=7.5)

    save_figure(fig, out_dir, "F06_pareto_cf_bill")


# FIGURE 7 - LP-OPF OPTIMALITY GAP
#  how close is each scenario to the centralised perfect-information
# LP-OPF lower bound on the demand charge?
def fig07_commfree_optimality_gap(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    df = data.df.copy()
    col = "comm_free_oracle_gap_pct"
    if col not in df.columns or df[col].notna().sum() == 0:
        log.warning(f"No valid {col} column -- skipping F07.")
        return
    df_ok = df[df[col].notna()].copy()
    scenarios = sorted(df_ok["scenario"].unique())
    pens = sorted(df_ok["pen_rate"].unique())

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.4))
    gs  = GridSpec(1, 2, figure=fig, wspace=0.30, width_ratios=[1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])

    # Panel (a): mean gap vs EV penetration, one line per scenario
    for scen in scenarios:
        sub = df_ok[df_ok["scenario"] == scen]
        means = [sub[sub["pen_rate"] == p][col].mean() for p in pens]
        ax_a.plot(pens, means, "-o", color=SCENARIO_COLORS[scen], ms=5, lw=1.5,
                  mfc="white", mec=SCENARIO_COLORS[scen], mew=1.3,
                  label=SCENARIO_NAMES_SHORT[scen])
    ax_a.set_xlabel("EV penetration")
    ax_a.set_ylabel("Gap to relaxed comm-free lower bound (%)")
    ax_a.set_xticks(pens)
    ax_a.legend(loc="best", fontsize=7.0, ncol=2, framealpha=0.9)
    add_panel_label(ax_a, "(a)")

    # Panel (b): gap distribution by scenario (boxplot, linear axis)
    box_data = [df_ok[df_ok["scenario"] == s][col].dropna().values for s in scenarios]
    bp = ax_b.boxplot(
        box_data,
        positions=range(len(scenarios)),
        widths=0.55, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=2.5, alpha=0.6,
                        markeredgecolor="none"),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        boxprops=dict(linewidth=0.6),
    )
    for patch, scen in zip(bp["boxes"], scenarios):
        patch.set_facecolor(SCENARIO_COLORS[scen])
        patch.set_alpha(0.55)
    ax_b.set_xticks(range(len(scenarios)))
    ax_b.set_xticklabels([SCENARIO_NAMES_SHORT[s] for s in scenarios])
    ax_b.set_xlabel("Scenario")
    ax_b.set_ylabel("Gap to relaxed comm-free lower bound (%)")
    add_panel_label(ax_b, "(b)")

    med = {s: float(df_ok[df_ok["scenario"] == s][col].median()) for s in scenarios}
    best = min(med, key=med.get)
    n_have = int(df[col].notna().sum())
    n_total = len(df)
    figure_title(fig, 
        f"Distance to a relaxed communication-free lower bound "
        f"(rate-uncapped water-fill; {n_have}/{n_total} conditions).\n"
        f"The bound is loose, so a large gap reflects bound looseness as much as "
        f"controller suboptimality; {SCENARIO_NAMES_SHORT[best]} sits closest "
        f"(median {med[best]:.0f}%).",
        y=1.07, fontsize=8.0,
    )
    save_figure(fig, out_dir, "F07_commfree_optimality_gap")


# FIGURE 8 - JAIN FAIRNESS INDEX
# distributed schemes can be inequitable.  Jain's fairness index
# bounds the cost equity across the 50 households.
def log_lp_opf_ratios(data: AnalysisData) -> None:
    """Log the per-scenario LP-OPF gaps quoted in Section 5.2 so they are
    auditable from the run log rather than hand-copied. No figure is emitted."""
    df = data.df
    if "optimality_gap_pct" not in df.columns:
        return
    hi = df[(df["pen_rate"] >= 0.50) & df["optimality_gap_pct"].notna()]
    if hi.empty:
        return
    log.info("  LP-OPF gap at eta>=0.50 (Section 5.2, secondary benchmark):")
    for s in sorted(hi["scenario"].unique()):
        sub = hi[hi["scenario"] == s]
        log.info(f"    {SCENARIO_NAMES_SHORT[s]}: mean {sub['optimality_gap_pct'].mean():7.1f}%  "
                 f"median {sub['optimality_gap_pct'].median():7.1f}%  (n={len(sub)})")


def fig08_jain_fairness(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    fig, (ax_hi, ax_lo) = plt.subplots(
        2, 1, sharex=True,
        figsize=(WIDTH_DOUBLE * 0.72, 3.4),
        gridspec_kw={"height_ratios": [2.0, 1.35], "hspace": 0.07},
    )

    pens = sorted(data.df["pen_rate"].unique())

    # Single shared RNG advances across the 5 x 4 = 20 bootstrap calls so each
    # call draws an independent resample sequence.
    rng_j = np.random.default_rng(seed=42)
    for scen in sorted(data.df["scenario"].unique()):
        means, los, his = [], [], []
        for p in pens:
            sub = data.df[(data.df["scenario"] == scen) & (data.df["pen_rate"] == p)]
            m, lo, hi = bootstrap_ci(sub["jain_fairness"].values,
                                     statistic=np.mean, rng=rng_j)
            means.append(m); los.append(lo); his.append(hi)
        means = np.array(means)
        err = np.vstack([means - np.array(los), np.array(his) - means])
        # The identical series is drawn on both axes; each axis clips to its
        # own y-band, so a curve appears only where its data actually lie.
        for ax in (ax_hi, ax_lo):
            ax.errorbar(
                pens, means, yerr=err,
                fmt="-o", color=SCENARIO_COLORS[scen],
                ecolor=SCENARIO_COLORS[scen], elinewidth=0.8, capsize=3,
                markersize=5, markerfacecolor="white", markeredgewidth=1.2,
                label=SCENARIO_NAMES_SHORT[scen],
                alpha=0.95,
            )

    # Band limits: upper frames the storage cluster and the J=1 asymptote,
    # lower frames the baselines down to their minimum.
    ax_hi.set_ylim(0.79, 1.06)
    ax_lo.set_ylim(0.16, 0.63)


    ax_hi.axhline(1.0, color="green", lw=0.8, ls=":", alpha=0.7)
    ax_hi.text(pens[-1] * 1.01, 1.003, "perfect equity", fontsize=7,
               ha="left", va="bottom", color="green")

    ax_hi.spines["bottom"].set_visible(False)
    ax_lo.spines["top"].set_visible(False)
    ax_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_lo.xaxis.tick_bottom()
    d = 0.012
    kw = dict(transform=ax_hi.transAxes, color="k", clip_on=False, lw=0.9)
    ax_hi.plot((-d, +d), (-d, +d), **kw)
    ax_hi.plot((1 - d, 1 + d), (-d, +d), **kw)
    dl = d * (2.0 / 1.35)
    kw.update(transform=ax_lo.transAxes)
    ax_lo.plot((-d, +d), (1 - dl, 1 + dl), **kw)
    ax_lo.plot((1 - d, 1 + d), (1 - dl, 1 + dl), **kw)

    # Shared x-axis and a single centered y-label spanning both bands.
    ax_lo.set_xticks(pens)
    ax_lo.set_xticklabels([PEN_LABELS[p] for p in pens])
    ax_lo.set_xlabel("EV penetration")
    fig.supylabel("Jain's Fairness Index (J)", fontsize=9)

    ax_hi.legend(loc="lower right", ncol=3, framealpha=0.92, fontsize=7.0,
                 title="Scenario", title_fontsize=7.5)
    axes_title(ax_hi, "Bill equity", fontsize=9.5)

    save_figure(fig, out_dir, "F08_jain_fairness")


# FIGURE 9 - CARBON FOOTPRINT
# did Algorithm 1's carbon-arbiter actually reduce emissions vs the baseline?
def fig09_carbon_emissions(data: AnalysisData, out_dir: Path) -> None:
    setup_style()
    fig, ax = plt.subplots(figsize=(WIDTH_DOUBLE * 0.72, 3.0))

    pens = sorted(data.df["pen_rate"].unique())

    # Baseline = S0 carbon
    s0_carbon = (data.df_s0.groupby("pen_rate")["E_carbon_kg"].mean())

    # Single shared RNG across the 5 × 4 bootstrap calls.
    rng_c = np.random.default_rng(seed=42)
    for scen in sorted(data.df["scenario"].unique()):
        sub = data.df[data.df["scenario"] == scen]
        carbon_pct = []
        carbon_lo, carbon_hi = [], []
        for p in pens:
            base = s0_carbon.loc[p]
            vals = sub[sub["pen_rate"] == p]["E_carbon_kg"].values
            if len(vals) == 0 or base <= 0:
                carbon_pct.append(np.nan)
                carbon_lo.append(np.nan); carbon_hi.append(np.nan)
                continue
            rel = (vals - base) / base * 100.0
            m, lo, hi = bootstrap_ci(rel, statistic=np.mean, rng=rng_c)
            carbon_pct.append(m); carbon_lo.append(lo); carbon_hi.append(hi)
        means = np.array(carbon_pct)
        err = np.vstack([means - np.array(carbon_lo),
                         np.array(carbon_hi) - means])
        ax.errorbar(
            pens, means, yerr=err,
            fmt="-o", color=SCENARIO_COLORS[scen],
            ecolor=SCENARIO_COLORS[scen], elinewidth=0.8, capsize=3,
            markersize=5, markerfacecolor="white", markeredgewidth=1.2,
            label=SCENARIO_NAMES_SHORT[scen],
            alpha=0.95,
        )

    ax.axhline(0, color="black", lw=0.8, ls="-", alpha=0.7)
    ax.set_xticks(pens)
    ax.set_xticklabels([PEN_LABELS[p] for p in pens])
    ax.set_xlabel("EV penetration")
    ax.set_ylabel(r"Carbon vs S0 baseline (% change)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.92, fontsize=7.0,
              title="Scenario", title_fontsize=7.5)
    axes_title(ax, "Carbon vs no-DR baseline", fontsize=9.5)

    save_figure(fig, out_dir, "F09_carbon_emissions")


# FIGURE 10 - METRIC DISTRIBUTIONS (KDE / VIOLIN GRID)
# a multi-panel fingerprint of all key metrics, showing distribution
# shape across all 36 conditions for each scenario.  Useful for the final
# results page of the paper - gives a holistic view at a glance.
def fig10_metric_distributions(data: AnalysisData, out_dir: Path) -> None:
    setup_style()

    metrics = [
        ("CF",                    "Coincidence Factor"),
        ("ramp_rate_kw_per_min",  "Ramp rate (kW/min)"),
        ("load_factor",           "Load Factor"),
        ("total_bill_usd",        "Annual bill (USD)"),
        ("optimality_gap_pct",    "Opt. gap (%)"),
        ("jain_fairness",         "Jain Fairness J"),
    ]

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 3.0 * rows))
    gs = GridSpec(rows, cols, figure=fig, wspace=0.35, hspace=0.50)

    scenarios = sorted(data.df["scenario"].unique())

    for i, (metric, label) in enumerate(metrics):
        r, c = divmod(i, cols)
        ax = fig.add_subplot(gs[r, c])

        # Skip if metric column is fully null
        if metric not in data.df.columns or data.df[metric].notna().sum() == 0:
            ax.text(0.5, 0.5, f"{label}\n(no data)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=8.5, color="gray")
            ax.set_axis_off()
            continue

        # Per-scenario violin
        violin_data, violin_pos, violin_colors = [], [], []
        for j, scen in enumerate(scenarios):
            vals = data.df[(data.df["scenario"] == scen)][metric].dropna().values
            if len(vals) > 1:
                violin_data.append(vals)
                violin_pos.append(j)
                violin_colors.append(SCENARIO_COLORS[scen])

        if violin_data:
            parts = ax.violinplot(
                violin_data, positions=violin_pos, widths=0.7,
                showmeans=False, showmedians=True, showextrema=False,
            )
            for body, color in zip(parts["bodies"], violin_colors):
                body.set_facecolor(color)
                body.set_alpha(0.55)
                body.set_edgecolor(color)
                body.set_linewidth(0.6)
            if "cmedians" in parts:
                parts["cmedians"].set_color("black")
                parts["cmedians"].set_linewidth(1.0)

        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels([SCENARIO_NAMES_SHORT[s] for s in scenarios],
                            fontsize=7.5)
        ax.set_title(label, fontsize=9, pad=4)
        # Add lowercase panel label
        add_panel_label(ax, f"({chr(ord('a') + i)})", x=-0.15, y=1.05)
        ax.tick_params(axis="y", labelsize=7)

        # Format y-axis for currency
        if "usd" in metric.lower() or "bill" in metric.lower():
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(
                    lambda v, _: f"{v/1000:.0f}k" if abs(v) >= 1000 else f"{v:.0f}"
                )
            )

    figure_title(fig, "Metric distributions by scenario",
                 y=1.00, fontsize=9.5)
    save_figure(fig, out_dir, "F10_metric_distributions")


# TABLE GENERATION

def _latex_escape(s: str) -> str:

    if not isinstance(s, str):
        return s
    if "$" in s or "\\" in s:
        return s            # already LaTeX
    # Order matters: escape backslash first (already excluded above)
    out = (s.replace("&",  r"\&")
            .replace("%",  r"\%")
            .replace("#",  r"\#")
            .replace("_",  r"\_")
            .replace("^",  r"\^{}")
            .replace("~",  r"\~{}"))
    return out


def _to_latex_booktabs(
    df: pd.DataFrame,
    caption: str,
    label: str,
    *,
    column_format: Optional[str] = None,
    float_format: str = "%.3f",
    note: Optional[str] = None,
    wide: bool = False,
) -> str:
    r"""
    Render a DataFrame as IEEE-style LaTeX (booktabs).  Avoids pandas's
    `to_latex` for fine-grained control over rules, column types, and
    captions.  Output is ready for direct \input{} into an IEEE template.

    Plain-text cells are escaped for LaTeX-special characters (%, &, _,
    #, ^, ~).  Cells that already contain math ($...$) or commands (\\)
    pass through unchanged.

    Set ``wide=True`` to emit a column-spanning `table*` environment
    rather than the default single-column `table`.  Use this for tables
    whose body is wider than ~3.5 inches (the single-column text width)
    or whose body has more than ~12 rows; otherwise the floating table
    will collide with the surrounding two-column prose.
    """
    n_cols = len(df.columns) + (1 if df.index.name else 0)
    if column_format is None:
        column_format = "l" + "c" * (n_cols - 1)

    env = "table*" if wide else "table"
    lines = []
    lines.append(rf"\begin{{{env}}}[!t]")
    # \arraystretch{1.2} keeps booktabs rows legible at journal size
    lines.append(r"\renewcommand{\arraystretch}{1.2}")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\centering")
    lines.append(rf"\begin{{tabular}}{{{column_format}}}")
    lines.append(r"\toprule")

    # Header row — escape headers as well
    headers = []
    if df.index.name:
        headers.append(_latex_escape(str(df.index.name)))
    headers.extend(_latex_escape(str(c)) for c in df.columns)
    lines.append(" & ".join(headers) + r" \\")
    lines.append(r"\midrule")

    # Data rows
    for idx, row in df.iterrows():
        cells = []
        if df.index.name:
            cells.append(_latex_escape(str(idx)))
        for v in row:
            if isinstance(v, (int, np.integer)):
                cells.append(f"{int(v):d}")
            elif isinstance(v, (float, np.floating)):
                cells.append("--" if np.isnan(v) else float_format % v)
            else:
                cells.append(_latex_escape(str(v)))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(rf"\\[2pt]\footnotesize {note}")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


def _save_table(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    caption: str,
    label: str,
    *,
    column_format: Optional[str] = None,
    float_format: str = "%.3f",
    note: Optional[str] = None,
    wide: bool = False,
) -> None:

    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{name}.csv"
    df.to_csv(csv_path, index=True)

    tex_path = out_dir / f"{name}.tex"
    tex_str = _to_latex_booktabs(
        df, caption=caption, label=label,
        column_format=column_format, float_format=float_format, note=note,
        wide=wide,
    )
    tex_path.write_text(tex_str, encoding="utf-8")

    log.info(f"  saved {name}.{{csv,tex}}")


# TABLE 1: Master summary by penetration
def tbl01_master_summary(data: AnalysisData, out_dir: Path) -> None:
    """
    Master table with one row per (penetration, scenario) summarising
    CF, total bill, demand charge, and Jain fairness.
    MIT is a condition-level quantity (CF(S1)/CF(S2) per condition
    """
    rows = []
    pens = sorted(data.df["pen_rate"].unique())
    scenarios = sorted(data.df["scenario"].unique())

    # Per-penetration MIT means (for the footnote)
    mit_by_pen = (data.df_s1.groupby("pen_rate")["MIT_ratio"]
                  .mean().reindex(pens))

    # Single shared RNG across the 5 * 4 = 20 bootstrap CIs.
    rng_t = np.random.default_rng(seed=42)
    for pen in pens:
        for scen in scenarios:
            sub = data.df[(data.df["pen_rate"] == pen) &
                          (data.df["scenario"] == scen)]
            if sub.empty:
                continue

            cf_m, cf_lo, cf_hi = bootstrap_ci(sub["CF"].values, rng=rng_t)
            bill_m            = float(sub["total_bill_usd"].mean())
            dc_m              = float(sub["demand_charge_usd"].mean())
            jain_m            = float(sub["jain_fairness"].mean())

            rows.append({
                "Pen.":        f"{int(pen*100)}%",
                "Scen.":       SCENARIO_NAMES_SHORT[scen],
                "CF":          fmt_value_ci(cf_m, cf_lo, cf_hi, decimals=3),
                "Bill (kUSD)": f"{bill_m/1000.0:.1f}",
                "Demand (kUSD)": f"{dc_m/1000.0:.1f}",
                "J":           f"{jain_m:.3f}",
            })

    df_tbl = pd.DataFrame(rows)
    df_tbl.set_index("Pen.", inplace=True)

    mit_str = "; ".join(f"$\\eta={p:.2f}$: ${mit_by_pen.loc[p]:.3f}$"
                        for p in pens)
    note = ("CF: point estimate [lo, hi], 95\\% percentile bootstrap ($n=5000$) "
            "over the 9 $(\\xi,\\rho)$ cells per bucket; Bill, Demand and $J$ are "
            "means over the same cells. MIT per penetration (condition-level, "
            f"identical across scenarios): {mit_str}.")

    _save_table(
        df_tbl, out_dir, "T01_master_summary",
        caption="Headline summary by penetration and scenario.",
        label="tab:t01_master_summary",
        column_format="ll" + "c" * (len(df_tbl.columns) - 1),  # Pen+Scen left, 4 data cols centered
        float_format="%.3f",
        note=note,
        # 20 data rows * 6 columns - too wide and long for IEEE single-column.
        # Emit as table* so the floating table spans the full page width and
        # does not collide with the surrounding two-column prose.
        wide=True,
    )


# TABLE 2: Sobol indices
def tbl02_sobol_indices(data: AnalysisData, out_dir: Path) -> None:
    p1 = data.sobol_p1
    if p1 is None:
        p1 = _fallback_factor_decomposition(data, target="SRE_ratio")

    factors = list(p1["factor"]) if "factor" in p1.columns else list(p1.index)
    factor_labels = {
        "pen_rate":   "EV penetration",
        "inel_rate":  "Inelasticity",
        "markov_rho": r"Markov $\rho$",
    }

    rows = []
    p2 = data.sobol_p2
    p1_idx = p1.set_index("factor") if "factor" in p1.columns else p1
    p2_idx = p2.set_index("factor") if (p2 is not None and "factor" in p2.columns) else None

    for f in factors:
        s1 = float(p1_idx.loc[f, "S1"])
        st = float(p1_idx.loc[f, "ST"]) if "ST" in p1_idx.columns and not pd.isna(p1_idx.loc[f, "ST"]) else np.nan
        s1c = float(p1_idx.loc[f, "S1_conf"]) if "S1_conf" in p1_idx.columns else 0.0
        stc = float(p1_idx.loc[f, "ST_conf"]) if "ST_conf" in p1_idx.columns else 0.0

        s1_str = f"{s1:.3f} $\\pm$ {s1c:.3f}" if s1c > 0 else f"{s1:.3f}"
        st_str = "--" if np.isnan(st) else (
            f"{st:.3f} $\\pm$ {stc:.3f}" if stc > 0 else f"{st:.3f}"
        )

        s1_p2_str = "--"
        st_p2_str = "--"
        if p2_idx is not None and f in p2_idx.index:
            s1_p2 = float(p2_idx.loc[f, "S1"])
            st_p2 = float(p2_idx.loc[f, "ST"])
            s1c_p2 = float(p2_idx.loc[f, "S1_conf"]) if "S1_conf" in p2_idx.columns else 0.0
            stc_p2 = float(p2_idx.loc[f, "ST_conf"]) if "ST_conf" in p2_idx.columns else 0.0
            s1_p2_str = "--" if np.isnan(s1_p2) else (f"{s1_p2:.3f} $\\pm$ {s1c_p2:.3f}" if s1c_p2 > 0 else f"{s1_p2:.3f}")
            st_p2_str = "--" if np.isnan(st_p2) else (f"{st_p2:.3f} $\\pm$ {stc_p2:.3f}" if stc_p2 > 0 else f"{st_p2:.3f}")

        rows.append({
            "Factor":               factor_labels.get(f, f),
            r"$S_1$ (real)":        s1_str,
            r"$S_T$ (real)":        st_str,
            r"$S_1$ (GP surr.)":    s1_p2_str,
            r"$S_T$ (GP surr.)":    st_p2_str,
        })

    df_tbl = pd.DataFrame(rows).set_index("Factor")

    # two entries read as sloppiness unless named as estimator artefacts.
    note = ("$S_1$ = first-order, $S_T$ = total-order Sobol index. "
            "Pass-1 uses real simulation outputs (36 cells); Pass-2 fits "
            "a Gaussian Process surrogate before Saltelli sampling. "
            "Two entries are small-design artefacts, not results: the negative "
            "$S_1(\\rho)$ reflects the Saltelli estimator's variance near zero, and "
            "the surrogate's $S_1(\\eta)$ interval exceeds the index's $[0,1]$ "
            "support. Read them as indistinguishable from zero and from one.")

    _save_table(
        df_tbl, out_dir, "T02_sobol_indices",
        caption=r"Sobol sensitivity decomposition of $\mathit{SRE}$.",
        label="tab:t02_sobol",
        column_format="lcccc",
        note=note,
        wide=True,
    )


# TABLE 3: Per-scenario summary
def tbl03_scenario_summary(data: AnalysisData, out_dir: Path) -> None:
    rows = []
    scenarios = sorted(data.df["scenario"].unique())
    # Single shared RNG across the 5 bootstrap calls.
    rng_t = np.random.default_rng(seed=42)
    for scen in scenarios:
        sub = data.df[data.df["scenario"] == scen]
        cf_m, cf_lo, cf_hi = bootstrap_ci(sub["CF"].values, rng=rng_t)
        rows.append({
            "Scenario":          SCENARIO_NAMES[scen],
            "CF (mean [95\\% CI])": fmt_value_ci(cf_m, cf_lo, cf_hi, decimals=3),
            "P. peak (kW)":      f"{sub['P_community_peak_kw'].mean():.1f}",
            "Bill (kUSD)":       f"{sub['total_bill_usd'].mean()/1000:.1f}",
            "Carbon (t/yr)":     f"{sub['E_carbon_kg'].mean()/1000:.2f}",
            "J":                 f"{sub['jain_fairness'].mean():.3f}",
        })

    df_tbl = pd.DataFrame(rows).set_index("Scenario")
    note = ("CF: point estimate [lo, hi], percentile bootstrap ($n=5000$); other "
            "columns are means over the 36 $(\\eta,\\xi,\\rho)$ conditions. S3 "
            "flattens the price signal to the time-weighted E-TOU-C average; its "
            "dispatch is otherwise identical to S0, hence the near-coincident rows.")
    _save_table(
        df_tbl, out_dir, "T03_scenario_summary",
        caption="Aggregate performance by scenario.",
        label="tab:t03_scenarios",
        column_format="lccccc",
        note=note,
        # 6 columns including a long-text Scenario column; spans more than
        wide=True,
    )


# TABLE 4: Statistical significance - S1 vs S2
def tbl04_significance_tests(data: AnalysisData, out_dir: Path) -> None:
    """
    Wilcoxon signed-rank test of CF(S1) vs CF(S2) per (pen, inel, rho)
    cell.  Each cell is matched (same condition), so paired test applies.
    """
    rows = []
    pens = sorted(data.df["pen_rate"].unique())
    for pen in pens:
        s1 = (data.df[(data.df["pen_rate"] == pen) &
                       (data.df["scenario"] == 1)]
              .sort_values(["inel_rate", "markov_rho"])["CF"].values)
        s2 = (data.df[(data.df["pen_rate"] == pen) &
                       (data.df["scenario"] == 2)]
              .sort_values(["inel_rate", "markov_rho"])["CF"].values)

        res = wilcoxon_test(s1, s2, alternative="less")
        rows.append({
            "Penetration":    PEN_LABELS[pen],
            "n":              res["n"],
            "Test":           res["test"].replace("_", " "),
            "Statistic":      f"{res['stat']:.1f}" if not np.isnan(res["stat"]) else "n/a",
            r"$p$-value":     fmt_p(res["p"]),
            "Cliff $\\delta$":  f"{res['delta']:+.3f}" if not np.isnan(res["delta"]) else "n/a",
            "Effect":         cliff_delta_label(res["delta"]),
        })

    # Add overall (all penetrations pooled)
    s1_all = data.df[data.df["scenario"] == 1].sort_values(
        ["pen_rate", "inel_rate", "markov_rho"])["CF"].values
    s2_all = data.df[data.df["scenario"] == 2].sort_values(
        ["pen_rate", "inel_rate", "markov_rho"])["CF"].values
    res = wilcoxon_test(s1_all, s2_all, alternative="less")
    rows.append({
        "Penetration":    "All (pooled)",
        "n":              res["n"],
        "Test":           res["test"].replace("_", " "),
        "Statistic":      f"{res['stat']:.1f}" if not np.isnan(res["stat"]) else "n/a",
        r"$p$-value":     fmt_p(res["p"]),
        "Cliff $\\delta$":  f"{res['delta']:+.3f}" if not np.isnan(res["delta"]) else "n/a",
        "Effect":         cliff_delta_label(res["delta"]),
    })

    df_tbl = pd.DataFrame(rows).set_index("Penetration")
    note = (r"One-sided paired Wilcoxon signed-rank test of $CF(\mathrm{S1}) < "
            r"CF(\mathrm{S2})$, matched per condition. Cliff $\delta$ is the "
            r"non-parametric effect size: $|\delta| < 0.147$ negligible, $< 0.33$ "
            r"small, $< 0.474$ medium, otherwise large \cite{61}.")
    _save_table(
        df_tbl, out_dir, "T04_significance_tests",
        caption="Statistical significance of mitigation: S1 vs S2.",
        label="tab:t04_significance",
        column_format="lcccccc",
        note=note,
        wide=True,
    )


# INTERACTIVE HTML REPORT (Plotly)
# Designed as an internal collaborator dashboard, NOT a publication artifact.
# Every plot has tooltips and a CSV export button.
def report_html(data: AnalysisData, out_dir: Path) -> None:
    if not PLOTLY_OK:
        log.warning("Plotly not available — skipping HTML report.")
        return

    df = data.df.copy()
    out_path = out_dir / "REPORT_publication.html"

    fig = make_subplots(
        rows=3, cols=2,
        specs=[
            [{"type": "scatter"}, {"type": "heatmap"}],
            [{"type": "bar"},     {"type": "scatter"}],
            [{"type": "box"},     {"type": "scatter"}],
        ],
        subplot_titles=(
            "MIT_ratio vs Penetration",
            "Coincidence Factor heatmap",
            "Sobol indices",
            "Pareto: Bill vs CF",
            "Optimality gap by scenario",
            "Carbon vs S0 baseline",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.10,
    )

    # (1,1) MIT scatter
    s1 = data.df_s1.copy()
    fig.add_trace(
        go.Scatter(
            x=s1["pen_rate"] * 100,
            y=s1["MIT_ratio"],
            mode="markers",
            marker=dict(color=COLORS_WONG["green"], size=8,
                        line=dict(width=1, color="white")),
            name="MIT (S1/S2)",
            hovertemplate=("Pen: %{x:.0f}%<br>"
                           "MIT: %{y:.3f}<br>"
                           "<extra></extra>"),
        ),
        row=1, col=1,
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS_WONG["vermillion"],
                  row=1, col=1)

    # (1,2) CF heatmap
    pivot = (df.groupby(["scenario", "pen_rate"])["CF"].mean()
               .unstack("pen_rate"))
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=[PEN_LABELS[p] for p in pivot.columns],
            y=[SCENARIO_NAMES_SHORT[s] for s in pivot.index],
            colorscale="Magma_r",
            zmin=0.4, zmax=0.95,
            colorbar=dict(title="CF", x=1.0, xanchor="left", thickness=14,
                          len=0.30, y=0.85),
            hovertemplate="Scenario: %{y}<br>Pen: %{x}<br>CF: %{z:.3f}<extra></extra>",
            showscale=True,
        ),
        row=1, col=2,
    )

    # (2,1) Sobol bars
    p1 = data.sobol_p1
    if p1 is None:
        p1 = _fallback_factor_decomposition(data, target="SRE_ratio")
    factors = list(p1["factor"]) if "factor" in p1.columns else list(p1.index)
    factor_labels = {"pen_rate": "Pen", "inel_rate": "Inel", "markov_rho": "Rho"}
    p1i = p1.set_index("factor") if "factor" in p1.columns else p1
    fig.add_trace(
        go.Bar(
            x=[factor_labels.get(f, f) for f in factors],
            y=p1i.loc[factors, "S1"].values,
            name="S1",
            marker=dict(color=COLORS_WONG["blue"]),
            showlegend=False,
        ),
        row=2, col=1,
    )
    if "ST" in p1i.columns and not p1i.loc[factors, "ST"].isna().all():
        fig.add_trace(
            go.Bar(
                x=[factor_labels.get(f, f) for f in factors],
                y=p1i.loc[factors, "ST"].values,
                name="ST",
                marker=dict(color=COLORS_WONG["skyblue"]),
                showlegend=False,
            ),
            row=2, col=1,
        )

    # (2,2) Pareto scatter
    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen]
        fig.add_trace(
            go.Scatter(
                x=sub["total_bill_usd"] / 1000.0,
                y=sub["CF"],
                mode="markers",
                marker=dict(
                    size=8 + (sub["pen_rate"] * 10).astype(int),
                    color=SCENARIO_COLORS[scen],
                    opacity=0.7, line=dict(width=0.7, color="white"),
                ),
                name=SCENARIO_NAMES_SHORT[scen],
                showlegend=False,
                hovertemplate=(f"Scenario: {SCENARIO_NAMES_SHORT[scen]}<br>"
                               "Bill: %{x:.1f} kUSD<br>"
                               "CF: %{y:.3f}<br>"
                               "<extra></extra>"),
            ),
            row=2, col=2,
        )

    # (3,1) Optimality gap boxplot
    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen]
        if sub["optimality_gap_pct"].notna().sum() == 0:
            continue
        fig.add_trace(
            go.Box(
                y=sub["optimality_gap_pct"],
                name=SCENARIO_NAMES_SHORT[scen],
                marker=dict(color=SCENARIO_COLORS[scen]),
                showlegend=False,
            ),
            row=3, col=1,
        )

    # (3,2) Carbon vs S0
    s0_carbon = data.df_s0.groupby("pen_rate")["E_carbon_kg"].mean()
    pens_arr = sorted(df["pen_rate"].unique())
    for scen in sorted(df["scenario"].unique()):
        sub = df[df["scenario"] == scen]
        means = []
        for p in pens_arr:
            base = s0_carbon.loc[p]
            vals = sub[sub["pen_rate"] == p]["E_carbon_kg"].mean()
            means.append((vals - base) / base * 100.0 if base > 0 else np.nan)
        fig.add_trace(
            go.Scatter(
                x=[p * 100 for p in pens_arr], y=means,
                mode="lines+markers",
                name=SCENARIO_NAMES_SHORT[scen],
                line=dict(color=SCENARIO_COLORS[scen]),
                marker=dict(size=8, line=dict(width=1, color="white")),
                showlegend=True,
                legendgroup=SCENARIO_NAMES_SHORT[scen],
            ),
            row=3, col=2,
        )

    # Layout & axis labels
    fig.update_xaxes(title_text="EV penetration (%)", row=1, col=1)
    fig.update_yaxes(title_text="MIT_ratio",          row=1, col=1)
    fig.update_xaxes(title_text="Factor",             row=2, col=1)
    fig.update_yaxes(title_text="Sobol index",        row=2, col=1)
    fig.update_xaxes(title_text="Bill (kUSD/yr)",     row=2, col=2)
    fig.update_yaxes(title_text="CF",                 row=2, col=2)
    fig.update_yaxes(title_text="Opt. gap (%)", type="log", row=3, col=1)
    fig.update_xaxes(title_text="EV penetration (%)", row=3, col=2)
    fig.update_yaxes(title_text="Carbon vs S0 (%)",   row=3, col=2)

    fig.update_layout(
        title=dict(
            text=("<b>IEEE TSG — Community SRE Analysis</b><br>"
                  "<sup>180 conditions × 5 scenarios × 50 houses · "
                  "SF Marine CZ3C · NBT · PG&amp;E E-TOU-C · "
                  "EnergyPlus 26.1</sup>"),
            font=dict(size=15),
            x=0.5, xanchor="center",
        ),
        font=dict(family="Times New Roman, serif", size=12),
        height=1100, width=1200,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.05,
                    xanchor="center", x=0.5),
        margin=dict(t=120, b=60, l=70, r=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")

    pio.write_html(
        fig, file=str(out_path), auto_open=False, include_plotlyjs="cdn",
        full_html=True,
    )
    log.info(f"  saved {out_path.name}")


# MAIN ENTRY POINT
# CHARGE-POWER DIMENSION — F11 / T05

EV_MEDIAN_VMT_MI:   float = 37.5    # OccupancyPlugin EV_NHTS_MEDIAN_VMT_MILES
EV_EPA_MI_PER_KWH:  float = 3.5     # OccupancyPlugin EV_EPA_CONSUMPTION_MI_PER_KWH
EV_CHARGE_ETA:      float = 0.95    # OccupancyPlugin EV_ETA_CHARGE
STAGGER_WINDOW_H:   float = 8.0     # OccupancyPlugin ALGO1_TAU_MAX_HARD_H (post clip)


def _median_session_hours(power_kw: float) -> float:
    """Charge-session length (h) for the median daily distance at a given power."""
    deficit_kwh = EV_MEDIAN_VMT_MI / EV_EPA_MI_PER_KWH
    return deficit_kwh / (power_kw * EV_CHARGE_ETA)


def _charge_power_stats(csv_path: Path) -> Dict[str, object]:
    """Per-cell rebound & mitigation summary for one power-tagged sweep CSV."""
    df = pd.read_csv(csv_path)
    if "P_peak_batt_kw" in df.columns and df["P_peak_batt_kw"].abs().max() > 1e-6:
        log.warning(f"    {csv_path.name}: battery is NOT off; the charge-power "
                    f"dimension is intended to be battery-free.")

    def cells(ratio: str, lo: str, hi: str, direction: str) -> Dict[str, float]:
        sub = df.dropna(subset=[ratio]).drop_duplicates(
            subset=["pen_rate", "inel_rate", "markov_rho"])
        n = len(sub)
        if n == 0:
            return dict(mean=np.nan, q1=np.nan, q3=np.nan, sig=0, n=0)
        sig = int((sub[lo] > 1.0).sum()) if direction == "gt" else int((sub[hi] < 1.0).sum())
        s = sub[ratio]
        return dict(mean=float(s.mean()), q1=float(s.quantile(0.25)),
                    q3=float(s.quantile(0.75)), sig=sig, n=n)

    reb = cells("SRE_ratio", "SRE_ci_lo", "SRE_ci_hi", "gt")
    mit = cells("MIT_ratio", "MIT_ci_lo", "MIT_ci_hi", "lt")
    g = df.groupby("scenario")["P_community_peak_kw"].mean()
    s1s0 = float(g.get(1, np.nan) / g.get(0, np.nan)) if 0 in g.index and 1 in g.index else float("nan")
    return dict(reb=reb, mit=mit, s1_vs_s0_peak=s1s0)


def _resolve_power_csvs(spec: Optional[List[str]], results_dir: Path) -> Dict[float, Path]:

    import re
    out: Dict[float, Path] = {}
    if spec:
        for item in spec:
            if ":" not in item:
                log.warning(f"  --power-csvs entry '{item}' is not 'kw:path'; skipping.")
                continue
            kw_str, path_str = item.split(":", 1)
            try:
                kw = float(kw_str)
            except ValueError:
                log.warning(f"  --power-csvs power '{kw_str}' is not numeric; skipping.")
                continue
            cand = Path(path_str)
            if not cand.exists() and not cand.is_absolute():
                cand = results_dir / path_str
            if cand.exists():
                out[kw] = cand
            else:
                log.warning(f"  --power-csvs path '{path_str}' not found; skipping.")
        return out
    for f in sorted(results_dir.glob("results_*kw.csv")):
        m = re.search(r"results_([0-9]+(?:\.[0-9]+)?)kw\.csv$", f.name)
        if m:
            kw = float(m.group(1))
            out[kw] = f
            log.info(f"  auto-discovered {f.name} -> assuming {kw} kW "
                     f"(use --power-csvs for exact fractional powers)")
    return out


def fig11_charge_power_sensitivity(power_csvs: Dict[float, Path], out_dir: Path) -> None:
    """F11: rebound and communication-free stagger mitigation vs EV charge power."""
    powers = sorted(power_csvs)
    if len(powers) < 2:
        log.warning("  F11 skipped: need >= 2 charge-power CSVs.")
        return
    st = {p: _charge_power_stats(power_csvs[p]) for p in powers}
    x = np.array(powers, dtype=float)
    c_reb = COLORS_WONG["vermillion"]
    c_mit = COLORS_WONG["blue"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.35))

    def panel(ax, key, color, marker, ylabel, annotate_on):
        m  = np.array([st[p][key]["mean"] for p in powers])
        q1 = np.array([st[p][key]["q1"]   for p in powers])
        q3 = np.array([st[p][key]["q3"]   for p in powers])
        ax.fill_between(x, q1, q3, color=color, alpha=0.15, linewidth=0,
                        label="interquartile range")
        ax.plot(x, m, marker + "-", color=color, mfc="white", mec=color,
                mew=1.4, ms=6, lw=1.6, label="mean across cells")
        ax.axhline(1.0, color="0.4", lw=0.8, ls=(0, (4, 3)))
        for p in powers:
            d = st[p][key]
            yv = d["q3"] if annotate_on == "top" else d["q1"]
            dy = 5 if annotate_on == "top" else -11
            ax.annotate(f"{d['sig']}/{d['n']}", (p, yv), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=7.3, color=color)
        ax.set_xlabel("EV charge power (kW)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)

    panel(axL, "reb", c_reb, "o", r"rebound  CF($S_2$)/CF($S_0$)", "top")
    panel(axR, "mit", c_mit, "s", r"mitigation  CF($S_1$)/CF($S_2$)", "bottom")
    add_panel_label(axL, "(a)", x=-0.16, y=1.04)
    add_panel_label(axR, "(b)", x=-0.16, y=1.04)
    axL.legend(loc="upper left", frameon=False, fontsize=7.5)
    axR.legend(loc="upper right", frameon=False, fontsize=7.5)

    for ax in (axL, axR):
        sec = ax.secondary_xaxis("top")
        sec.set_xticks(x)
        sec.set_xticklabels([f"{_median_session_hours(p):.2f}" for p in powers],
                            fontsize=7.6)
        sec.set_xlabel("median charge-session length (h)", fontsize=8.0)

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, out_dir, "F11_charge_power_sensitivity")


def tbl05_charge_power_sensitivity(power_csvs: Dict[float, Path], out_dir: Path) -> None:
    """T05: charge-power trend table (rebound, mitigation, feasibility proxy)."""
    powers = sorted(power_csvs)
    if len(powers) < 2:
        log.warning("  T05 skipped: need >= 2 charge-power CSVs.")
        return
    rows = []
    for p in powers:
        s = _charge_power_stats(power_csvs[p])
        sess = _median_session_hours(p)
        rows.append({
            "Power (kW)":       f"{p:.1f}",
            "Session (h)":      f"{sess:.2f}",
            "Window (\\%)":     f"{100*sess/STAGGER_WINDOW_H:.0f}",
            "Rebound $S_2/S_0$":  f"{s['reb']['mean']:.3f}",
            "Reb. sig.":        f"{s['reb']['sig']}/{s['reb']['n']}",
            "Mitig. $S_1/S_2$":   f"{s['mit']['mean']:.3f}",
            "Mit. sig.":        f"{s['mit']['sig']}/{s['mit']['n']}",
            "$S_1/S_0$ peak":     f"{s['s1_vs_s0_peak']:.3f}",
        })
    tdf = pd.DataFrame(rows).set_index("Power (kW)")
    note = ("Battery-free sweep, one level per EV charge power. Rebound is the mean "
            "$CF(\\mathrm{S2})/CF(\\mathrm{S0})$ and mitigation the mean "
            "$CF(\\mathrm{S1})/CF(\\mathrm{S2})$, over the 36 $(\\eta,\\xi,\\rho)$ "
            "cells. Significance counts cells whose bootstrap 95\\% CI excludes "
            "unity. Session length is at the median daily distance; the $S_1/S_0$ "
            "peak column is a point estimate carrying no CI.")
    _save_table(tdf, out_dir, "T05_charge_power_sensitivity",
                caption=("Rebound and mitigation versus EV charge power "
                         "(battery-free)."),
                label="tab:charge_power", note=note, wide=True)


BROADCAST_HETERO_COLOR: str = COLORS_WONG["orange"]   # heterogeneous-threshold S5
BROADCAST_NAIVE_COLOR:  str = "#7A1414"               # uniform-response S5 
BROADCAST_NAIVE_PEAK_KW_DEFAULT: float = 365.1


def _linfit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """Least-squares line through (x, y); returns (slope, intercept, R^2)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def _battpen_table(csv_path: Path) -> pd.DataFrame:
    """Per-(battery-penetration, scenario) mean peaks from the battery-pen sweep.
    In this sweep the ``pen_rate`` column denotes BATTERY penetration (the EV
    fleet is pinned), so every downstream label must read it as storage
    penetration rather than EV penetration.
    """
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if c != "scenario" and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return (df.groupby(["pen_rate", "scenario"])
              .agg(peak=("P_community_peak_kw", "mean"),
                   batt=("P_peak_batt_kw", "mean"),
                   ev=("P_peak_ev_kw", "mean"),
                   bld=("P_peak_building_kw", "mean"),
                   cf=("CF", "mean"))
              .reset_index())


def _scenario_peak_means(csv_path: Path) -> Dict[int, Dict[str, float]]:
    """Mean peak/CF/bill/attribution per scenario from a summary CSV, KEEPING S5.
    Unlike load_data(), this does not drop the S5 augmentation row; it is used
    only by the broadcast analysis, which must retain S5 explicitly.
    """
    df = pd.read_csv(csv_path)
    for c in df.columns:
        if c != "scenario" and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    out: Dict[int, Dict[str, float]] = {}
    for s, gg in df.groupby("scenario"):
        out[int(s)] = {
            "peak": gg["P_community_peak_kw"].mean(),
            "cf":   gg["CF"].mean(),
            "bill": gg["total_bill_usd"].mean() if "total_bill_usd" in gg else float("nan"),
            "batt": gg["P_peak_batt_kw"].mean() if "P_peak_batt_kw" in gg else float("nan"),
            "n":    len(gg),
        }
    return out


def fig14_storage_scaling(battpen_csv: Path, out_dir: Path) -> None:
    """F14: the rebound scales linearly with STORAGE penetration, and the same
    storage lowers the peak under self-consumption (S0) while raising it under
    arbitrage (S2). EV penetration fixed; battery penetration swept."""
    g = _battpen_table(battpen_csv)
    pens = sorted(g["pen_rate"].unique())
    if len(pens) < 2:
        log.warning("  F14 skipped: need >= 2 battery-penetration levels.")
        return
    x = np.array(pens, float)
    present = set(int(s) for s in g["scenario"].unique())

    def series(scn: int, col: str) -> np.ndarray:
        return np.array([g[(g.pen_rate == p) & (g.scenario == scn)][col].mean()
                         for p in pens], float)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.35))

    # Panel (a): total community peak vs storage -- the S0/S2 crossover.
    for scn, ls, mk in [(0, (0, (4, 3)), "o"), (1, "-", "s"), (2, "-", "D")]:
        if scn not in present:
            continue
        y = series(scn, "peak")
        axL.plot(x * 100, y, marker=mk, ls=ls, color=SCENARIO_COLORS[scn],
                 mfc="white", mec=SCENARIO_COLORS[scn], mew=1.4, ms=6, lw=1.6,
                 label=SCENARIO_NAMES_SHORT[scn])
    axL.set_xlabel("battery penetration (%)  [EV fixed at 25%]")
    axL.set_ylabel("community peak (kW)")
    axL.set_xticks(x * 100)
    axL.legend(loc="upper left", frameon=False, fontsize=7.5)
    add_panel_label(axL, "(a)", x=-0.16, y=1.04)

    # Panel (b): S2 battery-at-peak vs storage, with the linear fit.
    yb = series(2, "batt")
    sl, ic, r2 = _linfit(x, yb)
    xf = np.linspace(x.min(), x.max(), 50)
    axR.plot(xf * 100, sl * xf + ic, "-", color="0.55", lw=1.2, zorder=1,
             label=fr"linear fit ($R^2={r2:.4f}$)")
    axR.plot(x * 100, yb, "D", color=SCENARIO_COLORS[2], mfc="white",
             mec=SCENARIO_COLORS[2], mew=1.4, ms=6, zorder=2,
             label=r"S2 battery at $t^\star$")
    axR.set_xlabel("battery penetration (%)")
    axR.set_ylabel(r"S2 battery charging at $t^\star$ (kW)")
    axR.set_xticks(x * 100)
    axR.annotate(fr"$\approx${sl/50.0:.1f} kW per added pack",
                 xy=(0.05, 0.90), xycoords="axes fraction", fontsize=7.5,
                 color="0.35")
    axR.legend(loc="lower right", frameon=False, fontsize=7.5)
    add_panel_label(axR, "(b)", x=-0.16, y=1.04)

    fig.tight_layout(w_pad=1.8)
    save_figure(fig, out_dir, "F14_storage_scaling")


T06_IDX  = "Batt.\\ pen.\\ (\\%)"
T06_COLS = ["S0 peak", "S1 peak", "S2 peak",
            "S2 batt.\\ $@t^\\star$", "S2 EV $@t^\\star$", "S2 CF"]


def _tbl06_emit(tdf: "pd.DataFrame", out_dir: Path,
                pens: list, s0: list, s1: list, s2: list,
                batt: list, ev: list,
                r2b: float = None, r2t: float = None) -> None:
    """Build T06's note from the numbers in hand and emit the table.

    Split out of tbl06_storage_scaling() so that the note has exactly ONE
    definition. It previously had one, reachable only when --battpen-csv was
    supplied; when it was not, a T06_storage_scaling.tex generated by an older
    build simply survived on disk and was pulled into the manuscript verbatim. That is how
    the "S2 EV contribution stays near 18 kW" note -- refuted by its own
    21/14/21/14 column -- reached a compiled manuscript twice after being fixed.
    """
    x = np.array(pens, float)
    if r2b is None:
        _, _, r2b = _linfit(x, np.array(batt, float))
    if r2t is None:
        _, _, r2t = _linfit(x, np.array(s2, float))

    _ev = np.array(ev, float)
    ev_lo, ev_hi, ev_mean = _ev.min(), _ev.max(), _ev.mean()

    note = (
        "The penetration axis is BATTERY, not EV: the vehicle fleet is pinned at "
        f"25\\% while storage is swept, at one $(\\xi=0.30, \\rho=0.60)$ point, so "
        "peaks are cell means. The S2 battery term at $t^\\star$ is linear in "
        f"storage ($R^2={r2b:.4f}$) and the S2 peak inherits it ($R^2={r2t:.4f}$); "
        f"the S2 EV term neither grows nor trends, alternating {ev_lo:.0f}/{ev_hi:.0f}\\,kW "
        f"({ev_lo/7:.0f}--{ev_hi/7:.0f} vehicles at 7\\,kW, mean {ev_mean:.1f}) as $t^\\star$ "
        f"shifts, the fleet being fixed. At {pens[0]*100:.0f}\\% storage S1 "
        f"({s1[0]:.0f}\\,kW) exceeds S2 ({s2[0]:.0f}\\,kW): with so few packs there is "
        "almost no synchronized recharge to disperse, while the rate gate still "
        "charges delayed vehicles harder, so the mitigation is defined against a "
        "rebound that must first exist.")

    assert "stays near 18" not in note, "T06 note regression"
    _save_table(
        tdf, out_dir, "T06_storage_scaling",
        caption=("Storage-penetration scaling of the rebound at fixed EV "
                 "penetration."),
        label="tab:storage_scaling", note=note, wide=True)


def tbl06_refresh_from_emitted_csv(out_dir: Path) -> bool:
    """
    The raw battery-penetration sweep is a separate EnergyPlus run and is often
    not to hand, but T06_storage_scaling.csv carries every column the table
    needs. Refreshing from it means a stale .tex can never outlive a fix to the
    note, which is the failure this path exists to close. Returns True on
    success. The numbers are NOT recomputed - they are the ones already
    published - only the prose around them is rebuilt from current code.
    """
    csv = Path(out_dir) / "T06_storage_scaling.csv"
    if not csv.exists():
        return False
    try:
        d = pd.read_csv(csv)
        if T06_IDX not in d.columns or not set(T06_COLS) <= set(d.columns):
            log.warning(f"  T06 refresh skipped: {csv.name} lacks the expected columns.")
            return False
        d = d.sort_values(T06_IDX)
        pens = [float(v) / 100.0 for v in d[T06_IDX]]
        # preserve the published cell formatting: integers for kW, 3 dp for CF
        disp = {c: [f"{float(v):.0f}" for v in d[c]] for c in T06_COLS if c != "S2 CF"}
        disp["S2 CF"] = [f"{float(v):.3f}" for v in d["S2 CF"]]
        tdf = pd.DataFrame({c: disp[c] for c in T06_COLS},
                           index=[f"{v}" for v in d[T06_IDX]])
        tdf.index.name = T06_IDX
        # carry the EXACT published R^2 forward rather than refitting on rounded peaks
        r2b = r2t = None
        tex = Path(out_dir) / "T06_storage_scaling.tex"
        if tex.exists():
            found = re.findall(r"R\^2=([0-9.]+)", tex.read_text(errors="ignore"))
            if len(found) >= 2:
                r2b, r2t = float(found[0]), float(found[1])
        if r2b is None:
            log.warning("    T06 refresh: no prior R^2 found; refitting on the CSV's "
                        "rounded peaks, which perturbs the last digit. Supply "
                        "--battpen-csv for exact values.")
        _tbl06_emit(tdf, Path(out_dir), pens,
                    list(d["S0 peak"]), list(d["S1 peak"]), list(d["S2 peak"]),
                    list(d["S2 batt.\\ $@t^\\star$"]), list(d["S2 EV $@t^\\star$"]),
                    r2b=r2b, r2t=r2t)
        log.info(f"  T06 note REFRESHED from {csv.name} (numbers unchanged; prose rebuilt "
                 "from current code). Supply --battpen-csv to recompute the numbers too.")
        return True
    except Exception as e:
        log.warning(f"  T06 refresh failed: {type(e).__name__}: {e}")
        return False


def tbl06_storage_scaling(battpen_csv: Path, out_dir: Path) -> None:
    """T06: battery-penetration scaling table (peaks, attribution, linear fit)."""
    g = _battpen_table(battpen_csv)
    pens = sorted(g["pen_rate"].unique())
    if len(pens) < 2:
        log.warning("  T06 skipped: need >= 2 battery-penetration levels.")
        return

    def val(scn: int, col: str, p: float) -> float:
        return g[(g.pen_rate == p) & (g.scenario == scn)][col].mean()

    idx = T06_IDX
    rows = []
    for p in pens:
        rows.append({
            idx:                      f"{p*100:.0f}",
            "S0 peak":                f"{val(0,'peak',p):.0f}",
            "S1 peak":                f"{val(1,'peak',p):.0f}",
            "S2 peak":                f"{val(2,'peak',p):.0f}",
            "S2 batt.\\ $@t^\\star$":  f"{val(2,'batt',p):.0f}",
            "S2 EV $@t^\\star$":       f"{val(2,'ev',p):.0f}",
            "S2 CF":                  f"{val(2,'cf',p):.3f}",
        })
    tdf = pd.DataFrame(rows).set_index(idx)
    _tbl06_emit(tdf, out_dir, pens,
                [val(0, "peak", p) for p in pens],
                [val(1, "peak", p) for p in pens],
                [val(2, "peak", p) for p in pens],
                [val(2, "batt", p) for p in pens],
                [val(2, "ev",   p) for p in pens])


def fig15_broadcast_arc(broadcast_csv: Path, out_dir: Path,
                        naive_csv: Optional[Path] = None,
                        naive_peak_kw: float = BROADCAST_NAIVE_PEAK_KW_DEFAULT) -> None:
    """F15: the one-way broadcast cautionary arc. (a) mean-peak ranking with both
    broadcast variants; (b) the heterogeneous broadcast (S5) never beats the
    message-free stagger (S1) across EV penetration."""
    sc = _scenario_peak_means(broadcast_csv)
    if 5 not in sc:
        log.warning("  F15 skipped: no S5 (broadcast) rows in the supplied CSV; "
                    "pass --broadcast-csv pointing to the heterogeneous run.")
        return
    naive_peak = naive_peak_kw
    naive_src = "recorded default"
    if naive_csv is not None and Path(naive_csv).exists():
        try:
            naive_peak = _scenario_peak_means(Path(naive_csv))[5]["peak"]
            naive_src = "from --broadcast-naive-csv"
        except Exception as e:
            log.warning(f"  F15: could not read naive S5 from {naive_csv} "
                        f"({type(e).__name__}); using recorded default.")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.35))

    # Panel (a): horizontal mean-peak ranking, lowest (best) at top.
    bars = [("S0", sc[0]["peak"], SCENARIO_COLORS[0]),
            ("S1 stagger", sc[1]["peak"], SCENARIO_COLORS[1]),
            ("S4 valley-fill", sc[4]["peak"], SCENARIO_COLORS[4]),
            ("S5 broadcast\n(heterogeneous)", sc[5]["peak"], BROADCAST_HETERO_COLOR),
            ("S2 rebound", sc[2]["peak"], SCENARIO_COLORS[2]),
            ("S5 broadcast\n(uniform)", naive_peak, BROADCAST_NAIVE_COLOR)]
    bars.sort(key=lambda t: t[1])
    ypos = np.arange(len(bars))
    axL.barh(ypos, [b[1] for b in bars], color=[b[2] for b in bars],
             height=0.66, zorder=2)
    for yp, (_, v, _) in zip(ypos, bars):
        axL.annotate(f"{v:.0f}", (v, yp), xytext=(4, 0),
                     textcoords="offset points", va="center", fontsize=7.3)
    axL.set_yticks(ypos)
    axL.set_yticklabels([b[0] for b in bars], fontsize=7.2)
    axL.invert_yaxis()
    axL.set_xlabel("mean community peak (kW)")
    axL.set_xlim(0, max(b[1] for b in bars) * 1.14)
    add_panel_label(axL, "(a)", x=-0.42, y=1.04)

    # Panel (b): S1 vs S5(het) vs S2 peak by EV penetration.
    df = pd.read_csv(broadcast_csv)
    for c in df.columns:
        if c != "scenario" and df[c].dtype == object:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pen_rate" in df.columns and df[df.scenario == 5]["pen_rate"].nunique() >= 2:
        pens = sorted(df["pen_rate"].unique())
        xp = np.array(pens, float) * 100

        def by_pen(scn: int) -> np.ndarray:
            return np.array([df[(df.scenario == scn) & (df.pen_rate == p)]
                             ["P_community_peak_kw"].mean() for p in pens], float)
        for scn, col, mk, lab in [(2, SCENARIO_COLORS[2], "D", "S2 rebound"),
                                   (5, BROADCAST_HETERO_COLOR, "o", "S5 broadcast (het.)"),
                                   (1, SCENARIO_COLORS[1], "s", "S1 stagger")]:
            axR.plot(xp, by_pen(scn), marker=mk, ls="-", color=col, mfc="white",
                     mec=col, mew=1.4, ms=6, lw=1.6, label=lab)
        axR.set_xlabel("EV penetration (%)")
        axR.set_ylabel("community peak (kW)")
        axR.set_xticks(xp)
        # Lower-right is the empty corner: all three traces rise with penetration,
        # so "upper left" overlapped the S2 rebound and S5 curves.
        axR.legend(loc="lower right", frameon=False, fontsize=7.3)
    else:
        axR.text(0.5, 0.5, "per-penetration S5\nnot available", ha="center",
                 va="center", transform=axR.transAxes, fontsize=8, color="0.5")
        axR.set_axis_off()
    add_panel_label(axR, "(b)", x=-0.16, y=1.04)

    fig.tight_layout(w_pad=2.0)
    save_figure(fig, out_dir, "F15_broadcast_arc")
    log.info(f"  F15: uniform-broadcast S5 peak = {naive_peak:.1f} kW ({naive_src}); "
             f"heterogeneous S5 = {sc[5]['peak']:.1f} kW; stagger S1 = "
             f"{sc[1]['peak']:.1f} kW.")


def _het_vs_stagger_counts(broadcast_csv: Path) -> Tuple[int, float]:
    """Cells (of 36) in which the heterogeneous S5 peak exceeds the S1 stagger,
    with the paired Wilcoxon p-value on the matched per-condition peaks. Returns
    (n_above, p). Falls back to (36, <0.001) if the frame lacks matched rows."""
    try:
        df = pd.read_csv(broadcast_csv)
        key = ["pen_rate", "inel_rate", "markov_rho"]
        s1 = df[df["scenario"] == 1].sort_values(key).reset_index(drop=True)
        s5 = df[df["scenario"] == 5].sort_values(key).reset_index(drop=True)
        if len(s1) == 0 or len(s5) == 0 or len(s1) != len(s5):
            return (36, 0.0)
        p1 = s1["P_community_peak_kw"].to_numpy()
        p5 = s5["P_community_peak_kw"].to_numpy()
        n_above = int((p5 > p1).sum())
        try:
            pval = float(stats.wilcoxon(p5, p1, alternative="greater").pvalue)
        except Exception:
            pval = 0.0
        return (n_above, pval)
    except Exception:
        return (36, 0.0)


def tbl07_broadcast_arc(broadcast_csv: Path, out_dir: Path,
                        naive_csv: Optional[Path] = None,
                        naive_peak_kw: float = BROADCAST_NAIVE_PEAK_KW_DEFAULT) -> None:
    """T07: one-way broadcast comparison (both variants vs the stagger)."""
    sc = _scenario_peak_means(broadcast_csv)
    if 5 not in sc:
        log.warning("  T07 skipped: no S5 rows in the supplied CSV.")
        return
    naive = dict(peak=naive_peak_kw, cf=float("nan"), bill=float("nan"))
    if naive_csv is not None and Path(naive_csv).exists():
        try:
            n5 = _scenario_peak_means(Path(naive_csv))[5]
            naive = dict(peak=n5["peak"], cf=n5["cf"], bill=n5["bill"])
        except Exception:
            pass

    s1 = sc[1]

    def dpk(v: float) -> str:

        d = 100 * (v - s1["peak"]) / s1["peak"]
        return f"${d:+.1f}\\%$" if abs(d) < 0.5 else f"${d:+.0f}\\%$"

    def cf(v: float) -> str:
        return f"{v:.3f}" if v == v else "--"

    def bl(v: float) -> str:
        return f"{v/1000:.0f}" if v == v else "--"

    idx = "Scenario"
    rows = [
        {idx: "S1 stagger (comm-free)",       "Peak (kW)": f"{s1['peak']:.0f}",
         "vs S1": "--", "CF": cf(s1['cf']), "Bill (k\\$)": bl(s1['bill'])},
        {idx: "S5 broadcast, heterogeneous",  "Peak (kW)": f"{sc[5]['peak']:.0f}",
         "vs S1": dpk(sc[5]['peak']), "CF": cf(sc[5]['cf']), "Bill (k\\$)": bl(sc[5]['bill'])},
        {idx: "S5 broadcast, uniform",        "Peak (kW)": f"{naive['peak']:.0f}",
         "vs S1": dpk(naive['peak']), "CF": cf(naive['cf']), "Bill (k\\$)": bl(naive['bill'])},
        {idx: "S2 rebound (uncoordinated)",   "Peak (kW)": f"{sc[2]['peak']:.0f}",
         "vs S1": dpk(sc[2]['peak']), "CF": cf(sc[2]['cf']), "Bill (k\\$)": bl(sc[2]['bill'])},
    ]
    tdf = pd.DataFrame(rows).set_index(idx)

    het_above, het_p = _het_vs_stagger_counts(broadcast_csv)
    _below = 36 - het_above
    if het_above == 36:
        _het_clause = ("recovers that loss to a mean below the rebound but confers no "
                       "advantage over the message-free stagger, remaining above it in "
                       "all 36 cells")
    else:
        _het_clause = (f"recovers that loss but confers no systematic advantage over the "
                       f"message-free stagger, falling below it in {_below} of 36 cells "
                       f"and above in {het_above}")
    note = (
        "One-way community-load broadcast added as a sixth scenario (S5) over the "
        "36-cell design; peaks are design means. The uniform-response variant, every "
        "house throttling identically, re-synchronizes the fleet above the "
        "uncoordinated rebound. The heterogeneous-threshold variant, each house "
        "throttling only above a low-discrepancy personal threshold, "
        + _het_clause +
        f" (paired Wilcoxon $p {fmt_p(het_p)[1:]}). Where the uniform-run summary "
        "omits a downstream field, its peak is the recorded value and its CF and "
        "bill are omitted.")
    _save_table(
        tdf, out_dir, "T07_broadcast_arc",
        caption=("One-way broadcast versus the communication-free stagger."),
        label="tab:broadcast_arc", note=note, wide=True)


# FIGURE 16 / TABLE 8 - THE i.i.d.-RANDOMIZED STAGGER ABLATION
#   "Replacing the van der Corput construction with an i.i.d.-uniform delay,
#    holding every other layer fixed, RAISES the community peak in all 36
#    conditions.  The coincidence factor says the opposite, and it is wrong to
#    listen to it: CF falls only because the per-house maxima in its denominator
#    inflate faster than the aggregate in its numerator."
ABLATION_VDC_COLOR: str = SCENARIO_COLORS[1]
ABLATION_IID_COLOR: str = COLORS_WONG["orange"]

_ABLATION_KEY = ["pen_rate", "inel_rate", "markov_rho"]


def _ablation_arm_label(csv_path: Path) -> str:

    marker = Path(csv_path).parent / "run_mode.json"
    if marker.exists():
        try:
            sig = _json.loads(marker.read_text(encoding="utf-8"))
            for key in ("ablation_arm", "stagger_mode"):
                val = str(sig.get(key, "")).lower()
                if val in ("vdc", "iid"):
                    return val
        except Exception:
            pass
    name = Path(csv_path).name.lower()
    if "iid" in name:
        return "iid"
    if "vdc" in name:
        return "vdc"
    return "unknown"


def _load_ablation_arms(
    vdc_csv: Path,
    iid_csv: Path,
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:

    lab_v, lab_i = _ablation_arm_label(vdc_csv), _ablation_arm_label(iid_csv)
    if lab_v == lab_i and lab_v != "unknown":
        log.error("  F16/T08 REFUSED: both arguments point at the '%s' arm (%s, %s); "
                  "the ablation needs one of each.", lab_v,
                  Path(vdc_csv).name, Path(iid_csv).name)
        return None
    if lab_v == "iid" or lab_i == "vdc":
        log.error("  F16/T08 REFUSED: the ablation arms are SWAPPED.")
        log.error("    --ablation-vdc-csv resolves to the '%s' arm (%s)", lab_v,
                  Path(vdc_csv).name)
        log.error("    --ablation-iid-csv resolves to the '%s' arm (%s)", lab_i,
                  Path(iid_csv).name)
        log.error("    Swapped arms stay perfectly paired, so every check below "
                  "would pass and the figure would render with the sign reversed, "
                  "reporting the deployed construction as the worse one.")
        return None
    if "unknown" in (lab_v, lab_i):
        log.warning("  ablation arm could not be confirmed from run_mode.json or the "
                    "filename (vdc='%s', iid='%s'); proceeding on argument order "
                    "alone. Check the sign of the result against Table 8.",
                    lab_v, lab_i)

    v = pd.read_csv(vdc_csv)
    i = pd.read_csv(iid_csv)
    v = v[v["scenario"] == 1].sort_values(_ABLATION_KEY).reset_index(drop=True)
    i = i[i["scenario"] == 1].sort_values(_ABLATION_KEY).reset_index(drop=True)
    if len(v) == 0 or len(i) == 0 or len(v) != len(i):
        log.error("  F16/T08 skipped: arms have %d and %d scenario-1 rows.",
                  len(v), len(i))
        return None
    if not np.array_equal(v[_ABLATION_KEY].values, i[_ABLATION_KEY].values):
        log.error("  F16/T08 skipped: the two arms do not share a design grid.")
        return None

    fails: List[str] = []
    d_ne = i["L_n_elastic"].values - v["L_n_elastic"].values
    if np.any(d_ne != 0):
        fails.append(f"L_n_elastic differs in {int((d_ne != 0).sum())}/{len(v)} cells")
    if not np.allclose(v["E_pv_kwh"].values, i["E_pv_kwh"].values, rtol=1e-9):
        fails.append("E_pv_kwh differs")
    for col in ("E_ev_kwh", "E_annual_kwh"):
        rel = np.abs(i[col].values - v[col].values) / np.maximum(v[col].values, 1e-9)
        if rel.max() >= 0.02:
            fails.append(f"{col} moves {100 * rel.max():.2f}% (limit 2%)")
    if fails:
        log.error("  F16/T08 REFUSED: the arms are NOT paired (%s).", "; ".join(fails))
        log.error("    The two arms drew different fleets, so any difference mixes the "
                  "construction with seed noise. Re-run BOTH arms with a plugin that "
                  "pins the per-house seed (v14.2 SIM_SEED) before regenerating.")
        return None

    log.info("  ablation pairing verified: L_n_elastic identical in %d/%d cells, "
             "PV identical, energy conserved.", len(v), len(v))
    return v, i


def _ablation_stats(v: pd.DataFrame, i: pd.DataFrame) -> Dict[str, object]:
    """Paired statistics and the CF decomposition for F16/T08."""
    pk_v = v["P_community_peak_kw"].values
    pk_i = i["P_community_peak_kw"].values
    cf_v, cf_i = v["CF"].values, i["CF"].values
    # CF = P_agg / sum_i P_i^peak, so the denominator is recoverable exactly.
    den_v, den_i = pk_v / cf_v, pk_i / cf_i
    d_pk = pk_i - pk_v

    _, lo, hi = bca_ci(d_pk, np.mean)
    w_pk = wilcoxon_test(pk_v, pk_i, alternative="less")   # H1: vdC peak < iid peak
    w_cf = wilcoxon_test(cf_i, cf_v, alternative="less")   # H1: iid CF  < vdC CF
    w_dn = wilcoxon_test(den_v, den_i, alternative="less")
    return dict(
        n=len(v),
        pk_v=float(pk_v.mean()), pk_i=float(pk_i.mean()),
        d_pk=float(d_pk.mean()), d_pk_lo=float(lo), d_pk_hi=float(hi),
        d_pk_pct=float(100 * d_pk.mean() / pk_v.mean()),
        n_worse=int((d_pk > 0).sum()),
        p_pk=float(w_pk["p"]), cliff_pk=cliff_delta(pk_i, pk_v),
        cf_v=float(cf_v.mean()), cf_i=float(cf_i.mean()),
        d_cf=float((cf_i - cf_v).mean()), p_cf=float(w_cf["p"]),
        cf_pct=float(100 * (cf_i.mean() / cf_v.mean() - 1)),
        den_v=float(den_v.mean()), den_i=float(den_i.mean()),
        den_pct=float(100 * (den_i.mean() / den_v.mean() - 1)), p_den=float(w_dn["p"]),
        num_pct=float(100 * (pk_i.mean() / pk_v.mean() - 1)),
        evpk_v=float(v["P_peak_ev_kw"].mean()), evpk_i=float(i["P_peak_ev_kw"].mean()),
        bill_v=float(v["total_bill_usd"].mean()), bill_i=float(i["total_bill_usd"].mean()),
        jain_v=float(v["jain_fairness"].mean()), jain_i=float(i["jain_fairness"].mean()),
    )


def fig16_iid_ablation(vdc_csv: Path, iid_csv: Path, out_dir: Path) -> None:
    """F16: the paired i.i.d. ablation and the coincidence-factor trap."""
    arms = _load_ablation_arms(vdc_csv, iid_csv)
    if arms is None:
        return
    v, i = arms
    S = _ablation_stats(v, i)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.1, 3.15))

    # (a) the headline: raw peak by penetration, both arms 
    pens = sorted(v["pen_rate"].unique())
    xs = np.arange(len(pens))
    for df, col, mk, lab in [(v, ABLATION_VDC_COLOR, "s", "van der Corput (deployed)"),
                             (i, ABLATION_IID_COLOR, "o", "i.i.d. uniform (ablation)")]:
        means, los, his = [], [], []
        for p in pens:
            y = df.loc[df["pen_rate"] == p, "P_community_peak_kw"].values
            m, lo, hi = bca_ci(y, np.mean)
            means.append(m); los.append(m - lo); his.append(hi - m)
        axL.errorbar(xs, means, yerr=[los, his], marker=mk, ms=5, lw=1.5,
                     color=col, mfc="white", mec=col, mew=1.3, capsize=2.5,
                     label=lab, zorder=3)
    # per-condition paired points make "worse in every cell" visible, not asserted
    for df, col in [(v, ABLATION_VDC_COLOR), (i, ABLATION_IID_COLOR)]:
        jitter = -0.06 if col == ABLATION_VDC_COLOR else 0.06
        axL.scatter([pens.index(p) + jitter for p in df["pen_rate"]],
                    df["P_community_peak_kw"], s=5, color=col, alpha=0.35,
                    linewidths=0, zorder=2)
    axL.set_xticks(xs)
    axL.set_xticklabels([f"{p:.0%}" for p in pens])
    axL.set_xlabel("EV penetration")
    axL.set_ylabel("Community peak (kW)")
    axL.legend(loc="upper left", framealpha=0.92)
    axL.grid(axis="y", alpha=0.25, lw=0.4)
    add_panel_label(axL, "(a)", x=-0.17, y=1.03)
    axes_title(axL, "Randomization raises the peak at every penetration")

    # (b) the CF trap: which term of CF = P_agg / sum(P_i) actually moves 
    labels = [r"Numerator $P_{\mathrm{agg}}$",
              r"Denominator $\sum_i P_i^{\mathrm{peak}}$",
              r"Ratio CF"]
    vals = [S["num_pct"], S["den_pct"], S["cf_pct"]]
    cols = [ABLATION_IID_COLOR, ABLATION_IID_COLOR, COLORS_WONG["skyblue"]]
    ypos = np.arange(len(labels))[::-1]
    axR.barh(ypos, vals, color=cols, edgecolor="black", linewidth=0.5, height=0.6)
    axR.axvline(0, color="black", lw=0.8)
    for y, val in zip(ypos, vals):
        axR.text(val + (0.25 if val > 0 else -0.25), y, f"{val:+.1f}%",
                 va="center", ha="left" if val > 0 else "right", fontsize=7.5)
    axR.set_yticks(ypos)
    axR.set_yticklabels(labels)
    axR.set_xlabel("Change under i.i.d. randomization (%)")
    axR.set_xlim(min(vals) - 3.2, max(vals) + 3.2)
    axR.grid(axis="x", alpha=0.25, lw=0.4)

    add_panel_label(axR, "(b)", x=-0.42, y=1.03)
    axes_title(axR, "Why CF misreads the ablation")

    fig.tight_layout()
    save_figure(fig, out_dir, "F16_iid_ablation")


def tbl08_iid_ablation(vdc_csv: Path, iid_csv: Path, out_dir: Path) -> None:
    """T08: paired i.i.d. ablation, headline quantities and the CF decomposition."""
    arms = _load_ablation_arms(vdc_csv, iid_csv)
    if arms is None:
        return
    v, i = arms
    S = _ablation_stats(v, i)

    idx = "Quantity"
    rows: List[Dict[str, str]] = []

    # Per-penetration peaks: the every cell claim, cell by cell.
    for p in sorted(v["pen_rate"].unique()):
        a = v.loc[v["pen_rate"] == p, "P_community_peak_kw"].values
        b = i.loc[i["pen_rate"] == p, "P_community_peak_kw"].values

        wp = wilcoxon_test(a, b, alternative="less")
        rows.append({
            idx: f"\\quad Community peak, $\\eta = {p:.2f}$ (kW)",
            "vdC": f"{a.mean():.1f}", "i.i.d.": f"{b.mean():.1f}",
            "$\\Delta$": f"${b.mean() - a.mean():+.1f}$",
            "iid worse": f"{int((b > a).sum())}/{len(a)}",
            "$p$": fmt_p(float(wp["p"])),
        })

    rows.append({
        idx: "Community peak, pooled (kW)",
        "vdC": f"{S['pk_v']:.1f}", "i.i.d.": f"{S['pk_i']:.1f}",
        "$\\Delta$": f"${S['d_pk']:+.2f}$ [${S['d_pk_lo']:+.2f}$, ${S['d_pk_hi']:+.2f}$]",
        "iid worse": f"{S['n_worse']}/{S['n']}", "$p$": fmt_p(S["p_pk"]),
    })
    rows.append({
        idx: r"Denominator $\sum_i P_i^{\mathrm{peak}}$ (kW)",
        "vdC": f"{S['den_v']:.1f}", "i.i.d.": f"{S['den_i']:.1f}",
        "$\\Delta$": f"${S['den_pct']:+.1f}\\%$", "iid worse": "--",
        "$p$": fmt_p(S["p_den"]),
    })
    rows.append({
        idx: "Per-house EV peak (kW)",
        "vdC": f"{S['evpk_v']:.1f}", "i.i.d.": f"{S['evpk_i']:.1f}",
        "$\\Delta$": f"${S['evpk_i'] - S['evpk_v']:+.1f}$", "iid worse": "--", "$p$": "--",
    })
    rows.append({
        idx: "Coincidence factor",
        "vdC": f"{S['cf_v']:.3f}", "i.i.d.": f"{S['cf_i']:.3f}",
        "$\\Delta$": f"${S['d_cf']:+.3f}$", "iid worse": "--", "$p$": fmt_p(S["p_cf"]),
    })
    rows.append({
        idx: "Annual bill (kUSD)",
        "vdC": f"{S['bill_v'] / 1000:.1f}", "i.i.d.": f"{S['bill_i'] / 1000:.1f}",
        "$\\Delta$": f"${(S['bill_i'] - S['bill_v']) / 1000:+.1f}$",
        "iid worse": "--", "$p$": "--",
    })

    rows.append({
        idx: "Jain fairness",
        "vdC": f"{S['jain_v']:.4f}", "i.i.d.": f"{S['jain_i']:.4f}",
        "$\\Delta$": f"${S['jain_i'] - S['jain_v']:+.4f}$", "iid worse": "--", "$p$": "--",
    })

    tdf = pd.DataFrame(rows).set_index(idx)
    note = (
        "Paired ablation over the 36-cell design: both arms sweep the same "
        "conditions under the same per-house seeds, so the fleets are identical in "
        "elasticity, occupancy, and vehicle demand and the only difference is the "
        "Layer-1 construction. Pairing is verified before the table is emitted "
        "(identical elastic-house counts and photovoltaic generation; EV and annual "
        "energy conserved). The $\\Delta$ column reports i.i.d.\\ minus van der "
        "Corput, with a BCa 95\\% confidence interval on the pooled peak and paired "
        "Wilcoxon signed-rank $p$-values. Randomization raises the community peak in "
        "every condition, yet lowers CF, because the per-house maxima in the CF "
        "denominator inflate faster than the aggregate in its numerator; the raw "
        "peak, not the ratio, is the quantity a planner sizes to.")
    _save_table(
        tdf, out_dir, "T08_iid_ablation",
        caption=("The i.i.d.-randomized stagger ablation against the deployed "
                 "van der Corput construction."),
        label="tab:iid_ablation", note=note, wide=True)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=("Applied Energy — Publication-grade analysis of the SRE "
                     "sensitivity sweep results."),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("./Results_Sensitivity"),
        help="Directory containing SRE_metrics_summary.csv (and optional "
             "Sobol_*.csv files).",
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="Publication",
        help="Subdirectory under --results-dir to write publication assets.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip figure generation (tables only).",
    )
    parser.add_argument(
        "--skip-tables",
        action="store_true",
        help="Skip table generation (figures only).",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="Skip the interactive HTML dashboard.",
    )
    parser.add_argument(
        "--power-csvs",
        type=str,
        nargs="+",
        default=None,
        metavar="KW:PATH",
        help="Charge-power sweep CSVs as 'power:path' pairs, e.g. "
             "7.0:results_7kw.csv 11.5:results_11kw.csv 19.2:results_19kw.csv. "
             "If omitted, auto-discovers results_<N>kw.csv in --results-dir. "
             "Drives the F11/T05 charge-power dimension (>= 2 levels required).",
    )
    parser.add_argument(
        "--battpen-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="SRE_metrics_summary.csv from the BATTERY-PENETRATION sweep "
             "(ENABLE_S5=False, BATTERY_PEN_SWEEP=True), whose pen_rate column "
             "denotes battery penetration. Drives the F14/T06 storage-scaling "
             "figure and table (the S0/S2 crossover). Rendered only if supplied.",
    )
    parser.add_argument(
        "--broadcast-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="SRE_metrics_summary.csv from the ONE-WAY BROADCAST run "
             "(ENABLE_S5=True) containing the heterogeneous-threshold S5 rows. "
             "Drives the F15/T07 broadcast cautionary arc. Rendered only if "
             "supplied and if it contains a scenario-5 row.",
    )
    parser.add_argument(
        "--broadcast-naive-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="Optional SRE_metrics_summary.csv from the UNIFORM-response "
             "broadcast run, used to recompute the uniform S5 peak in F15/T07. "
             f"If omitted, the recorded value "
             f"({BROADCAST_NAIVE_PEAK_KW_DEFAULT:.1f} kW) is used.",
    )
    parser.add_argument(
        "--ablation-vdc-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="SRE_metrics_summary_ablation_vdc.csv from the PAIRED i.i.d. "
             "ablation control arm (Run_Manager ABLATION_ARM='vdc'). Must be "
             "paired with --ablation-iid-csv: same seeds, same 36 conditions. "
             "Drives the F16/T08 ablation figure and table.",
    )
    parser.add_argument(
        "--ablation-iid-csv",
        type=Path,
        default=None,
        metavar="PATH",
        help="SRE_metrics_summary_ablation_iid.csv from the PAIRED i.i.d. "
             "ablation test arm (Run_Manager ABLATION_ARM='iid', plugin "
             "ALGO1_STAGGER_MODE='iid'). Both arms must come from a plugin that "
             "pins the per-house seed (v14.2 SIM_SEED); F16/T08 verify this and "
             "refuse to render if the arms drew different fleets.",
    )
    parser.add_argument(
        "--draft-titles",
        action="store_true",
        help="Draw the data-derived titles ON the artwork. OFF by default: "
             "Elsevier requires the title in the caption and not on the figure "
             "itself. Use for internal review PNGs only, never for submission.",
    )
    args = parser.parse_args(argv)

    global DRAW_ON_ARTWORK_TITLES
    DRAW_ON_ARTWORK_TITLES = bool(args.draft_titles)
    if DRAW_ON_ARTWORK_TITLES:
        log.warning("--draft-titles: on-artwork titles ENABLED. Do NOT submit "
                    "these files; Elsevier requires captions to carry the title.")

    results_dir = args.results_dir.resolve()
    out_dir     = (results_dir / args.out_subdir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info(f"IEEE_Sensitivity_Analysis v1.8 | results_dir = {results_dir}")
    log.info(f"                                 out_dir     = {out_dir}")
    log.info("=" * 70)

    # Load data 
    data = load_data(results_dir)
    setup_style()

    # Figures
    if not args.skip_figures:
        log.info("[1/3] Generating publication figures...")
        log_lp_opf_ratios(data)
        for fn in [
            fig01_mitigation_headline,
            fig02_cf_heatmap,
            fig03_sobol_indices,
            fig04_bill_decomposition,
            fig05_peak_attribution,
            fig06_pareto_cf_bill,
            fig07_commfree_optimality_gap,
            fig08_jain_fairness,
            fig09_carbon_emissions,
            fig10_metric_distributions,
        ]:
            try:
                fn(data, out_dir)
            except Exception as e:
                log.error(f"  ✗ {fn.__name__} failed: {type(e).__name__}: {e}",
                          exc_info=False)

    # Tables
    if not args.skip_tables:
        log.info("[2/3] Generating publication tables...")
        for fn in [
            tbl01_master_summary,
            tbl02_sobol_indices,
            tbl03_scenario_summary,
            tbl04_significance_tests,
        ]:
            try:
                fn(data, out_dir)
            except Exception as e:
                log.error(f"  ✗ {fn.__name__} failed: {type(e).__name__}: {e}",
                          exc_info=False)

    # Charge-power dimension (F11 / T05)
    power_csvs = _resolve_power_csvs(args.power_csvs, results_dir)
    if len(power_csvs) >= 2:
        log.info(f"[+] Charge-power dimension: {len(power_csvs)} level(s) at "
                 f"{sorted(power_csvs)} kW")
        if not args.skip_figures:
            try:
                fig11_charge_power_sensitivity(power_csvs, out_dir)
            except Exception as e:
                log.error(f"  ✗ fig11 failed: {type(e).__name__}: {e}")
        if not args.skip_tables:
            try:
                tbl05_charge_power_sensitivity(power_csvs, out_dir)
            except Exception as e:
                log.error(f"  ✗ tbl05 failed: {type(e).__name__}: {e}")
    else:
        log.info("[+] Charge-power dimension: < 2 power CSVs found; skipping F11/T05 "
                 "(pass --power-csvs or place results_<N>kw.csv in --results-dir).")

    # Battery-penetration dimension (F14 / T06)
    if args.battpen_csv is None and not args.skip_tables:
        stale = out_dir / "T06_storage_scaling.tex"
        log.warning("[!] --battpen-csv not supplied: F14 and T06 numbers will NOT be recomputed.")
        if tbl06_refresh_from_emitted_csv(out_dir):
            pass
        elif stale.exists():
            txt = stale.read_text(errors="ignore")
            if "stays near 18" in txt:
                log.error("    ✗ %s on disk carries the RETRACTED note (\"stays near 18 kW\"), "
                          "which its own 21/14/21/14 column refutes, and neither "
                          "--battpen-csv nor T06_storage_scaling.csv is available to rebuild "
                          "it. DO NOT SUBMIT with this file. Supply --battpen-csv.", stale.name)
            else:
                log.warning("    %s is present and will be used AS IS; it was not "
                            "regenerated by this run.", stale.name)

    if args.battpen_csv is not None:
        bp = args.battpen_csv
        if not bp.is_absolute() and not bp.exists():
            bp = results_dir / bp
        if bp.exists():
            log.info(f"[+] Battery-penetration dimension: {bp.name}")
            if not args.skip_figures:
                try:
                    fig14_storage_scaling(bp, out_dir)
                except Exception as e:
                    log.error(f"  ✗ fig14 failed: {type(e).__name__}: {e}")
            if not args.skip_tables:
                try:
                    tbl06_storage_scaling(bp, out_dir)
                except Exception as e:
                    log.error(f"  ✗ tbl06 failed: {type(e).__name__}: {e}")
        else:
            log.warning(f"  --battpen-csv '{args.battpen_csv}' not found; "
                        "skipping F14/T06.")

    # i.i.d. stagger ablation (F16 / T08)
    abl_v, abl_i = args.ablation_vdc_csv, args.ablation_iid_csv
    if (abl_v is None) != (abl_i is None):
        log.warning("[!] Only one ablation arm supplied (--ablation-%s-csv); F16/T08 "
                    "need BOTH the vdc control and the iid test arm. Skipping.",
                    "vdc" if abl_v is not None else "iid")
    elif abl_v is not None:
        def _resolve(p: Path) -> Path:
            return p if (p.is_absolute() or p.exists()) else results_dir / p
        pv, pi = _resolve(abl_v), _resolve(abl_i)
        if not pv.exists() or not pi.exists():
            missing = [str(p) for p in (pv, pi) if not p.exists()]
            log.warning("  ablation CSV(s) not found: %s; skipping F16/T08.",
                        ", ".join(missing))
        else:
            log.info(f"[+] i.i.d. ablation: {pv.name} vs {pi.name}")
            if not args.skip_figures:
                try:
                    fig16_iid_ablation(pv, pi, out_dir)
                except Exception as e:
                    log.error(f"  ✗ fig16 failed: {type(e).__name__}: {e}")
            if not args.skip_tables:
                try:
                    tbl08_iid_ablation(pv, pi, out_dir)
                except Exception as e:
                    log.error(f"  ✗ tbl08 failed: {type(e).__name__}: {e}")

    # One-way broadcast dimension (F15 / T07)
    if args.broadcast_csv is not None:
        bc = args.broadcast_csv
        if not bc.is_absolute() and not bc.exists():
            bc = results_dir / bc
        nv = args.broadcast_naive_csv
        if nv is not None and not nv.is_absolute() and not nv.exists():
            nv = results_dir / nv
        if bc.exists():
            log.info(f"[+] One-way broadcast dimension: {bc.name}"
                     + (f" (uniform ref: {nv.name})" if nv and nv.exists() else ""))
            if not args.skip_figures:
                try:
                    fig15_broadcast_arc(bc, out_dir, naive_csv=nv)
                except Exception as e:
                    log.error(f"  ✗ fig15 failed: {type(e).__name__}: {e}")
            if not args.skip_tables:
                try:
                    tbl07_broadcast_arc(bc, out_dir, naive_csv=nv)
                except Exception as e:
                    log.error(f"  ✗ tbl07 failed: {type(e).__name__}: {e}")
        else:
            log.warning(f"  --broadcast-csv '{args.broadcast_csv}' not found; "
                        "skipping F15/T07.")

    # Interactive HTML dashboard
    if not args.skip_html and PLOTLY_OK:
        log.info("[3/3] Generating interactive HTML report...")
        try:
            report_html(data, out_dir)
        except Exception as e:
            log.error(f"  ✗ report_html failed: {type(e).__name__}: {e}",
                      exc_info=False)
    elif not args.skip_html:
        log.warning("[3/3] Plotly not installed — skipping HTML report.")

    # Summary
    log.info("=" * 70)
    log.info("KEY FINDINGS (auto-summary):")

    # Mitigation success rate + headline BCa 95% CI (10,000 resamples, seed 42)
    mit = data.df_s1["MIT_ratio"].dropna()
    n_mit = (mit < 1.0).sum()
    _, mit_lo, mit_hi = bca_ci(mit.values)
    log.info(f"  • MIT < 1 in {n_mit}/{len(mit)} conditions  "
             f"(mean MIT = {mit.mean():.3f}, BCa 95% CI [{mit_lo:.3f}, {mit_hi:.3f}], "
             f"range [{mit.min():.3f}, {mit.max():.3f}])")

    # SRE severity + headline BCa 95% CI
    sre = data.df_s1["SRE_ratio"].dropna()
    _, sre_lo, sre_hi = bca_ci(sre.values)
    log.info(f"  • SRE severity: mean = {sre.mean():.3f}, "
             f"BCa 95% CI [{sre_lo:.3f}, {sre_hi:.3f}]")

    # Bill savings S1 vs S2
    s1_bill = data.df_s1["total_bill_usd"].mean()
    s2_bill = data.df_s2["total_bill_usd"].mean()
    log.info(f"  • S1 vs S2 annual bill: ${s1_bill:,.0f} vs ${s2_bill:,.0f}  "
             f"(savings ${s2_bill - s1_bill:,.0f}, {100*(s2_bill-s1_bill)/s2_bill:.1f}%)")

    gap = data.df_s1["comm_free_oracle_gap_pct"].dropna()
    if len(gap):
        log.info(f"  • S1 distance to relaxed comm-free lower bound "
                 f"(rate-uncapped, loose): median={gap.median():.1f}%, "
                 f"mean={gap.mean():.1f}%")

    # Jain fairness + headline BCa 95% CI
    jain_arr = data.df_s1["jain_fairness"].dropna()
    _, jain_lo, jain_hi = bca_ci(jain_arr.values)
    log.info(f"  • S1 Jain fairness: mean = {jain_arr.mean():.3f}, "
             f"BCa 95% CI [{jain_lo:.3f}, {jain_hi:.3f}]")

    log.info("=" * 70)
    log.info(f"All outputs in: {out_dir}")
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
