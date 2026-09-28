from __future__ import annotations

from collections import Counter
import re
from typing import Optional
from pathlib import Path

import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Wong colour-blind-safe palette
WONG = {
    "black":  "#000000",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

DPI = 300

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "font.family": "serif",
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
})


DRAW_ON_ARTWORK_TITLES: bool = False


def figure_title(fig: plt.Figure, text: str, **kw) -> None:
    """Figure-level title; drawn only in draft mode."""
    if DRAW_ON_ARTWORK_TITLES:
        fig.suptitle(text, **kw)


def axes_title(ax, text: str, **kw) -> None:
    """Single-axes figure title (ax.set_title IS the figure title); draft only."""
    if DRAW_ON_ARTWORK_TITLES:
        ax.set_title(text, **kw)



import json as _json, datetime as _dt

PROVENANCE_OWNER = "make_schematic_figures1.py"


def _claim_output(base_name: str) -> None:
    """Refuse to overwrite a figure owned by a different generator."""
    side = Path(f"{base_name}.provenance.json")
    if side.exists():
        try:
            prev = _json.loads(side.read_text()).get("owner")
        except Exception:
            prev = None
        if prev and prev != PROVENANCE_OWNER:
            raise SystemExit(
                f"\nREFUSING TO OVERWRITE: '{base_name}' is owned by {prev}, "
                f"not by {PROVENANCE_OWNER}.\n"
                f"Two generators are competing for one base name. Rename one of "
                f"them rather than letting the last run win -- a silent overwrite "
                f"here produces a figure that contradicts its caption.\n"
                f"If this is deliberate, delete {side} and re-run.\n")
    side.write_text(_json.dumps(
        {"owner": PROVENANCE_OWNER,
         "written": _dt.datetime.now().isoformat(timespec="seconds")}, indent=2))



CAPTION_CONTRACTS: dict = {

    "Fig5_LP_OPF_Bound": [
        "Community baseline",                                  # light-blue region
        "Staggered EV charging (illustrative)",                # solid orange line
        "Peak-minimizing placement of the same energy (bound)", # dashed green line
        "Hour of day (illustrative day)",                      # illustrative day
        "optimality",                                          # the optimality gap
    ],
}

CAPTION_ANTI_CONTRACTS: dict = {
    "Fig5_LP_OPF_Bound": [
        "lower bound on demand charge",   # log-log scatter x-axis
        "1:1 (zero gap)",                 # log-log scatter legend
    ],
}


def _check_caption_contract(base_name: str) -> None:
    """Read the emitted PDF back and hold it to its caption's claims."""
    must = CAPTION_CONTRACTS.get(base_name)
    if not must:
        return
    try:
        from pypdf import PdfReader
    except ImportError:
        print(f"    NOTE: pypdf not installed; caption contract for {base_name} unchecked.")
        return
    try:
        drawn = PdfReader(f"{base_name}.pdf").pages[0].extract_text() or ""
    except Exception as e:
        print(f"    NOTE: could not read back {base_name}.pdf ({e}); contract unchecked.")
        return
    missing = [c for c in must if c not in drawn]
    forbidden = [c for c in CAPTION_ANTI_CONTRACTS.get(base_name, []) if c in drawn]
    if missing or forbidden:
        raise SystemExit(
            f"\nCAPTION CONTRACT VIOLATED for {base_name}.\n"
            + (f"  the caption claims these, but they are NOT drawn: {missing}\n" if missing else "")
            + (f"  the figure draws these, which the caption contradicts: {forbidden}\n" if forbidden else "")
            + "  Either the figure changed and the caption must follow, or the wrong\n"
              "  figure is being written to this base name. Do not ship the mismatch.\n")
    print(f"    caption contract OK ({len(must)}/{len(must)} claims drawn)")


def _despine(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

def _save_multi_format(fig: plt.Figure, base_name: str) -> None:
    """Helper to save the figure in both PNG and PDF formats."""
    _claim_output(base_name)
    fig.tight_layout()
    fig.savefig(f"{base_name}.png")
    fig.savefig(f"{base_name}.pdf")
    plt.close(fig)
    print(f"  wrote {base_name}.png and {base_name}.pdf")
    _check_caption_contract(base_name)


MANUSCRIPT_N_REFERENCES: int = 107  # verified: 107 unique \bibitem entries

VERIFIED_YEAR_TOTALS: dict = {1935: 1, 1945: 1, 1970: 1, 1972: 1, 1981: 1, 1984: 1, 1992: 1, 1993: 2, 1996: 1, 2001: 2, 2004: 2, 2005: 2, 2006: 1, 2008: 3, 2009: 1, 2010: 3, 2011: 3, 2013: 2, 2014: 1, 2015: 4, 2016: 3, 2017: 6, 2018: 7, 2019: 3, 2020: 8, 2021: 7, 2022: 9, 2023: 9, 2024: 10, 2025: 9, 2026: 2}

THEMATIC_CLUSTERS = [
    ("Microgrid foundations\n& EMS reviews",                10, WONG["skyblue"]),
    ("Synchronization &\nre-synchronization",               15, WONG["green"]),
    ("Synchronization\nrebound effect (SRE)",               10, WONG["orange"]),
    ("Cyber-physical\nresilience",                          9,  WONG["vermillion"]),
    ("Distributed coordination\n(ADMM, MARL, transactive)", 10, WONG["skyblue"]),
    ("Co-simulation &\nhardware-in-the-loop",               4,  WONG["green"]),
    ("Mathematical\nfoundations",                           23, WONG["purple"]),
    ("Tools, datasets,\ntariffs, standards",                26, WONG["yellow"]),
]

YEAR_COUNTS = {
    "Mathematical foundations & tools": {1935: 1, 1945: 1, 1970: 1, 1972: 1,
        1981: 1, 1984: 1, 1992: 1, 1993: 2, 1996: 1, 2001: 2, 2004: 2, 2005: 2,
        2006: 1, 2008: 3, 2009: 1, 2010: 3, 2011: 3, 2013: 1, 2014: 1, 2015: 3,
        2017: 2, 2018: 2, 2019: 1, 2020: 3, 2022: 2, 2023: 1, 2025: 2, 2026: 2},
    "Control / coordination / resilience": {2013: 1, 2015: 1, 2016: 2, 2017: 4,
        2018: 3, 2019: 1, 2020: 4, 2021: 6, 2022: 5, 2023: 6, 2024: 8, 2025: 7,
        2026: 2},
    "SRE & rebound": {2016: 1, 2018: 1, 2019: 1, 2020: 1, 2021: 1, 2022: 2,
        2023: 1, 2024: 2},
}



REFERENCE_THEME_CODES = "RRRTTTTRRDDDDDGCCCCCCCCCRDMMTTMMMMMMMMRMYYYYYYDDGGGGGGGGYYYYYYYYYRRRRDHHHHMMMTTTTTTTTGTRGTTTTGGGMMMMMMMTTTT"
REFERENCE_YEARS = [2024, 2020, 2023, 2026, 2011, 2023, 2022, 2024, 2022, 2013, 2016, 2019, 2024, 2022, 2025, 2018, 2021, 2021, 2018, 2023, 2024, 2021, 2024, 2025, 2022, 2024, 1935, 2010, 2001, 2018, 1984, 1996, 1993, 2006, 2010, 2001, 2014, 2009, 2015, 2010, 2020, 2023, 2024, 2021, 2022, 2024, 2023, 2025, 2017, 2023, 2023, 2022, 2016, 2023, 2021, 2022, 2024, 2020, 2017, 2022, 2024, 2025, 2020, 2017, 2017, 2018, 2021, 2016, 2019, 2021, 2015, 2025, 2025, 2025, 2008, 2013, 1993, 2008, 2020, 2004, 2020, 2020, 2019, 2020, 1970, 2015, 2022, 2011, 2005, 2008, 2011, 2018, 2026, 2017, 2018, 2023, 1992, 2004, 2015, 1972, 1981, 2005, 1945, 2018, 2017, 2025, 2025]
_CLUSTER_INDEX = {"G": 0, "Y": 1, "R": 2, "C": 3, "D": 4, "H": 5, "M": 6, "T": 7}
_STRATUM_OF = {"M": "Mathematical foundations & tools", "T": "Mathematical foundations & tools", "R": "SRE & rebound"}
assert len(REFERENCE_THEME_CODES) == len(REFERENCE_YEARS) == MANUSCRIPT_N_REFERENCES
assert Counter(REFERENCE_YEARS) == Counter(VERIFIED_YEAR_TOTALS), "per-reference years must reproduce the verified totals"
THEMATIC_CLUSTERS = [(name, sum(_CLUSTER_INDEX[c] == i for c in REFERENCE_THEME_CODES), col)
                     for i, (name, _n, col) in enumerate(THEMATIC_CLUSTERS)]
_yc = {k: {} for k in YEAR_COUNTS}
for _c, _y in zip(REFERENCE_THEME_CODES, REFERENCE_YEARS):
    _s = _STRATUM_OF.get(_c, "Control / coordination / resilience")
    _yc[_s][_y] = _yc[_s].get(_y, 0) + 1
YEAR_COUNTS = {k: dict(sorted(v.items())) for k, v in _yc.items()}

def derive_year_totals_from_tex(tex_path: Optional[Path] = None) -> dict:
 
    if tex_path is None:
        for _cand in (Path("main_reconciled_FINAL.tex"),
                      Path("main_reconciled__3___1_.tex"),
                      Path("main_reconciled.tex")):
            if _cand.exists():
                tex_path = _cand
                break
        else:
            tex_path = Path("main_reconciled.tex")
    if not Path(tex_path).exists():
        return {}
    src = Path(tex_path).read_text(encoding="utf-8", errors="replace")
    totals: dict = {}
    for _, body in re.findall(r"\\bibitem\{(\d+)\}(.*)", src):
        body = re.sub(r"\\url\{[^}]*\}|\(accessed[^)]*\)", " ", body)  # v14.8: ignore URLs, DOIs and access dates
        years = re.findall(r"\b(19[3-9]\d|20[0-2]\d)\b", body)
        if years:
            y = max(int(v) for v in years)
            totals[y] = totals.get(y, 0) + 1
    return dict(sorted(totals.items()))



def reference_first_citation_report(tex_path: Optional[Path] = None) -> None:

    import csv
    tex_path = tex_path or next((p for p in (Path("main_reconciled_FINAL.tex"),) if p.exists()), None)
    if tex_path is None:
        return
    s = tex_path.read_text(encoding="utf-8")
    bib = s[s.index(r"\begin{thebibliography}"):]
    body = s[:s.index(r"\begin{thebibliography}")]
    rows = []
    for item in re.split(r"\\bibitem\{", bib)[1:]:
        key, text = item.split("}", 1)
        yr = re.search(r"\b(19\d{2}|20[0-2]\d)\b", text)
        m = re.search(r"\\cite\{[^}]*\b" + re.escape(key) + r"\b[^}]*\}", body)
        sec = re.findall(r"\\(?:sub)?section\*?\{([^}]*)\}", body[:m.start()]) if m else []
        rows.append([key, yr.group(1) if yr else "", sec[-1] if sec else "", " ".join(text.split())[:90]])
    with open("reference_first_citation.csv", "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([["ref", "year", "first_cited_in", "entry"]] + rows)
    print(f"  wrote reference_first_citation.csv ({len(rows)} references)")

def reference_count_report() -> None:
  
    fig1_total = sum(count for _, count, _ in THEMATIC_CLUSTERS)
    fig2_total = sum(sum(by_year.values()) for by_year in YEAR_COUNTS.values())
    verified_total = sum(VERIFIED_YEAR_TOTALS.values())
    print(f"[reference counts] manuscript = {MANUSCRIPT_N_REFERENCES} references; "
          f"VERIFIED_YEAR_TOTALS = {verified_total}; "
          f"Fig1 (by theme) = {fig1_total}; Fig2 (by year) = {fig2_total}.")

    derived = derive_year_totals_from_tex()
    if derived:
        n = sum(derived.values())
        if n != MANUSCRIPT_N_REFERENCES:
            print(f"  WARNING: the .tex bibliography holds {n} references, not "
                  f"{MANUSCRIPT_N_REFERENCES}; update MANUSCRIPT_N_REFERENCES.")
        if derived != VERIFIED_YEAR_TOTALS:
            diff = {y: (VERIFIED_YEAR_TOTALS.get(y, 0), derived.get(y, 0))
                    for y in set(derived) | set(VERIFIED_YEAR_TOTALS)
                    if derived.get(y, 0) != VERIFIED_YEAR_TOTALS.get(y, 0)}
            print(f"  WARNING: VERIFIED_YEAR_TOTALS disagrees with the .tex "
                  f"(year: snapshot vs actual) -> {diff}")
        else:
            print("  OK: VERIFIED_YEAR_TOTALS matches the .tex bibliography.")
    else:
        print("  NOTE: main_reconciled.tex not found; using the stored snapshot.")

    if verified_total != MANUSCRIPT_N_REFERENCES:
        print(f"  WARNING: VERIFIED_YEAR_TOTALS sums to {verified_total}, not "
              f"{MANUSCRIPT_N_REFERENCES}.")
    for name, total in (("Fig1 THEMATIC_CLUSTERS", fig1_total),
                        ("Fig2 YEAR_COUNTS", fig2_total)):
        if total != MANUSCRIPT_N_REFERENCES:
            print(f"  WARNING: {name} sums to {total}, not "
                  f"{MANUSCRIPT_N_REFERENCES}; redistribute it to match.")

    columns: dict = {}
    for by_year in YEAR_COUNTS.values():
        for year, count in by_year.items():
            columns[year] = columns.get(year, 0) + count
    bad = {y: (columns.get(y, 0), VERIFIED_YEAR_TOTALS.get(y, 0))
           for y in set(columns) | set(VERIFIED_YEAR_TOTALS)
           if columns.get(y, 0) != VERIFIED_YEAR_TOTALS.get(y, 0)}
    if bad:
        print(f"  WARNING: Fig2 per-year stacks disagree with the verified "
              f"totals (year: stacked vs verified) -> {bad}")
    else:
        print("  OK: Fig2 per-year stacks match the verified per-year totals.")
    _cl = {n.replace("\n", " "): c for n, c, _ in THEMATIC_CLUSTERS}
    _pool = {"Mathematical foundations & tools": _cl["Mathematical foundations"] + _cl["Tools, datasets, tariffs, standards"],
             "SRE & rebound": _cl["Synchronization rebound effect (SRE)"]}
    _pool["Control / coordination / resilience"] = sum(_cl.values()) - sum(_pool.values())
    _diff = {k: (sum(YEAR_COUNTS[k].values()), v) for k, v in _pool.items() if sum(YEAR_COUNTS[k].values()) != v}
    if not _diff:
        print("  OK: Fig2 strata equal the pooled Fig1 clusters (one per-reference classification).")
    if _diff:
        print("  NOTE: Fig2 strata differ from pooled Fig1 clusters (stratum, Fig1): " + str(_diff)
              + " -- UNEXPECTED: both figures derive from REFERENCE_THEME_CODES.")


# FIG 1 / FIG_INTRO 
def _place_legend_clear_lines(ax, prefer: str = "upper right", **kw):

    from matplotlib.lines import Line2D
    from matplotlib.collections import PathCollection, PolyCollection
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = []

    def _dense(xy):
        if len(xy) < 2:
            return xy
        seg = [np.linspace(xy[k], xy[k + 1], 24, endpoint=False) for k in range(len(xy) - 1)]
        return np.vstack(seg + [xy[-1:]])

    for art in ax.get_children():
        if not art.get_visible():
            continue
        if isinstance(art, Line2D) and art.get_transform() == ax.transData:
            xy = np.column_stack(art.get_data()).astype(float)
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy):
                pts.append(_dense(ax.transData.transform(xy)))
        elif isinstance(art, PathCollection):
            off = art.get_offsets()
            if len(off):
                pts.append(art.get_offset_transform().transform(off))
        elif isinstance(art, PolyCollection):
            for path in art.get_paths():
                v = path.vertices[np.isfinite(path.vertices).all(axis=1)]
                if len(v):
                    pts.append(_dense(ax.transData.transform(v)))
    P = np.vstack(pts) if pts else np.empty((0, 2))
    axbb = ax.get_window_extent(rend)
    pad = 3.0

    def _ok(leg):
        fig.canvas.draw()
        bb = leg.get_window_extent(rend)
        inside = (bb.x0 >= axbb.x0 and bb.x1 <= axbb.x1 and bb.y0 >= axbb.y0 and bb.y1 <= axbb.y1)
        clear = not len(P) or not ((P[:, 0] >= bb.x0 - pad) & (P[:, 0] <= bb.x1 + pad) &
                                   (P[:, 1] >= bb.y0 - pad) & (P[:, 1] <= bb.y1 + pad)).any()
        return inside and clear

    order = [prefer] + [c for c in ("upper right", "upper left", "lower left", "lower right",
                                    "center right", "center left", "upper center", "lower center")
                        if c != prefer]
    for loc in order:
        leg = ax.legend(loc=loc, **kw)
        if _ok(leg):
            return leg
    for fy in (0.88, 0.78, 0.68, 0.58, 0.48, 0.38, 0.28, 0.18):
        for fx in (0.75, 0.65, 0.55, 0.35, 0.25):
            leg = ax.legend(loc="center", bbox_to_anchor=(fx, fy), **kw)
            if _ok(leg):
                return leg
    return ax.legend(loc=prefer, **kw)         

def fig_intro_sre_mechanism(base_name: str = "Fig_Intro_SRE_Mechanism") -> None:
    t = np.linspace(0, 24, 24 * 12 + 1)

    # Baseline residential demand: morning + evening bumps
    def bump(c, w, a):
        return a * np.exp(-0.5 * ((t - c) / w) ** 2)
    baseline = 60 + bump(8, 1.6, 28) + bump(19, 2.2, 78)

    # Uncoordinated EV charging: a synchronized block at the 21:00 edge
    ev_energy = 1500.0  # arbitrary illustrative kWh-equivalent area
    unco = baseline.copy()
    block = (t >= 21) & (t < 23.5)
    unco[block] += 150.0  # tall synchronized rebound

    # vdC-staggered: same area spread across the off-peak window 21:00->07:00.
    stag = baseline.copy()
    win = ((t >= 21) | (t < 7))
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    added_unco = _trapz(np.where(block, 150.0, 0.0), t)
    spread_height = added_unco / ((24 - 21) + 7)  # over a 10h window
    stag[win] += spread_height

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(t, 0, baseline, color=WONG["skyblue"], alpha=0.35,
                    label="Baseline residential demand", zorder=1)
    ax.plot(t, baseline, color=WONG["skyblue"], lw=1.4, zorder=2)
    ax.plot(t, unco, color=WONG["orange"], lw=1.8,
            label="Uncoordinated EV charging (21:00 price-drop edge)", zorder=3)
    ax.plot(t, stag, color=WONG["green"], lw=1.8, ls="--",
            label="van der Corput staggered EV charging", zorder=4)

    ax.axvline(21, color=WONG["black"], lw=0.8, ls=":", alpha=0.6)
    ax.annotate("21:00 price-drop edge\n(largest single TOU step)",
                xy=(21, 165), xytext=(15.5, 95),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color=WONG["black"], lw=0.8))
    ax.annotate("synchronized\nrebound peak",
                xy=(21.2, 258), xytext=(23.3, 210), fontsize=8, ha="center",
                color=WONG["vermillion"],
                arrowprops=dict(arrowstyle="->", color=WONG["vermillion"], lw=0.9))

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 290)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Community demand (illustrative, kW)")
    axes_title(ax, "Rebound and staggered dispatch")
    ax.legend(loc="upper left", framealpha=0.9)
    _despine(ax)
    _save_multi_format(fig, base_name)


# FIG 1 (landscape) 
def fig_thematic_landscape(base_name: str = "Fig1_Thematic_Landscape") -> None:
    clusters = THEMATIC_CLUSTERS   # single source; see reference_count_report()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Central node
    cx, cy = 5, 5
    cbox = FancyBboxPatch((cx - 1.45, cy - 0.7), 2.9, 1.4,
                          boxstyle="round,pad=0.08", linewidth=1.6,
                          edgecolor=WONG["black"], facecolor=WONG["orange"],
                          alpha=0.9, zorder=5)
    ax.add_patch(cbox)
    ax.text(cx, cy, "Mitigating the\nSynchronization\nRebound Effect",
            ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=6)

    # Eight surrounding nodes
    n = len(clusters)
    R = 3.7
    for i, (label, count, col) in enumerate(clusters):
        ang = np.pi / 2 - i * 2 * np.pi / n
        x = cx + R * np.cos(ang)
        y = cy + R * np.sin(ang) * 0.95
        w, h = 2.1, 0.95
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                             boxstyle="round,pad=0.06", linewidth=1.1,
                             edgecolor=WONG["black"],
                             facecolor=tuple(0.55 * c + 0.45 for c in mcolors.to_rgb(col)),
                             alpha=1.0, zorder=4)
        ax.add_patch(box)
        ax.text(x, y + 0.12, label, ha="center", va="center", fontsize=7.4,
                zorder=5)
        ax.text(x, y - 0.30, f"$n = {count}$", ha="center", va="center",
                fontsize=7.6, style="italic", zorder=5)
        arrow = FancyArrowPatch((x, y), (cx, cy),
                                connectionstyle="arc3,rad=0.0",
                                arrowstyle="-|>", mutation_scale=11,
                                linewidth=0.9, color=WONG["black"], alpha=0.5,
                                shrinkA=18, shrinkB=40, zorder=2)
        ax.add_patch(arrow)

    axes_title(ax, "Thematic landscape", fontsize=11,
                 pad=2)
    _save_multi_format(fig, base_name)


# FIG 2 : distribution of surveyed references by year and theme (stacked)
def fig_year_distribution(base_name: str = "Fig2_Year_Distribution") -> None:
    years = list(range(1935, 2027))
    strata = {name: np.zeros(len(years)) for name in YEAR_COUNTS}
    for name, by_year in YEAR_COUNTS.items():
        for y, k in by_year.items():
            strata[name][years.index(y)] += k
    math_found = strata["Mathematical foundations & tools"]
    control = strata["Control / coordination / resilience"]
    sre = strata["SRE & rebound"]

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.bar(years, math_found, color=WONG["purple"], label="Mathematical foundations & tools",
           width=0.8, zorder=3)
    ax.bar(years, control, bottom=math_found, color=WONG["blue"],
           label="Control / coordination / resilience", width=0.8, zorder=3)
    ax.bar(years, sre, bottom=math_found + control, color=WONG["orange"],
           label="SRE & rebound", width=0.8, zorder=3)

    ax.axvline(2019.5, color=WONG["black"], lw=0.8, ls="--", alpha=0.6)
    ax.text(2019.2, ax.get_ylim()[1] * 0.92, r"over half from 2020 onward $\rightarrow$",
            fontsize=7.6, va="top", ha="right")
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Number of references")
    axes_title(ax, "References by year and theme")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_xlim(1933, 2028)
    _despine(ax)
    _save_multi_format(fig, base_name)


# FEEDER TOPOLOGY : the Section 1.2 problem statement
def fig_feeder_topology(base_name: str = "Fig_Feeder_Topology") -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.6, bold=False, alpha=0.55,
            ls="solid", ec=None, tc=None):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                           linewidth=1.1, edgecolor=(ec or WONG["black"]),
                           facecolor=fc, alpha=alpha, linestyle=ls, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=("bold" if bold else "normal"),
                color=(tc or "black"), zorder=4)
        return (x + w / 2, y + h / 2)

    BUS_Y = 5.55

    # ---------------------------------------------------------------- power side
    box(4.30, 7.05, 3.40, 0.68, "Substation / grid", WONG["skyblue"], fs=8)
    box(4.30, 5.92, 3.40, 0.78,
        "Community billing meter\n(point of common coupling)",
        WONG["vermillion"], fs=7.4, bold=True)
    ax.add_patch(FancyArrowPatch((6.00, 6.70), (6.00, 7.05),
                 arrowstyle="<|-|>", mutation_scale=10, lw=1.1,
                 color=WONG["black"], alpha=0.65, zorder=5))
    ax.text(8.00, 6.92,
            r"$P_{\mathrm{agg}}(t)=\sum_i\!\left[\ell_i+u_i^{\mathrm{ev}}"
            r"+u_i^{\mathrm{batt}}-p_i^{\mathrm{pv}}\right]$",
            ha="left", va="center", fontsize=7.6)

    # The absent central dispatcher, drawn struck through.
    dx, dy, dw, dh = 0.40, 5.92, 3.20, 0.78
    box(dx, dy, dw, dh, "Central dispatcher", "#FFFFFF", fs=7.6,
        alpha=0.9, ls=(0, (4, 2)), ec="#9A9A9A", tc="#8A8A8A")
    ax.plot([dx + 0.18, dx + dw - 0.18], [dy + 0.15, dy + dh - 0.15],
            color=WONG["vermillion"], lw=1.6, zorder=6)
    ax.plot([dx + 0.18, dx + dw - 0.18], [dy + dh - 0.15, dy + 0.15],
            color=WONG["vermillion"], lw=1.6, zorder=6)
    ax.text(dx + dw / 2, dy + dh + 0.20, "no central dispatcher",
            ha="center", va="center", fontsize=7.2, style="italic",
            color=WONG["vermillion"])

    # Low-voltage feeder bus.
    ax.plot([0.45, 11.55], [BUS_Y, BUS_Y], color=WONG["black"], lw=2.4,
            solid_capstyle="butt", zorder=2)
    ax.add_patch(FancyArrowPatch((6.00, 5.92), (6.00, BUS_Y + 0.02),
                 arrowstyle="-|>", mutation_scale=10, lw=1.1,
                 color=WONG["black"], alpha=0.75, zorder=5))
    ax.text(11.55, BUS_Y + 0.16, "common low-voltage feeder",
            ha="right", va="bottom", fontsize=7.4, style="italic")

    hx = [0.45, 2.42, 4.39, 6.36, 8.33]
    hw, hy, hh = 1.80, 2.15, 2.95
    top = hy + hh
    labels = ["House 1", "House 2", "House 3", "House 4", r"House $N$"]
    has_ev = [True, True, False, True, True]
    ems_mid = hy + hh - 0.86 + 0.25

    for k, (x, lab) in enumerate(zip(hx, labels)):
        ax.add_patch(FancyBboxPatch((x, hy), hw, hh, boxstyle="round,pad=0.05",
                                    linewidth=1.2, edgecolor=WONG["black"],
                                    facecolor="none", zorder=3))
        ax.text(x + hw / 2, top - 0.19, lab, ha="center", va="center",
                fontsize=7.6, fontweight="bold")
        ax.add_patch(FancyArrowPatch((x + hw / 2, BUS_Y), (x + hw / 2, top),
                     arrowstyle="<|-|>", mutation_scale=8, lw=0.9,
                     color=WONG["black"], alpha=0.6, zorder=4))
        ax.text(x + hw / 2 + 0.09, (BUS_Y + top) / 2,
                (rf"$P_{{{k+1}}}$" if k < 4 else r"$P_N$"),
                ha="left", va="center", fontsize=6.8)
        box(x + 0.14, top - 0.86, hw - 0.28, 0.50,
            "Local EMS", WONG["orange"], fs=7.0, bold=True)
        comps = [(r"$\ell_i$  load", WONG["skyblue"]),
                 ("PV", WONG["yellow"]),
                 ("Battery", WONG["green"])]
        comps.append(("EV charger", WONG["green"]) if has_ev[k]
                     else ("no EV", "#FFFFFF"))
        for j, (t, fc) in enumerate(comps):
            yy = top - 1.46 - j * 0.42
            box(x + 0.14, yy, hw - 0.28, 0.34, t, fc, fs=6.6,
                alpha=(0.30 if t == "no EV" else 0.55),
                ls=((0, (3, 2)) if t == "no EV" else "solid"),
                tc=("#8A8A8A" if t == "no EV" else None))

    # continuation to N houses, clear of the house row
    ax.text(10.72, hy + hh / 2 + 0.25, r"$\cdots$", ha="center",
            va="center", fontsize=15)
    ax.text(10.25, hy + hh / 2 - 0.35, r"$N=50$ houses", ha="left",
            va="center", fontsize=7.4, style="italic")
    ax.text(10.25, hy + hh / 2 - 0.80, r"EV penetration $\eta$", ha="left",
            va="center", fontsize=7.0, style="italic", color="#6A6A6A")

    for x0, x1 in zip(hx[:-1], hx[1:]):
        a, b = x0 + hw, x1
        ax.plot([a, b], [ems_mid, ems_mid], color="#9A9A9A", lw=1.0,
                linestyle=(0, (3, 2)), zorder=2)
        ax.plot([(a + b) / 2], [ems_mid], marker="x", markersize=6.5,
                markeredgewidth=1.7, color=WONG["vermillion"], zorder=6)
    # Keyed off to the right of the house row rather than led with a pointer,
    # which would have to cross House N to reach the nearest cross.
    ax.plot([10.32], [ems_mid], marker="x", markersize=6.5,
            markeredgewidth=1.7, color=WONG["vermillion"], zorder=6)
    ax.text(10.50, ems_mid, "no inter-house\ncommunication",
            ha="left", va="center", fontsize=7.2, style="italic",
            color=WONG["vermillion"])

    RAIL_Y = 1.55
    ax.plot([0.45, 11.55], [RAIL_Y, RAIL_Y], color=WONG["blue"], lw=1.4,
            linestyle=(0, (5, 2)), zorder=2)
    box(0.45, 0.58, 3.60, 0.64,
        r"TOU price $\pi(t)$, carbon $\psi(t)$", WONG["blue"], fs=7.4)
    ax.add_patch(FancyArrowPatch((2.25, 1.22), (2.25, RAIL_Y - 0.02),
                 arrowstyle="-|>", mutation_scale=8, lw=1.0,
                 color=WONG["blue"], alpha=0.85, zorder=5))
    for x in hx:
        ax.add_patch(FancyArrowPatch((x + hw / 2, RAIL_Y), (x + hw / 2, hy),
                     arrowstyle="-|>", mutation_scale=8, lw=1.0,
                     color=WONG["blue"], alpha=0.85,
                     linestyle=(0, (4, 2)), zorder=4))
    ax.text(11.55, 1.08,
            "one-way broadcast: an identical signal reaches every house",
            ha="right", va="center", fontsize=7.2, style="italic",
            color=WONG["blue"])

    axes_title(ax, "Community topology and the two absent coordination channels",
               fontsize=10, pad=2)
    _save_multi_format(fig, base_name)


# FIG 3 : system architecture block diagram
def fig_system_architecture(base_name: str = "Fig3_System_Architecture") -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, text, fc, fs=7.8, bold=False):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06",
                           linewidth=1.1, edgecolor=WONG["black"],
                           facecolor=fc, alpha=0.55, zorder=3)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight=("bold" if bold else "normal"),
                zorder=4)
        return (x + w / 2, y + h / 2)

    # House boundary
    house = FancyBboxPatch((0.3, 0.5), 6.6, 7.0, boxstyle="round,pad=0.05",
                           linewidth=1.3, edgecolor=WONG["black"],
                           facecolor="none", linestyle="--", zorder=2)
    ax.add_patch(house)
    ax.text(3.6, 7.2, r"House $i$ (one of $N=50$, no inter-house messages)",
            ha="center", fontsize=8, style="italic")

    # Four components
    c1 = box(0.6, 5.6, 2.3, 0.9, "Non-deferrable load\n(HVAC+light+equip)", WONG["skyblue"])
    c2 = box(0.6, 4.3, 2.3, 0.9, "Rooftop PV", WONG["yellow"])
    c3 = box(0.6, 3.0, 2.3, 0.9, "Battery\n13.5 kWh / 5 kW", WONG["green"])
    c4 = box(0.6, 1.7, 2.3, 0.9, "EV charger\n7 kW / 50 kWh", WONG["green"])

    # Scheduler
    sched = box(3.5, 3.3, 2.8, 1.6, "Local scheduler\n(vdC-Stagger)\nemits $u^{ev}_i$, $u^{batt}_i$",
                WONG["orange"], fs=8, bold=True)

    # Exogenous signals
    box(3.5, 5.7, 2.8, 0.8, r"TOU price $\pi(t)$, CO$_2$ $\psi(t)$", WONG["skyblue"], fs=7.4)
    box(3.5, 1.5, 2.8, 0.7, r"Occupancy $m_i(t)$", WONG["skyblue"], fs=7.4)

    for (cx, cy), sty in zip((c1, c2, c3, c4), ("-|>", "-|>", "<|-|>", "<|-|>")):
        ax.add_patch(FancyArrowPatch((cx + 1.15, cy), (3.5, 4.1),
                     arrowstyle=sty, mutation_scale=9, lw=0.8,
                     color=WONG["black"], alpha=0.5, shrinkA=2, shrinkB=4))

    ax.add_patch(FancyArrowPatch((4.9, 5.7), (4.9, 4.9), arrowstyle="-|>",
                 mutation_scale=9, lw=0.8, color=WONG["black"], alpha=0.5,
                 shrinkA=2, shrinkB=3))
    ax.add_patch(FancyArrowPatch((4.9, 2.2), (4.9, 3.3), arrowstyle="-|>",
                 mutation_scale=9, lw=0.8, color=WONG["black"], alpha=0.5,
                 shrinkA=2, shrinkB=3))

    # Community meter
    meter = box(8.0, 3.3, 3.2, 1.6, "Community\nbilling meter\n(E-TOU-C, B-19\ndemand, NBT)",
                WONG["vermillion"], fs=7.8, bold=True)
    # Grid
    box(8.0, 5.7, 3.2, 0.9, "Substation / grid", WONG["skyblue"])
    # Per-house bill
    box(8.0, 1.5, 3.2, 0.9, r"Per-house bill $J_i$" + "\nTOU+demand$-$NBT", WONG["purple"])

    # Net power arrow from scheduler to meter
    ax.add_patch(FancyArrowPatch((6.95, 4.1), (8.0, 4.1), arrowstyle="-|>",  
                 mutation_scale=12, lw=1.4, color=WONG["orange"]))
    ax.text(7.45, 4.34, r"$P_i(t)$", ha="center", va="bottom", fontsize=8,
            zorder=6, bbox=dict(boxstyle="square,pad=0.12", facecolor="white",
                                edgecolor="none"))
    ax.text(7.45, 3.88, "net power", ha="center", va="top",
            fontsize=6.4, color="#5A5A5A", zorder=6)

    ax.add_patch(FancyArrowPatch((9.6, 4.9), (9.6, 5.7), arrowstyle="<|-|>",
                 mutation_scale=10, lw=1.0, color=WONG["black"], alpha=0.6))
    ax.text(9.78, 5.38, r"$P_{\mathrm{agg}}(t)=\sum_i P_i(t)$", ha="left",
            va="center", fontsize=7.4)
    ax.text(9.78, 5.08, "community aggregate", ha="left", va="center",
            fontsize=6.4, color="#5A5A5A")
    # Meter -> bill
    ax.add_patch(FancyArrowPatch((9.6, 3.3), (9.6, 2.4), arrowstyle="-|>",
                 mutation_scale=10, lw=1.0, color=WONG["black"], alpha=0.6))

    axes_title(ax, "System architecture",
                 fontsize=10, pad=2)
    _save_multi_format(fig, base_name)


# FIG 4 : experimental design (factorial), 3 panels
def fig_experimental_design(base_name: str = "Fig4_Experimental_Design") -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.0),
                             gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})

    # Panel (a): 36 factor cells as a 4 (eta) x 9 (xi*rho) grid
    axa = axes[0]
    etas = [0.25, 0.50, 0.75, 1.00]
    cmap = [WONG["green"], WONG["yellow"], WONG["orange"], WONG["vermillion"]]
    for r, eta in enumerate(etas):
        for c in range(9):
            axa.add_patch(Rectangle((c, r), 0.92, 0.92, facecolor=cmap[r],
                          alpha=0.55, edgecolor=WONG["black"], lw=0.4))
    axa.set_xlim(0, 9)
    axa.set_ylim(0, 4)
    axa.set_yticks([i + 0.5 for i in range(4)])
    axa.set_yticklabels([f"$\\eta$={e:g}" for e in etas], fontsize=7)
    axa.set_xticks([1.5, 4.5, 7.5])
    axa.set_xticklabels([r"$\xi$=0.1", r"$\xi$=0.3", r"$\xi$=0.5"], fontsize=7)
    axa.set_title("(a) 36 factor cells\n" + r"$(\eta, \xi, \rho)$", fontsize=8.5)
    axa.text(4.5, -1.05, r"3 $\rho$ levels nested per $\xi$ block",
             ha="center", fontsize=6.8, style="italic")
    for s in ("top", "right", "left", "bottom"):
        axa.spines[s].set_visible(False)
    axa.tick_params(length=0)

    # Panel (b): 5 scenarios -> 180 rows
    axb = axes[1]
    scen = ["S0 Uncontrolled", "S1 vdC-Stagger", "S2 TOU rebound",
            "S3 Flat tariff", "S4 Valley-fill"]
    scol = [WONG["skyblue"], WONG["green"], WONG["orange"], WONG["blue"],
            WONG["purple"]]
    for i, (s, c) in enumerate(zip(scen, scol)):
        axb.add_patch(FancyBboxPatch((0.1, 4 - i * 0.9 + 0.05), 3.6, 0.7,
                      boxstyle="round,pad=0.03", facecolor=c, alpha=0.55,
                      edgecolor=WONG["black"], lw=0.7))
        axb.text(1.9, 4 - i * 0.9 + 0.4, s, ha="center", va="center", fontsize=7.4)
    axb.set_xlim(0, 3.8)
    axb.set_ylim(-0.5, 5)
    axb.axis("off")
    axb.set_title("(b) 5 scenarios\n" + r"$\rightarrow$ 180 rows", fontsize=8.5)

    # Panel (c): 50 houses per row -> 9000 runs
    axc = axes[2]
    for i in range(50):
        r, c = divmod(i, 10)
        axc.add_patch(Rectangle((c * 0.9, r * 0.9), 0.7, 0.7,
                      facecolor=WONG["blue"], alpha=0.5, edgecolor=WONG["black"],
                      lw=0.3))
    axc.set_xlim(-0.3, 9.3)
    axc.set_ylim(-0.3, 4.8)
    axc.axis("off")
    axc.set_title("(c) 50 houses/row\n" + r"$\rightarrow$ 9,000 runs", fontsize=8.5)
    axc.invert_yaxis()

    figure_title(fig, r"Experimental design: $36 \times 5 \times 50$",
                 fontsize=10, y=1.04)
    _save_multi_format(fig, base_name)


# FIG 5 : LP-OPF single-day exhibit 
def fig_lp_opf_bound(base_name: str = "Fig5_LP_OPF_Bound") -> None:

    h = np.arange(24)
    baseline = np.array([55,50,46,43,40,42,48,62,70,60,40,25,
                         18,20,28,40,58,78,110,118,120,95,80,68], float)
    ev_add = {21: 40, 22: 36, 23: 30, 0: 24, 1: 18, 2: 12, 3: 8}   # staggered after 21:00
    actual = baseline.copy()
    for hr, add in ev_add.items():
        actual[hr] += add
    ev_energy = float(sum(ev_add.values()))                        # kWh (1-h slots)
    window = list(range(21, 24)) + list(range(0, 7))               # off-peak availability
    b = baseline[window]
    lo, hi = float(b.min()), float(b.max()) + ev_energy
    for _ in range(80):                                            # bisection on water level
        lvl = 0.5 * (lo + hi)
        if np.maximum(0.0, lvl - b).sum() > ev_energy:
            hi = lvl
        else:
            lo = lvl
    opf = baseline.copy()
    opf[window] = np.maximum(b, lo)
    assert abs((opf - baseline).sum() - ev_energy) < 1e-6, "bound must place the same EV energy"
    pk_a, pk_o = int(np.argmax(actual)), int(np.argmax(opf))
    assert actual[pk_a] > opf[pk_o] + 1.0, "the exhibit must show a positive peak gap"
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.fill_between(h, 0, baseline, color=WONG["skyblue"], alpha=0.35,
                    label=r"Community baseline ($\ell_i - p_i^{pv}$)", step="mid")
    ax.step(h, actual, where="mid", color=WONG["orange"], lw=1.8,
            label="Staggered EV charging (illustrative)")
    ax.step(h, opf, where="mid", color=WONG["green"], lw=1.8, ls="--",
            label="Peak-minimizing placement of the same energy (bound)")
    ax.axhline(opf[pk_o], color=WONG["green"], lw=0.8, ls=":", alpha=0.8)
    ax.annotate("", xy=(pk_a, actual[pk_a]), xytext=(pk_a, opf[pk_o]),
                arrowprops=dict(arrowstyle="<|-|>", color=WONG["vermillion"], lw=1.2))
    ax.text(pk_a - 0.45, (actual[pk_a] + opf[pk_o]) / 2, "optimality\ngap", fontsize=7.6,
            color=WONG["vermillion"], ha="right", va="center")
    ax.set_xlim(0, 23)
    ax.set_ylim(0, 160)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (illustrative day)")
    ax.set_ylabel("Community demand (kW)")
    axes_title(ax, "LP-OPF lower bound (illustrative)")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=7.6)
    _despine(ax)
    _save_multi_format(fig, base_name)


# FIG 6 : vdC-Stagger components 
def _van_der_corput(n: int, base: int = 2) -> float:
    q, denom = 0.0, 1.0
    while n > 0:
        denom *= base
        q += (n % base) / denom
        n //= base
    return q


def fig_algorithm_components(base_name: str = "Fig6_Algorithm1_Components") -> None:
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    # (a) vdC vs iid uniform spread on [0,1]
    N = 50
    vdc = np.array([_van_der_corput(i + 1) for i in range(N)])
    rng = np.random.default_rng(7)
    unif = rng.uniform(0, 1, N)
    axa.scatter(vdc, np.ones(N) * 1.0, marker="o", s=18, color=WONG["green"],
                label=r"van der Corput $\varphi_2(i)$")
    axa.scatter(unif, np.ones(N) * 0.5, marker="x", s=18, color=WONG["vermillion"],
                label="i.i.d. uniform random")
    axa.set_xlim(0, 1)

    for k in range(1, 48):
        axa.axvline(k / 48, color="0.86", lw=0.4, zorder=0)
    busy_vdc = int(np.bincount(np.minimum(np.floor(vdc * 48).astype(int), 47), minlength=48).max())
    busy_uni = int(np.bincount(np.minimum(np.floor(unif * 48).astype(int), 47), minlength=48).max())
    axa.text(0.99, 1.13, f"busiest slot: {busy_vdc}", ha="right", fontsize=6.6, color=WONG["green"])
    axa.text(0.99, 0.63, f"busiest slot: {busy_uni}", ha="right", fontsize=6.6, color=WONG["vermillion"])
    axa.set_ylim(0.2, 1.3)
    axa.set_yticks([])
    axa.set_xlabel("Stagger offset (normalised)")
    axa.set_title("(a) Low-discrepancy vs random spread\n" +
                  r"$D^*_N = \mathcal{O}(\log N/N)$ vs $\mathcal{O}(N^{-1/2})$",
                  fontsize=8.2)
    axa.legend(loc="lower center", fontsize=6.8, framealpha=0.9)
    _despine(axa)

    # (b) Layer-3 rate envelope
    ploc = np.linspace(0.75, 1.75, 400)
    rarb = np.where(ploc < 1.0, 1.0,
            np.where(ploc < 1.4, 1 - 0.7 * (ploc - 1.0) / 0.4, 0.30))
    axb.plot(ploc, rarb, color=WONG["blue"], lw=2.0)
    axb.axhline(0.10, color=WONG["vermillion"], lw=1.0, ls=":",
                label=r"DRM floor $r_{\min}=0.10$")
    axb.axvline(1.0, color=WONG["black"], lw=0.7, ls="--", alpha=0.5)
    axb.axvline(1.4, color=WONG["black"], lw=0.7, ls="--", alpha=0.5)
    axb.text(1.0, 1.09, r"$\pi_{LO}$", ha="center", fontsize=7.5)
    axb.text(1.4, 1.09, r"$\pi_{HI}$", ha="center", fontsize=7.5)
    axb.set_xlim(0.75, 1.75)
    axb.set_ylim(0, 1.18)
    axb.set_xlabel(r"Local shadow price $\pi_{loc}$")
    axb.set_ylabel(r"Rate fraction $r_{ARB}$")
    axb.set_title("(b) Shadow-price rate envelope", fontsize=8.2)
    axb.legend(loc="lower left", fontsize=6.8, framealpha=0.9)
    _despine(axb)

    _save_multi_format(fig, base_name)


# FIG 11 : tariff geometries across three jurisdictions
def fig_tariff_geometries(base_name: str = "Fig11_Tariff_Geometries") -> None:
    h = np.arange(25)

    # PG&E E-TOU-C: two-tier schedule, on-peak 16:00-21:00 
    pge = np.full(24, 0.399)          # off-peak (summer total)
    pge[16:21] = 0.522                 # on-peak 16:00-21:00 (hours 16-20)
    pge = np.append(pge, pge[-1])

    rng = np.random.default_rng(11)
    hh = np.arange(0, 24.5, 0.5)
    shape = (0.13 + 0.05 * np.exp(-((hh - 8.0) ** 2) / 3.0)
             - 0.02 * np.exp(-((hh - 13.0) ** 2) / 4.0)
             + 0.15 * ((hh >= 16) & (hh < 19)))
    octopus = np.clip(shape + rng.normal(0, 0.008, hh.size), 0.04, 0.40)
    assert 16 <= hh[int(np.argmax(octopus))] < 19, "stylised Agile must peak 16:00-19:00"

    aus = np.full(24, 0.08)
    aus[15:21] = 0.22
    aus = np.append(aus, aus[-1])

    flat = np.full(25, float(np.mean(pge[:24])))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.step(h, pge, where="post", color=WONG["blue"], lw=2.0,
            label="PG&E E-TOU-C, summer (USD/kWh, this study)")
    ax.step(hh, octopus, where="post", color=WONG["orange"], lw=1.5, ls="--",
            label="Octopus Agile-style, stylised (GBP/kWh)")
    ax.step(h, aus, where="post", color=WONG["green"], lw=1.5, ls=":",
            label="Ausgrid TOU, peak season, stylised levels (AUD/kWh)")
    ax.step(h, flat, where="post", color=WONG["black"], lw=1.0, alpha=0.5,
            label="Flat tariff, summer (S3 reference)")

    ax.annotate(r"PG&E 21:00 cliff" + "\n($0.123/kWh drop)",
                xy=(21, 0.455), xytext=(12.5, 0.30), fontsize=7,
                ha="center",
                arrowprops=dict(arrowstyle="->", color=WONG["blue"], lw=0.8))

    ax.set_xlim(0, 24)
    ax.set_ylim(0, 0.58)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Import price (local currency / kWh)")
    axes_title(ax, "TOU tariff geometries")
    ax.legend(loc="upper left", fontsize=7.0, framealpha=0.9)
    _despine(ax)
    _save_multi_format(fig, base_name)


# FIG (Proposition 1)
def _star_discrepancy_1d(points: np.ndarray) -> float:
    """Exact 1D star discrepancy D*_N = max_i max(i/N - x_(i), x_(i) - (i-1)/N)."""
    x = np.sort(np.asarray(points, dtype=float))
    N = x.size
    i = np.arange(1, N + 1)
    return float(np.max(np.maximum(i / N - x, x - (i - 1) / N)))


def fig_active_set_apriori(base_name: str = "F_active_set_apriori") -> None:

    Ns = np.logspace(np.log10(10), np.log10(2000), 60)
    Ns = np.unique(np.round(np.append(Ns, 50)).astype(int))

    realized = np.array([_star_discrepancy_1d(
        np.array([_van_der_corput(i) for i in range(int(N))])) for N in Ns])
    floor_log2 = np.floor(np.log2(Ns)) + 1.0

    _N_DEP = 50
    _dep = np.array([
        _star_discrepancy_1d(np.array([
            _van_der_corput(((i + 7919 * d) % 1024) or 1) for i in range(1, _N_DEP + 1)
        ]))
        for d in range(1024)
    ])
    dep_lo, dep_med, dep_hi = _dep.min(), np.median(_dep), _dep.max()
    print(f"    deployed day-permuted D*_50 over the 1024-day cycle: "
          f"min {_N_DEP*dep_lo:.3f}, median {_N_DEP*dep_med:.3f}, max {_N_DEP*dep_hi:.3f} "
          f"(N*D*; a-priori bound {int(np.floor(np.log2(_N_DEP)))+1}) -- "
          f"days exceeding the bound: {(_N_DEP*_dep > np.floor(np.log2(_N_DEP))+1).sum()}/1024")
    bound = floor_log2 / Ns                       # closed-form D*_N bound
    vdc_overshoot = 2.0 * floor_log2 / Ns         # relative overshoot, O(log N / N)
    uni_overshoot = np.sqrt(np.log(Ns) / Ns)      # relative overshoot, O(sqrt(log N / N))

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.4, 3.2))

    axa.loglog(Ns, bound, color=WONG["vermillion"], lw=1.6, ls="--",
               label=r"Bound $(\lfloor\log_2 N\rfloor+1)/N$")
    axa.loglog(Ns, realized, color=WONG["blue"], lw=1.8,
               label=r"Realized $D^*_N$ (vdC)")
    axa.fill_between(Ns, realized, bound, color=WONG["skyblue"], alpha=0.18,
                     label="Slack")
    axa.vlines(_N_DEP, dep_lo, dep_hi, color=WONG["purple"], lw=2.4, alpha=0.95,
               zorder=5,
               label=r"Deployed, $N{=}50$")
    axa.plot([_N_DEP], [dep_med], "_", color=WONG["purple"], ms=9, mew=2.0, zorder=6)
    axa.set_xlabel(r"Number of houses $N$")
    axa.set_ylabel(r"Star discrepancy $D^*_N$")
    axa.set_title(r"(a) $D^*_N$ below the a priori bound")

    _place_legend_clear_lines(axa, prefer="upper right", framealpha=0.9, fontsize=6.2,
                              handlelength=1.6, borderpad=0.35, labelspacing=0.3)
    axa.grid(True, which="both", ls=":", lw=0.4, alpha=0.45)
    _despine(axa)

    axb.loglog(Ns, uni_overshoot, color=WONG["orange"], lw=1.8, ls="-.",
               label=r"Uniform delay $\mathcal{O}(\sqrt{\log N/N})$")
    axb.loglog(Ns, vdc_overshoot, color=WONG["green"], lw=1.8,
               label=r"van der Corput $\mathcal{O}(\log N/N)$")
    # annotate the manuscript's N=50 reference (12-house absolute overshoot)
    j = int(np.where(Ns == 50)[0][0])
    axb.plot(50, vdc_overshoot[j], "o", color=WONG["green"], ms=4.5)
    axb.annotate(r"12 houses at $N{=}50$", xy=(50, vdc_overshoot[j]),
                 xytext=(95, vdc_overshoot[j] * 2.1), fontsize=7.0,
                 color=WONG["green"],
                 arrowprops=dict(arrowstyle="->", color=WONG["green"], lw=0.8))
    axb.set_xlabel(r"Number of houses $N$")
    axb.set_ylabel(r"Relative active-set overshoot")
    axb.set_title("(b) Overshoot vanishes faster for vdC")
    axb.legend(loc="lower left", framealpha=0.9, fontsize=7.2)
    axb.grid(True, which="both", ls=":", lw=0.4, alpha=0.45)
    _despine(axb)

    _save_multi_format(fig, base_name)



# GRAPHICAL ABSTRACT
GA_CF_S0        = 0.390     # T03, compiled Table 7: uncontrolled
GA_CF_S2        = 0.772     # T03, compiled Table 7: TOU rebound
GA_CF_S1        = 0.456     # T03, compiled Table 7: vdC-Stagger
GA_SRE          = 2.010     # abstract; per-cell mean of CF(S2)/CF(S0)
GA_MIT          = 0.597     # abstract; per-cell mean of CF(S1)/CF(S2)
GA_MIT_CI       = (0.568, 0.625)   # BCa 95%, 10,000 resamples
GA_MIT_CELLS    = 36        # MIT < 1 in 36 of 36
GA_BILL_USD     = 75132     # abstract; S2 minus S1 annual community bill
GA_BILL_PCT     = 32.3
GA_CO2_PCT      = 5.4       # abstract, vs the rebound
GA_JAIN_S1      = 0.672
GA_JAIN_S2      = 0.670
GA_PEAK_S2_EV   = 191.6     # battery-free S2 peak: the vehicle block at the 21:00 edge
GA_PEAK_S2      = 309.2
GA_BLDG_EV_PK   = 45.9      # building load at the battery-free S2 peak (21:00 block)
GA_BLDG_S2_PK   = 32.1      # building load at the S2 peak (midnight battery block), SRE_metrics_summary.csv


def fig_graphical_abstract(base_name: str = "Graphical_Abstract") -> None:

    assert GA_CF_S0 < GA_CF_S1 < GA_CF_S2, "CF ordering must be S0 < S1 < S2"
    assert abs(GA_CF_S2 / GA_CF_S0 - GA_SRE) < 0.06, "SRE must match the CF bars"
    assert abs(GA_CF_S1 / GA_CF_S2 - GA_MIT) < 0.01, "MIT must match the CF bars"
    assert GA_MIT_CI[0] < GA_MIT < GA_MIT_CI[1], "MIT must lie inside its own CI"
    assert GA_JAIN_S1 >= GA_JAIN_S2, "Jain: the stagger is level with the rebound"

    _claim_output(base_name)

    from matplotlib import font_manager as _fm
    _have = {f.name for f in _fm.fontManager.ttflist}
    _face = next((f for f in ("Arial", "Liberation Sans", "Helvetica") if f in _have), None)
    if _face is None:
        print("    WARNING: neither Arial nor Liberation Sans is installed; falling back "
              "to DejaVu Sans, which is NOT on Elsevier's permitted font list.")
        _face = "DejaVu Sans"
    SANS = {"family": _face}

    W_PX, H_PX, DPI_GA = 2200, 880, 300          # 2.50:1, above the 1328x531 floor
    _ga_rc = {"font.family": "sans-serif", "font.sans-serif": [_face],
              "mathtext.fontset": "stix"}
    _ctx = matplotlib.rc_context(_ga_rc); _ctx.__enter__()
    fig = plt.figure(figsize=(W_PX / DPI_GA, H_PX / DPI_GA), dpi=DPI_GA)
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.960, "Mitigating the Synchronization Rebound Effect with a "
             "Communication-Free van der Corput Stagger",
             ha="center", va="center", fontsize=11.0, fontweight="bold",
             color="0.05", **SANS)

    fig.text(0.5, 0.905, "Deterministic, open-loop, no messaging    |    "
             r"a priori $\mathcal{O}(\log N)$ diversity bound    |    "
             "9,000-run EnergyPlus sweep",
             ha="center", va="center", fontsize=9.6, color="0.34", **SANS)

    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.00, 0.92, 1.12],
                  left=0.092, right=0.982, top=0.700, bottom=0.175, wspace=0.42)
    ax0, ax1 = (fig.add_subplot(gs[0, i]) for i in range(2))
    _p = gs[0, 2].get_position(fig)
    ax2 = fig.add_axes([_p.x0, 0.500, _p.width, 0.700 - 0.500])

    # Panel 1: the problem
    h = np.linspace(17, 27, 700)

    def _base(x):
        return GA_BLDG_S2_PK + 36.0 * np.exp(-((x - 19.2) ** 2) / 6.0)
    base = _base(h)

    assert abs(float(_base(21.6)) - GA_BLDG_EV_PK) < 1.0, "base must match building load at the 21:00 peak"
    assert abs(float(_base(24.9)) - GA_BLDG_S2_PK) < 1.0, "base must match building load at the midnight peak"
    _ev = (GA_PEAK_S2_EV - _base(21.6)) * np.exp(-((h - 21.6) ** 2) / 0.55)
    _plateau = 1.0 / (1.0 + np.exp(-(h - 24.05) / 0.08)) / (1.0 + np.exp((h - 25.9) / 0.22))
    _bat = (GA_PEAK_S2 - _base(24.9)) * _plateau
    reb = base + _ev + _bat
    ax0.fill_between(h, 0, base, color=WONG["skyblue"], alpha=0.40, lw=0)
    ax0.plot(h, reb, color=WONG["vermillion"], lw=2.6)
    ax0.axvline(21, color="0.15", ls="--", lw=1.4)
    ax0.axvline(24, ymax=0.715, color="0.15", ls=":", lw=1.2)   # stops below its label
    ax0.annotate("21:00\nprice drop:\nvehicles", xy=(21.3, 178), xytext=(17.35, 226),
                 fontsize=7.2, color="0.15", **SANS, linespacing=1.15,
                 arrowprops=dict(arrowstyle="->", color="0.15", lw=0.9))
    ax0.annotate("00:00\ncarbon gate:\nbatteries", xy=(24.3, 300), xytext=(23.05, 338),
                 fontsize=7.2, color="0.15", **SANS, linespacing=1.15,
                 arrowprops=dict(arrowstyle="->", color="0.15", lw=0.9))
    ax0.set_xlim(17, 27); ax0.set_ylim(0, max(355, GA_PEAK_S2 * 1.42))
    ax0.set_yticks([0, 100, 200, 300])
    ax0.set_xticks([18, 21, 24, 27]); ax0.set_xticklabels(["18", "21", "0", "3"])
    ax0.set_xlabel("Hour", fontsize=9.6, **SANS)
    ax0.set_ylabel("Community power (kW)", fontsize=8.6, **SANS)
    ax0.set_title(f"Problem: a TOU tariff synchronizes\nflexible loads "
                  f"(CF {GA_CF_S0:.2f} $\\rightarrow$ {GA_CF_S2:.2f}, SRE {GA_SRE:.1f})",
                  fontsize=9.2, fontweight="bold", **SANS, pad=6)
    ax0.tick_params(labelsize=8.6); _despine(ax0)
    ax0.legend(handles=[mpatches.Patch(facecolor=WONG["skyblue"], alpha=0.40,
                                       label="Building load"),
                        mlines.Line2D([], [], color=WONG["vermillion"], lw=2.4,
                                      label="Rebound (S2)")],
               loc="upper left", fontsize=6.8, framealpha=0.92, handlelength=1.4,
               borderpad=0.3, prop={"family": _face, "size": 6.8})

    # Panel 2: the method 
    n_show = 14
    starts = [21 + _van_der_corput(i + 1) * 6.0 for i in range(n_show)]
    for k, s in enumerate(starts):
        ax1.barh(k, 1.55, left=s, height=0.56, color=WONG["green"],
                 alpha=0.88, edgecolor="white", lw=0.7)
        ax1.plot([s], [k], "o", color=WONG["blue"], ms=2.6, zorder=5)
    ax1.axvline(21, color="0.15", ls="--", lw=1.4)
    ax1.set_xlim(20.6, 28.4); ax1.set_ylim(-3.4, n_show - 0.1)  
    ax1.set_xticks([21, 23, 25, 27]); ax1.set_xticklabels(["21", "23", "1", "3"])
    ax1.set_yticks([])

    ax1.set_title(r"Method: $\varphi_2$ stagger keyed to"
                  "\n" r"house ID, $D_N^*=\mathcal{O}(\log N/N)$",
                  fontsize=9.2, fontweight="bold", **SANS, pad=6)
    ax1.tick_params(labelsize=8.6)
    for s in ("top", "right", "left"): ax1.spines[s].set_visible(False)
    ax1.set_xlabel("Hour", fontsize=8.6, labelpad=2, **SANS)
    ax1.text(0.5, 0.035, "feasibility gate +\ncarbon-aware shadow-price arbiter",
             transform=ax1.transAxes, ha="center", va="bottom", fontsize=7.0,
             color="0.34", linespacing=1.3, **SANS)

    # Panel 3: the result
    ax2.bar([0, 1], [GA_CF_S2, GA_CF_S1], width=0.56,
            color=[WONG["vermillion"], WONG["green"]], edgecolor="black", lw=0.7)
    for x, v in [(0, GA_CF_S2), (1, GA_CF_S1)]:
        ax2.text(x, v + 0.028, f"{v:.2f}", ha="center", fontsize=10.2,
                 fontweight="bold", **SANS)
    ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Rebound\n(S2)", "Stagger\n(S1)"])
    ax2.set_ylim(0, 1.0); ax2.set_xlim(-0.62, 1.62)
    ax2.set_ylabel("Coincidence factor", fontsize=9.0, **SANS)
    ax2.set_title(f"Result: CF cut in {GA_MIT_CELLS}/36 conditions,\nno communication required",
                  fontsize=9.2, fontweight="bold", **SANS, pad=6)
    ax2.tick_params(labelsize=8.6); _despine(ax2)


    card_x, card_w = 0.687, 0.295
    cards = [
        (0.070, WONG["green"],  "mitigation ",
         f"{GA_MIT:.3f}   BCa [{GA_MIT_CI[0]:.3f}, {GA_MIT_CI[1]:.3f}]"),
        (0.000, WONG["blue"],   "bill ",
         f"$-$\\${GA_BILL_USD:,} ({GA_BILL_PCT}%)    CO$_2$ $-${GA_CO2_PCT}%"),
        (-0.070, WONG["yellow"], "Jain ",
         f"{GA_JAIN_S1:.3f}   vs {GA_JAIN_S2:.3f} under the rebound"),
    ]
    for dy, col, head, body in cards:
        y = 0.258 + dy
        fig.patches.append(mpatches.FancyBboxPatch(
            (card_x, y), card_w, 0.056, transform=fig.transFigure,
            boxstyle="round,pad=0.003,rounding_size=0.010",
            linewidth=0, facecolor=col, alpha=0.17, zorder=0))
        fig.text(card_x + 0.014, y + 0.028, head, fontsize=8.0,
                 fontweight="bold", color="0.05", va="center", **SANS)
        fig.text(card_x + 0.014 + len(head) * 0.0072, y + 0.028, body,
                 fontsize=8.0, color="0.12", va="center", **SANS)

    fig.text(card_x + card_w / 2, 0.108,
             "all versus the TOU rebound (S2),\nthe operative counterfactual under a tariff",
             ha="center", va="center", fontsize=6.6, color="0.42",
             linespacing=1.35, style="italic", **SANS)

    for xa in (0.348, 0.652):
        fig.text(xa, 0.415, r"$\blacktriangleright$", ha="center", va="center",
                 fontsize=12, color="0.15")

    with matplotlib.rc_context({"savefig.bbox": None, "savefig.pad_inches": 0.0}):
        for ext in ("png", "pdf", "tif"):
            fig.savefig(f"{base_name}.{ext}", dpi=DPI_GA, facecolor="white")
    import matplotlib.text as _mtext
    _rend = fig.canvas.get_renderer()
    _fig_texts = [o for o in fig.findobj(_mtext.Text)]

    _ticks = {id(l) for ax in fig.axes for l in ax.get_xticklabels() + ax.get_yticklabels()}
    _major = [x for x in _fig_texts if len(x.get_text().strip()) > 1]
    _collide = []
    for _i in range(len(_major)):
        for _j in range(_i + 1, len(_major)):
            if id(_major[_i]) in _ticks and id(_major[_j]) in _ticks:
                continue
            try:
                _b1 = _major[_i].get_window_extent(renderer=_rend)
                _b2 = _major[_j].get_window_extent(renderer=_rend)
            except Exception:
                continue
            _ox = min(_b1.x1, _b2.x1) - max(_b1.x0, _b2.x0)
            _oy = min(_b1.y1, _b2.y1) - max(_b1.y0, _b2.y0)
            if _ox > 3 and _oy > 3:
                _collide.append((_major[_i].get_text()[:40].replace("\n", " / "),
                                 _major[_j].get_text()[:40].replace("\n", " / "),
                                 round(_ox), round(_oy)))

    _overflow = []
    for _txt in _fig_texts:
        s = _txt.get_text().strip()
        if not s:
            continue
        try:
            bb = _txt.get_window_extent(renderer=_rend)
        except Exception:
            continue
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W_PX + 0.5 or bb.y1 > H_PX + 0.5:
            _overflow.append((s[:44], round(bb.x0), round(bb.x1)))
    plt.close(fig)
    _ctx.__exit__(None, None, None)
    if _overflow:
        raise SystemExit(
            "\nGRAPHICAL ABSTRACT: text is clipped by the canvas.\n"
            + "".join(f"    {s!r}  spans x=[{x0}, {x1}]  (canvas 0..{W_PX})\n"
                      for s, x0, x1 in _overflow)
            + "  Shrink the type or widen the panel; do not ship it.\n")
    if _collide:
        raise SystemExit(
            "\nGRAPHICAL ABSTRACT: labels overlap each other.\n"
            + "".join(f"    {a!r}\n      <-> {b!r}   overlap {ox}x{oy} px\n"
                      for a, b, ox, oy in _collide)
            + "  These collide on the page even though both fit the canvas.\n")

    from PIL import Image as _Im
    w, hpx = _Im.open(f"{base_name}.png").size
    spec_ok = (w >= 1328 and hpx >= 531 and abs(w / hpx - 2.5) < 0.02)
    print(f"  wrote {base_name}.{{png,pdf,tif}}  [{w}x{hpx} px, ratio {w/hpx:.2f}:1, {DPI_GA} dpi]")
    print(f"    Elsevier spec (>=1328x531 px, 5:2, >=300 dpi): "
          f"{'PASS' if spec_ok else '** FAIL **'}   font: {_face}")
    print(f"    no clipped text: PASS ({len(_fig_texts)} text artists checked)")


def main() -> None:
    print("Generating the 9 schematic figures + the Elsevier graphical abstract (Wong palette, 300 dpi):")
    reference_count_report()
    reference_first_citation_report()
    fig_intro_sre_mechanism()
    fig_thematic_landscape()
    fig_year_distribution()
    fig_feeder_topology()
    fig_system_architecture()
    fig_experimental_design()
    fig_lp_opf_bound()
    fig_algorithm_components()
    fig_active_set_apriori()
    fig_tariff_geometries()
    fig_graphical_abstract()
    print("Done. These complement F01..F10 from IEEE_Sensitivity_Analysis.py,")
    print("so every figure in the manuscript now has a reproducible source.")
    print("\nNOTE: figures captioned 'illustrative' (SRE mechanism, LP-OPF exhibit)")
    print("depict mechanism shape, not telemetry; the tariff figure uses the exact")
    print("PG&E levels with stylised Octopus/Ausgrid archetypes, as the caption states.")


if __name__ == "__main__":
    main()