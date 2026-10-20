#!/usr/bin/env python3
"""Charts for the first C3 (EuqVid-Text) results. Black and white, for slides.

    python plot_lexical.py [--root ...]

Reads   derived/c3/c3_lexical_features.csv
        equvid/components/C3_Text/results/c3_lexical_corpus_summary.csv
        equvid/components/C3_Text/results/c3_agency_power_summary.csv (if present)
Writes  equvid/components/C3_Text/results/
          fig1_corpus_parity.png     the C3 equation features: pooled value and 95% CI
          fig2_words_vs_time.png     F1 against C2's speaking-time parity, per clip:
                                     the evidence for dropping F1 from the equation
          fig3_clip_coverage.png     share of clips where each feature can be computed
          fig4_agency_power.png      agency and power: pooled value and 95% CI
                                     (only once agency_power_features.py has run)

Every number drawn here is also in the CSVs, which are the table view.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE.parents[2]

INK, MUTED, GRID = "#1a1a1a", "#6b6b6b", "#d9d9d9"
STYLE = {  # identity by marker shape and fill, so it survives grayscale printing
    "MELD": dict(marker="o", ms=8, mfc=INK, mec=INK),
    "AMI": dict(marker="s", ms=8, mfc="white", mec=INK, mew=1.6),
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": INK,
    "axes.labelcolor": INK, "text.color": INK, "savefig.dpi": 200,
})

PANELS = [  # feature, title, unit of the value label -- the C3 equation features only.
    # F1 (words per person) is not here: it left the C3 equation on 19 Oct 2026
    # because it repeats C2 (fig2). It is still computed, as a pipeline check.
    ("length_log_ratio", "Words per utterance (F3)", "log"),
    ("same_gender_excess_diff", "Same-gender replies above chance", "pts"),
    ("ref_log_parity", "Referred to as he / she,\nper person (new)", "log"),
]


AP_PANELS = [  # agency_power_features.py: the two equation features
    ("agency", "Agency of the speaker's own verbs\n(\"I decided\" +1, \"I waited\" \u22121)", "diff"),
    ("power", "Power of the speaker's own verbs\n(\"I ordered\" +1, \"I obeyed\" \u22121)", "diff"),
]


def value_label(v, unit):
    if unit == "pts":
        txt = f"{100 * v:+.1f} pts"
    elif unit == "diff":
        txt = "0.00" if abs(v) < 0.005 else f"{v:+.2f}"   # no "-0.00"
    else:
        txt = f"{v:+.2f} ({100 * (np.exp(v) - 1):+.0f}%)"
    return txt.replace("-", "\u2212")          # typographic minus, as on the axis


def clean(ax):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def dot_whisker(summary, panels, suptitle, out, note):
    """One panel per feature: pooled value and 95% CI for each corpus."""
    fig, axes = plt.subplots(1, len(panels), figsize=(3.5 * len(panels), 3.4), sharey=True)
    corpora = ["AMI", "MELD"]                    # drawn bottom-up, so MELD is on top
    for ax, (feat, title, unit) in zip(axes, panels):
        s = summary[summary.feature == feat].set_index("corpus")
        span = np.nanmax(np.abs(s[["ci_low", "ci_high"]].to_numpy())) * 1.25
        ax.axvline(0, color=INK, linewidth=1.0, zorder=1)
        for y, c in enumerate(corpora):
            r = s.loc[c]
            ax.plot([r.ci_low, r.ci_high], [y, y], color=INK, linewidth=1.6,
                    solid_capstyle="round", zorder=2)
            ax.plot(r.estimate, y, linestyle="none", zorder=3, **STYLE[c])
            ax.annotate(value_label(r.estimate, unit), (r.estimate, y),
                        xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK, zorder=4,
                        bbox=dict(fc="white", ec="none", pad=0.6))
            ax.annotate(note(r), (span, y), xytext=(0, -13),
                        textcoords="offset points", ha="right", fontsize=7.5, color=MUTED)
        ax.set_xlim(-span, span)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))  # no crowded ticks
        ax.set_ylim(-0.6, 1.6)
        ax.set_yticks([0, 1], corpora)
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlabel("← men higher     women higher →", fontsize=8.5, color=MUTED)
        clean(ax)
    fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig1(summary, out):
    dot_whisker(summary, PANELS,
                "C3 features without a parser: women vs men per corpus, 95% CI "
                "(bootstrap over series)", out, lambda r: f"n = {int(r.n_clips)} clips")


def fig4(summary, out):
    dot_whisker(summary[summary.role == "equation"], AP_PANELS,
                "Agency and power (Sap et al. lexicon): women minus men per corpus, 95% CI",
                out, lambda r: f"verbs: F {int(r.frames_F):,} / M {int(r.frames_M):,}")


def fig2(clips, out):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    v = clips[clips.status == "valid"]
    lim = 2.8
    ax.plot([-lim, lim], [-lim, lim], color=MUTED, linewidth=1.0, zorder=1)
    ax.axhline(0, color=GRID, linewidth=0.8, zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.8, zorder=0)
    for c in ("MELD", "AMI"):
        d = v[v.corpus == c].dropna(subset=["time_log_parity", "volume_log_parity"])
        r = d.time_log_parity.corr(d.volume_log_parity)
        st = dict(STYLE[c], ms=5.5)
        ax.plot(d.time_log_parity, d.volume_log_parity, linestyle="none", alpha=0.85,
                label=f"{c}: {len(d)} clips, r = {r:.2f}", zorder=2, **st)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("C2: speaking time per person, log(F / M)")
    ax.set_ylabel("C3 F1: words per person, log(F / M)")
    ax.set_title("F1 repeats C2, so it leaves the C3 equation", loc="left", fontsize=11.5)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.annotate("line = identical", (1.75, 2.05), fontsize=8, color=MUTED, rotation=45)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig3(clips, out):
    feats = [("length_log_ratio", "Words per utterance (F3)"),
             ("same_gender_excess_diff", "Same-gender replies"),
             ("ref_log_parity", "Referred to as he / she")]
    v = clips[clips.status == "valid"]
    fig, ax = plt.subplots(figsize=(7.2, 2.7))
    shades = {"MELD": INK, "AMI": "#a3a3a3"}
    h = 0.34
    for i, (f, label) in enumerate(feats):
        for j, c in enumerate(("MELD", "AMI")):
            d = v[v.corpus == c]
            share = d[f].notna().mean()
            y = i + (j - 0.5) * (h + 0.04)
            ax.barh(y, 100 * share, height=h, color=shades[c],
                    label=c if i == 0 else None)
            ax.annotate(f"{100 * share:.0f}%  ({d[f].notna().sum()}/{len(d)})",
                        (100 * share, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=8.5)
    ax.set_yticks(range(len(feats)), [l for _, l in feats])
    ax.invert_yaxis()
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("% of scorable clips where the feature can be computed on one clip")
    ax.set_title("Pronoun features need pooling over many clips", loc="left", fontsize=11.5)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    clean(ax)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    res = args.root / "equvid" / "components" / "C3_Text" / "results"
    summary = pd.read_csv(res / "c3_lexical_corpus_summary.csv")
    clips = pd.read_csv(args.root / "derived" / "c3" / "c3_lexical_features.csv")
    fig1(summary, res / "fig1_corpus_parity.png")
    fig2(clips, res / "fig2_words_vs_time.png")
    fig3(clips, res / "fig3_clip_coverage.png")
    ap_summary = res / "c3_agency_power_summary.csv"
    if ap_summary.exists():                      # written by agency_power_features.py
        fig4(pd.read_csv(ap_summary), res / "fig4_agency_power.png")
    print("wrote the figures to", res)


if __name__ == "__main__":
    main()
