#!/usr/bin/env python3
"""Mixed-tier paper figure & table generator.

Reads CSVs produced by the four mixed-tier ARC drivers and emits paper-ready
PDF figures + LaTeX tables.

Usage:
    python scripts/make_paper_figures.py \
        --baseline   results/mixed_tier_v35_50seed_5143812 \
        --n2k        results/mixed_tier_v35_n2k_50seed_5143876 \
        --mechanism  results/mixed_tier_mechanism_5143889 \
        --placebo    results/mixed_tier_placebo_5143888 \
        --refutation results/mixed_tier_refutation_v3_<job> \
        --output     results/paper_figures_<ts>
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

TIER_ORDER = ["none", "basic", "advanced", "premium"]
TIER_LABELS = {"none": "No AI", "basic": "Basic AI",
               "advanced": "Advanced AI", "premium": "Premium AI"}
TIER_COLORS = {"none": "#6c757d", "basic": "#0d6efd",
               "advanced": "#fd7e14", "premium": "#dc3545"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 100,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
})


def fig_survival_by_tier(baseline_dir, output_dir):
    df = pd.read_csv(baseline_dir / "summary_stats.csv").set_index("tier")
    fig, ax = plt.subplots(figsize=(6, 4))
    means = [df.loc[t, "mean_survival_rate"] * 100 for t in TIER_ORDER]
    stds = [df.loc[t, "std_survival_rate"] * 100 for t in TIER_ORDER]
    ns = [df.loc[t, "n_runs"] for t in TIER_ORDER]
    se = [s / np.sqrt(n) for s, n in zip(stds, ns)]
    ci95 = [1.96 * s for s in se]
    colors = [TIER_COLORS[t] for t in TIER_ORDER]
    bars = ax.bar(range(4), means, yerr=ci95, capsize=6,
                  color=colors, edgecolor="black", linewidth=0.7)
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2, mean + 1.5,
                f"{mean:.1f}%\n(±{std:.1f})", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("60-Round Survival Rate (%)")
    ax.set_title("Mixed-Tier Survival by AI Level\n(1000 agents, 250/tier, 50 seeds)")
    ax.set_ylim(0, 70)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(3.45, 50.5, "BLS band\n50–55%", fontsize=7, color="gray", alpha=0.7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_survival_by_tier.pdf")
    plt.close(fig)


def fig_survival_trajectories(baseline_dir, output_dir):
    df = pd.read_csv(baseline_dir / "survival_trajectories.csv")
    per_run = pd.read_csv(baseline_dir / "per_run_data.csv")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rounds = df["round"].values
    for tier in TIER_ORDER:
        mean_traj = df[tier].values * 100
        # approx SE from per_run final survival std (band only at endpoint;
        # for cleaner viz just plot the mean trajectories)
        ax.plot(rounds, mean_traj, color=TIER_COLORS[tier],
                label=TIER_LABELS[tier], linewidth=2)
    ax.set_xlabel("Round (Months)")
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Survival Trajectories Under Competitive Mixing")
    ax.set_xlim(1, 60)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_survival_trajectories.pdf")
    plt.close(fig)


def fig_treatment_effects(baseline_dir, output_dir):
    df = pd.read_csv(baseline_dir / "treatment_effects.csv").set_index("tier")
    treated = ["basic", "advanced", "premium"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    tes = [df.loc[t, "treatment_effect"] for t in treated]
    ses = [df.loc[t, "std_error"] * 100 for t in treated]
    ci95 = [1.96 * s for s in ses]
    y = np.arange(len(treated))
    colors = [TIER_COLORS[t] for t in treated]
    ax.errorbar(tes, y, xerr=ci95, fmt="o", color="black",
                ecolor="gray", capsize=5, markersize=8,
                markerfacecolor="white")
    for yi, te, color in zip(y, tes, colors):
        ax.scatter([te], [yi], s=120, color=color, zorder=5,
                   edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([TIER_LABELS[t] for t in treated])
    ax.set_xlabel("Treatment Effect vs No AI (percentage points)")
    ax.set_title("Survival Treatment Effects (mean ± 95% CI)")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_treatment_effects.pdf")
    plt.close(fig)


def fig_wealth_distributions(baseline_dir, output_dir):
    df = pd.read_csv(baseline_dir / "per_run_data.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    positions = []
    data = []
    colors = []
    labels = []
    for i, tier in enumerate(TIER_ORDER):
        sub = df[df.tier == tier]
        for j, q in enumerate(["p50_capital", "p90_capital", "p95_capital"]):
            positions.append(i * 4 + j)
            data.append(sub[q].values / 1e6)
            colors.append(TIER_COLORS[tier])
        labels.append(TIER_LABELS[tier])
    bp = ax.boxplot(data, positions=positions, widths=0.7,
                    patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
    ax.set_xticks([i * 4 + 1 for i in range(4)])
    ax.set_xticklabels(labels)
    ax.set_ylabel("Survivor Capital ($M)")
    ax.set_title("Survivor Wealth Distribution by Tier (P50, P90, P95)")
    legend_elements = [
        Patch(facecolor="white", edgecolor="black", label="left=P50"),
        Patch(facecolor="white", edgecolor="black", label="mid=P90"),
        Patch(facecolor="white", edgecolor="black", label="right=P95"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_wealth_distributions.pdf")
    plt.close(fig)


def fig_four_winners(baseline_dir, output_dir):
    summary = pd.read_csv(baseline_dir / "summary_stats.csv").set_index("tier")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))

    metrics = [
        ("mean_survival_rate", "Survival Rate", lambda v: v * 100, "%", "basic"),
        ("mean_innovations_per_agent", "Innovation Volume", lambda v: v,
         "per agent", "advanced"),
        ("mean_p50_capital", "Median Survivor Wealth", lambda v: v / 1e6,
         "$M", "none"),
        ("mean_p95_capital", "Right-Tail Wealth (P95)", lambda v: v / 1e6,
         "$M", "premium"),
    ]
    for ax, (col, title, transform, units, winner) in zip(axes, metrics):
        vals = [transform(summary.loc[t, col]) for t in TIER_ORDER]
        colors = [TIER_COLORS[t] if t == winner else
                  (TIER_COLORS[t] + "60") for t in TIER_ORDER]
        edges = ["black" if t == winner else "gray" for t in TIER_ORDER]
        widths = [1.5 if t == winner else 0.5 for t in TIER_ORDER]
        bars = ax.bar(range(4), vals, color=colors,
                      edgecolor=edges, linewidth=widths)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2,
                    v + max(vals) * 0.02,
                    f"{v:.1f}", ha="center", fontsize=8)
        ax.set_xticks(range(4))
        ax.set_xticklabels(["None", "Basic", "Adv", "Prem"], fontsize=8)
        ax.set_title(f"{title}\n(winner: {TIER_LABELS[winner]})", fontsize=10)
        ax.set_ylabel(units, fontsize=9)
    fig.suptitle("Four Metrics, Four Winners (Mixed-Tier, N=1000, 50 seeds)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_four_winners.pdf")
    plt.close(fig)


def fig_mediation_paths(mech_dir, output_dir):
    df = pd.read_csv(mech_dir / "mediation_analysis.csv")
    paths_show = [
        ("tier_to_survival", "Tier -> Survival\n(total)"),
        ("tier_to_innovate", "Tier -> Innovate"),
        ("tier_to_competition", "Tier -> Competition"),
        ("tier_to_niches", "Tier -> Niches"),
        ("innovate_to_survival", "Innovate -> Survival"),
        ("competition_to_survival", "Competition -> Survival"),
        ("niches_to_survival", "Niches -> Survival"),
        ("indirect_via_innovate", "Indirect via Innovate"),
        ("indirect_via_competition", "Indirect via Competition"),
        ("indirect_via_niches", "Indirect via Niches"),
    ]
    pd_map = dict(zip(df.path, df.value))
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [lab for _, lab in paths_show]
    vals = [pd_map[k] for k, _ in paths_show]
    colors = ["#dc3545" if v < 0 else "#0d6efd" for v in vals]
    y = np.arange(len(vals))
    ax.barh(y, vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + (0.02 if v >= 0 else -0.02), yi,
                f"{v:+.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Correlation (r)")
    ax.set_title("Mediation Pathways: Tier -> Survival\n(within-run cross-tier panel, N=200 rows)")
    ax.set_xlim(-1, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_mediation_paths.pdf")
    plt.close(fig)


def fig_placebo(placebo_dir, output_dir):
    perm = pd.read_csv(placebo_dir / "permutation_null_distribution.csv")
    summary = pd.read_csv(placebo_dir / "placebo_summary.csv").set_index("test")
    early = pd.read_csv(placebo_dir / "early_period_ate.csv")

    fig = plt.figure(figsize=(13, 4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.9, 1.0], wspace=0.4)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # Permutation null
    ax1.hist(perm.null_te_pp, bins=40, color="#cccccc",
             edgecolor="black", linewidth=0.4)
    actual = summary.loc["permutation_premium_vs_none", "actual_pp"]
    ci_lo = summary.loc["permutation_premium_vs_none", "ci_lo_pp"]
    ci_hi = summary.loc["permutation_premium_vs_none", "ci_hi_pp"]
    p_val = summary.loc["permutation_premium_vs_none", "p_value"]
    ax1.axvline(actual, color=TIER_COLORS["premium"], linewidth=2.5,
                label=f"Actual ATE = {actual:+.2f} pp")
    ax1.axvline(ci_lo, color="black", linestyle="--", linewidth=1)
    ax1.axvline(ci_hi, color="black", linestyle="--", linewidth=1,
                label=f"95% null CI [{ci_lo:+.1f},{ci_hi:+.1f}]")
    ax1.set_xlabel("Premium − None ATE (pp)")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"A. Permutation Test (p = {p_val:.4f})")
    ax1.legend(fontsize=8, framealpha=0.9)

    # Placebo tier comparison
    actual_premium = summary.loc["permutation_premium_vs_none", "actual_pp"]
    placebo_pair = summary.loc["placebo_basic_vs_advanced", "actual_pp"]
    ax2.bar([0, 1], [abs(actual_premium), abs(placebo_pair)],
            color=[TIER_COLORS["premium"], TIER_COLORS["basic"]],
            edgecolor="black")
    for i, v in enumerate([abs(actual_premium), abs(placebo_pair)]):
        ax2.text(i, v + 0.1, f"{v:.2f} pp", ha="center", fontsize=9)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Actual\n(Prem vs None)", "Placebo\n(Basic vs Adv)"])
    ax2.set_ylabel("|Treatment Effect| (pp)")
    ax2.set_title(f"B. Placebo Tier Test\n(ratio: {abs(actual_premium)/max(abs(placebo_pair),1e-3):.1f}×)")

    # Early-period growth
    ax3.plot(early["round"], early["ate_pp"], "o-",
             color=TIER_COLORS["premium"], markersize=8, linewidth=2)
    ax3.axhline(0, color="black", linewidth=0.6, alpha=0.5)
    ax3.set_xlabel("Round (Months)")
    ax3.set_ylabel("Premium − None ATE (pp)")
    ax3.set_title("C. Effect Growth Over Time")
    ax3.set_xticks(early["round"])

    fig.suptitle("Placebo Tests Support the Mixed-Tier Paradox",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_placebo.pdf")
    plt.close(fig)


def fig_n_robustness(baseline_dir, n2k_dir, output_dir):
    b = pd.read_csv(baseline_dir / "summary_stats.csv").set_index("tier")
    n2 = pd.read_csv(n2k_dir / "summary_stats.csv").set_index("tier")
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(4)
    w = 0.35
    b_means = [b.loc[t, "mean_survival_rate"] * 100 for t in TIER_ORDER]
    b_stds = [b.loc[t, "std_survival_rate"] * 100 for t in TIER_ORDER]
    n_means = [n2.loc[t, "mean_survival_rate"] * 100 for t in TIER_ORDER]
    n_stds = [n2.loc[t, "std_survival_rate"] * 100 for t in TIER_ORDER]
    colors = [TIER_COLORS[t] for t in TIER_ORDER]
    ax.bar(x - w/2, b_means, w, yerr=b_stds, capsize=4,
           color=[c + "99" for c in colors], edgecolor="black", label="N=1000")
    ax.bar(x + w/2, n_means, w, yerr=n_stds, capsize=4,
           color=colors, edgecolor="black", label="N=2000")
    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in TIER_ORDER])
    ax.set_ylabel("Survival Rate (%)")
    ax.set_title("Scale Robustness: Pattern Preserved at N=2000")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 75)
    fig.tight_layout()
    fig.savefig(output_dir / "fig_n_robustness.pdf")
    plt.close(fig)


def fig_refutation_grid(refut_dir, output_dir):
    df = pd.read_csv(refut_dir / "refutation_v3_mixed_summary.csv")
    df = df.sort_values(["category", "test"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 0.32 * len(df) + 1.2))
    y = np.arange(len(df))
    tes = df.te_premium_pp.values
    colors = ["#dc3545" if v < -1 else ("#fd7e14" if v < 1 else "#0d6efd")
              for v in tes]
    ax.barh(y, tes, color=colors, edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linewidth=0.7)
    for yi, v, n in zip(y, tes, df.test):
        ax.text(v + (0.2 if v >= 0 else -0.2), yi,
                f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"[{c}] {n}" for c, n in zip(df.category, df.test)], fontsize=7
    )
    ax.invert_yaxis()
    ax.set_xlabel("Premium − None Treatment Effect (pp)")
    ax.set_title("Refutation Test Suite v3 — Mixed-Tier (31 conditions)")
    fig.tight_layout()
    fig.savefig(output_dir / "fig_refutation_grid.pdf")
    plt.close(fig)


def write_table_summary(baseline_dir, n2k_dir, output_dir):
    b = pd.read_csv(baseline_dir / "summary_stats.csv").set_index("tier")
    n2 = pd.read_csv(n2k_dir / "summary_stats.csv").set_index("tier") \
        if n2k_dir else None
    rows = []
    for t in TIER_ORDER:
        row = {
            "Tier": TIER_LABELS[t],
            "Survival N=1000 (mean ± std)":
                f"{b.loc[t, 'mean_survival_rate']:.3f} ± {b.loc[t, 'std_survival_rate']:.3f}",
            "P50 ($M)": f"{b.loc[t, 'mean_p50_capital']/1e6:.2f}",
            "P95 ($M)": f"{b.loc[t, 'mean_p95_capital']/1e6:.2f}",
            "Innov/Agent": f"{b.loc[t, 'mean_innovations_per_agent']:.2f}",
            "Innov Share": f"{b.loc[t, 'mean_innovate_share']*100:.1f}%",
        }
        if n2 is not None:
            row["Survival N=2000 (mean ± std)"] = (
                f"{n2.loc[t, 'mean_survival_rate']:.3f} ± "
                f"{n2.loc[t, 'std_survival_rate']:.3f}"
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "table_3_summary.csv", index=False)
    with open(output_dir / "table_3_summary.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True))


def write_table_refutation(refut_dir, output_dir):
    df = pd.read_csv(refut_dir / "refutation_v3_mixed_summary.csv")
    out = df[["test", "category", "description",
              "none_survival_mean", "premium_survival_mean",
              "te_premium_pp", "te_premium_pp_std",
              "te_advanced_pp", "te_basic_pp"]].copy()
    out.columns = ["Test", "Category", "Description",
                   "None", "Premium", "TE Prem (pp)",
                   "TE Prem SD", "TE Adv (pp)", "TE Basic (pp)"]
    for col in ["None", "Premium"]:
        out[col] = out[col].apply(lambda x: f"{x:.3f}")
    for col in ["TE Prem (pp)", "TE Adv (pp)", "TE Basic (pp)"]:
        out[col] = out[col].apply(lambda x: f"{x:+.2f}")
    out["TE Prem SD"] = out["TE Prem SD"].apply(lambda x: f"{x:.2f}")
    out.to_csv(output_dir / "table_4_refutation.csv", index=False)
    with open(output_dir / "table_4_refutation.tex", "w") as f:
        f.write(out.to_latex(index=False, escape=True, longtable=True))


def write_table_mediation(mech_dir, output_dir):
    df = pd.read_csv(mech_dir / "mediation_analysis.csv")
    df["value"] = df["value"].apply(lambda x: f"{x:+.3f}")
    df.columns = ["Path", "Correlation"]
    df.to_csv(output_dir / "table_5_mediation.csv", index=False)
    with open(output_dir / "table_5_mediation.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True))


def write_table_placebo(placebo_dir, output_dir):
    summary = pd.read_csv(placebo_dir / "placebo_summary.csv")
    summary["actual_pp"] = summary["actual_pp"].apply(lambda x: f"{x:+.2f}")
    summary["null_mean_pp"] = summary["null_mean_pp"].apply(lambda x: f"{x:+.2f}")
    summary["ci_lo_pp"] = summary["ci_lo_pp"].apply(lambda x: f"{x:+.2f}")
    summary["ci_hi_pp"] = summary["ci_hi_pp"].apply(lambda x: f"{x:+.2f}")
    summary["p_value"] = summary["p_value"].apply(lambda x: f"{x:.4f}")
    summary.to_csv(output_dir / "table_6_placebo.csv", index=False)
    with open(output_dir / "table_6_placebo.tex", "w") as f:
        f.write(summary.to_latex(index=False, escape=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, type=Path)
    p.add_argument("--n2k", type=Path, default=None)
    p.add_argument("--mechanism", type=Path, default=None)
    p.add_argument("--placebo", type=Path, default=None)
    p.add_argument("--refutation", type=Path, default=None)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures and tables in: {args.output}")

    print("  fig_survival_by_tier")
    fig_survival_by_tier(args.baseline, args.output)
    print("  fig_survival_trajectories")
    fig_survival_trajectories(args.baseline, args.output)
    print("  fig_treatment_effects")
    fig_treatment_effects(args.baseline, args.output)
    print("  fig_wealth_distributions")
    fig_wealth_distributions(args.baseline, args.output)
    print("  fig_four_winners")
    fig_four_winners(args.baseline, args.output)
    print("  table_3_summary")
    write_table_summary(args.baseline, args.n2k, args.output)

    if args.n2k:
        print("  fig_n_robustness")
        fig_n_robustness(args.baseline, args.n2k, args.output)

    if args.mechanism:
        print("  fig_mediation_paths")
        fig_mediation_paths(args.mechanism, args.output)
        print("  table_5_mediation")
        write_table_mediation(args.mechanism, args.output)

    if args.placebo:
        print("  fig_placebo")
        fig_placebo(args.placebo, args.output)
        print("  table_6_placebo")
        write_table_placebo(args.placebo, args.output)

    if args.refutation and (args.refutation / "refutation_v3_mixed_summary.csv").exists():
        print("  fig_refutation_grid")
        fig_refutation_grid(args.refutation, args.output)
        print("  table_4_refutation")
        write_table_refutation(args.refutation, args.output)
    elif args.refutation:
        print(f"  refutation CSV not yet present at {args.refutation}; skipping")

    print("Done.")


if __name__ == "__main__":
    main()
