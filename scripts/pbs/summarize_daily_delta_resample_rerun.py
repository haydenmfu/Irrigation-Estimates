#!/usr/bin/env python3
"""Summarize retrieved Week 7 daily delta-SM sequential resample/rerun output."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
plt.style.use(str(PROJECT / "research_report.mplstyle"))


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize a retrieved Week 7 resample/rerun PBS output.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def read_window_outputs(run_root):
    post_root = run_root / "postprocess"
    summary_frames = []
    daily_frames = []
    weight_frames = []
    for window_dir in sorted(post_root.glob("window_*")):
        if (window_dir / "window_summary.csv").exists():
            summary_frames.append(pd.read_csv(window_dir / "window_summary.csv"))
        if (window_dir / "posterior_daily_irrigation.csv").exists():
            daily = pd.read_csv(window_dir / "posterior_daily_irrigation.csv")
            if summary_frames:
                window_id = int(pd.read_csv(window_dir / "window_summary.csv").iloc[0]["window_id"])
                daily["window_id"] = window_id
            daily_frames.append(daily)
        if (window_dir / "particle_weights.csv").exists():
            weights = pd.read_csv(window_dir / "particle_weights.csv")
            if summary_frames:
                window_id = int(pd.read_csv(window_dir / "window_summary.csv").iloc[0]["window_id"])
                weights["window_id"] = window_id
            weight_frames.append(weights)
    if not summary_frames:
        raise FileNotFoundError("No postprocess/window_*/window_summary.csv files found under {}".format(run_root))
    summary = pd.concat(summary_frames, ignore_index=True).sort_values("window_id")
    daily = pd.concat(daily_frames, ignore_index=True).sort_values("date") if daily_frames else pd.DataFrame()
    weights = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    return summary, daily, weights


def read_resampling_tables(run_root):
    frames = []
    for path in sorted((run_root / "inputs").glob("particle_initial_states_window_*.csv")):
        frame = pd.read_csv(path)
        frame["resample_file"] = path.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_outputs(out_dir, summary, daily, resampling):
    if not daily.empty:
        d = daily.copy()
        d["date_dt"] = pd.to_datetime(d["date"])
        fig, ax = plt.subplots(figsize=(13.0, 5.0), constrained_layout=True)
        ax.bar(d["date_dt"], d["posterior_mean_irrigation_mm"], width=0.75, color="#365C8D", label="Posterior mean")
        ax.plot(d["date_dt"], d["prior_mean_irrigation_mm"], marker="o", color="#5C6773", label="Prior mean")
        ax.set_title("Sequential resample/rerun daily delta-SM posterior irrigation", pad=38)
        ax.set_ylabel("Irrigation (mm/day)")
        ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))
        fig.savefig(out_dir / "posterior_vs_prior_daily_irrigation_resample_rerun.png", dpi=300)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    ax.bar(summary["window_id"].astype(str), summary["effective_sample_size"], color="#365C8D")
    ax.set_xlabel("Window")
    ax.set_ylabel("ESS")
    ax.set_title("Effective sample size by sequential rerun window")
    fig.savefig(out_dir / "effective_sample_size_by_window_resample_rerun.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    ax.bar(summary["window_id"].astype(str), summary["max_particle_weight"], color="#B58D1D")
    ax.set_xlabel("Window")
    ax.set_ylabel("Max particle weight")
    ax.set_title("Max particle weight by sequential rerun window")
    fig.savefig(out_dir / "max_particle_weight_by_window_resample_rerun.png", dpi=300)
    plt.close(fig)

    if not resampling.empty:
        counts = (
            resampling.groupby(["resample_file", "parent_particle"], as_index=False)
            .size()
            .rename(columns={"size": "offspring_count"})
        )
        top = counts.sort_values(["resample_file", "offspring_count"], ascending=[True, False])
        top.to_csv(out_dir / "resampling_parent_offspring_counts.csv", index=False)
        diversity = (
            counts.groupby("resample_file", as_index=False)
            .agg(unique_parent_particles=("parent_particle", "nunique"), max_offspring=("offspring_count", "max"))
        )
        diversity.to_csv(out_dir / "resampling_diversity.csv", index=False)
        fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
        ax.bar(diversity["resample_file"], diversity["unique_parent_particles"], color="#1F9E89")
        ax.set_ylabel("Unique parent particles")
        ax.set_title("Resampling parent diversity")
        ax.tick_params(axis="x", labelrotation=20)
        fig.savefig(out_dir / "resampling_parent_diversity.png", dpi=300)
        plt.close(fig)


def write_report(out_dir, run_root, summary, daily, resampling):
    posterior_total = float(summary["posterior_irrigation_sum_mm"].sum())
    prior_total = float(summary["prior_irrigation_sum_mm"].sum())
    lines = [
        "# Week 7 Daily Delta-SM Sequential Resample/Rerun Summary",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Run Totals",
        "",
        f"- Posterior total irrigation: `{posterior_total:.4f} mm`",
        f"- Prior total irrigation: `{prior_total:.4f} mm`",
        f"- Minimum ESS: `{summary['effective_sample_size'].min():.2f}`",
        f"- Maximum particle weight: `{summary['max_particle_weight'].max():.3f}`",
        "",
        "## Window Summary",
        "",
        "| Window | Dates | Target | Sigma | Daily obs rows | ESS | Max weight | Best particle | Posterior irr mm | Prior irr mm |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row.window_id)} | {row.window_start} to {row.window_end} | {row.target_mode} | "
            f"{row.delta_sigma_m3m3:.3f} | {int(row.n_daily_observation_rows)} | "
            f"{row.effective_sample_size:.2f} | {row.max_particle_weight:.3f} | {int(row.best_particle)} | "
            f"{row.posterior_irrigation_sum_mm:.4f} | {row.prior_irrigation_sum_mm:.4f} |"
        )
    if not resampling.empty:
        diversity_path = out_dir / "resampling_diversity.csv"
        if diversity_path.exists():
            diversity = pd.read_csv(diversity_path)
            lines.extend(["", "## Resampling Diversity", "", "| Next-window state table | Unique parents | Max offspring |", "|---|---:|---:|"])
            for _, row in diversity.iterrows():
                lines.append(
                    f"| {row.resample_file} | {int(row.unique_parent_particles)} | {int(row.max_offspring)} |"
                )
    lines.extend(
        [
            "",
            "## Plots",
            "",
            "![Posterior versus prior daily irrigation](./posterior_vs_prior_daily_irrigation_resample_rerun.png)",
            "",
            "![ESS by window](./effective_sample_size_by_window_resample_rerun.png)",
            "",
            "![Max particle weight by window](./max_particle_weight_by_window_resample_rerun.png)",
            "",
            "![Resampling parent diversity](./resampling_parent_diversity.png)",
            "",
        ]
    )
    (out_dir / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")
    metadata = {
        "run_root": str(run_root),
        "posterior_total_irrigation_mm": posterior_total,
        "prior_total_irrigation_mm": prior_total,
        "min_effective_sample_size": float(summary["effective_sample_size"].min()),
        "max_particle_weight": float(summary["max_particle_weight"].max()),
    }
    (out_dir / "summary_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    run_root = args.run_root.resolve()
    if not run_root.exists():
        raise FileNotFoundError(run_root)
    out_dir = args.out_dir or (run_root / "local_summary")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary, daily, weights = read_window_outputs(run_root)
    resampling = read_resampling_tables(run_root)
    summary.to_csv(out_dir / "window_summary_combined.csv", index=False)
    daily.to_csv(out_dir / "posterior_daily_irrigation_combined.csv", index=False)
    weights.to_csv(out_dir / "particle_weights_combined.csv", index=False)
    if not resampling.empty:
        resampling.to_csv(out_dir / "resampling_parent_state_tables_combined.csv", index=False)
    plot_outputs(out_dir, summary, daily, resampling)
    write_report(out_dir, run_root, summary, daily, resampling)
    print(out_dir)


if __name__ == "__main__":
    main()
