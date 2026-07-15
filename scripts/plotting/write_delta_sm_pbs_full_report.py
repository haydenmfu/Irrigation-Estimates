#!/usr/bin/env python3
"""Create plots and a full report for the HUC8 SMAP-L3 delta-SM PBS run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
plt.style.use(str(PROJECT / "research_report.mplstyle"))
DELTA_RUN = (
    PROJECT
    / "Week 7"
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "diagnostics"
    / "huc8_smap_l3_delta_sm_pbs_20260708_192928"
)
ABS_RUN = (
    PROJECT
    / "Week 6"
    / "outputs"
    / "fresh_4week_sequential_pbs_N100"
    / "pbs4week_absoluteSM_mlpSatSM_threshold_20260706_152742"
)
RESAMPLE_SUMMARY = (
    PROJECT
    / "Week 7"
    / "data"
    / "daily_delta_resample_rerun_N100"
    / "extracted"
    / "week7_daily_deltaSM_resample_rerun_N100_20260709"
    / "local_summary"
)
VIC_DAILY = (
    PROJECT
    / "Week 3"
    / "data"
    / "VIC_basin0_outputs"
    / "dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv"
)
SMAP_ENDPOINTS = (
    PROJECT
    / "Week 7"
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_weekly_vic_aligned"
    / "satellite_weekly"
    / "huc8_smap_l3_endpoint_observations.csv"
)
CDL_SUMMARY = (
    PROJECT
    / "Week 7"
    / "outputs"
    / "delta_sm_pre_pbs"
    / "huc8_10180009_cdl_crop30_power_rain_partial_current"
    / "huc8_smap_cell_cropland_summary.csv"
)
REPORT_PATH = DELTA_RUN / "DELTA_SM_PBS_FULL_REPORT.md"

CROP_COLOR = "#365C8D"
CONTROL_COLOR = "#1F9E89"
ABS_COLOR = "#D8576B"
PRIOR_COLOR = "#5C6773"
PRECIP_COLOR = "#9AB9D6"
MIXED_COLOR = "#6D5ACF"


def rel(path):
    path = Path(path)
    if path.parent == DELTA_RUN:
        return f"./{path.name}"
    return path.as_posix()


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def endpoint_sm_series(pairs, mask_class):
    subset = pairs[pairs["mask_class"].eq(mask_class)].copy()
    rows = []
    for endpoint, date_col, actual_col, sm_col, offset_col in [
        ("t0", "target_t0", "sat_t0", "SMAP_SM_t0", "sat_t0_offset_days"),
        ("t1", "target_t1", "sat_t1", "SMAP_SM_t1", "sat_t1_offset_days"),
    ]:
        for (window_id, target_date), group in subset.groupby(["window_id", date_col]):
            rows.append(
                {
                    "window_id": int(window_id),
                    "endpoint": endpoint,
                    "date": pd.Timestamp(target_date),
                    "soil_moisture": weighted_mean(group[sm_col], group["overlap_area_km2"]),
                    "n_pairs": int(len(group)),
                    "mean_abs_offset_days": float(group[offset_col].abs().mean()),
                    "actual_dates": ";".join(sorted(group[actual_col].astype(str).unique())),
                    "mask_class": mask_class,
                }
            )
    return pd.DataFrame(rows).sort_values(["date", "endpoint", "window_id"])


def load_inputs():
    delta_win = pd.read_csv(DELTA_RUN / "window_summary.csv")
    abs_win = pd.read_csv(ABS_RUN / "window_summary.csv")
    resample_win = pd.read_csv(RESAMPLE_SUMMARY / "window_summary_combined.csv")
    delta_daily = pd.read_csv(DELTA_RUN / "posterior_daily_irrigation.csv", parse_dates=["date"])
    abs_daily = pd.read_csv(ABS_RUN / "posterior_daily_irrigation.csv", parse_dates=["date"])
    resample_daily = pd.read_csv(RESAMPLE_SUMMARY / "posterior_daily_irrigation_combined.csv", parse_dates=["date"])
    pairs = pd.read_csv(DELTA_RUN / "huc8_smap_l3_weekly_delta_selected_pairs.csv")
    particle_deltas = pd.read_csv(DELTA_RUN / "particle_weekly_delta_sm.csv")
    weights = pd.read_csv(DELTA_RUN / "pbs_particle_weights.csv")
    reference = pd.read_csv(DELTA_RUN / "basin0_open_loop_vic_and_rainfall_weekly_reference.csv")
    vic = pd.read_csv(VIC_DAILY, parse_dates=["time"])
    return delta_win, abs_win, resample_win, delta_daily, abs_daily, resample_daily, pairs, particle_deltas, weights, reference, vic


def plot_pbs_comparison(abs_win, resample_win, abs_daily, resample_daily):
    out = DELTA_RUN / "absolute_vs_delta_sm_pbs_comparison.png"
    fig, axes = plt.subplots(3, 1, figsize=(13.8, 11.2), constrained_layout=True)

    ax = axes[0]
    ax.plot(
        resample_daily["date"],
        resample_daily["prior_mean_irrigation_mm"],
        color=PRIOR_COLOR,
        linewidth=1.8,
        label="Prior mean",
    )
    ax.plot(
        resample_daily["date"],
        resample_daily["posterior_mean_irrigation_mm"],
        color=CROP_COLOR,
        linewidth=2.2,
        marker="o",
        label="Daily Delta-SM posterior",
    )
    ax.plot(
        abs_daily["date"],
        abs_daily["posterior_mean_irrigation_mm"],
        color=ABS_COLOR,
        linewidth=2.0,
        marker="s",
        label="Absolute-SM posterior",
    )
    ax.set_title("Posterior irrigation: daily Delta-SM sequential PBS vs absolute-SM weighting", pad=38)
    ax.set_ylabel("Irrigation (mm/day)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    x = np.arange(len(resample_win))
    width = 0.36
    ax = axes[1]
    ax.bar(x - width / 2, abs_win["effective_sample_size"], width, color=ABS_COLOR, label="Absolute SM")
    ax.bar(x + width / 2, resample_win["effective_sample_size"], width, color=CROP_COLOR, label="Daily Delta SM")
    ax.set_xticks(x)
    ax.set_xticklabels(resample_win["window_id"].astype(int).astype(str))
    ax.set_title("Effective sample size by weekly window", pad=34)
    ax.set_xlabel("Window")
    ax.set_ylabel("ESS")
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    ax = axes[2]
    totals = pd.DataFrame(
        {
            "method": ["Prior", "Absolute-SM posterior", "Daily Delta-SM posterior"],
            "irrigation_mm": [
                resample_daily["prior_mean_irrigation_mm"].sum(),
                abs_daily["posterior_mean_irrigation_mm"].sum(),
                resample_daily["posterior_mean_irrigation_mm"].sum(),
            ],
        }
    )
    colors = [PRIOR_COLOR, ABS_COLOR, CROP_COLOR]
    ax.bar(totals["method"], totals["irrigation_mm"], color=colors)
    ax.set_title("Four-week irrigation totals", pad=18)
    ax.set_ylabel("Total irrigation (mm)")
    for i, value in enumerate(totals["irrigation_mm"]):
        ax.text(i, value + max(totals["irrigation_mm"]) * 0.02, f"{value:.2f}", ha="center")

    for axis in axes:
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d")) if axis is axes[0] else None
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_no_pbs_rainfall_sm(delta_win, pairs, reference, vic):
    out = DELTA_RUN / "no_pbs_rainfall_vs_sm_diagnostic.png"
    crop = endpoint_sm_series(pairs, "cropland_like")
    control = endpoint_sm_series(pairs, "noncropland_control")
    crop.to_csv(DELTA_RUN / "no_pbs_cropland_endpoint_sm.csv", index=False)
    control.to_csv(DELTA_RUN / "no_pbs_control_endpoint_sm.csv", index=False)

    start = pd.Timestamp(delta_win["window_start"].min())
    end = pd.Timestamp(delta_win["window_end"].max())
    vic_daily = vic[(vic["time"] >= start) & (vic["time"] <= end)].copy()

    fig, axes = plt.subplots(3, 1, figsize=(13.0, 11.0), constrained_layout=True)

    ax = axes[0]
    ax_rain = ax.twinx()
    ax_rain.spines["right"].set_visible(True)
    ax_rain.bar(
        vic_daily["time"],
        vic_daily["basin_mean_out_prec"],
        color=PRECIP_COLOR,
        width=0.78,
        alpha=0.55,
        label="basin0 VIC forcing precip",
    )
    ax.plot(vic_daily["time"], vic_daily["basin_mean_vic_soil_moist_layer1_m3m3"], color="black", linestyle="--", linewidth=1.7, label="basin0 VIC open-loop SM")
    ax.plot(crop["date"], crop["soil_moisture"], color=CROP_COLOR, marker="o", linewidth=2.0, label="HUC8 cropland SMAP SM")
    ax.plot(control["date"], control["soil_moisture"], color=CONTROL_COLOR, marker="s", linewidth=2.0, label="HUC8 control SMAP SM")
    ax.set_title("No-PBS check: VIC forcing precipitation and absolute SM time series", pad=38)
    ax.set_ylabel("Soil moisture (m3/m3)")
    ax_rain.set_ylabel("Precipitation (mm/day)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    ax = axes[1]
    ax_rain = ax.twinx()
    ax_rain.spines["right"].set_visible(True)
    x = pd.to_datetime(delta_win["window_end"])
    ax_rain.bar(x, delta_win["basin0_vic_precip_sum_mm"], color=PRECIP_COLOR, width=3.0, alpha=0.55, label="Weekly VIC forcing precip")
    ax.axhline(0, color="#525252", linewidth=0.9)
    ax.plot(x, delta_win["cropland_delta_SM_satellite"], marker="o", linewidth=2.1, color=CROP_COLOR, label="Cropland SMAP delta")
    ax.plot(x, delta_win["control_delta_SM_satellite"], marker="s", linewidth=2.1, color=CONTROL_COLOR, label="Control SMAP delta")
    ax.plot(x, delta_win["basin0_open_loop_delta_SM"], marker="^", linewidth=2.0, color="black", linestyle="--", label="basin0 VIC delta")
    ax.set_title("Weekly delta SM without PBS", pad=38)
    ax.set_ylabel("Delta SM (m3/m3)")
    ax_rain.set_ylabel("Precipitation (mm/week)")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    ax = axes[2]
    ax.axhline(0, color="#525252", linewidth=0.9)
    for y_col, label, color, marker in [
        ("cropland_delta_SM_satellite", "Cropland SMAP", CROP_COLOR, "o"),
        ("control_delta_SM_satellite", "Control SMAP", CONTROL_COLOR, "s"),
        ("basin0_open_loop_delta_SM", "basin0 VIC", "black", "^"),
    ]:
        ax.scatter(delta_win["basin0_vic_precip_sum_mm"], delta_win[y_col], s=82, color=color, marker=marker, label=label)
        for _, row in delta_win.iterrows():
            ax.text(row["basin0_vic_precip_sum_mm"] + 0.12, row[y_col], f"W{int(row.window_id)}", color=color, fontsize=8)
    ax.set_title("VIC forcing precipitation vs weekly delta SM: irrigation-signal sanity check", pad=38)
    ax.set_xlabel("basin0 VIC forcing precipitation (mm/week)")
    ax.set_ylabel("Delta SM (m3/m3)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    for axis in axes[:2]:
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_simple_no_pbs_rainfall_delta_sm(delta_win):
    out = DELTA_RUN / "no_pbs_rainfall_vs_weekly_delta_sm_simple.png"
    fig, ax = plt.subplots(figsize=(9.2, 6.2), constrained_layout=True)
    ax.axhline(0, color="#525252", linewidth=0.9)

    series = [
        ("cropland_delta_SM_satellite", "Cropland SMAP", CROP_COLOR, "o", 110),
        ("control_delta_SM_satellite", "Control SMAP", CONTROL_COLOR, "s", 90),
        ("basin0_open_loop_delta_SM", "basin0 VIC open-loop", "black", "^", 95),
    ]
    for col, label, color, marker, size in series:
        ax.scatter(
            delta_win["basin0_vic_precip_sum_mm"],
            delta_win[col],
            s=size,
            color=color,
            marker=marker,
            label=label,
            zorder=3,
        )

    for _, row in delta_win.iterrows():
        for col, _, color, _, _ in series:
            ax.annotate(
                f"W{int(row.window_id)}",
                (row["basin0_vic_precip_sum_mm"], row[col]),
                xytext=(6, 4),
                textcoords="offset points",
                color=color,
                fontsize=9,
            )

    ax.set_title("No-PBS check: weekly precipitation vs soil-moisture change", pad=38)
    ax.set_xlabel("basin0 VIC forcing precipitation (mm/week)")
    ax.set_ylabel("Weekly delta soil moisture (m3/m3)")
    ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.02))

    note = (
        "Irrigation-like signal would appear as cropland SMAP delta above\n"
        "control/VIC at similar or low VIC forcing precipitation."
    )
    ax.text(
        0.02,
        0.03,
        note,
        transform=ax.transAxes,
        fontsize=9,
        color="#3F4A54",
        bbox={"facecolor": "white", "edgecolor": "#D8DEE6", "alpha": 0.88, "boxstyle": "round,pad=0.35"},
    )

    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_no_pbs_all_smap_readings(delta_win, vic):
    out = DELTA_RUN / "no_pbs_all_smap_l3_readings_vs_rainfall.png"

    obs = pd.read_csv(SMAP_ENDPOINTS, parse_dates=["date"])
    cdl = pd.read_csv(CDL_SUMMARY)
    cdl = cdl[cdl["year"].eq(2020)][["row", "col", "mask_class"]].copy()
    obs = obs.merge(cdl, on=["row", "col"], how="left")
    obs["mask_class"] = obs["mask_class"].fillna("unknown")
    obs["retrieval_recommended_bool"] = obs["retrieval_recommended"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    obs = obs[
        obs["mask_class"].isin(["cropland_like", "noncropland_control"])
        & obs["soil_moisture"].notna()
        & obs["retrieval_recommended_bool"]
    ].copy()

    rows = []
    for (date, period, mask_class), group in obs.groupby(["date", "period", "mask_class"]):
        rows.append(
            {
                "date": date,
                "period": period,
                "mask_class": mask_class,
                "soil_moisture": weighted_mean(group["soil_moisture"], group["overlap_area_km2"]),
                "n_cells": int(len(group)),
                "xdate": date
                + pd.Timedelta(hours=-5 if period == "AM" else 5),
            }
        )
    smap_series = pd.DataFrame(rows).sort_values(["date", "period", "mask_class"])
    smap_series.to_csv(DELTA_RUN / "no_pbs_all_smap_l3_readings_by_date_period_mask.csv", index=False)

    start = pd.Timestamp(delta_win["window_start"].min())
    end = pd.Timestamp(delta_win["window_end"].max())
    vic_daily = vic[(vic["time"] >= start) & (vic["time"] <= end)].copy()

    fig, ax = plt.subplots(figsize=(12.5, 6.4), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.13, top=0.72)
    ax_rain = ax.twinx()
    ax_rain.spines["right"].set_visible(True)
    ax_rain.bar(
        vic_daily["time"],
        vic_daily["basin_mean_out_prec"],
        width=0.78,
        color=PRECIP_COLOR,
        alpha=0.45,
        label="basin0 VIC forcing precip",
        zorder=1,
    )
    ax.plot(
        vic_daily["time"],
        vic_daily["basin_mean_vic_soil_moist_layer1_m3m3"],
        color="black",
        linestyle="--",
        linewidth=1.8,
        label="basin0 VIC open-loop SM",
        zorder=3,
    )

    style = {
        ("cropland_like", "AM"): (CROP_COLOR, "o", "Cropland SMAP AM"),
        ("cropland_like", "PM"): ("#56B4E9", "^", "Cropland SMAP PM"),
        ("noncropland_control", "AM"): (CONTROL_COLOR, "s", "Control SMAP AM"),
        ("noncropland_control", "PM"): ("#7CCBA2", "v", "Control SMAP PM"),
    }
    for (mask_class, period), group in smap_series.groupby(["mask_class", "period"]):
        color, marker, label = style[(mask_class, period)]
        ax.scatter(
            group["xdate"],
            group["soil_moisture"],
            s=58,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            zorder=5,
        )

    for _, row in delta_win.iterrows():
        ax.axvspan(
            pd.Timestamp(row["window_start"]),
            pd.Timestamp(row["window_end"]),
            color="#F6F8FA" if int(row["window_id"]) % 2 else "#FFFFFF",
            alpha=0.5,
            zorder=0,
        )
        ax.text(
            pd.Timestamp(row["window_start"]) + pd.Timedelta(days=0.25),
            ax.get_ylim()[1],
            f"W{int(row.window_id)}",
            va="top",
            ha="left",
            color="#53606D",
            fontsize=9,
        )

    fig.suptitle("No-PBS check: all available SMAP-L3 readings and VIC forcing precipitation", y=0.98)
    ax.set_ylabel("Soil moisture (m3/m3)")
    ax_rain.set_ylabel("Precipitation (mm/day)")
    ax.set_xlabel("Date")
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_rain.get_legend_handles_labels()
    fig.legend(
        lines + lines2,
        labels + labels2,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_particle_likelihood(delta_win, particle_deltas):
    out = DELTA_RUN / "delta_sm_particle_likelihood_diagnostics.png"
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.6), constrained_layout=True)
    axes = axes.ravel()
    for ax, (_, row) in zip(axes, delta_win.iterrows()):
        sub = particle_deltas[particle_deltas["window_id"].eq(row["window_id"])]
        ax.hist(sub["particle_delta_SM"], bins=22, color="#C7CDD6", edgecolor="white")
        ax.axvline(row["cropland_delta_SM_satellite"], color=CROP_COLOR, linewidth=2.2, label="Cropland target")
        ax.axvline(row["control_delta_SM_satellite"], color=CONTROL_COLOR, linewidth=2.0, linestyle="-.", label="Control")
        ax.axvline(row["basin0_open_loop_delta_SM"], color="black", linewidth=2.0, linestyle="--", label="basin0 open-loop")
        ax.set_title(f"Window {int(row.window_id)}: {row.window_start} to {row.window_end}", pad=12)
        ax.set_xlabel("Particle weekly delta SM (m3/m3)")
        ax.set_ylabel("Particles")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.03))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_satellite_delta_by_mask(pairs):
    out = DELTA_RUN / "satellite_delta_sm_by_mask_boxplots.png"
    keep = pairs[pairs["mask_class"].isin(["cropland_like", "noncropland_control", "mixed", "unknown"])].copy()
    fig, axes = plt.subplots(1, 4, figsize=(14.0, 4.6), sharey=True, constrained_layout=True)
    order = ["cropland_like", "noncropland_control", "mixed", "unknown"]
    colors = [CROP_COLOR, CONTROL_COLOR, MIXED_COLOR, PRIOR_COLOR]
    for ax, window_id in zip(axes, sorted(keep["window_id"].unique())):
        data = [keep[(keep["window_id"].eq(window_id)) & (keep["mask_class"].eq(mask))]["delta_SM_satellite"].dropna() for mask in order]
        box = ax.boxplot(data, tick_labels=["crop", "control", "mixed", "unknown"], patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.axhline(0, color="#525252", linewidth=0.9)
        ax.set_title(f"Window {int(window_id)}", pad=12)
        ax.tick_params(axis="x", rotation=25)
    axes[0].set_ylabel("Satellite delta SM (m3/m3)")
    fig.suptitle("SMAP-L3 cell-level weekly delta SM by CDL mask")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def markdown_table(frame, columns, formats=None):
    formats = formats or {}
    lines = []
    headers = [label for _, label in columns]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---" for _ in headers]) + "|")
    for _, row in frame.iterrows():
        values = []
        for col, _ in columns:
            value = row[col]
            fmt = formats.get(col)
            if fmt and pd.notna(value):
                values.append(fmt.format(value))
            elif pd.isna(value):
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(delta_win, abs_win, delta_daily, abs_daily, plots):
    comparison = delta_win[
        [
            "window_id",
            "window_start",
            "window_end",
            "cropland_delta_SM_satellite",
            "control_delta_SM_satellite",
            "basin0_open_loop_delta_SM",
            "basin0_vic_precip_sum_mm",
            "effective_sample_size",
            "posterior_irrigation_sum_mm",
        ]
    ].merge(
        abs_win[["window_id", "effective_sample_size", "posterior_irrigation_sum_mm"]],
        on="window_id",
        suffixes=("_delta", "_absolute"),
    )
    total_prior = float(delta_daily["prior_mean_irrigation_mm"].sum())
    total_delta = float(delta_daily["posterior_mean_irrigation_mm"].sum())
    total_abs = float(abs_daily["posterior_mean_irrigation_mm"].sum())

    lines = [
        "# HUC8 SMAP-L3 Delta-SM PBS Full Report",
        "",
        "## Objective",
        "",
        "This report documents the 4-week PBS iteration that retained particle soil-moisture and irrigation estimates, but changed the particle likelihood from absolute satellite SM to weekly delta SM.",
        "",
        "The intent was to reduce sensitivity to absolute SMAP/VIC bias and instead weight particles by whether their week-to-week soil-moisture changes match the HUC8 cropland satellite signal.",
        "",
        "## Data And Domain",
        "",
        "- HUC8: `10180009`, Middle North Platte-Scotts Bluff.",
        "- Satellite product: native SMAP L3 enhanced passive soil moisture, `SPL3SMP_E`.",
        "- Raw SMAP-L3 download: `Week 7/data/smap_l3_raw_huc8_weekly_endpoints`.",
        "- Downloaded candidates: 28 granules from 2020-06-29 through 2020-07-30, covering each PBS endpoint plus +/-2 days.",
        "- Extracted observation table: `satellite_weekly/huc8_smap_l3_endpoint_observations.csv` with 12,320 rows.",
        "- CDL split: 2020 CDL crop30 mask from `huc8_smap_cell_cropland_summary.csv`.",
        "- Particle ensemble: existing 4-week N100 VIC ensemble from Week 6.",
        "",
        "## How Delta SM Was Computed",
        "",
        "### VIC Side",
        "",
        "The open-loop reference uses the existing Dry Spottedtail `basin0` daily VIC summary:",
        "",
        "`Week 3/data/VIC_basin0_outputs/dry_spottedtail_creek_vic_basin_daily_summary_for_satellite.csv`",
        "",
        "For each weekly window, VIC delta SM was computed from layer-1 volumetric soil moisture:",
        "",
        "`delta_SM_VIC = basin_mean_vic_soil_moist_layer1_m3m3(t1) - basin_mean_vic_soil_moist_layer1_m3m3(t0)`",
        "",
        "Precipitation context uses the existing basin0 VIC forcing precipitation (`basin_mean_out_prec`) summed over `(t0, t1]`. It is plotted for interpretation but is not included directly in the delta-SM likelihood.",
        "",
        "For particles, the same delta idea was applied to each particle's basin-mean VIC layer-1 SM:",
        "",
        "`delta_SM_particle_i,w = mean_SM_particle_i(t1) - mean_SM_particle_i(t0)`",
        "",
        "This particle delta is the simulated quantity used in the likelihood.",
        "",
        "### Satellite Side",
        "",
        "For each HUC8 SMAP cell and period, the satellite endpoint search selected the nearest same-period SMAP-L3 observation within +/-2 days of the target weekly endpoint. High-quality mode retained finite retrievals with `retrieval_qual_flag` equal to `0` or `8`. This follows the NSIDC SPL3SMP_E Version 6 User Guide, which treats both values as high quality; value `8` is included because a failed freeze/thaw retrieval does not affect the soil-moisture retrieval.",
        "",
        "Cell-level satellite delta was computed as:",
        "",
        "`delta_SM_satellite_cell = SMAP_SM(t1_endpoint) - SMAP_SM(t0_endpoint)`",
        "",
        "The cell deltas were then area-weighted by HUC8 overlap area within each CDL mask class. The PBS likelihood used the area-weighted `cropland_like` delta. The `noncropland_control` delta was retained as a comparison.",
        "",
        "## Particle Weighting",
        "",
        "For each window `w` and particle `i`, the likelihood was Gaussian in delta-SM residual:",
        "",
        "`residual_i,w = delta_SM_satellite_cropland,w - delta_SM_particle_i,w`",
        "",
        "`log L_i,w = -0.5 * (residual_i,w / 0.035)^2`",
        "",
        "The `0.035 m3/m3` scale should be read as a working assumption for this first-pass comparison, not as an independently calibrated weekly delta-SM error model. It was kept close to the prior absolute-SM PBS scale for comparability; a later sensitivity test should rerun the comparison over plausible delta-SM sigmas.",
        "",
        "Weights were updated sequentially by week:",
        "",
        "`posterior_weight_i,w proportional_to prior_weight_i,w * L_i,w`",
        "",
        "The posterior irrigation estimate was not computed from satellite SM directly. It was retained from the particle irrigation input table and averaged with the posterior particle weights.",
        "",
        "## PBS Results",
        "",
        markdown_table(
            delta_win,
            [
                ("window_id", "Window"),
                ("window_start", "Start"),
                ("window_end", "End"),
                ("cropland_delta_SM_satellite", "Crop dSM"),
                ("cropland_n_pairs", "Crop n"),
                ("control_delta_SM_satellite", "Control dSM"),
                ("control_n_pairs", "Control n"),
                ("basin0_open_loop_delta_SM", "VIC dSM"),
                ("basin0_vic_precip_sum_mm", "VIC P mm"),
                ("effective_sample_size", "ESS"),
                ("posterior_irrigation_sum_mm", "Post irr mm"),
            ],
            {
                "cropland_delta_SM_satellite": "{:.4f}",
                "control_delta_SM_satellite": "{:.4f}",
                "basin0_open_loop_delta_SM": "{:.4f}",
                "basin0_vic_precip_sum_mm": "{:.2f}",
                "effective_sample_size": "{:.2f}",
                "posterior_irrigation_sum_mm": "{:.4f}",
            },
        ),
        "",
        f"- Prior total irrigation over the four weeks: `{total_prior:.4f} mm`.",
        f"- Absolute-SM posterior total irrigation: `{total_abs:.4f} mm`.",
        f"- Delta-SM posterior total irrigation: `{total_delta:.4f} mm`.",
        "",
        "The delta-SM run did not collapse onto zero-irrigation particles. Its posterior total is lower than the prior, but it keeps a substantial irrigation signal.",
        "",
        "## Comparison With Absolute-SM Weighting",
        "",
        markdown_table(
            comparison,
            [
                ("window_id", "Window"),
                ("effective_sample_size_absolute", "ESS abs-SM"),
                ("effective_sample_size_delta", "ESS delta-SM"),
                ("posterior_irrigation_sum_mm_absolute", "Abs post irr mm"),
                ("posterior_irrigation_sum_mm_delta", "Delta post irr mm"),
            ],
            {
                "effective_sample_size_absolute": "{:.2f}",
                "effective_sample_size_delta": "{:.2f}",
                "posterior_irrigation_sum_mm_absolute": "{:.4f}",
                "posterior_irrigation_sum_mm_delta": "{:.4f}",
            },
        ),
        "",
        "The absolute-SM run used daily absolute satellite SM with the previous MLP mapping/filtering workflow. It strongly concentrated the posterior weights by windows 3-4 and produced almost zero posterior irrigation. The delta-SM run had healthier ESS values and preserved a nonzero posterior irrigation estimate because it scored weekly changes instead of absolute level agreement.",
        "",
        "## No-PBS Precipitation / SM Check",
        "",
        "The main no-PBS diagnostic asks a simple question: do the raw satellite soil-moisture readings show cropland wetting that is not easily explained by the precipitation already supplied to VIC? The clearest version is a time-series plot with all available high-quality SMAP-L3 aggregate readings, daily basin0 VIC forcing precipitation bars, and the basin0 VIC open-loop SM line. This avoids compressing each week into only an endpoint pair.",
        "",
        "- Window 2 had low basin0 VIC forcing precipitation (`2.98 mm`) but positive cropland delta SM (`+0.0309`), larger than control (`+0.0188`) and basin0 VIC (`+0.0103`).",
        "- Window 3 had modest basin0 VIC forcing precipitation (`5.08 mm`), but cropland delta (`+0.0797`) still exceeded control (`+0.0450`) and basin0 VIC (`+0.0068`).",
        "- Window 4 had no basin0 VIC forcing precipitation and negative satellite and VIC deltas, so it is not evidence for irrigation-like wetting.",
        "",
        "That pattern is consistent with possible nonzero irrigation in at least some windows, especially where cropland increases exceed both control and open-loop VIC. It is not proof: this is only four weekly windows, SMAP endpoint dates can be offset by up to two days, and the precipitation reference is basin0 VIC forcing while the satellite aggregation is HUC8-scale.",
        "",
        "## Plots",
        "",
        f"![No-PBS all SMAP-L3 readings versus VIC forcing precipitation]({rel(plots['no_pbs_all_readings'])})",
        "",
        f"![Abolafia/Rosenzweig-style delta-SM PBS diagnostic]({rel(DELTA_RUN / 'abolafia_rosenzweig_style_delta_sm_pbs_diagnostic.png')})",
        "",
        f"![Delta-SM satellite/VIC/precipitation comparison]({rel(DELTA_RUN / 'huc8_satellite_vic_rainfall_delta_comparison.png')})",
        "",
        f"![Absolute vs delta-SM PBS comparison]({rel(plots['pbs_comparison'])})",
        "",
        f"![Supplemental no-PBS precipitation versus weekly delta SM sanity check]({rel(plots['no_pbs_simple'])})",
        "",
        f"![Supplemental no-PBS precipitation and SM time-series diagnostic]({rel(plots['no_pbs'])})",
        "",
        f"![Particle delta-SM likelihood diagnostics]({rel(plots['particle_likelihood'])})",
        "",
        f"![Satellite delta SM by CDL mask]({rel(plots['mask_boxplots'])})",
        "",
        f"![Particle weights by window]({rel(DELTA_RUN / 'particle_weights_by_window.png')})",
        "",
        f"![Posterior versus prior daily irrigation]({rel(DELTA_RUN / 'posterior_vs_prior_daily_irrigation_delta_sm_pbs.png')})",
        "",
        "## Output Tables",
        "",
        "- `huc8_smap_l3_weekly_delta_selected_pairs.csv`: selected same-period endpoint pairs per HUC8 cell.",
        "- `huc8_smap_l3_weekly_delta_by_mask.csv`: area-weighted satellite deltas by CDL mask.",
        "- `particle_weekly_delta_sm.csv`: weekly delta SM for each particle.",
        "- `pbs_particle_weights.csv`: sequential posterior particle weights.",
        "- `posterior_daily_irrigation.csv`: posterior and prior daily irrigation estimates.",
        "- `window_summary.csv`: compact window-level results.",
    ]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    delta_win, abs_win, resample_win, delta_daily, abs_daily, resample_daily, pairs, particle_deltas, weights, reference, vic = load_inputs()
    plots = {
        "pbs_comparison": plot_pbs_comparison(abs_win, resample_win, abs_daily, resample_daily),
        "no_pbs": plot_no_pbs_rainfall_sm(delta_win, pairs, reference, vic),
        "no_pbs_simple": plot_simple_no_pbs_rainfall_delta_sm(delta_win),
        "no_pbs_all_readings": plot_no_pbs_all_smap_readings(delta_win, vic),
        "particle_likelihood": plot_particle_likelihood(delta_win, particle_deltas),
        "mask_boxplots": plot_satellite_delta_by_mask(pairs),
    }
    write_report(delta_win, abs_win, delta_daily, abs_daily, plots)
    print(REPORT_PATH)
    for path in plots.values():
        print(path)


if __name__ == "__main__":
    main()
