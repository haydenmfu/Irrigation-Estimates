#!/usr/bin/env python3
"""Create an Abolafia-Rosenzweig Figure-2-style PBS diagnostic.

Default inputs target the Week 6 AdaPBS-style pooled run, but the script also
accepts generic CSVs with particle soil moisture, particle irrigation,
observation soil moisture, optional open-loop soil moisture, optional
validation irrigation, and PBS window metadata.

The figure has two panels:

  (a) particle soil-moisture trajectories colored by PBS weight, weighted
      ensemble mean, satellite observations, optional open-loop VIC.
  (b) posterior irrigation estimate with posterior uncertainty and optional
      validation/reference irrigation.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STYLE = PROJECT / "config" / "research_report.mplstyle"
if STYLE.exists():
    plt.style.use(str(STYLE))
DEFAULT_RUN = (
    PROJECT
    / "outputs"
    / "adapbs_4week_N100_pooled"
    / "adapbs4week_N100plusN100_pooled_20260706_153939"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot a two-panel PBS diagnostic in the style of Abolafia-Rosenzweig Figure 2."
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--particle-sm", type=Path, default=None)
    parser.add_argument("--particle-irrigation", type=Path, default=None)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--observations", type=Path, default=None)
    parser.add_argument("--window-summary", type=Path, default=None)
    parser.add_argument("--open-loop-sm", type=Path, default=None)
    parser.add_argument("--precipitation", type=Path, default=None)
    parser.add_argument("--validation-irrigation", type=Path, default=None)
    parser.add_argument("--out-prefix", type=Path, default=None)

    parser.add_argument("--particle-id-col", default="particle")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--particle-sm-col", default=None)
    parser.add_argument("--particle-irrigation-col", default="irrigation_mm_day")
    parser.add_argument("--weight-col", default="weight")
    parser.add_argument("--obs-sm-col", default=None)
    parser.add_argument("--open-loop-sm-col", default="soil_moisture")
    parser.add_argument("--precipitation-col", default="precip_mm")
    parser.add_argument("--precipitation-label", default="Precipitation")
    parser.add_argument("--precipitation-rug-fraction", type=float, default=0.16)
    parser.add_argument("--precipitation-color", default="#56B4E9")
    parser.add_argument("--validation-irrigation-col", default="irrigation_mm")

    parser.add_argument("--zoom-start", default=None)
    parser.add_argument("--zoom-end", default=None)
    parser.add_argument("--color-weight-window", type=int, default=None)
    parser.add_argument("--max-particles", type=int, default=None, help="Optional cap for overplotting; defaults to all particles.")
    parser.add_argument("--uncertainty", choices=["quantile", "std"], default="quantile")
    parser.add_argument(
        "--cmap",
        default="viridis_r",
        help="Colormap for particle weight. Default is viridis_r: light/yellow=low weight, dark/purple=high weight. "
        "A multi-hue map like this reads with much more contrast than a single-hue map (e.g. Purples) once "
        "you have ~100 overlapping lines, since the eye distinguishes hue changes far more easily than "
        "small luminance differences within one hue.",
    )
    parser.add_argument("--cmap-min", type=float, default=0.0, help="Lower fraction of the colormap to use.")
    parser.add_argument("--cmap-max", type=float, default=1.0, help="Upper fraction of the colormap to use.")
    parser.add_argument(
        "--weight-color-scale",
        choices=["auto", "linear", "log"],
        default="auto",
        help="Color normalization for particle weights. Auto uses log scaling for highly skewed weights.",
    )
    parser.add_argument(
        "--weight-color-scope",
        choices=["window", "global"],
        default="window",
        help="Use an adaptive color range within each PBS window, or one global range for the full plot.",
    )
    parser.add_argument(
        "--weight-vmax-percentile",
        type=float,
        default=99.0,
        help="Robust upper percentile for the particle-weight color range.",
    )
    parser.add_argument(
        "--weight-floor-fraction",
        type=float,
        default=1e-3,
        help="For log color scaling, lower color limit is at least this fraction of the robust upper limit.",
    )
    parser.add_argument(
        "--min-particle-alpha",
        type=float,
        default=0.50,
        help="Kept close to max-particle-alpha by default so opacity doesn't also fade low-weight lines to "
        "invisibility on top of the colormap fade — color alone should carry the weight signal.",
    )
    parser.add_argument("--max-particle-alpha", type=float, default=0.80)
    parser.add_argument(
        "--alpha-gamma",
        type=float,
        default=1.0,
        help="Gamma applied to scaled particle weight before mapping to line opacity. 1.0=linear; values <1 "
        "compound with the colormap fade and disproportionately hide low-weight particles.",
    )
    parser.add_argument("--min-particle-linewidth", type=float, default=0.55)
    parser.add_argument("--max-particle-linewidth", type=float, default=1.15)
    parser.add_argument(
        "--linewidth-gamma",
        type=float,
        default=1.0,
        help="Gamma applied to scaled particle weight before mapping to line width.",
    )
    parser.add_argument(
        "--highlight-min-scaled-weight",
        type=float,
        default=0.85,
        help="Redraw particles above this scaled weight as a darker/thicker highlight pass. Raised from 0.72 "
        "so only genuinely dominant particles get the highlight treatment, matching the reference figure's "
        "sparse, distinct high-weight trajectories rather than a broad highlighted band.",
    )
    parser.add_argument("--highlight-linewidth-boost", type=float, default=0.55)
    parser.add_argument("--highlight-alpha", type=float, default=0.95)
    parser.add_argument("--sm-units-label", default="Soil Moisture (cm3 cm-3)")
    parser.add_argument("--irrigation-units-label", default="Daily Irrigation (mm)")
    parser.add_argument("--title", default="PBS Diagnostic")
    return parser.parse_args()


def first_existing(*paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path)
    return None


def default_paths(run_dir):
    matches = first_existing(
        run_dir / "adapbs_particle_satellite_matches.csv",
        run_dir / "pbs_particle_satellite_matches.csv",
    )
    weights = first_existing(
        run_dir / "adapbs_particle_weights.csv",
        run_dir / "pbs_particle_weights.csv",
    )
    irrigation = run_dir / "posterior_daily_irrigation.csv"
    window_summary = run_dir / "window_summary.csv"
    return matches, weights, irrigation, window_summary


def build_default_particle_irrigation_from_metadata(run_dir):
    meta_path = run_dir / "pbs_run_metadata.json"
    if not meta_path.exists():
        return None
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    out_path = run_dir / "particle_irrigation_inputs_for_abolafia_diagnostic.csv"
    if out_path.exists():
        return out_path

    frames = []
    if metadata.get("round1_root") and metadata.get("round2_root"):
        for proposal_round, offset_key, root_key in [
            (1, 0, "round1_root"),
            (2, 100000, "round2_root"),
        ]:
            root = Path(metadata[root_key])
            path = root / "particle_irrigation_inputs.csv"
            if not path.exists():
                return None
            frame = pd.read_csv(path)
            frame["original_particle"] = frame["particle"].astype(int)
            frame["proposal_round"] = proposal_round
            frame["particle"] = frame["original_particle"] + offset_key
            frames.append(frame)
    elif metadata.get("vic_input_output_directory"):
        path = Path(metadata["vic_input_output_directory"]) / "particle_irrigation_inputs.csv"
        if path.exists():
            frames.append(pd.read_csv(path))

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_path, index=False)
    return out_path


def infer_col(frame, candidates, label):
    for col in candidates:
        if col in frame.columns:
            return col
    raise ValueError(f"Could not infer {label}. Tried {candidates}; columns are {list(frame.columns)}")


def normalize_window_weights(weights, particle_col, weight_col):
    weights = weights.copy()
    if "window_id" not in weights.columns:
        weights["window_id"] = 1
    weights[weight_col] = pd.to_numeric(weights[weight_col], errors="coerce").fillna(0.0)
    sums = weights.groupby("window_id")[weight_col].transform("sum")
    weights["normalized_weight"] = np.where(sums > 0, weights[weight_col] / sums, 1.0 / weights.groupby("window_id")[particle_col].transform("count"))
    return weights


def load_particle_sm(path, args):
    frame = pd.read_csv(path)
    date_col = args.date_col if args.date_col in frame.columns else infer_col(frame, ["date", "time"], "date column")
    particle_col = args.particle_id_col if args.particle_id_col in frame.columns else infer_col(frame, ["particle", "particle_id"], "particle id column")
    if args.particle_sm_col:
        sm_col = args.particle_sm_col
    else:
        sm_col = infer_col(
            frame,
            ["mlp_mapped_vic_m3m3", "soil_moisture", "soil_moisture_m3m3", "vic_surface_sm", "predicted_vic_cell_m3m3"],
            "particle soil moisture column",
        )
    out = frame[[date_col, particle_col, sm_col]].copy()
    out.columns = ["date", "particle", "soil_moisture"]
    out["date"] = pd.to_datetime(out["date"])
    out["soil_moisture"] = pd.to_numeric(out["soil_moisture"], errors="coerce")
    return (
        out.dropna(subset=["soil_moisture"])
        .groupby(["date", "particle"], as_index=False)
        .agg(soil_moisture=("soil_moisture", "mean"))
    )


def load_observations(path, args):
    frame = pd.read_csv(path)
    date_col = args.date_col if args.date_col in frame.columns else infer_col(frame, ["date", "time"], "date column")
    if args.obs_sm_col:
        sm_col = args.obs_sm_col
    else:
        sm_col = infer_col(frame, ["satellite_m3m3", "soil_moisture", "soil_moisture_m3m3", "smap_sm_raw"], "observation soil moisture column")
    out = frame[[date_col, sm_col]].copy()
    out.columns = ["date", "soil_moisture"]
    out["date"] = pd.to_datetime(out["date"])
    out["soil_moisture"] = pd.to_numeric(out["soil_moisture"], errors="coerce")
    return (
        out.dropna(subset=["soil_moisture"])
        .drop_duplicates()
        .groupby("date", as_index=False)
        .agg(soil_moisture=("soil_moisture", "mean"))
    )


def load_particle_irrigation(path, args, weights):
    frame = pd.read_csv(path)
    if {"posterior_mean_irrigation_mm", "prior_mean_irrigation_mm"}.issubset(frame.columns):
        return None, load_posterior_daily_irrigation(frame)

    date_col = args.date_col if args.date_col in frame.columns else infer_col(frame, ["date", "time"], "date column")
    particle_col = args.particle_id_col if args.particle_id_col in frame.columns else infer_col(frame, ["particle", "particle_id"], "particle id column")
    irr_col = args.particle_irrigation_col if args.particle_irrigation_col in frame.columns else infer_col(frame, ["irrigation_mm", "irrigation_mm_day"], "particle irrigation column")
    out = frame[[date_col, particle_col, irr_col]].copy()
    out.columns = ["date", "particle", "irrigation_mm"]
    out["date"] = pd.to_datetime(out["date"])
    out["irrigation_mm"] = pd.to_numeric(out["irrigation_mm"], errors="coerce").fillna(0.0)
    if "window_id" not in out.columns and {"window_id", "window_start", "window_end"}.issubset(weights.columns):
        windows = (
            weights[["window_id", "window_start", "window_end"]]
            .drop_duplicates()
            .sort_values("window_id")
            .copy()
        )
        windows["window_start"] = pd.to_datetime(windows["window_start"])
        windows["window_end"] = pd.to_datetime(windows["window_end"])
        out = assign_windows_by_date(out, windows)
    return out, summarize_weighted_irrigation(out, weights)


def load_posterior_daily_irrigation(frame):
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"])
    rename = {}
    if "posterior_mean_irrigation_mm" in out.columns:
        rename["posterior_mean_irrigation_mm"] = "pbs_mean"
    if "prior_mean_irrigation_mm" in out.columns:
        rename["prior_mean_irrigation_mm"] = "prior_mean"
    out = out.rename(columns=rename)
    if "pbs_mean" not in out.columns:
        raise ValueError("Posterior irrigation file must include posterior_mean_irrigation_mm or particle irrigation rows.")
    if "lower" not in out.columns:
        out["lower"] = np.nan
    if "upper" not in out.columns:
        out["upper"] = np.nan
    return out[["date", "pbs_mean", "lower", "upper"] + (["prior_mean"] if "prior_mean" in out.columns else [])]


def assign_windows_by_date(frame, windows):
    out = frame.copy()
    out["window_id"] = np.nan
    for _, row in windows.iterrows():
        start = pd.Timestamp(row["window_start"])
        end = pd.Timestamp(row["window_end"])
        out.loc[(out["date"] >= start) & (out["date"] <= end), "window_id"] = int(row["window_id"])
    return out


def weighted_quantile(values, weights, quantiles):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    values = values[mask]
    weights = weights[mask]
    if len(values) == 0:
        return [np.nan for _ in quantiles]
    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) - 0.5 * weights
    cdf = np.clip(cdf, 0, 1)
    return np.interp(quantiles, cdf, values).tolist()


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    mask = np.isfinite(values) & np.isfinite(weights)
    values = values[mask]
    weights = weights[mask]
    total = weights.sum()
    if len(values) == 0 or total <= 0:
        return np.nan
    return float(np.sum(values * weights) / total)


def weighted_std(values, weights):
    mean = weighted_mean(values, weights)
    if not np.isfinite(mean):
        return np.nan
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    if total <= 0:
        return np.nan
    return float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))


def summarize_weighted_sm(particle_sm, weights, windows):
    sm = assign_windows_by_date(particle_sm, windows)
    merged = sm.merge(weights[["particle", "window_id", "normalized_weight"]], on=["particle", "window_id"], how="left")
    merged["normalized_weight"] = merged["normalized_weight"].fillna(0.0)
    summary = (
        merged.groupby("date", as_index=False)
        .apply(lambda g: pd.Series({"weighted_mean": weighted_mean(g["soil_moisture"], g["normalized_weight"])}), include_groups=False)
    )
    return summary, merged


def summarize_weighted_irrigation(particle_irrigation, weights):
    merged = particle_irrigation.merge(weights[["particle", "window_id", "normalized_weight"]], on=["particle", "window_id"], how="left")
    merged["normalized_weight"] = merged["normalized_weight"].fillna(0.0)
    rows = []
    for date, group in merged.groupby("date"):
        mean = weighted_mean(group["irrigation_mm"], group["normalized_weight"])
        if group["normalized_weight"].sum() <= 0:
            lower = upper = np.nan
        else:
            lower, upper = weighted_quantile(group["irrigation_mm"], group["normalized_weight"], [0.10, 0.90])
        std = weighted_std(group["irrigation_mm"], group["normalized_weight"])
        rows.append({"date": date, "pbs_mean": mean, "lower": lower, "upper": upper, "std": std})
    return pd.DataFrame(rows).sort_values("date")


def load_windows(path, start_date, end_date, window_days=7):
    if path and Path(path).exists():
        frame = pd.read_csv(path)
        needed = {"window_id", "window_start", "window_end"}
        if needed.issubset(frame.columns):
            out = frame[["window_id", "window_start", "window_end"]].drop_duplicates().copy()
            out["window_start"] = pd.to_datetime(out["window_start"])
            out["window_end"] = pd.to_datetime(out["window_end"])
            return out.sort_values("window_id")
    dates = pd.date_range(start_date, end_date, freq=f"{window_days}D")
    rows = []
    cur = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    idx = 1
    while cur <= end:
        win_end = min(cur + pd.Timedelta(days=window_days - 1), end)
        rows.append({"window_id": idx, "window_start": cur, "window_end": win_end})
        cur = win_end + pd.Timedelta(days=1)
        idx += 1
    return pd.DataFrame(rows)


def load_simple_series(path, value_col):
    if path is None or not Path(path).exists():
        return None
    frame = pd.read_csv(path)
    date_col = "date" if "date" in frame.columns else infer_col(frame, ["time"], "date column")
    val_col = value_col if value_col in frame.columns else infer_col(frame, [value_col, "soil_moisture", "irrigation_mm"], "value column")
    out = frame[[date_col, val_col]].copy()
    out.columns = ["date", "value"]
    out["date"] = pd.to_datetime(out["date"])
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out.dropna(subset=["value"]).groupby("date", as_index=False).agg(value=("value", "mean"))


def build_weight_norm(plot_weights, args):
    weights = pd.to_numeric(pd.Series(plot_weights), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    positive = weights[weights > 0]
    if positive.empty:
        return Normalize(vmin=0.0, vmax=1.0, clip=True), 0.0, 1.0, "linear"

    percentile = float(np.clip(args.weight_vmax_percentile, 50.0, 100.0))
    vmax = float(np.nanpercentile(positive, percentile))
    vmax = max(vmax, float(positive.max()) * 1e-6, np.finfo(float).tiny)

    dynamic_range = float(positive.max() / max(float(positive[positive > 0].min()), np.finfo(float).tiny))
    use_log = args.weight_color_scale == "log" or (args.weight_color_scale == "auto" and dynamic_range > 100.0)
    if use_log:
        floor_fraction = max(float(args.weight_floor_fraction), np.finfo(float).eps)
        vmin = max(vmax * floor_fraction, np.finfo(float).tiny)
        if vmin >= vmax:
            vmin = max(vmax * 0.01, np.finfo(float).tiny)
        return LogNorm(vmin=vmin, vmax=vmax, clip=True), vmin, vmax, "log"

    return Normalize(vmin=0.0, vmax=vmax, clip=True), 0.0, vmax, "linear"


def scaled_weight(weight, norm, norm_vmin, color_scale):
    weight = float(weight) if np.isfinite(weight) else 0.0
    if color_scale == "log":
        if weight <= 0:
            return 0.0
        value = max(weight, norm_vmin)
    else:
        value = max(weight, 0.0)
    scaled = norm(value)
    if np.ma.is_masked(scaled):
        return 0.0
    return float(np.clip(scaled, 0.0, 1.0))


def particle_alpha(weight, norm, norm_vmin, color_scale, args):
    lo = float(np.clip(args.min_particle_alpha, 0.0, 1.0))
    hi = float(np.clip(args.max_particle_alpha, lo, 1.0))
    gamma = max(float(args.alpha_gamma), np.finfo(float).eps)
    return lo + (hi - lo) * (scaled_weight(weight, norm, norm_vmin, color_scale) ** gamma)


def particle_linewidth(weight, norm, norm_vmin, color_scale, args):
    lo = max(float(args.min_particle_linewidth), 0.01)
    hi = max(float(args.max_particle_linewidth), lo)
    gamma = max(float(args.linewidth_gamma), np.finfo(float).eps)
    return lo + (hi - lo) * (scaled_weight(weight, norm, norm_vmin, color_scale) ** gamma)


def truncate_colormap(cmap, lower, upper, n=256):
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, lower + 1e-6, 1.0))
    colors = cmap(np.linspace(lower, upper, n))
    return LinearSegmentedColormap.from_list(f"{cmap.name}_truncated", colors)


def plot_diagnostic(args):
    run_dir = args.run_dir.resolve()
    default_matches, default_weights, default_irrigation, default_windows = default_paths(run_dir)
    particle_sm_path = args.particle_sm or default_matches
    observations_path = args.observations or default_matches
    weights_path = args.weights or default_weights
    particle_irrigation_path = args.particle_irrigation or build_default_particle_irrigation_from_metadata(run_dir) or default_irrigation
    window_path = args.window_summary or default_windows

    if particle_sm_path is None or weights_path is None or observations_path is None:
        raise FileNotFoundError("Could not resolve required particle SM, weights, and observation files.")

    particle_sm = load_particle_sm(particle_sm_path, args)
    observations = load_observations(observations_path, args)
    weights = pd.read_csv(weights_path)
    weights = weights.rename(columns={args.particle_id_col: "particle", args.weight_col: "weight"})
    weights = normalize_window_weights(weights, "particle", "weight")

    start = particle_sm["date"].min()
    end = particle_sm["date"].max()
    windows = load_windows(window_path, start, end)
    sm_summary, particle_sm_weighted = summarize_weighted_sm(particle_sm, weights, windows)

    if particle_irrigation_path is not None and Path(particle_irrigation_path).exists():
        particle_irr, irr_summary = load_particle_irrigation(particle_irrigation_path, args, weights)
    else:
        particle_irr, irr_summary = None, pd.DataFrame(columns=["date", "pbs_mean", "lower", "upper"])

    zoom_start = pd.Timestamp(args.zoom_start) if args.zoom_start else start
    zoom_end = pd.Timestamp(args.zoom_end) if args.zoom_end else end
    zoom_sm = particle_sm_weighted[(particle_sm_weighted["date"] >= zoom_start) & (particle_sm_weighted["date"] <= zoom_end)].copy()
    zoom_obs = observations[(observations["date"] >= zoom_start) & (observations["date"] <= zoom_end)].copy()
    zoom_mean = sm_summary[(sm_summary["date"] >= zoom_start) & (sm_summary["date"] <= zoom_end)].copy()

    # Color each particle by the weight it held *within its own assimilation
    # window*, not by a single snapshot window applied to the whole
    # trajectory. This is what produces the paper's shifting purple->yellow
    # patches across successive ~week-long PBS windows, instead of one flat
    # color per particle for the entire period.
    zoom_sm = zoom_sm.rename(columns={"normalized_weight": "plot_weight"})
    zoom_sm["plot_weight"] = zoom_sm["plot_weight"].fillna(0.0)

    if args.color_weight_window:
        # Optional override: force a single window's weights (old behavior).
        color_weights = weights[weights["window_id"].eq(args.color_weight_window)][["particle", "normalized_weight"]]
        color_weights = color_weights.rename(columns={"normalized_weight": "plot_weight"})
        zoom_sm = zoom_sm.drop(columns=["plot_weight"]).merge(color_weights, on="particle", how="left")
        zoom_sm["plot_weight"] = zoom_sm["plot_weight"].fillna(0.0)

    if args.max_particles and zoom_sm["particle"].nunique() > args.max_particles:
        keep = (
            zoom_sm.groupby("particle")["plot_weight"].max().sort_values(ascending=False).head(args.max_particles).index
        )
        zoom_sm = zoom_sm[zoom_sm["particle"].isin(keep)].copy()

    open_loop = load_simple_series(args.open_loop_sm, args.open_loop_sm_col)
    precipitation = load_simple_series(args.precipitation, args.precipitation_col)
    validation_irr = load_simple_series(args.validation_irrigation, args.validation_irrigation_col)

    if args.out_prefix:
        out_prefix = args.out_prefix
    else:
        out_prefix = run_dir / "abolafia_rosenzweig_style_pbs_diagnostic"
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.4), sharex=False, constrained_layout=True)
    ax = axes[0]
    # Default Purples: light purple = low weight, dark purple = high weight.
    # Highly skewed weight distributions use an adaptive log scale. By default
    # that scale is recomputed within each PBS window so the color range is
    # spent on meaningful posterior mass in every window, not only in the
    # window containing the global maximum weight.
    cmap = truncate_colormap(plt.get_cmap(args.cmap), args.cmap_min, args.cmap_max)

    global_norm_info = build_weight_norm(zoom_sm["plot_weight"], args)
    if args.weight_color_scope == "window":
        norm_info_by_window = {
            window_id: build_weight_norm(group["plot_weight"], args)
            for window_id, group in zoom_sm.groupby("window_id")
        }
    else:
        norm_info_by_window = {}

    def norm_info_for(window_id):
        return norm_info_by_window.get(window_id, global_norm_info)

    # Build one line segment per (particle, window). Stitch consecutive
    # windows together (prepend the previous window's last point) so the
    # trajectory reads as continuous even though its color changes at each
    # window boundary. Then draw all segments sorted by ascending scaled weight
    # so high-weight (dark/thick) particles are drawn last and stay visible on top,
    # instead of being buried under a stack of low-weight lines.
    segments = []
    for particle, pgroup in zoom_sm.groupby("particle"):
        pgroup = pgroup.sort_values(["window_id", "date"])
        prev_last = None
        for window_id, wgroup in pgroup.groupby("window_id"):
            wgroup = wgroup.sort_values("date")
            dates = wgroup["date"].tolist()
            values = wgroup["soil_moisture"].tolist()
            window_id_value = int(window_id) if pd.notna(window_id) else 1
            weight = float(wgroup["plot_weight"].iloc[0])
            norm, norm_vmin, _, color_scale = norm_info_for(window_id_value)
            color_value = scaled_weight(weight, norm, norm_vmin, color_scale)
            if prev_last is not None and prev_last[0] < dates[0]:
                dates = [prev_last[0]] + dates
                values = [prev_last[1]] + values
            segments.append((color_value, window_id_value, weight, dates, values))
            prev_last = (wgroup["date"].iloc[-1], wgroup["soil_moisture"].iloc[-1])

    segments.sort(key=lambda seg: seg[0])
    for color_value, window_id, weight, dates, values in segments:
        norm, norm_vmin, _, color_scale = norm_info_for(window_id)
        line_width = particle_linewidth(weight, norm, norm_vmin, color_scale, args)
        ax.plot(
            dates,
            values,
            color=cmap(color_value),
            linewidth=line_width,
            alpha=particle_alpha(weight, norm, norm_vmin, color_scale, args),
            zorder=2,
        )

    highlight_threshold = float(np.clip(args.highlight_min_scaled_weight, 0.0, 1.0))
    for color_value, window_id, weight, dates, values in segments:
        if color_value < highlight_threshold:
            continue
        norm, norm_vmin, _, color_scale = norm_info_for(window_id)
        line_width = particle_linewidth(weight, norm, norm_vmin, color_scale, args) + max(args.highlight_linewidth_boost, 0.0)
        ax.plot(
            dates,
            values,
            color=cmap(min(1.0, max(color_value, 0.92))),
            linewidth=line_width,
            alpha=float(np.clip(args.highlight_alpha, 0.0, 1.0)),
            zorder=4,
        )

    ax.plot(zoom_mean["date"], zoom_mean["weighted_mean"], color="black", linewidth=2.0, label="Weighted ensemble mean", zorder=6)
    ax.scatter(zoom_obs["date"], zoom_obs["soil_moisture"], color="#D62728", s=34, marker="o", label="Satellite SM", zorder=7)
    if open_loop is not None:
        sub = open_loop[(open_loop["date"] >= zoom_start) & (open_loop["date"] <= zoom_end)]
        ax.plot(sub["date"], sub["value"], color="black", linewidth=1.7, linestyle="--", label="Open-loop/no-irrigation VIC")

    rain_axis = None
    if precipitation is not None:
        rain = precipitation[(precipitation["date"] >= zoom_start) & (precipitation["date"] <= zoom_end)].copy()
        rain["value"] = rain["value"].clip(lower=0.0)
        rain = rain[rain["value"] > 0]
        if not rain.empty and rain["value"].max() > 0:
            rain_color = args.precipitation_color
            rain_axis = ax.twinx()
            rain_axis.vlines(
                rain["date"],
                0.0,
                rain["value"],
                color=rain_color,
                linewidth=2.2,
                alpha=0.78,
                label=args.precipitation_label,
                zorder=5,
            )
            rain_axis.set_ylim(0.0, float(rain["value"].max()) * 1.18)
            rain_axis.set_ylabel("Precipitation (mm/day)", color=rain_color)
            rain_axis.tick_params(axis="y", colors=rain_color)
            rain_axis.spines["right"].set_visible(True)
            rain_axis.spines["right"].set_color(rain_color)
            rain_axis.spines["top"].set_visible(False)
            rain_axis.grid(False)

    for _, row in windows.iterrows():
        if row["window_start"] >= zoom_start and row["window_start"] <= zoom_end:
            ax.axvline(row["window_start"], color="#8C8C8C", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.text(0.012, 0.94, "A", transform=ax.transAxes, fontsize=15, fontweight="bold", family="sans-serif")
    ax.set_ylabel(args.sm_units_label)
    ax.set_title(args.title, pad=34)
    ax.grid(alpha=0.20)
    ax.spines[["top", "right"]].set_visible(False)
    # Anchor the legend just above the axes (bbox y=1.02, in axes-fraction
    # coords) rather than loc="best" inside the plot area. constrained_layout
    # accounts for legend artists placed this way and reserves space for it,
    # so it never sits on top of the particle lines and stays clear of the
    # colorbar since it's centered rather than right-anchored.
    handles, labels = ax.get_legend_handles_labels()
    if rain_axis is not None:
        rain_handles, rain_labels = rain_axis.get_legend_handles_labels()
        handles += rain_handles
        labels += rain_labels
    ax.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        columnspacing=1.4,
        handlelength=1.8,
    )
    norm_infos_for_summary = list(norm_info_by_window.values()) if norm_info_by_window else [global_norm_info]
    summary_color_scale = ",".join(sorted({info[3] for info in norm_infos_for_summary}))
    summary_norm_vmin = min(info[1] for info in norm_infos_for_summary)
    summary_norm_vmax = max(info[2] for info in norm_infos_for_summary)

    if args.weight_color_scope == "window":
        cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap), ax=ax, pad=0.075 if rain_axis is not None else 0.014, fraction=0.035)
        cbar_label = "Relative Particle Weight"
    else:
        norm, _, _, _ = global_norm_info
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.075 if rain_axis is not None else 0.014, fraction=0.035)
        cbar_label = "Particle Weight"
    if "log" in summary_color_scale:
        cbar_label += " (log color scale)"
    if args.weight_color_scope == "window":
        cbar_label += ", within window"
    cbar.set_label(cbar_label, fontsize=10.5)
    cbar.ax.tick_params(labelsize=9)

    ax2 = axes[1]
    irr_summary = irr_summary.copy()
    if not irr_summary.empty:
        irr_summary["date"] = pd.to_datetime(irr_summary["date"])
        if args.uncertainty == "std" and "std" in irr_summary.columns:
            lower = irr_summary["pbs_mean"] - irr_summary["std"]
            upper = irr_summary["pbs_mean"] + irr_summary["std"]
        else:
            lower = irr_summary["lower"]
            upper = irr_summary["upper"]
        if np.isfinite(lower).any() and np.isfinite(upper).any():
            ax2.fill_between(irr_summary["date"], lower, upper, color="#BDBDBD", alpha=0.42, label="PBS uncertainty")
        ax2.plot(irr_summary["date"], irr_summary["pbs_mean"], color="black", linewidth=2.2, label="PBS irrigation")

    if validation_irr is not None:
        ax2.plot(validation_irr["date"], validation_irr["value"], color="#D62728", linewidth=1.8, label="Reference irrigation")

    for _, row in windows.iterrows():
        ax2.axvline(row["window_start"], color="#8C8C8C", linestyle="--", linewidth=1.0, alpha=0.75)
    ax2.text(0.012, 0.90, "B", transform=ax2.transAxes, fontsize=15, fontweight="bold", family="sans-serif")
    ax2.set_ylabel(args.irrigation_units_label)
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.20)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        columnspacing=1.4,
        handlelength=1.8,
    )
    span_days = max((zoom_end - zoom_start).days, 1)
    if span_days > 90:
        ax2.xaxis.set_major_locator(mdates.MonthLocator())
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    elif span_days > 45:
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    else:
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    for axis in axes:
        axis.tick_params(axis="x", rotation=0)

    png = out_prefix.with_suffix(".png")
    pdf = out_prefix.with_suffix(".pdf")
    # 300 dpi PNG is the usual minimum for print/grant-application figures;
    # PDF is vector and is what you'd actually want to embed if the
    # application format allows it (no resolution loss when scaled).
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "particle_sm": str(particle_sm_path),
        "weights": str(weights_path),
        "observations": str(observations_path),
        "particle_irrigation": str(particle_irrigation_path) if particle_irrigation_path else "",
        "open_loop_sm": str(args.open_loop_sm) if args.open_loop_sm else "",
        "precipitation": str(args.precipitation) if args.precipitation else "",
        "validation_irrigation": str(args.validation_irrigation) if args.validation_irrigation else "",
        "zoom_start": zoom_start.date().isoformat(),
        "zoom_end": zoom_end.date().isoformat(),
        "color_weight_window": args.color_weight_window if args.color_weight_window else "per-window (varies)",
        "weight_color_scope": args.weight_color_scope,
        "weight_color_scale": summary_color_scale,
        "weight_color_vmin": summary_norm_vmin,
        "weight_color_vmax": summary_norm_vmax,
        "min_particle_alpha": args.min_particle_alpha,
        "max_particle_alpha": args.max_particle_alpha,
        "n_particles_plotted": int(zoom_sm["particle"].nunique()),
        "png": str(png),
        "pdf": str(pdf),
    }
    pd.DataFrame([summary]).to_csv(out_prefix.with_name(out_prefix.name + "_inputs.csv"), index=False)
    print(png)
    print(pdf)


def main():
    plot_diagnostic(parse_args())


if __name__ == "__main__":
    main()
