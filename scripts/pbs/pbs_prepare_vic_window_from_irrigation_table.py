#!/usr/bin/env python3
"""Prepare one VIC PBS window from explicit irrigation and optional parent states."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare one VIC window from an irrigation table.")
    parser.add_argument("--basin-name", default="basin0")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--irrigation-table", type=Path, required=True)
    parser.add_argument("--particle-initial-states", type=Path, default=None)
    parser.add_argument("--initial-state", default=None)
    parser.add_argument(
        "--allow-cold-start",
        action="store_true",
        help="Allow VIC to start without INIT_STATE. Intended for open-loop spin-up/state-generation runs.",
    )
    parser.add_argument("--vic-exe", required=True)
    parser.add_argument("--global-template", required=True)
    parser.add_argument("--domain-file", required=True)
    parser.add_argument("--parameter-file", required=True)
    parser.add_argument("--forcing-prefix", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--mpi-procs", type=int, default=20)
    return parser.parse_args()


def read_text(path):
    return Path(path).read_text(encoding="utf-8")


def write_text(path, text):
    Path(path).write_text(text, encoding="utf-8")


def set_global_value(text, key, value):
    lines = text.splitlines()
    out = []
    found = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(key + " ") or stripped.startswith(key + "\t") or stripped == key:
            out.append("{} {}".format(key, value))
            found = True
        elif stripped.startswith("##" + key + " ") or stripped.startswith("# " + key + " ") or stripped.startswith("#" + key + " "):
            out.append("{} {}".format(key, value))
            found = True
        else:
            out.append(line)
    if not found:
        out.append("{} {}".format(key, value))
    return "\n".join(out) + "\n"


def comment_global_value(text, key):
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(key + " ") or stripped.startswith(key + "\t") or stripped == key:
            out.append("##{}".format(line))
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def date_range(start_date, end_date):
    return pd.date_range(start_date, end_date, freq="D")


def years_in_window(start_date, end_date):
    return sorted(set(date_range(start_date, end_date).year))


def patch_global(template_text, args, particle_dir, initial_state):
    start = datetime.strptime(args.start_date, "%Y-%m-%d")
    end = datetime.strptime(args.end_date, "%Y-%m-%d")
    input_prefix = particle_dir / "input" / "{}_forcing_".format(args.basin_name)
    result_dir = particle_dir / "results"
    state_prefix = particle_dir / "state" / "state"
    log_dir = particle_dir / "logs"

    text = template_text
    text = text.replace("${VIC_SAMPLE_DATA}", str(particle_dir))
    text = text.replace("${VIC_SAMPLE_RESULTS}", str(particle_dir))
    replacements = {
        "STARTYEAR": start.year,
        "STARTMONTH": start.month,
        "STARTDAY": start.day,
        "ENDYEAR": end.year,
        "ENDMONTH": end.month,
        "ENDDAY": end.day,
        "DOMAIN": args.domain_file,
        "PARAMETERS": args.parameter_file,
        "FORCING1": str(input_prefix),
        "RESULT_DIR": str(result_dir),
        "OUTFILE": "fluxes",
        "LOG_DIR": str(log_dir),
        "STATENAME": str(state_prefix),
        "STATEYEAR": end.year,
        "STATEMONTH": end.month,
        "STATEDAY": end.day,
    }
    for key, value in replacements.items():
        text = set_global_value(text, key, value)
    if initial_state:
        text = set_global_value(text, "INIT_STATE", initial_state)
    elif args.allow_cold_start:
        text = comment_global_value(text, "INIT_STATE")
    return text


def copy_and_modify_forcing(args, particle_dir, irrigation):
    input_dir = particle_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    irrigation = irrigation.copy()
    irrigation["date"] = pd.to_datetime(irrigation["date"]).dt.date

    for year in years_in_window(args.start_date, args.end_date):
        src = Path("{}{}.nc".format(args.forcing_prefix, year))
        if not src.exists():
            raise FileNotFoundError("Forcing file not found: {}".format(src))
        dst = input_dir / "{}_forcing_{}.nc".format(args.basin_name, year)

        ds = xr.open_dataset(src).load()
        if "prec" not in ds:
            raise KeyError("'prec' variable not found in {}".format(src))

        times = pd.to_datetime(ds["time"].values)
        step_dates = pd.Series(times.date)
        daily = dict(zip(irrigation["date"], irrigation["irrigation_mm_day"]))
        add = np.zeros(ds["prec"].shape, dtype="float32")
        for date, amount in daily.items():
            if float(amount) == 0.0:
                continue
            idx = np.where(step_dates == date)[0]
            if len(idx) == 0:
                continue
            add[idx, :, :] = float(amount) / len(idx)

        ds["prec"] = ds["prec"] + xr.DataArray(add, dims=ds["prec"].dims, coords=ds["prec"].coords)
        ds.to_netcdf(dst)


def write_particle_run_script(args, work_root):
    run_script = work_root / "run_all_particles.sh"
    vic_exe = Path(args.vic_exe).resolve()
    lines = [
        "#!/bin/bash",
        "set -e",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'cd "${SCRIPT_DIR}"',
        "",
        "VIC_EXE={!r}".format(str(vic_exe)),
        "MPI_PROCS={}".format(args.mpi_procs),
        "",
        "shopt -s nullglob",
        "global_files=(particle_*/global_file.txt)",
        'if [ "${#global_files[@]}" -eq 0 ]; then',
        '  echo "No particle global files found under ${SCRIPT_DIR}" >&2',
        "  exit 1",
        "fi",
        "",
        'for global_file in "${global_files[@]}"; do',
        '  echo "Running VIC particle: ${global_file}"',
        '  mpirun -np "${MPI_PROCS}" --bind-to core --map-by core "${VIC_EXE}" -g "${global_file}"',
        "done",
        "",
    ]
    write_text(run_script, "\n".join(lines))
    run_script.chmod(0o755)
    return run_script


def read_initial_state_map(args):
    if args.particle_initial_states is None:
        return {}
    states = pd.read_csv(args.particle_initial_states)
    required = {"particle", "initial_state"}
    missing = required - set(states.columns)
    if missing:
        raise ValueError("Particle initial-state table missing columns: {}".format(sorted(missing)))
    return {int(row.particle): str(row.initial_state) for row in states.itertuples(index=False)}


def main():
    args = parse_args()
    work_root = Path(args.work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    irrigation = pd.read_csv(args.irrigation_table)
    required = {"particle", "date", "irrigation_mm_day"}
    missing = required - set(irrigation.columns)
    if missing:
        raise ValueError("Irrigation table missing columns: {}".format(sorted(missing)))
    irrigation["date"] = pd.to_datetime(irrigation["date"]).dt.date.astype(str)
    irrigation = irrigation[(irrigation["date"] >= args.start_date) & (irrigation["date"] <= args.end_date)].copy()
    irrigation["particle"] = irrigation["particle"].astype(int)
    particles = sorted(irrigation["particle"].unique())
    if not particles:
        raise ValueError("No particles found in irrigation table for requested dates.")

    initial_state_map = read_initial_state_map(args)
    if initial_state_map:
        missing_particles = sorted(set(particles) - set(initial_state_map))
        if missing_particles:
            raise ValueError("Missing per-particle initial states for particles: {}".format(missing_particles[:10]))
    elif not args.initial_state and not args.allow_cold_start:
        raise ValueError("Provide --initial-state or --particle-initial-states, or use --allow-cold-start for spin-up.")

    template_text = read_text(args.global_template)
    irrigation.to_csv(work_root / "particle_irrigation_inputs.csv", index=False)

    for particle in particles:
        particle_dir = work_root / "particle_{:04d}".format(int(particle))
        for subdir in ["input", "results", "state", "logs"]:
            (particle_dir / subdir).mkdir(parents=True, exist_ok=True)
        particle_irrigation = irrigation[irrigation["particle"] == particle]
        copy_and_modify_forcing(args, particle_dir, particle_irrigation)
        initial_state = initial_state_map.get(int(particle), args.initial_state)
        if initial_state and not Path(initial_state).exists():
            raise FileNotFoundError("Initial state for particle {} not found: {}".format(particle, initial_state))
        write_text(particle_dir / "global_file.txt", patch_global(template_text, args, particle_dir, initial_state))

    run_script = write_particle_run_script(args, work_root)
    print("Prepared {} VIC particles under {}".format(len(particles), work_root))
    print("Run all particles with: {}".format(run_script))


if __name__ == "__main__":
    main()
