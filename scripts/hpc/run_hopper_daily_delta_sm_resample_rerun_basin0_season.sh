#!/bin/bash
#SBATCH -J dSMseason_basin0
#SBATCH -p normal
#SBATCH -n 20
#SBATCH -c 1
#SBATCH -o season_delta_resample_rerun_%j.out
#SBATCH -e season_delta_resample_rerun_%j.err

export OMP_NUM_THREADS=1
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

source ~/.bashrc || true

if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/home/fs01/hmf63/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/home/fs01/hmf63/miniconda3/etc/profile.d/conda.sh"
else
    echo "Could not find conda. Edit this script to source conda.sh before conda activate." >&2
    exit 1
fi

conda activate base
set -u

basin_name="${BASIN_NAME:-basin0}"
n_particles="${N_PARTICLES:-100}"
mpi_procs="${MPI_PROCS:-20}"
seed="${PBS_SEED:-20260712}"
window_days="${WINDOW_DAYS:-7}"

# Main experiment defaults: full assumed irrigation season plus one-year warmup.
pbs_start="${PBS_START:-2020-05-01}"
pbs_end="${PBS_END:-2020-09-30}"
spinup_start="${SPINUP_START:-2019-05-01}"
spinup_end="${SPINUP_END:-2020-04-30}"

target_mode="${TARGET_MODE:-cropland_only}"
delta_sigma="${DELTA_SIGMA:-0.075}"

vic_root="${VIC_ROOT:-/home/fs01/hmf63/Local_Irrigation/VICFiles}"
sample_data="${SAMPLE_DATA:-${vic_root}/samples/p2_cal}"
run_id="${RUN_ID:-season_daily_deltaSM_resample_rerun_${basin_name}_${pbs_start}_${pbs_end}_N${n_particles}}"
run_root="${RUN_ROOT:-${vic_root}/pbs_runs_season/${run_id}}"
target_csv="${TARGET_CSV:-${vic_root}/daily_delta_targets_for_hopper_season_latest.csv}"

vic_exe="${VIC_EXE:-./vic_image.exe}"
global_template="${GLOBAL_TEMPLATE:-${sample_data}/global_p2_${basin_name}.txt}"
domain_file="${DOMAIN_FILE:-${sample_data}/basin/domain_${basin_name}.nc}"
parameter_file="${PARAMETER_FILE:-${sample_data}/param/veg_p0_soilgrid.nc}"
forcing_prefix="${FORCING_PREFIX:-${sample_data}/Input/CONUS_${basin_name}_}"
initial_state_override="${INITIAL_STATE:-}"

mkdir -p "${run_root}/inputs" "${run_root}/postprocess" "${run_root}/logs"

echo "Seasonal daily Delta-SM sequential PBS"
echo "Run id: ${run_id}"
echo "PBS window: ${pbs_start} to ${pbs_end}; window days: ${window_days}"
echo "Spin-up: ${spinup_start} to ${spinup_end}"
echo "Particles: ${n_particles}; MPI procs per VIC run: ${mpi_procs}"
echo "Target mode: ${target_mode}; Delta sigma: ${delta_sigma}"
echo "Target CSV: ${target_csv}"
echo "Run root: ${run_root}"

for required in "$target_csv" "$global_template" "$domain_file" "$parameter_file"; do
    if [ ! -f "$required" ]; then
        echo "Missing required file: $required" >&2
        exit 2
    fi
done

date_token() {
    echo "$1" | tr -d '-'
}

find_spinup_state() {
    local root="$1"
    local token="$2"
    find "${root}/particle_0000/state" -maxdepth 1 -type f -name "state.${token}_*.nc" 2>/dev/null | sort | head -n 1
}

if [ -n "$initial_state_override" ]; then
    initial_state="$initial_state_override"
    echo "Using caller-provided initial state: ${initial_state}"
else
    spinup_root="${run_root}/spinup_${spinup_start}_${spinup_end}"
    spinup_token="$(date_token "$spinup_end")"
    initial_state="$(find_spinup_state "$spinup_root" "$spinup_token" || true)"

    if [ -z "$initial_state" ]; then
        echo ""
        echo "No spin-up state found. Running open-loop spin-up to generate ${spinup_end} state..."
        spinup_irrigation="${run_root}/inputs/spinup_zero_irrigation_${spinup_start}_${spinup_end}.csv"
        python pbs_generate_irrigation_window_table.py \
          --start-date "$spinup_start" \
          --end-date "$spinup_end" \
          --n-particles 1 \
          --event-probability 0.0 \
          --seed "$seed" \
          --out-csv "$spinup_irrigation"

        python pbs_prepare_vic_window_from_irrigation_table.py \
          --basin-name "$basin_name" \
          --start-date "$spinup_start" \
          --end-date "$spinup_end" \
          --irrigation-table "$spinup_irrigation" \
          --vic-exe "$vic_exe" \
          --global-template "$global_template" \
          --domain-file "$domain_file" \
          --parameter-file "$parameter_file" \
          --forcing-prefix "$forcing_prefix" \
          --work-root "$spinup_root" \
          --mpi-procs "$mpi_procs" \
          --allow-cold-start

        bash "${spinup_root}/run_all_particles.sh"
        initial_state="$(find_spinup_state "$spinup_root" "$spinup_token" || true)"
    fi

    if [ -z "$initial_state" ]; then
        echo "Spin-up completed but no state file was found for ${spinup_end} under ${spinup_root}" >&2
        exit 3
    fi
    echo "Using generated spin-up state: ${initial_state}"
fi

if [ ! -f "$initial_state" ]; then
    echo "Initial state file does not exist: ${initial_state}" >&2
    exit 4
fi

mapfile -t windows < <(python - "$pbs_start" "$pbs_end" "$window_days" <<'PY'
import sys
import pandas as pd

start = pd.Timestamp(sys.argv[1])
end = pd.Timestamp(sys.argv[2])
window_days = int(sys.argv[3])
cur = start
while cur <= end:
    win_end = min(cur + pd.Timedelta(days=window_days - 1), end)
    print(f"{cur.date().isoformat()},{win_end.date().isoformat()}")
    cur = win_end + pd.Timedelta(days=1)
PY
)

if [ "${#windows[@]}" -eq 0 ]; then
    echo "No PBS windows generated for ${pbs_start} to ${pbs_end}" >&2
    exit 5
fi

parent_state_table=""
for idx in "${!windows[@]}"; do
    window_id=$((idx + 1))
    IFS=',' read -r window_start window_end <<< "${windows[$idx]}"
    window_root="${run_root}/window_${window_id}_${window_start}_${window_end}"
    irrigation_table="${run_root}/inputs/irrigation_window_${window_id}.csv"
    post_dir="${run_root}/postprocess/window_${window_id}"

    echo ""
    echo "=== Window ${window_id}/${#windows[@]}: ${window_start} to ${window_end} ==="

    python pbs_generate_irrigation_window_table.py \
      --start-date "$window_start" \
      --end-date "$window_end" \
      --n-particles "$n_particles" \
      --seed $((seed + window_id * 1000)) \
      --out-csv "$irrigation_table"

    prepare_args=(
      --basin-name "$basin_name"
      --start-date "$window_start"
      --end-date "$window_end"
      --irrigation-table "$irrigation_table"
      --vic-exe "$vic_exe"
      --global-template "$global_template"
      --domain-file "$domain_file"
      --parameter-file "$parameter_file"
      --forcing-prefix "$forcing_prefix"
      --work-root "$window_root"
      --mpi-procs "$mpi_procs"
    )
    if [ -n "$parent_state_table" ]; then
        prepare_args+=(--particle-initial-states "$parent_state_table")
    else
        prepare_args+=(--initial-state "$initial_state")
    fi

    python pbs_prepare_vic_window_from_irrigation_table.py "${prepare_args[@]}"

    echo "Launching VIC particles for window ${window_id}..."
    bash "${window_root}/run_all_particles.sh"

    if [ "$window_id" -lt "${#windows[@]}" ]; then
        next_parent_state_table="${run_root}/inputs/particle_initial_states_window_$((window_id + 1)).csv"
        resample_args=(--resample-out "$next_parent_state_table")
    else
        next_parent_state_table=""
        resample_args=()
    fi

    python score_hopper_daily_delta_sm_window.py \
      --window-root "$window_root" \
      --target-csv "$target_csv" \
      --window-id "$window_id" \
      --window-start "$window_start" \
      --window-end "$window_end" \
      --target-mode "$target_mode" \
      --delta-sigma-m3m3 "$delta_sigma" \
      --out-dir "$post_dir" \
      --seed $((seed + window_id * 2000)) \
      "${resample_args[@]}"

    parent_state_table="$next_parent_state_table"
done

archive="${run_root}_results.tgz"
echo ""
echo "Creating archive: ${archive}"
rm -f "$archive"
tar -C "$(dirname "$run_root")" -czf "$archive" "$(basename "$run_root")"

echo "Seasonal sequential PBS complete."
echo "Run directory: ${run_root}"
echo "Archive: ${archive}"
