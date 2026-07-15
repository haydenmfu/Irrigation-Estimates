#!/bin/bash
#SBATCH -J dSMresamp_basin0
#SBATCH -p normal
#SBATCH -n 20
#SBATCH -c 1
#SBATCH -o week7_daily_delta_resample_rerun_20260709.out
#SBATCH -e week7_daily_delta_resample_rerun_20260709.err

export OMP_NUM_THREADS=1
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"

export BASHRCSOURCED="${BASHRCSOURCED:-0}"
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

basin_name="basin0"
n_particles=100
mpi_procs=20
seed=20260709

# Recommended first Hopper test from the local sensitivity run:
# daily delta observations, cropland-only target, loose sigma to avoid immediate degeneracy.
target_mode="${TARGET_MODE:-cropland_only}"
delta_sigma="${DELTA_SIGMA:-0.075}"

vic_root="/home/fs01/hmf63/Local_Irrigation/VICFiles"
sample_data="${vic_root}/samples/p2_cal"
run_id="week7_daily_deltaSM_resample_rerun_N100_20260709"
run_root="${vic_root}/pbs_runs_week7/${run_id}"
target_csv="${vic_root}/daily_delta_targets_for_hopper_latest.csv"

vic_exe="./vic_image.exe"
global_template="${sample_data}/global_p2_${basin_name}.txt"
domain_file="${sample_data}/basin/domain_${basin_name}.nc"
parameter_file="${sample_data}/param/veg_p0_soilgrid.nc"
forcing_prefix="${sample_data}/Input/CONUS_${basin_name}_"
initial_state="${vic_root}/pbs_state_generation/${basin_name}_state_20200630/state/state_${basin_name}.20200630_00000.nc"

starts=("2020-07-01" "2020-07-08" "2020-07-15" "2020-07-22")
ends=("2020-07-07" "2020-07-14" "2020-07-21" "2020-07-28")

mkdir -p "${run_root}/inputs" "${run_root}/postprocess"

echo "Week 7 daily delta-SM sequential resample/rerun PBS"
echo "Run id: ${run_id}"
echo "Target mode: ${target_mode}"
echo "Delta sigma: ${delta_sigma}"
echo "Target CSV: ${target_csv}"
echo "Run root: ${run_root}"

if [ ! -f "$target_csv" ]; then
    echo "Missing target CSV: ${target_csv}" >&2
    exit 2
fi
if [ ! -f "$initial_state" ]; then
    echo "Missing initial state file: ${initial_state}" >&2
    exit 3
fi

parent_state_table=""
for idx in "${!starts[@]}"; do
    window_id=$((idx + 1))
    window_start="${starts[$idx]}"
    window_end="${ends[$idx]}"
    window_root="${run_root}/window_${window_id}_${window_start}_${window_end}"
    irrigation_table="${run_root}/inputs/irrigation_window_${window_id}.csv"
    post_dir="${run_root}/postprocess/window_${window_id}"

    echo ""
    echo "=== Window ${window_id}: ${window_start} to ${window_end} ==="

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

    if [ "$window_id" -lt "${#starts[@]}" ]; then
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

archive="${vic_root}/pbs_runs_week7/${run_id}_results.tgz"
echo ""
echo "Creating archive: ${archive}"
rm -f "$archive"
tar -C "$(dirname "$run_root")" -czf "$archive" "$(basename "$run_root")"

echo "Sequential resample/rerun PBS complete."
echo "Run directory: ${run_root}"
echo "Archive: ${archive}"

