#!/bin/bash

#SBATCH --nodes=10
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:20:00
#SBATCH --mail-type=END,FAIL
#SBATCH --account=PCS0294
#SBATCH --job-name=MonteCarloPi

# Load modules with error checking
module load openmpi/5.0.2 || { echo "Error: Failed to load openmpi/5.0.2"; exit 1; }
module load boost/1.83.0 || { echo "Error: Failed to load boost/1.83.0"; exit 1; }
module load cmake/3.25.2 || { echo "Error: Failed to load cmake/3.25.2"; exit 1; }
module load oneapi/2024.1.0 || { echo "Error: Failed to load oneapi/2024.1.0"; exit 1; }

# Build process with error handling
cd "$SLURM_SUBMIT_DIR" || { echo "Error: Failed to change to $SLURM_SUBMIT_DIR"; exit 1; }
mkdir -p build || { echo "Error: Failed to create build directory"; exit 1; }
cd build || { echo "Error: Failed to change to build directory"; exit 1; }
cmake -D CMAKE_CXX_COMPILER=icpx -G "Unix Makefiles" .. || { echo "Error: CMake configuration failed"; exit 1; }
make || { echo "Error: Make failed"; exit 1; }

# CSV output with unique filename
RESULTS_CSV="benchmark_results_${SLURM_JOB_ID}.csv"
echo "N,p,nodes,ppn,trial,seconds,pi_estimate" > "$RESULTS_CSV"

# Serial Run Block
P_SERIAL=1
NODES_SERIAL=1
PPN_SERIAL=1
TRIAL_CONFIG_NAME="Serial Run"
NUM_TRIALS=10
EXECUTABLE="$SLURM_SUBMIT_DIR/build/montecarlo_pi"
TOTAL_SAMPLES=100000000

echo "Benchmarking ${TRIAL_CONFIG_NAME} (P=${P_SERIAL})..."
SRUN_ARGS_SERIAL="--ntasks=${P_SERIAL} --nodes=${NODES_SERIAL} --ntasks-per-node=${PPN_SERIAL}"

for TRIAL in $(seq 1 "${NUM_TRIALS}"); do
    echo "  Running Trial ${TRIAL}/${NUM_TRIALS} for Serial..."
    TWO_FIELD_OUTPUT=$(srun ${SRUN_ARGS_SERIAL} "${EXECUTABLE}" -n "${TOTAL_SAMPLES}" -t serial) || { echo "Error: srun failed for serial trial ${TRIAL}"; exit 1; }
    if [[ -z "$TWO_FIELD_OUTPUT" ]]; then
        echo "Error: No output from srun for serial trial ${TRIAL}"
        exit 1
    fi
    echo "${TOTAL_SAMPLES},${P_SERIAL},${NODES_SERIAL},${PPN_SERIAL},${TRIAL},${TWO_FIELD_OUTPUT}" >> "$RESULTS_CSV"
done
echo ""

# Parallel Run Block
for P in 1 2 3 4 5 6 7 8 9 10; do
    NODES=$P
    PPN=1
    SRUN_ARGS_MPI="--ntasks=${P} --nodes=${NODES} --ntasks-per-node=${PPN} --distribution=cyclic"
    TRIAL_CONFIG_NAME="Parallel Run"
    echo "Benchmarking ${TRIAL_CONFIG_NAME} (P=${P})..."

    for TRIAL in $(seq 1 "${NUM_TRIALS}"); do
        echo "  Running Trial ${TRIAL}/${NUM_TRIALS} for MPI P=${P}..."
        TWO_FIELD_OUTPUT=$(srun ${SRUN_ARGS_MPI} "${EXECUTABLE}" -n "${TOTAL_SAMPLES}" -t mpi) || { echo "Error: srun failed for MPI trial ${TRIAL}, P=${P}"; exit 1; }
        if [[ -z "$TWO_FIELD_OUTPUT" ]]; then
            echo "Error: No output from srun for MPI trial ${TRIAL}, P=${P}"
            exit 1
        fi
        echo "${TOTAL_SAMPLES},${P},${NODES},${PPN},${TRIAL},${TWO_FIELD_OUTPUT}" >> "$RESULTS_CSV"
    done
    echo ""
done