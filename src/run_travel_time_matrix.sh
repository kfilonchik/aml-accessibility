#!/usr/bin/env bash
# run_travel_time_matrix.sh
#
# Launches the r5py travel-time-matrix pipeline (02_build_network.py smoke
# test, then 03_travel_time_matrices.py) in the background, using the
# "r5geo" conda env, and survives the terminal/SSH session closing.
#
# Usage:
#   ./run_travel_time_matrix.sh          # start in background
#   tail -f <printed log path>           # watch progress
#   kill $(cat <printed pid path>)       # stop it
#
# Safe to re-run after a crash/interruption: 03_travel_time_matrices.py
# checkpoints each origin batch to a .parquet file and skips batches that
# already exist on disk.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV_PATH="/home/20230692/miniconda3/envs/r5geo"
LOG_DIR="${SCRIPT_DIR}/../output/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_DIR}/travel_time_matrix_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/travel_time_matrix.pid"

# r5py needs a JDK 21 JVM. The system-wide `java` on PATH on this machine is
# JDK 17, which r5py cannot load (fails with "incorrect Java version") — the
# r5geo env ships its own JDK 21 under its prefix, so JAVA_HOME must point
# there explicitly. GDAL_DATA/PROJ_LIB are pinned too, since a background
# non-interactive shell won't necessarily inherit the same env a `conda
# activate`-ed interactive shell would.
export JAVA_HOME="${CONDA_ENV_PATH}"
export GDAL_DATA="${CONDA_ENV_PATH}/share/gdal"
export PROJ_LIB="${CONDA_ENV_PATH}/share/proj"
export PATH="${CONDA_ENV_PATH}/bin:${PATH}"

PYTHON_BIN="${CONDA_ENV_PATH}/bin/python"

cd "${SCRIPT_DIR}"

if ! nohup "${PYTHON_BIN}" -u 02_build_network.py > "${LOG_FILE}" 2>&1; then
    echo "02_build_network.py failed — last 30 lines of ${LOG_FILE}:" >&2
    tail -n 30 "${LOG_FILE}" >&2
    exit 1
fi
echo "[02_build_network.py OK, see ${LOG_FILE}]" >> "${LOG_FILE}"

nohup "${PYTHON_BIN}" -u 03_travel_time_matrices.py >> "${LOG_FILE}" 2>&1 &
echo $! > "${PID_FILE}"

echo "Started 03_travel_time_matrices.py in background."
echo "  PID:  $(cat "${PID_FILE}")"
echo "  log:  ${LOG_FILE}"
echo "  stop: kill \$(cat ${PID_FILE})"
