#!/usr/bin/env bash
#
# Build the ORB-SLAM3 image and run it once against the local data/ folder.
# The container's ENTRYPOINT (entrypoint.sh) handles all sequences x runs
# in a single invocation; this script is just a build-and-test helper.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-endovis-orbslam-submission:ci}"
INPUT_HOST="${INPUT_HOST:-${SCRIPT_DIR}/../data}"
OUTPUT_HOST="${OUTPUT_HOST:-/tmp/endovis-orbslam-output}"
EXECUTIONS="${EXECUTIONS:-5}"

mkdir -p "${OUTPUT_HOST}"

if [ ! -d "${SCRIPT_DIR}/ORB-SLAM3" ]; then
  echo "ORB-SLAM3 source not found at ${SCRIPT_DIR}/ORB-SLAM3" >&2
  echo "To use a different ORB-SLAM3 checkout, replace ORBSLAM/ORB-SLAM3/ with it." >&2
  exit 1
fi

echo "== Build ORB-SLAM3 Docker image =="
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "== GPU sanity check =="
if docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
else
  echo "GPU runtime not available; running without GPU."
  GPU_ARGS=()
fi

echo "== Verify ORB-SLAM3 install =="
docker run --rm --entrypoint /bin/bash "${IMAGE}" -lc '
  set -e
  test -s /opt/ORB_SLAM3/lib/libORB_SLAM3.so
  test -s /opt/ORB_SLAM3/Vocabulary/ORBvoc.txt
'

if [ ! -d "${INPUT_HOST}" ]; then
  echo "Input directory does not exist: ${INPUT_HOST}" >&2
  exit 1
fi

mapfile -t INPUT_VIDEOS < <(find "${INPUT_HOST}" -maxdepth 1 -type f -iname '*.mp4' | sort)
if [ "${#INPUT_VIDEOS[@]}" -eq 0 ]; then
  echo "Missing input: no .mp4 files found in ${INPUT_HOST}" >&2
  exit 1
fi

echo "== Run ORB-SLAM3 submission =="
docker run --rm "${GPU_ARGS[@]}" \
  --user "$(id -u)":"$(id -g)" \
  -e EXECUTIONS="${EXECUTIONS}" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  "${IMAGE}"

echo "== Done. Results in: ${OUTPUT_HOST} =="
find "${OUTPUT_HOST}" -type f | sort
