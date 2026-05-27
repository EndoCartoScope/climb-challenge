#!/usr/bin/env bash
#
# Build the toy-example image and run it once against the local data/ folder.
# The container itself produces the full submission tree (all runs × sequences)
# in a single invocation; this script is just a build-and-test helper.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-endovis-toy-example-submission:ci}"
INPUT_HOST="${INPUT_HOST:-${SCRIPT_DIR}/../data}"
OUTPUT_HOST="${OUTPUT_HOST:-/tmp/endovis-toy-example-output}"
NUM_RUNS="${NUM_RUNS:-5}"

mkdir -p "${OUTPUT_HOST}"

mapfile -t INPUT_VIDEOS < <(find "${INPUT_HOST}" -maxdepth 1 -type f -iname '*.mp4' | sort)
if [ "${#INPUT_VIDEOS[@]}" -eq 0 ]; then
  echo "Missing example input videos: no .mp4 files found in ${INPUT_HOST}" >&2
  exit 1
fi

echo "== Build toy-example Docker image =="
docker build -t "${IMAGE}" "${SCRIPT_DIR}"

echo "== Optional GPU sanity check =="
if docker run --rm --gpus all "${IMAGE}" --help >/dev/null 2>&1; then
  GPU_ARGS=(--gpus all)
else
  echo "GPU runtime is not available; running the toy example on CPU."
  GPU_ARGS=()
fi

echo "== Run toy-example submission =="
docker run --rm "${GPU_ARGS[@]}" \
  --user "$(id -u)":"$(id -g)" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  "${IMAGE}" \
  --input /input \
  --output /output \
  --num-runs "${NUM_RUNS}"

echo "== Validate expected files =="
for video_path in "${INPUT_VIDEOS[@]}"; do
  video_name="$(basename "${video_path}")"
  sequence="${video_name%.*}"
  for run_id in $(seq 1 "${NUM_RUNS}"); do
    test -s "${OUTPUT_HOST}/${sequence}/${run_id}/camera_trajectory/cam_traj_map_000.txt"
    test -s "${OUTPUT_HOST}/${sequence}/${run_id}/3D_maps/000/points3D.txt"
  done
done

echo "Toy-example output written to: ${OUTPUT_HOST}"
find "${OUTPUT_HOST}" -type f | sort
