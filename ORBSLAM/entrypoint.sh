#!/usr/bin/env bash
#
# ORB-SLAM3 submission entrypoint.
# Processes every .mp4 under <input>/ and produces the full submission tree
# (<input>/<seq>.mp4 -> <output>/<seq>/{1..EXECUTIONS}/{3D_maps,camera_trajectory}/...)
# in a single container invocation.
#
set -euo pipefail

cd /opt/ORB_SLAM3

input_dir="${1:-/input}"
pathResults="${2:-/output}"
initial_index="${INITIAL_INDEX:-0}"
executions="${EXECUTIONS:-5}"

if [[ ! -d "${input_dir}" ]]; then
    echo "Input directory does not exist: ${input_dir}" 1>&2
    exit 1
fi

mkdir -p "${pathResults}"

shopt -s nullglob nocaseglob
Sequences=("${input_dir}"/*.mp4)
shopt -u nocaseglob

if [[ "${#Sequences[@]}" -eq 0 ]]; then
    echo "No .mp4 files found in ${input_dir}" 1>&2
    exit 1
fi

t_total_start=${SECONDS}

for pathVideo in "${Sequences[@]}"; do
    video_seq="$(basename "${pathVideo}")"
    name_seq="${video_seq%.*}"
    pathResultSeq="${pathResults}/${name_seq}"
    mkdir -p "${pathResultSeq}"

    t_seq_start=${SECONDS}

    for i in $(seq 1 "${executions}"); do
        echo "Running ${name_seq} --> ${i}/${executions}"
        echo "Video path: ${pathVideo}"
        if ! (
            set -e
            rm -rf output

            ./Examples/Monocular/mono_endo_hculb \
                ./Vocabulary/ORBvoc.txt \
                ./Examples/Monocular/HCULB_Endoscope_Depth.yaml \
                "${pathVideo}"

            index=$((initial_index + i))
            pathResultExp="${pathResultSeq}/${index}"
            rm -rf "${pathResultExp}"
            mkdir -p "${pathResultExp}"

            shopt -s nullglob
            mv output/* "${pathResultExp}"
            shopt -u nullglob
        ); then
            echo "[WARN] ${name_seq} execution ${i} failed; continuing with next" >&2
        fi
    done

    echo "  ${name_seq}: runs=${executions} runtime=$((SECONDS - t_seq_start))s"
done

echo "Total: $((SECONDS - t_total_start))s"
