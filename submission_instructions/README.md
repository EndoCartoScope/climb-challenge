# CLiMB Submission Instructions

This document is the authoritative contract for submissions to the CLiMB challenge. If anything in the [top-level README](../README.md) appears to contradict what's here, this file wins.

## What you submit

A **single Docker image** that, when invoked as described below, reads videos from `/input` and writes the full submission tree to `/output`. The image must be self-contained: all weights, models, vocabularies, and dependencies baked in at build time.

You do **not** submit source code, host-side scripts, or a `docker-run.sh`. We will not call your `docker-run.sh` (if any). Only the image's `ENTRYPOINT` / `CMD` is invoked.

## The container contract

We will run your image with **exactly** this command:

```bash
docker run --rm \
  --gpus all \
  --network=none \
  --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  -v "${INPUT_HOST}:/input:ro" \
  -v "${OUTPUT_HOST}:/output" \
  <your-image>:<tag>
```

No extra flags, no extra args. Your `ENTRYPOINT` (with default `CMD`) must:

1. Discover every `*.mp4` directly under `/input/`.
2. For each video `<seq>.mp4`, produce **5 independent runs** under `/output/<seq>/{1,2,3,4,5}/`.
3. Each run folder must contain the full submission tree (see [Output layout](#output-layout)).
4. Exit with status `0` on success.

The "5 runs" requirement exists because SLAM systems are typically non-deterministic; evaluation metrics are averaged across runs to mitigate variance. Your container is responsible for producing all 5 — we will not loop externally.

### Input layout

```
/input/
  Seq_001_a.mp4
  Seq_001_c.mp4
  ...
```

Flat directory of `.mp4` files. Sequence name = video filename without extension.

### Output layout

```
/output/
  <seq>/                                   # e.g. Seq_001_a
    <run_id>/                              # 1, 2, 3, 4, 5
      3D_maps/
        <map_id>/                          # integer, e.g. 0 or 000
          points3D.txt                     # COLMAP-format 3D points
      camera_trajectory/
        cam_traj_map_<map_id:03d>.txt      # e.g. cam_traj_map_000.txt
      runtime.txt                          # per-run timing — see Runtime reporting
```

A method that produces multiple sub-maps per sequence should write one folder per `<map_id>` under `3D_maps/`, with a matching `cam_traj_map_<map_id:03d>.txt` in `camera_trajectory/`.

#### File format notes

Beyond the example layouts in the [top-level README](../README.md#trajectory-file-format), the evaluator is intentionally tolerant in a few places. These are guarantees, not just current behaviour:

- **Map folder names**: the folder under `3D_maps/` must be an integer. Zero-padding is optional — `0`, `00`, and `000` are all accepted and treated as the same map id. The corresponding trajectory file, however, **must be zero-padded to 3 digits** (`cam_traj_map_000.txt`).
- **Trajectory columns**: the first 9 comma-separated fields (`timestamp, name_image, tx, ty, tz, qw, qx, qy, qz`) are required and must be in that order. 
- **Frame numbering**: frame IDs (the numeric prefix of `name_image`) are **1-based** — the first frame of every sequence is frame ID `1` (`000001.png`), not `0`. This must hold for every method so IDs line up with the COLMAP reference.
- `**points3D.txt` headers**: any line starting with `#` is treated as a comment. Multi-line COLMAP-style headers, single-line headers, or no headers at all are all fine.

#### Runtime reporting (`runtime.txt`)

Each `<run_id>/` folder must contain a `runtime.txt` recording the wall-clock for that single run, split into two phases:

```
init_seconds=<float>
processing_seconds=<float>
```

- `**init_seconds**` everything that runs **before the image/video feed loop**: model/weights load, SLAM vocabulary load, BoW build, GPU warmup, opening the input video, parsing settings, and any one-time precomputation.
- `**processing_seconds`** the per-frame loop itself, from the first frame read to the last pose produced. Post-processing steps (SLAM shutdown, trajectory/map serialization) are **excluded** from both numbers.

The ranking metric uses only `processing_seconds`; per-frame runtime is computed as `processing_seconds / N`, where `N` is the number of processed frames in the sequence. `init_seconds` is logged for audit and is not used in scoring.

A missing or unparseable `runtime.txt` is treated as a failed run under the [Missing outputs and failures](../README.md) rules.

## Runtime constraints


| Resource   | Limit                                                                                             |
| ---------- | ------------------------------------------------------------------------------------------------- |
| GPU        | 1× NVIDIA RTX 5090                                                                                |
| CUDA       | 12.8 host driver                                                                                  |
| RAM        | 64 GB                                                                                             |
| Network    | available at **build** time (for `apt`, `pip`, `git clone`); `**--network=none`** at **run** time |
| Filesystem | read-only `/input`, writable `/output`. Nothing else persists.                                    |
| Wall-clock | to be confirmed — currently unbounded                                                             |


Because the container runs with `--network=none`, **bake all weights, model files, vocabularies, and any other downloaded assets into the image at build time**. Any runtime download attempt will fail.

The formal timing contract is the per-run `runtime.txt` artifact described in [Runtime reporting](#runtime-reporting-runtimetxt). Printing per-sequence and total wall-clock to stdout in addition is optional but encouraged for easier log inspection. Both reference implementations show the pattern: [toy-example/run.py](../toy-example/run.py) (Python, `time.perf_counter()`) and [ORBSLAM/entrypoint.sh](../ORBSLAM/entrypoint.sh) (bash, `${SECONDS}`).

## Submission

**TBD**. Full instructions for tagging, pushing, and submitting the Docker image to the CLiMB Synapse evaluation queue will be published here once the queue is live.

## Local testing

Two reference implementations live in the repo and follow this exact contract:

- **[toy-example/](../toy-example/)** — minimal constant-velocity stub. Verifies the I/O contract without running a real SLAM system. Good first smoke-test target. Build and run with `bash toy-example/docker-run.sh`.
- **[ORBSLAM/](../ORBSLAM/)** — full ORB-SLAM3 baseline (vendored source). Build and run with `bash ORBSLAM/docker-run.sh`.

Both `docker-run.sh` scripts are **developer convenience helpers**, not part of the contract. They build the image, do a GPU sanity check, then run the container with `INPUT_HOST` / `OUTPUT_HOST` mounts. Use them as a template for testing your own image:

```bash
docker build -t my-submission .
docker run --rm --gpus all --network=none --memory=64g \
  --user "$(id -u)":"$(id -g)" \
  -v "$(pwd)/data:/input:ro" \
  -v "$(pwd)/output:/output" \
  my-submission
```

If that command produces a tree that matches [Output layout](#output-layout), you're done. Run `python evaluation/slam_evaluation.py --colmap_path ... --slam_path $(pwd)/output --results_file results.json` to confirm the evaluator can score it before pushing.