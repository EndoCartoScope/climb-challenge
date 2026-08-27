# CLiMB Submission Instructions

This document is the authoritative contract for submissions to the CLiMB challenge. 

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

If your method produces a **single map** per sequence, you still use this same structure with `map_id = 0`: the map goes in `3D_maps/0/points3D.txt` and its trajectory in `camera_trajectory/cam_traj_map_000.txt`.

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
| Wall-clock | **5× the real-time duration of the job** (all sequences × 5 runs). For the current test set this is ≈ 6 h per scoring submission. Exceeding it aborts the run → `INVALID` (`timeout`). |


Because the container runs with `--network=none`, **bake all weights, model files, vocabularies, and any other downloaded assets into the image at build time**. Any runtime download attempt will fail.

The formal timing contract is the per-run `runtime.txt` artifact described in [Runtime reporting](#runtime-reporting-runtimetxt). Printing per-sequence and total wall-clock to stdout in addition is optional but encouraged for easier log inspection. Both reference implementations show the pattern: [toy-example/run.py](../toy-example/run.py) (Python, `time.perf_counter()`) and [ORBSLAM/entrypoint.sh](../ORBSLAM/entrypoint.sh) (bash, `${SECONDS}`).

## Submission

Submissions are delivered as **Docker images** pushed to the CLiMB Synapse Docker
registry and submitted through **two evaluation queues**:

| Queue | Purpose | What it runs | Result |
|---|---|---|---|
| **CLiMB Validation** | Fast smoke test | Your image on **one short clip**, then a format check of the output tree | `VALIDATED` or `INVALID` (+ reason) |
| **CLiMB Scoring** | Official evaluation | Your image on the **full test set** (all sequences × 5 runs), then metrics | `SCORED` (+ ATE/RPE/TFR/Runtime) or `INVALID` (+ reason) |

> Submission opens **July 15, 2026**. Until then the queues are closed to
> participants.

### 0. Register for the challenge

Before anything else, complete the registration steps on the
[challenge wiki](https://www.synapse.org/Synapse:syn74370700/wiki/639980) (Synapse
account, challenge registration / participant team, and Synapse **certified user**
status — required to create projects and push Docker images). Without these, the
steps below will fail with permission errors.

### 1. Push your image to your own project's registry

Docker images are pushed to the registry of **your own Synapse project** (create one
from your Synapse home page if you don't have one — it can stay private). You do
**not** push to the CLiMB project; you only *submit* the image entity to our queues.

```bash
# Tag for YOUR project's registry: docker.synapse.org/<your_project_synID>/<method>:<tag>
docker tag my-submission:dev docker.synapse.org/syn<YOUR_PROJECT_ID>/my-method:v1

# Log in (password = a Synapse Personal Access Token, NOT your account password)
docker login docker.synapse.org

# Push
docker push docker.synapse.org/syn<YOUR_PROJECT_ID>/my-method:v1
```

After the push, the image appears under the **Docker** tab of your project as an
entity — that entity is what you submit. Tag each new submission with a fresh version
(`v1`, `v2`, …) so previous submissions stay reproducible. Do not overwrite a tag that
has already been evaluated.

### 2. Submit to **Validation** first

Always submit to the **CLiMB Validation** queue first. In the Synapse web UI: open your
Docker image entity → **Submit to Challenge** → choose the **CLiMB Validation** queue.
The validation runner executes your image on a single short clip and checks that the
output tree matches [Output layout](#output-layout): 5 run folders, integer map folders
with matching zero-padded `cam_traj_map_<id:03d>.txt`, 1-based frame IDs, and a
parseable `runtime.txt`.

- **`VALIDATED`** — the format is correct; you're clear to submit to scoring.
- **`INVALID`** — see the `error_message` annotation on your submission for the exact
  problem (e.g. `Seq_001_a: missing run folder 3/`, or a container crash / timeout), fix
  it, push a new version, and re-validate.

Validation runs your container on a tiny input, so it catches contract mistakes in
minutes instead of burning a multi-hour scoring slot.

### 3. Once `VALIDATED`, submit the same image to **Scoring**

When your submission reaches `VALIDATED`, submit the **same image** to the
**CLiMB Scoring** queue. The scoring runner executes your container on the full test
set (all sequences × 5 runs) on the reference RTX 5090 and evaluates it.

- **`SCORED`** — metrics are attached as annotations (mean ATE mm, RPE rotational deg at
  δ=40, TFR %, Runtime s/frame, success count) and appear on the leaderboard, ranked by
  the final **`climb_score`** — see [Final score and ranking](../README.md#final-score-and-ranking)
  for the exact formula.
- **`INVALID`** — the `error_message` annotation explains what failed (pull error,
  container crash, wall-clock timeout, malformed output, or a scoring error).

> Tip: set a **submission alias** in the submit form — it is shown on the leaderboard
> next to your team name and helps you tell your own attempts apart.

### Where you see the result

The outcome shows up as the **submission status + annotations** on your submission in the
Synapse UI, and `SCORED` metrics on the challenge leaderboard. (The evaluator does **not**
email you — check your submission in the web UI.)

### Please include a short method description

Along with **each** submission (Synapse submission form or accompanying text):

- Method name and a one-line description.
- Approximate per-sequence wall-clock on the EndoMapper short sequences from your local runs.

### Final submission: method write-up (mandatory for the ranking)

For your **final** submission, a short **method write-up (max 3 pages, excluding
references)** is required, submissions without it do not qualify for the official
leaderboard. It must cover:

- Title, authors, affiliations, team name, and whether you agree to make the write-up
  public (co-authorship on the joint challenge publication follows the Publication
  Policy on the challenge wiki).
- **Background**: motivation and novelty of the approach.
- **Methods**: description detailed enough to be reproducible: pipeline diagram,
  pre-processing, training details, GPU resources used.
- **Data disclosure**: any pre-trained models and public data used for training.
  EndoMapper is public: using the **test sequences (or reconstructions of them)** for
  training or tuning is **not allowed** and must be explicitly ruled out here.
- **Discussion**: strengths/weaknesses, expected failure cases.
- Citations and, if available, a link to your source code (encouraged).

**How to deliver it:** send the write-up as a **PDF by email** to
**endocartoscope@unizar.es** with the subject `[CLiMB 2026] Write-up <TeamName>`,
by **September 12, 2026, 23:59 AoE**. In the email, state the **Synapse submission ID**
of your team's entry that the write-up describes.

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

Then check the produced tree against the contract with the self-check tool (no
dependencies, same checks the Validation queue runs):

```bash
python evaluation/validate_submission.py --output $(pwd)/output --input $(pwd)/data
```

If it prints `VALID`, your submission will not be rejected for format reasons.

As an **additional check**, you can run the metric evaluator itself on the sample
sequences, their COLMAP reconstructions are provided with the challenge samples data, so the
full pipeline (alignment + ATE/RPE/TFR) runs end-to-end on your side:

```bash
python evaluation/slam_evaluation.py \
    --colmap_path /path/to/sample_colmaps \
    --slam_path   $(pwd)/output \
    --results_file results.json
```

This confirms your trajectories match frame IDs with the COLMAP reference and produce
sane metrics before you submit. The test-set ground truth itself remains private.
