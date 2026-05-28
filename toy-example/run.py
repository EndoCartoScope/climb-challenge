#!/usr/bin/env python3
"""Toy-example EndoVis submission that writes a constant-velocity camera trajectory."""

import argparse
import time
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".m4v"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate fake ECS-format camera trajectories for input videos."
    )
    parser.add_argument("--input", default="/input", help="Directory containing input videos.")
    parser.add_argument("--output", default="/output", help="Directory where trajectories are written.")
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of submission runs to produce per sequence (folders 1..N under each sequence).",
    )
    parser.add_argument(
        "--translation-step",
        type=float,
        default=0.001,
        help="Constant translation increment per frame in world units.",
    )
    parser.add_argument(
        "--default-frames",
        type=int,
        default=10,
        help="Fallback frame count when OpenCV cannot read a video.",
    )
    return parser.parse_args()


def find_videos(input_dir: Path):
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def sequence_name(input_dir: Path, video_path: Path):
    parent = video_path.parent
    if parent == input_dir:
        return video_path.stem
    return parent.relative_to(input_dir).parts[0]


def count_frames(video_path: Path, default_frames: int):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            return max(1, default_frames)

        frames = 0
        while True:
            ok, _frame = capture.read()
            if not ok:
                break
            frames += 1

        return max(1, frames or default_frames)
    finally:
        capture.release()


def write_points3d(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write("# POINT3D_ID X Y Z R G B ERROR\n")
        file.write("1 0.000000 0.000000 0.050000 255 0 0 0.0\n")
        file.write("2 0.010000 0.000000 0.050000 0 255 0 0.0\n")
        file.write("3 0.000000 0.010000 0.050000 0 0 255 0.0\n")
        file.write("4 0.010000 0.010000 0.050000 255 255 255 0.0\n")


def write_runtime(path: Path, init_seconds: float, processing_seconds: float):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(f"init_seconds={init_seconds:.6f}\n")
        file.write(f"processing_seconds={processing_seconds:.6f}\n")


def write_run_output(output_dir: Path, sequence: str, run_id: int, num_frames: int, translation_step: float):
    sequence_root = output_dir / sequence / str(run_id)
    map_root = sequence_root / "3D_maps" / "000"
    trajectory_root = sequence_root / "camera_trajectory"

    # --- Init: everything before the per-frame feed loop. For a real VO/SLAM/learning-based this is
    #     model/vocabulary load, GPU warmup, opening the input video, etc. Here we
    #     just open the trajectory file and write its header.
    t_init_start = time.perf_counter()

    trajectory_path = trajectory_root / "cam_traj_map_000.txt"
    trajectory_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_file = trajectory_path.open("w", encoding="utf-8")
    trajectory_file.write("# timestamp, name_image, tx, ty, tz, qw, qx, qy, qz\n")

    # --- Processing: the per-frame feed loop. Only this interval is used by the
    #     ranking metric (processing_seconds / N_frames).
    t_processing_start = time.perf_counter()

    for index in range(num_frames):
        frame_id = index + 1  # frame IDs are 1-based: the first frame must be ID 1
        timestamp = float(index)
        image_name = f"{frame_id:06d}.png"
        tx = index * translation_step
        ty = 0.0
        tz = 0.0
        qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
        trajectory_file.write(
            f"{timestamp:.6f},{image_name},{tx:.9f},{ty:.9f},{tz:.9f},"
            f"{qw:.9f},{qx:.9f},{qy:.9f},{qz:.9f}\n"
        )

    t_processing_end = time.perf_counter()

    # --- Post-processing (shutdown, map serialization). Excluded from both timers.
    trajectory_file.close()
    write_points3d(map_root / "points3D.txt")

    write_runtime(
        sequence_root / "runtime.txt",
        init_seconds=t_processing_start - t_init_start,
        processing_seconds=t_processing_end - t_processing_start,
    )


def main():
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    videos = find_videos(input_dir)
    if not videos:
        raise SystemExit(f"No input videos found below {input_dir}")

    sequence_to_frames = {}
    for video_path in videos:
        name = sequence_name(input_dir, video_path)
        frames = count_frames(video_path, args.default_frames)
        sequence_to_frames[name] = max(sequence_to_frames.get(name, 0), frames)

    t0 = time.perf_counter()
    for sequence, frames in sorted(sequence_to_frames.items()):
        seq_t0 = time.perf_counter()
        for run_id in range(1, args.num_runs + 1):
            write_run_output(output_dir, sequence, run_id, frames, args.translation_step)
        print(
            f"  {sequence}: frames={frames} runs={args.num_runs} "
            f"runtime={time.perf_counter() - seq_t0:.2f}s",
            flush=True,
        )
    print(f"Total: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
