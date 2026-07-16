"""
Self-check a CLiMB submission output tree BEFORE submitting.

Run your container locally (see submission_instructions/README.md), then point this
script at the produced /output directory. It performs the same structural checks the
challenge Validation queue runs, so passing here means your submission will not be
rejected for format reasons:

    python evaluation/validate_submission.py --output /path/to/output
    python evaluation/validate_submission.py --output /path/to/output --input /path/to/videos

With --input (the folder of .mp4 files you mounted at /input), it also verifies that
every sequence produced an output folder. Without it, whatever sequence folders exist
under --output are validated.

Checks per sequence:
  - all 5 run folders (1..5) exist
  - each run has 3D_maps/<integer map_id>/points3D.txt
  - each map has a matching camera_trajectory/cam_traj_map_<map_id:03d>.txt
    (zero-padded to 3 digits)
  - trajectories have >= 9 comma-separated fields per pose row
  - frame IDs (numeric prefix of name_image) are 1-based
  - runtime.txt exists and parses (init_seconds=<float>, processing_seconds=<float>)

Exit code 0 = valid, 1 = problems found (listed on stdout).

Requires Python 3.8+, no third-party dependencies.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

EXPECTED_RUN_IDS = ("1", "2", "3", "4", "5")
REQUIRED_TRAJ_FIELDS = 9  # timestamp, name_image, tx, ty, tz, qw, qx, qy, qz


def _parse_runtime_txt(path: Path) -> list:
    errs = []
    if not path.exists():
        return [f"missing {path.name}"]
    kv = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            errs.append(f"{path.name}:{lineno}: expected key=value, got '{line[:40]}'")
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    for key in ("init_seconds", "processing_seconds"):
        if key not in kv:
            errs.append(f"{path.name}: missing '{key}'")
            continue
        try:
            float(kv[key])
        except ValueError:
            errs.append(f"{path.name}: '{key}' is not a float ('{kv[key]}')")
    return errs


def _validate_trajectory(path: Path) -> list:
    errs = []
    rows = 0
    min_frame_id = None
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [t.strip() for t in line.split(",")]
        if len(parts) < REQUIRED_TRAJ_FIELDS:
            errs.append(
                f"{path.name}:{lineno}: expected >={REQUIRED_TRAJ_FIELDS} "
                f"comma-separated fields, got {len(parts)}"
            )
            continue
        name_image = parts[1]
        num = ""
        for ch in name_image:
            if ch.isdigit():
                num += ch
            else:
                break
        if not num:
            errs.append(f"{path.name}:{lineno}: cannot read frame id from '{name_image}'")
        else:
            fid = int(num)
            min_frame_id = fid if min_frame_id is None else min(min_frame_id, fid)
        try:
            [float(x) for x in parts[2:9]]
        except ValueError as e:
            errs.append(f"{path.name}:{lineno}: non-numeric pose field ({e})")
        rows += 1

    if rows == 0:
        errs.append(f"{path.name}: no pose rows")
    if min_frame_id is not None and min_frame_id < 1:
        errs.append(
            f"{path.name}: frame IDs must be 1-based; found id {min_frame_id} "
            f"(first frame must be 1, not 0)"
        )
    return errs


def _validate_run(run_dir: Path) -> list:
    errs = []
    seq_run = f"{run_dir.parent.name}/{run_dir.name}"

    errs += [f"{seq_run}: {e}" for e in _parse_runtime_txt(run_dir / "runtime.txt")]

    maps_root = run_dir / "3D_maps"
    traj_root = run_dir / "camera_trajectory"
    if not maps_root.is_dir():
        errs.append(f"{seq_run}: missing 3D_maps/")
        return errs
    if not traj_root.is_dir():
        errs.append(f"{seq_run}: missing camera_trajectory/")
        return errs

    map_dirs = [d for d in sorted(maps_root.iterdir()) if d.is_dir()]
    if not map_dirs:
        errs.append(f"{seq_run}: 3D_maps/ has no map folders")
        return errs

    valid_maps = 0
    for map_dir in map_dirs:
        try:
            map_id = int(map_dir.name)
        except ValueError:
            errs.append(f"{seq_run}: map folder '{map_dir.name}' is not an integer")
            continue
        if not (map_dir / "points3D.txt").is_file():
            errs.append(f"{seq_run}: map {map_id} missing points3D.txt")
        traj_file = traj_root / f"cam_traj_map_{map_id:03d}.txt"
        if not traj_file.is_file():
            errs.append(
                f"{seq_run}: map {map_id} missing trajectory "
                f"cam_traj_map_{map_id:03d}.txt (must be zero-padded to 3 digits)"
            )
            continue
        traj_errs = _validate_trajectory(traj_file)
        errs += [f"{seq_run}: {e}" for e in traj_errs]
        if not traj_errs:
            valid_maps += 1

    if valid_maps == 0:
        errs.append(f"{seq_run}: no valid (map + trajectory) pair")
    return errs


def validate_tree(output_root: Path, input_root: Path = None) -> list:
    """Return a list of problems; empty list means the tree is valid."""
    errors = []

    if not output_root.is_dir():
        return [f"output directory not found: {output_root}"]

    if input_root is not None:
        expected = sorted(p.stem for p in input_root.glob("*.mp4"))
        if not expected:
            return [f"no *.mp4 files found under {input_root}"]
    else:
        expected = sorted(d.name for d in output_root.iterdir() if d.is_dir())
        if not expected:
            return [f"no sequence folders found under {output_root}"]

    for seq in expected:
        seq_dir = output_root / seq
        if not seq_dir.is_dir():
            errors.append(f"{seq}: missing output folder")
            continue
        for run_id in EXPECTED_RUN_IDS:
            run_dir = seq_dir / run_id
            if not run_dir.is_dir():
                errors.append(f"{seq}: missing run folder {run_id}/")
                continue
            errors += _validate_run(run_dir)

    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate a CLiMB submission output tree against the contract."
    )
    parser.add_argument("--output", required=True,
                        help="Path to the /output tree produced by your container.")
    parser.add_argument("--input", default=None,
                        help="Optional path to the input videos folder (checks that "
                             "every *.mp4 has a matching output).")
    args = parser.parse_args()

    errors = validate_tree(Path(args.output),
                           Path(args.input) if args.input else None)

    if errors:
        print(f"INVALID: {len(errors)} problem(s) found:\n")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print("VALID: the output tree matches the CLiMB submission contract.")
    sys.exit(0)


if __name__ == "__main__":
    main()
