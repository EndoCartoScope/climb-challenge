import argparse
import os
import numpy as np

from colmap_utils import (list_colmap_maps_for_sequence, build_colmap_image_name_set,
                          read_colmap_points)
from utils import qvec2rotmat


# A SLAM sub-map is only considered for evaluation if it overlaps the COLMAP
# reference by at least this many image IDs. At the EndoMapper short-sequence
# frame rate (40-50 fps), 100 frames is 2-2.5 seconds of video — below that
# a map is too short to be metrically meaningful, and a multi-map system that
# emits many tiny maps would otherwise game ATE with cherry-picked fragments.
MIN_COMMON_FRAMES = 100


def parse_endomapper_args():
    parser = argparse.ArgumentParser(
        description="Evaluation and visualization of COLMAP vs SLAM on endomapper sequences."
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        required=True,
        help="Path to the COLMAP sequences folder (where each folder is a sequence)."
    )
    parser.add_argument(
        "--slam_path",
        type=str,
        required=True,
        help="Path to the SLAM sequences folder  (where each folder is a sequence)."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="If activated, show info. By default False."
    )
    parser.add_argument(
        "--results_file",
        type=str,
        default=None,
        help="Optional path to save evaluation results as JSON."
    )
    parser.add_argument(
        "--save_ply",
        action="store_true",
        help="Also write the (large) visualization PLYs per map/run. "
             "Off by default; metrics do not depend on them. Use for debugging."
    )

    args = parser.parse_args()

    if args.verbose:
        print("COLMAP path:", args.colmap_path)
        print("SLAM path:", args.slam_path)
        print("Verbose:", "yes" if args.verbose else "no")
        print("Results file:", args.results_file if args.results_file else "not set")

    return args

def list_slam_maps_for_sequence(slam_seq_path):
    """
    Discover SLAM maps inside a given SLAM sequence folder.

    Expected structure:
        <slam_seq>/<exp_name>/3D_maps/<map_id>/points3D.txt
        <slam_seq>/<exp_name>/camera_trajectory/cam_traj_map_<map_id>.txt

    Returns
    -------
    experiments : dict[str, dict[int, dict[str, str]]]
        Nested dictionary storing per-experiment, per-map file paths.

        Structure
        ---------
        experiments[exp_name][map_id] = {
            "points_file": <str>,
            "traj_file":   <str>,
        }

        Where:
        - exp_name (str) is the experiment identifier/name.
        - map_id (int) identifies a specific map within that experiment.
        - The inner dict contains the file paths associated with that map.
    num_existing_folders: int
        Number of existing folders within the given SLAM sequence where each experiment is saved
        (If they are greater than the elements in experiments, it is because they are not valid experiments).
    """
    slam_exp = {}
    num_existing_folders = 0

    if not os.path.isdir(slam_seq_path):
        print(f"[WARN] SLAM sequence folder not found: {slam_seq_path}")
        return slam_exp

    # Each subfolder of slam_seq_path is an experiment folder
    for exp_name in sorted(os.listdir(slam_seq_path)):
        exp_path = os.path.join(slam_seq_path, exp_name)
        if not os.path.isdir(exp_path):
            continue
        num_existing_folders += 1

        maps_root = os.path.join(exp_path, "3D_maps")
        traj_root = os.path.join(exp_path, "camera_trajectory")

        if not os.path.isdir(maps_root):
            print(f"[WARN] SLAM 3D_maps folder not found: {maps_root}")
            continue
        if not os.path.isdir(traj_root):
            print(f"[WARN] SLAM trajectory folder not found: {traj_root}")
            continue

        slam_maps = {}
        # Inside 3D_maps/<map_id>/
        for map_name in sorted(os.listdir(maps_root)):
            map_dir = os.path.join(maps_root, map_name)
            if not os.path.isdir(map_dir):
                continue

            # map_id is the folder name; we expect an integer
            try:
                map_id = int(map_name)
            except ValueError:
                print(f"[WARN] Map folder '{map_name}' is not a valid integer ID, skipping.")
                continue

            points_file = os.path.join(map_dir, "points3D.txt")

            # SLAM trajectory file. Naming: cam_traj_<map_id>.txt
            traj_file = os.path.join(traj_root, f"cam_traj_map_{map_id:03d}.txt")

            if not os.path.isfile(points_file):
                print(f"[WARN] SLAM points3D.txt not found for map_id={map_id}: {points_file}")
                continue
            if not os.path.isfile(traj_file):
                print(f"[WARN] SLAM trajectory file not found for map_id={map_id}: {traj_file}")
                continue

            slam_maps[map_id] = {
                "points_file": points_file,
                "traj_file": traj_file,
            }

        slam_exp[exp_name] = slam_maps

    return slam_exp, num_existing_folders


def build_image_name_set(traj_file, initial_id=0, use_timestamp=False):
    """
    Build a set of integer image IDs from a SLAM trajectory file.

    The trajectory file has lines of the form:
        timestamp, name_image, tx, ty, tz, qw, qx, qy, qz,
        frame_state, track_state, dataset_id, merged_frame_id

    Example of 'name_image':
        "23.jpg"
        "145_color.png"
        "0012_kf"
        "84"

    The function extracts the *numeric prefix* of name_image.

    Parameters
    ----------
    traj_file : str
        Path to the SLAM trajectory .txt file.
    initial_id : int (optional)
        Initial image ID to start with, defaults to 0.
    use_timestamp : bool (optional)
        If true, use timestamp instead of filename to get image IDs.
        Happens in some dataset as NR-SLAM

    Returns
    -------
    ids : set[int]
        Set of integer image IDs extracted from the trajectory.
    """

    if not os.path.isfile(traj_file):
        raise FileNotFoundError(f"Trajectory file not found: {traj_file}")

    image_ids = set()

    with open(traj_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = [t.strip() for t in line.split(",")]
            if len(tokens) < 2:
                continue

            name_image = tokens[1] # Namefile field
            if use_timestamp:
                name_image = tokens[0] # Timestamp field

            # Remove extension if present
            base = os.path.splitext(name_image)[0]

            # Extract leading number sequence
            num_str = ""
            for ch in base:
                if ch.isdigit():
                    num_str += ch

            if num_str:
                try:
                    image_ids.add(int(num_str) + initial_id)
                except ValueError:
                    pass

    return image_ids


def get_slam_seq_directories(path):
    """
    Verifies that `path` exists and is a directory,
    and returns a list of subdirectories.

    Args:
        path (str): input directory path

    Returns:
        list[str]: list of absolute paths of directories

    Raises:
        ValueError: if the path does not exist or is not a directory
    """
    if not os.path.exists(path):
        raise ValueError(f"Path does not exist: {path}")

    if not os.path.isdir(path):
        raise ValueError(f"Path is not a directory: {path}")

    # Check if the current frame is a defined sequence
    path = os.path.normpath(path)  # Erase last "/" if exists

    seq_dirs = []
    for name in os.listdir(path):
        full_path = os.path.join(path, name)
        if os.path.isdir(full_path):
            seq_dirs.append(full_path)

    seq_dirs.sort()

    return seq_dirs


def match_with_slam_maps(slam_maps, ref_ids_set, use_timestamp=False):
    matched_slam_exp = {}
    for slam_exp_name, slam_exp_data in slam_maps.items():
        # print(f"[INFO] SLAM exp {slam_exp_name} has={slam_exp_data}")
        matched_slam_maps = {}
        for map_id, slam_map in slam_exp_data.items():
            # print(f"[INFO] SLAM map id {map_id} has={slam_map}")
            sm_set = build_image_name_set(slam_map["traj_file"], use_timestamp=use_timestamp)

            # Drop sub-maps that overlap the COLMAP reference by fewer than
            # MIN_COMMON_FRAMES image IDs — those are too short to be meaningful.
            if len(ref_ids_set.intersection(sm_set)) >= MIN_COMMON_FRAMES:
                matched_slam_maps[map_id] = slam_map

        if matched_slam_maps:
            matched_slam_exp[slam_exp_name] = matched_slam_maps

    return matched_slam_exp


def match_colmap_slam_maps_by_images(colmap_seq_path, slam_seq_path, initial_id_colmap=0, initial_id_slam=0, verbose=False):
    """
    For a given pair of (COLMAP sequence, SLAM sequence), find all map pairs
    such that the set of image names is exactly the same.

    A single COLMAP map can match multiple SLAM maps, and vice versa.

    Args:
        colmap_seq_path (str): path to the COLMAP sequence folder
        slam_seq_path   (str): path to the SLAM sequence folder
        initial_id_colmap (int): optional offset for COLMAP read_colmap_images
        initial_id_slam   (int): optional offset for SLAM read_colmap_images
        verbose (bool): plot additional information (default: False)

    Returns:
        matches: list of dicts, each:
            {
                "ref_map": {
                    "map_id": str,
                    "images_file": str,
                    "points_file": str or None,
                },
                "slam_maps": [
                    {
                        exp_name: {
                            map_id: {
                            "points_file": str,
                            "traj_file":  str,
                            }
                        }
                    },
                    ...
                ],
                "num_slam_exp": int,
                "num_existing_exp": int
            }
    """
    colmap_maps = list_colmap_maps_for_sequence(colmap_seq_path)
    if verbose:
        print(f"colmap_maps: {colmap_maps}")
    if not colmap_maps:
        print(f"[WARN] No COLMAP sequences found in {colmap_seq_path}")
        return []
    slam_maps, num_existing_exp = list_slam_maps_for_sequence(slam_seq_path)
    if verbose:
        print(f"slam_maps: {slam_maps}")
    num_slam_exp = len(slam_maps)
    # print(f"number of valid experiments: {num_slam_exp} of {num_existing_exp}")
    if not slam_maps:
        print(f"[WARN] No SLAM sequences found in {slam_seq_path}")
        return []

    matches = []
    for cm in colmap_maps:
        cm_set = build_colmap_image_name_set(cm['images_file'], initial_id=initial_id_colmap)

        matched_slam_exp = match_with_slam_maps(slam_maps, cm_set)
        if matched_slam_exp:
            matches.append(
                {
                    "ref_map": cm,
                    "slam_maps": matched_slam_exp,
                    "num_slam_exp": num_slam_exp,
                    "num_existing_exp": num_existing_exp,
                }
            )

    return matches


def read_slam_trajectory(traj_file, initial_id=0):
    """
    Read a SLAM trajectory file with format:

        # timestamp, name_image, tx, ty, tz, qw, qx, qy, qz

    Every line is treated as a keyframe pose in camera-to-world format
    (T_wc), i.e. (tx,ty,tz) is the camera center C_w in world coordinates
    and (qw,qx,qy,qz) is the camera-to-world rotation.

    Returns:
        Dictionary with fields:
          - timestamps: (N,)
          - names: list of image-name strings
          - ids: (N,) integer IDs extracted from the numeric prefix of each name
          - centers_wc: (N, 3) camera centers in world coordinates
          - rotations_wc: (N, 3, 3) camera-to-world rotation matrices
    """

    timestamps = []
    names = []
    image_ids = []
    centers_wc = []
    rotations_wc = []

    with open(traj_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split(",")
            if len(tokens) < 9:
                # malformed line
                continue

            # 0: timestamp
            ts = float(tokens[0])

            # 1: name_image (contains image id)
            name = tokens[1]
            # Extract numeric prefix
            num_str = ""
            for ch in name:
                if ch.isdigit():
                    num_str += ch
                else:
                    break

            img_id = int(num_str) if num_str else None

            # 2-4: tx, ty, tz
            tx, ty, tz = map(float, tokens[2:5])
            if not np.isfinite([tx, ty, tz]).all():
                continue
            t_wc = np.array([tx, ty, tz], dtype=float)  # C_w = t_wc

            # 5-8: qw, qx, qy, qz
            qvec = np.array(tokens[5:9], dtype=float)
            R_wc = qvec2rotmat(qvec)

            timestamps.append(ts)
            names.append(name)
            image_ids.append(img_id)
            rotations_wc.append(R_wc)
            centers_wc.append(t_wc)

    return {
        "timestamps": np.array(timestamps, dtype=float),
        "names": names,
        "ids": np.array([(int(n) + initial_id) for n in image_ids], dtype=int),
        "centers_wc": np.vstack(centers_wc) if centers_wc else np.empty((0, 3)),
        "rotations_wc": np.stack(rotations_wc) if rotations_wc else np.empty((0, 3, 3)),
    }


def read_slam_data(slam_map, verbose=False):
    slam_traj_file = slam_map['traj_file']
    slam_points_file = slam_map['points_file']

    if verbose:
        print(f"SLAM traj file: {slam_traj_file}")
        print(f"SLAM points file: {slam_points_file}")

    # Read SLAM trajectory (every pose is treated as a keyframe).
    slam_traj = read_slam_trajectory(slam_traj_file)
    slam_traj_pose_wc = slam_traj['centers_wc']
    slam_traj_rot_wc = slam_traj['rotations_wc']
    slam_traj_ids = slam_traj['ids']
    slam_traj_names = slam_traj['names']

    # Read 3D points
    slam_points_3D, _, _, slam_points_color = read_colmap_points(slam_points_file)
    slam_points_3D_array = np.copy(np.array(slam_points_3D))
    slam_points_color_array = np.copy(np.array(slam_points_color))

    out_dir = os.path.dirname(os.path.normpath(slam_points_file))
    # Trajectory poses double as keyframe poses — every frame is a keyframe.
    return (slam_traj_pose_wc, slam_traj_rot_wc, slam_traj_ids,
            slam_traj_pose_wc, slam_traj_rot_wc, slam_traj_ids, slam_traj_names,
            slam_points_3D_array, slam_points_color_array, out_dir)