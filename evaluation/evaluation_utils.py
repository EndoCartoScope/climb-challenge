import os
import numpy as np
import copy
import open3d as o3d
from collections import defaultdict

from evaluate_ate_scale import build_correspondence_matrices, align
from colmap_utils import read_colmap_data, read_colmap_data_as_slam
from slam_utils import read_slam_data, match_colmap_slam_maps_by_images
from utils import (load_video_frame_counts, load_sequence_scales, load_sequence_trajectory_lengths,
                   compute_rpe_metrics, draw_in_files_ply, merge_triangle_meshes, compute_pose_pair_errors,
                   set_save_ply)


def match_sequences_by_folder(gt_list, slam_list):
    """
    Given two lists of directory paths (GT and SLAM),
    matches sequences by their last folder name.
    """
    gt_map = {os.path.basename(os.path.normpath(p)): p for p in gt_list}
    slam_map = {os.path.basename(os.path.normpath(p)): p for p in slam_list}

    matched_gt = []
    matched_slam = []

    for seq_name, gt_path in gt_map.items():
        if seq_name in slam_map:
            matched_gt.append(gt_path)
            matched_slam.append(slam_map[seq_name])

    return matched_gt, matched_slam


def build_result_record(exp_name, map_id, align_results, seq_data, result_rpe):
    ratio_loc_frames = -1.0
    ref_ratio_loc_frames = -1.0

    if 'total_video_frames' in seq_data and seq_data['total_video_frames'] > 0:
        ratio_loc_frames = align_results['total_frames'] / seq_data['total_video_frames']
        ref_ratio_loc_frames = seq_data['ref_num_images'] / seq_data['total_video_frames']

    return {
        "exp_name": exp_name,
        "map_id": map_id,
        "mean_ate": float(align_results['mean_ate']),
        "std_ate": float(align_results['std_ate']),
        "median_ate": float(align_results['median_ate']),
        "num_frames": align_results['total_frames'],
        "num_kfs": align_results['total_kfs'],
        "ratio_loc_frames": ratio_loc_frames * 100.0,
        "num_points": align_results['num_points'],
        "num_matched_poses": align_results['num_matched_poses'],
        "ref_images": seq_data['ref_num_images'],
        "ref_points": seq_data['ref_num_points'],
        "ref_ratio_loc_frames": ref_ratio_loc_frames * 100.0,
        "trans_rpe_1frame": float(result_rpe[1]['trans_mean']),
        "rot_rpe_deg_1frame": float(result_rpe[1]['rot_mean']),
        "trans_rpe_10frame": float(result_rpe[10]['trans_mean']),
        "rot_rpe_deg_10frame": float(result_rpe[10]['rot_mean']),
        "trans_rpe_20frame": float(result_rpe[20]['trans_mean']),
        "rot_rpe_deg_20frame": float(result_rpe[20]['rot_mean']),
        "mean_trans_rpe_40frame": float(result_rpe[40]['trans_mean']),
        "std_trans_rpe_40frame": float(result_rpe[40]['trans_std']),
        "mean_rot_rpe_deg_40frame": float(result_rpe[40]['rot_mean']),
        "std_rot_rpe_deg_40frame": float(result_rpe[40]['rot_std']),
    }


def get_pondered_maps_result(results_map):
    if not results_map:
        return None

    if len(results_map) == 1:
        only_result = copy.deepcopy(next(iter(results_map.values())))
        only_result["num_maps"] = 1
        return only_result

    num_maps = len(results_map)
    maps = list(results_map.values())

    total_matched = sum(m["num_matched_poses"] for m in maps)
    if total_matched <= 0:
        total_matched = 1

    weighted_keys = [
        "mean_ate", "std_ate", "median_ate",
        "trans_rpe_1frame", "rot_rpe_deg_1frame",
        "trans_rpe_10frame", "rot_rpe_deg_10frame",
        "trans_rpe_20frame", "rot_rpe_deg_20frame",
        "mean_trans_rpe_40frame", "std_trans_rpe_40frame",
        "mean_rot_rpe_deg_40frame", "std_rot_rpe_deg_40frame",
    ]

    sum_keys = [
        "ratio_loc_frames", "num_frames", "num_kfs",
        "num_points", "num_matched_poses",
    ]

    out = {"num_maps": num_maps}

    for key in weighted_keys:
        out[key] = float(sum(m[key] * m["num_matched_poses"] for m in maps) / total_matched)

    for key in sum_keys:
        out[key] = int(round(sum(m[key] for m in maps)))

    out["ref_images"] = maps[-1]["ref_images"]
    out["ref_points"] = maps[-1]["ref_points"]
    out["ref_ratio_loc_frames"] = float(maps[-1]["ref_ratio_loc_frames"])

    return out


def get_seq_traj_lengths(dict_seq_traj_lengths, seq_name):
    seq_traj_mm = -1.0
    dist_traj_mm = -1.0
    if dict_seq_traj_lengths and seq_name in dict_seq_traj_lengths:
        dict_seq_length = dict_seq_traj_lengths[seq_name]
        if dict_seq_length and "total_trajectory_mm" in dict_seq_length:
            seq_traj_mm = dict_seq_length['total_trajectory_mm']
        if dict_seq_length and "displacement_mm" in dict_seq_length:
            dist_traj_mm = dict_seq_length['displacement_mm']
    return seq_traj_mm, dist_traj_mm


def get_mean_seq_result(results, dict_seq_traj_lengths, seq_name, num_valid_exp=1, total_exp=1):
    agg = defaultdict(list)

    for _, r in results.items():
        for k, v in r.items():
            if isinstance(v, (int, float, np.number)):
                agg[k].append(float(v))

    dist_traj_mm = -1
    if dict_seq_traj_lengths:
        _, dist_traj_mm = get_seq_traj_lengths(dict_seq_traj_lengths, seq_name)

    return {
        "dist_traj_mm": dist_traj_mm,

        "mean_ate": np.mean(agg["mean_ate"]),
        "std_mean_ate": np.std(agg["mean_ate"]),
        "median_ate": np.mean(agg["median_ate"]),

        "num_maps": np.mean(agg["num_maps"]),
        "num_frames": round(np.mean(agg["num_frames"])),
        "ratio_loc_frames": np.mean(agg["ratio_loc_frames"]),
        "num_kfs": round(np.mean(agg["num_kfs"])),
        "num_points": round(np.mean(agg["num_points"])),
        "num_matched_poses": round(np.mean(agg["num_matched_poses"])),

        "ref_images": round(np.mean(agg["ref_images"])),
        "ref_points": round(np.mean(agg["ref_points"])),
        "ref_ratio_loc_frames": np.mean(agg["ref_ratio_loc_frames"]),

        "trans_rpe_1frame": np.mean(agg["trans_rpe_1frame"]),
        "rot_rpe_deg_1frame": np.mean(agg["rot_rpe_deg_1frame"]),
        "trans_rpe_10frame": np.mean(agg["trans_rpe_10frame"]),
        "rot_rpe_deg_10frame": np.mean(agg["rot_rpe_deg_10frame"]),
        "trans_rpe_20frame": np.mean(agg["trans_rpe_20frame"]),
        "rot_rpe_deg_20frame": np.mean(agg["rot_rpe_deg_20frame"]),

        "mean_trans_rpe_40frame": np.mean(agg["mean_trans_rpe_40frame"]),
        "std_trans_rpe_40frame": np.std(agg["mean_trans_rpe_40frame"]),
        "mean_rot_rpe_deg_40frame": np.mean(agg["mean_rot_rpe_deg_40frame"]),
        "std_rot_rpe_deg_40frame": np.std(agg["mean_rot_rpe_deg_40frame"]),

        "num_slam_exp": num_valid_exp,
        "num_existing_exp": total_exp,
    }


def plot_seq_results(seq_name, seq_results, line_spaces=82):
    print(f"\nResults for sequence: {seq_name}")
    if seq_results is None:
        print("No results found")
        return

    print("-" * line_spaces)
    print(f"{'Experiment':<12} {'# Maps':>8} {'Frames':>12} {'TFR (%)':>12} "
          f"{'Mean ATE(mm)':>16} {'RPE(deg) (d=40)':>17}")
    print("-" * line_spaces)

    for exp_name, r in seq_results.items():
        matched_frames_percent = (r['num_matched_poses'] / r['ref_images'] * 100.0) if r['ref_images'] else 0.0
        ate_str = f"{r['mean_ate']:.2f}+-{r['std_ate']:.2f}"
        rpe_str = f"{r['mean_rot_rpe_deg_40frame']:.2f}+-{r['std_rot_rpe_deg_40frame']:.2f}"
        print(f"{exp_name:<12} {r['num_maps']:>8d} {r['num_frames']:>12d} {matched_frames_percent:>12.2f} "
              f"{ate_str:>16} {rpe_str:>17}")

    print("-" * line_spaces)
    mean_ates = np.array([r['mean_ate'] for k, r in seq_results.items()])
    seq_mean_ate = np.mean(mean_ates)
    print(f"Mean ATE total for {seq_name}: {seq_mean_ate:.2f}\n")


def get_and_plot_mean_table_results(results, dict_seq_scale=None, dict_seq_traj_lengths=None, line_spaces=93):
    valid_sequence = False
    for exp_name, r in results.items():
        if r:
            valid_sequence = True
    if not valid_sequence:
        print("No valid sequences found to plot a table")
        return None
    if dict_seq_scale is None:
        dict_seq_scale = {}

    mean_num_maps = []
    mean_frame_loc = []
    mean_matched_frames_percent = []
    mean_ates = []
    mean_rpe_deg_40frame = []
    success_count = 0

    print("\n======================== GLOBAL MEAN SUMMARY ========================")
    print(f"{'Sequence':<14} {'# Maps':>8} {'Frames':>12} {'TFR (%)':>12} "
          f"{'Mean ATE(mm)':>16} {'RPE(deg) (d=40)':>17} {'Success':>8}")
    print("-" * line_spaces)
    show_valid_msg = False

    for seq_name, seq_results in results.items():
        mark_scaled = ''
        if dict_seq_scale and seq_name not in dict_seq_scale:
            mark_scaled = '*'

        mark_all_valid = ''
        sequence_incomplete = False

        if seq_results is None:
            seq_col = f"{seq_name}{mark_scaled}"
            print(f"{seq_col:<14} {'-':>8} {'-':>12} {'-':>12} "
                  f"{'-':>16} {'-':>17} {'✗':>8}")
            continue

        if "num_slam_exp" in seq_results and "num_existing_exp" in seq_results:
            if seq_results["num_slam_exp"] < seq_results["num_existing_exp"]:
                mark_all_valid = '+'
                show_valid_msg = True
                sequence_incomplete = True

        matched_frames_percent = (seq_results['num_matched_poses'] / seq_results['ref_images'] * 100.0) if seq_results['ref_images'] else 0.0

        seq_col = f"{seq_name}{mark_scaled}{mark_all_valid}"
        ate_str = f"{seq_results['mean_ate']:.2f}+-{seq_results['std_mean_ate']:.2f}"
        rpe_str = f"{seq_results['mean_rot_rpe_deg_40frame']:.2f}+-{seq_results['std_rot_rpe_deg_40frame']:.2f}"
        is_success = matched_frames_percent > 50.0
        if is_success:
            success_count += 1
        success_mark = "✓" if is_success else "✗"
        print(f"{seq_col:<14} {seq_results['num_maps']:>8.1f} {seq_results['num_frames']:>12d} "
              f"{matched_frames_percent:>12.2f} {ate_str:>16} {rpe_str:>17} {success_mark:>8}")

        if sequence_incomplete:
            continue
        mean_num_maps.append(seq_results["num_maps"])
        mean_frame_loc.append(seq_results["num_frames"])
        mean_matched_frames_percent.append(matched_frames_percent)
        mean_ates.append(seq_results['mean_ate'])
        mean_rpe_deg_40frame.append(seq_results['mean_rot_rpe_deg_40frame'])

    print("-" * line_spaces)

    global_mean = {
        "mean_num_maps": float(np.mean(mean_num_maps)) if mean_num_maps else 0.0,
        "mean_frames": int(round(np.mean(mean_frame_loc))) if mean_frame_loc else 0,
        "mean_matched_frames_percent": float(np.mean(mean_matched_frames_percent)) if mean_matched_frames_percent else 0.0,
        "mean_ates": float(np.mean(mean_ates)) if mean_ates else 0.0,
        "std_ates": float(np.std(mean_ates)) if mean_ates else 0.0,
        "mean_rpe_deg_40frame": float(np.mean(mean_rpe_deg_40frame)) if mean_rpe_deg_40frame else 0.0,
        "std_rpe_deg_40frame": float(np.std(mean_rpe_deg_40frame)) if mean_rpe_deg_40frame else 0.0,
        "success_count": success_count,
        "num_sequences": len(results),
    }

    mean_ate_str = f"{global_mean['mean_ates']:.2f}+-{global_mean['std_ates']:.2f}"
    mean_rpe_str = f"{global_mean['mean_rpe_deg_40frame']:.2f}+-{global_mean['std_rpe_deg_40frame']:.2f}"
    success_str = f"{success_count}/{len(results)}"
    print(f"{'Mean':<14} {global_mean['mean_num_maps']:>8.1f} {global_mean['mean_frames']:>12d} "
          f"{global_mean['mean_matched_frames_percent']:>12.2f} {mean_ate_str:>16} "
          f"{mean_rpe_str:>17} {success_str:>8}")
    print("-" * line_spaces)

    if show_valid_msg:
        print("+: Sequence where some experiments failed to complete\n")

    return global_mean


MATCHERS_REF = {
    "COLMAP_SLAM": match_colmap_slam_maps_by_images,
}

READER_REF = {
    "COLMAP": read_colmap_data,
}

READER_SLAM = {
    "SLAM": read_slam_data,
    "COLMAP": read_colmap_data_as_slam,
}


def is_valid_matcher_type(matcher_type):
    return matcher_type in MATCHERS_REF


def is_valid_ref_type(ref_type):
    return ref_type in READER_REF


def is_valid_slam_type(slam_type):
    return slam_type in READER_SLAM


def get_ref_matcher(matcher_type):
    try:
        return MATCHERS_REF[matcher_type]
    except KeyError:
        raise ValueError(
            f"Unsupported matcher: {matcher_type}. "
            f"Valid options are: {sorted(MATCHERS_REF.keys())}"
        )


def get_ref_reader(ref_type):
    try:
        return READER_REF[ref_type]
    except KeyError:
        raise ValueError(
            f"Unsupported ref_type: {ref_type}. "
            f"Valid options are: {sorted(READER_REF.keys())}"
        )


def get_slam_reader(slam_type):
    try:
        return READER_SLAM[slam_type]
    except KeyError:
        raise ValueError(
            f"Unsupported slam_type: {slam_type}. "
            f"Valid options are: {sorted(READER_SLAM.keys())}"
        )


def get_num_points(points_3D):
    if points_3D is None:
        return 0
    if hasattr(points_3D, "points"):
        return len(points_3D.points)
    return len(points_3D)


def match_and_align_sequences(ref_seqs, slam_seqs, ref_type, slam_type, verbose: bool = False,
                              file_num_frames: str = None, file_scales: str = None,
                              file_traj_lengths: str = None, save_ply: bool = False):
    """
    Matches REF (GT) and SLAM sequences, aligns their corresponding maps,
    computes pose metrics (ATE, RPE), and exports aligned results as PLY files.
    """
    if not is_valid_ref_type(ref_type):
        raise ValueError(
            f"Unsupported ref_type: {ref_type}. "
            f"Valid options are: {sorted(READER_REF.keys())}"
        )
    if not is_valid_slam_type(slam_type):
        raise ValueError(
            f"Unsupported slam_type: {slam_type}. "
            f"Valid options are: {sorted(READER_SLAM.keys())}"
        )
    matcher_type = f"{ref_type}_{slam_type}"
    if not is_valid_matcher_type(matcher_type):
        raise ValueError(
            f"Unsupported matcher: {matcher_type}. "
            f"Valid options are: {sorted(MATCHERS_REF.keys())}"
        )

    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    # Gate all visualization-PLY writing (large; not needed for metrics).
    set_save_ply(save_ply)

    all_results = {}
    all_results_mean = {}
    global_mean = {}
    dict_video_num_frames = None
    if file_num_frames:
        dict_video_num_frames = load_video_frame_counts(file_num_frames)
    dict_seq_scale_gt = None
    if file_scales:
        dict_seq_scale_gt = load_sequence_scales(file_scales)
    dict_seq_traj_lengths = None
    if file_traj_lengths:
        dict_seq_traj_lengths = load_sequence_trajectory_lengths(file_traj_lengths)

    for ref_seq, slam_seq in zip(ref_seqs, slam_seqs):
        seq_name = os.path.basename(os.path.normpath(ref_seq))
        vprint("--------")
        vprint("Sequence: ", seq_name)

        total_video_frames = None
        if dict_video_num_frames and seq_name in dict_video_num_frames:
            total_video_frames = dict_video_num_frames[seq_name]
            vprint("Total video frames: ", total_video_frames)

        matcher_ref_fn = get_ref_matcher(matcher_type)
        matched_maps = matcher_ref_fn(ref_seq, slam_seq, verbose=verbose)

        if not matched_maps:
            all_results[seq_name] = None
            all_results_mean[seq_name] = None
            print("No matches found for sequence:", seq_name)
            continue
        vprint(f"Matched maps: {matched_maps}")

        scale_factor_ref = None
        if dict_seq_scale_gt and seq_name in dict_seq_scale_gt:
            scale_factor_ref = dict_seq_scale_gt[seq_name]
            vprint("Reference scale factor: ", scale_factor_ref)

        for dict_maps in matched_maps:
            if "slam_maps" not in dict_maps or "ref_map" not in dict_maps:
                print("Invalid matched map structure for sequence:", seq_name)
                continue

            ref_matched_map = dict_maps['ref_map']
            if not ref_matched_map:
                print(f"[WARN] No reference maps found in {ref_seq}")
                continue

            ref_reader_fn = get_ref_reader(ref_type)
            ref_rot_wc, ref_center_wc, ref_ids, _, ref_points_3D, ref_points_color = (
                ref_reader_fn(ref_matched_map))

            if scale_factor_ref:
                ref_points_3D = ref_points_3D * scale_factor_ref
                ref_center_wc = ref_center_wc * scale_factor_ref

            ref_num_images = len(ref_center_wc)
            ref_num_points = get_num_points(ref_points_3D)
            seq_data = {
                "total_video_frames": total_video_frames if total_video_frames else -1,
                "ref_num_images": ref_num_images,
                "ref_num_points": ref_num_points
            }

            slam_matched_maps = dict_maps['slam_maps']

            vprint(f" Found {ref_num_images} images in reference sequence: {ref_seq}")
            vprint(f" Found {ref_num_points} points in reference sequence: {ref_seq}")
            vprint(f"Number of matched experiments: {len(slam_matched_maps)}")

            num_slam_exp = dict_maps['num_slam_exp']
            num_existing_exp = dict_maps['num_existing_exp']
            vprint(f" Found {num_slam_exp} experiments in reference sequence: {ref_seq}")
            vprint(f" There are {num_existing_exp} experiments in reference sequence: {ref_seq}")

            results_exp = {}
            for slam_exp_name, slam_map in slam_matched_maps.items():
                results_map = {}
                for slam_map_id, slam_map_dicts in slam_map.items():
                    vprint("-" * 15)
                    vprint(f"SLAM map: {slam_map_id}")

                    reader_slam_fn = get_slam_reader(slam_type)
                    (slam_traj_pose_wc, slam_traj_rot_wc, slam_traj_ids,
                     slam_image_poses_wc, slam_image_rotations_wc, slam_image_ids, slam_image_names,
                     slam_points_3D_array, slam_points_color_array,
                     out_dir) = reader_slam_fn(slam_map_dicts, verbose)

                    vprint(f"SLAM exp name: {slam_exp_name}")
                    vprint(f"SLAM frames: {len(slam_traj_pose_wc)}")
                    vprint(f"SLAM keyframes: {len(slam_image_poses_wc)}")
                    vprint(f"SLAM points: {len(slam_points_3D_array)}")

                    model, data, common_ids = build_correspondence_matrices(
                        slam_traj_pose_wc, slam_traj_ids,
                        ref_center_wc, ref_ids
                    )

                    rot, transGT, trans_errorGT, trans, trans_error, s = align(model, data)
                    mean_ate = trans_errorGT.mean()
                    std_ate = np.std(trans_errorGT)
                    median_ate = float(np.median(trans_errorGT))
                    num_matched_poses = model.shape[1]

                    vprint("Number of correspondences:", num_matched_poses)
                    vprint(f"Mean ATE (with scale): {mean_ate:.4f}")
                    vprint(f"Median ATE (with scale): {median_ate:.4f}")

                    slam_traj_rot_aligned_wc = np.array([rot @ R_s for R_s in slam_traj_rot_wc])
                    slam_traj_pose_aligned_wc = (s * rot @ slam_traj_pose_wc.T + transGT).T

                    slam_image_rotations_aligned_wc = np.array([rot @ R_s for R_s in slam_image_rotations_wc])
                    slam_image_poses_aligned_wc = (s * rot @ slam_image_poses_wc.T + transGT).T

                    slam_traj_matched_poses_aligned_wc = (s * rot @ model + transGT).T

                    slam_points_3D_aligned = (s * (rot @ slam_points_3D_array.T) + transGT).T

                    align_results = {
                        'mean_ate': mean_ate,
                        'std_ate': std_ate,
                        'median_ate': median_ate,
                        'total_frames': len(slam_traj_pose_wc),
                        'total_kfs': len(slam_image_poses_wc),
                        'num_points': len(slam_points_3D_array),
                        'num_matched_poses': num_matched_poses,
                    }

                    result_rpe = compute_rpe_metrics(ref_rot_wc, ref_center_wc, ref_ids,
                                                     slam_traj_rot_aligned_wc, slam_traj_pose_aligned_wc,
                                                     slam_traj_ids,
                                                     deltas=(1, 10, 20, 40), verbose=verbose)

                    vprint(f"Saving results to {out_dir}")

                    # -- REF --
                    ref_name = ref_type.lower()
                    draw_in_files_ply(ref_rot_wc, ref_center_wc, ref_ids,
                                      points_3D=ref_points_3D, points_color=ref_points_color, out_dir=out_dir,
                                      traj_filename=f"{ref_name}_trajectory.ply",
                                      pyramid_filename=f"{ref_name}_pyramid.ply",
                                      cam_axis_filename=f"{ref_name}_cam_axis.ply",
                                      points_filename=f"{ref_name}_points.ply",
                                      pyramid_color=[1, 0, 0])

                    # -- SLAM original --
                    slam_points_3D_orig_slam = s * slam_points_3D_array
                    points_color_orig_slam = slam_points_color_array.astype(float) / 255.0
                    draw_in_files_ply(slam_traj_rot_wc, slam_traj_pose_wc, slam_traj_ids,
                                      points_3D=slam_points_3D_orig_slam, points_color=points_color_orig_slam,
                                      out_dir=out_dir,
                                      traj_filename="slam_original_trajectory.ply",
                                      cam_axis_filename="slam_original_cam_axis.ply",
                                      points_filename="slam_original_points.ply",
                                      traj_color=[0, 1, 0])

                    draw_in_files_ply(slam_image_rotations_wc, slam_image_poses_wc, slam_image_ids,
                                      out_dir=out_dir,
                                      cam_axis_filename="slam_original_keyframe_axis.ply")

                    # -- SLAM aligned --
                    points_color_slam = slam_points_color_array.astype(float) / 255.0
                    draw_in_files_ply(slam_traj_rot_aligned_wc, slam_traj_pose_aligned_wc, slam_traj_ids,
                                      points_3D=slam_points_3D_aligned, points_color=points_color_slam,
                                      out_dir=out_dir,
                                      traj_filename="slam_trajectory.ply",
                                      pyramid_filename="slam_pyramid.ply",
                                      cam_axis_filename="slam_cam_axis.ply",
                                      points_filename="slam_points.ply",
                                      traj_color=[0, 1, 0],
                                      pyramid_color=[0, 0, 1])

                    draw_in_files_ply(slam_image_rotations_aligned_wc, slam_image_poses_aligned_wc, slam_image_ids,
                                      out_dir=out_dir,
                                      pyramid_filename="slam_keyframes.ply",
                                      cam_axis_filename="slam_keyframe_axis.ply",
                                      pyramid_color=[0, 0, 1])

                    draw_in_files_ply(None, slam_traj_matched_poses_aligned_wc, common_ids,
                                      out_dir=out_dir,
                                      traj_filename="slam_trajectory_matched.ply",
                                      traj_color=[0, 1, 0])

                    # -- Line error cylinders (visualization only) --
                    if save_ply:
                        (common_ids_err, trans_errors, rot_errors_deg, line_errors) = compute_pose_pair_errors(
                            ref_rot_wc, ref_center_wc, ref_ids,
                            slam_image_rotations_aligned_wc, slam_image_poses_aligned_wc, slam_image_ids,
                            color=[1, 0, 0], radius=0.05)
                        if common_ids_err is not None:
                            line_error_merged = merge_triangle_meshes(line_errors)
                            line_error_out_file = os.path.join(out_dir, "line_error.ply")
                            o3d.io.write_triangle_mesh(line_error_out_file, line_error_merged)

                    results_map[slam_map_id] = build_result_record(
                        exp_name=slam_exp_name,
                        map_id=slam_map_id,
                        align_results=align_results,
                        seq_data=seq_data,
                        result_rpe=result_rpe,
                    )

                if results_map:
                    results_exp[slam_exp_name] = get_pondered_maps_result(results_map)

            plot_seq_results(seq_name, results_exp)
            results_mean = get_mean_seq_result(results_exp, dict_seq_traj_lengths, seq_name,
                                               num_slam_exp, num_existing_exp)

            all_results[seq_name] = results_exp
            all_results_mean[seq_name] = results_mean

    global_mean = get_and_plot_mean_table_results(all_results_mean, dict_seq_scale_gt, dict_seq_traj_lengths)

    return all_results, all_results_mean, global_mean
