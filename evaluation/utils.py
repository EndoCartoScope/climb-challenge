import os
import json
from datetime import datetime, timezone
import numpy as np
import open3d as o3d
from typing import Dict
import csv
from pathlib import Path


# When False, draw_in_files_ply() is a no-op so the evaluator skips writing the
# (large) visualization PLYs. Metrics do not depend on them. Toggle via
# set_save_ply(); default True preserves standalone behaviour unless a caller
# opts out (e.g. slam_evaluation.py without --save_ply).
_SAVE_PLY = True


def set_save_ply(enabled: bool) -> None:
    global _SAVE_PLY
    _SAVE_PLY = bool(enabled)


def qvec2rotmat(qvec):
    """
        Convert a quaternion into a 3×3 rotation matrix, following the
        convention for quaternion ordering: (qw, qx, qy, qz).

        The returned rotation matrix maps world coordinates into
        camera coordinates (R_cw), consistent with:

            X_cam = R_cw * X_world + t_cw

        Parameters
        ----------
        qvec : array_like of shape (4,)
            Quaternion (qw, qx, qy, qz). The function normalizes the quaternion
            internally to ensure numerical stability.

        Returns
        -------
        R : ndarray of shape (3, 3)
            Rotation matrix corresponding to the input quaternion.
    """
    # Normalize
    qvec = qvec / np.linalg.norm(qvec)
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y]
    ], dtype=float)


def create_camera_wireframe(R_wc, C, scale=0.3, aspect=1.5, fov_deg=60.0, color=None):
    """
    Returns a pyramid lineset to represent the camera.
    """
    if color is None:
        color = [1, 0, 0]
    fov = np.deg2rad(fov_deg)
    depth = scale

    z = depth

    half_h = depth * np.tan(fov / 2.0)
    half_w = half_h * aspect

    # Pyramid in camera coordinates (in front of the camera = +Z)
    vertices_cam = np.array([
        [0,       0,        0],
        [-half_w, -half_h,  z],
        [ half_w, -half_h,  z],
        [ half_w,  half_h,  z],
        [-half_w,  half_h,  z],
    ])

    # world = R_wc * cam + C, where C = t_wc
    vertices_world = (R_wc @ vertices_cam.T).T + C.reshape(1,3)

    # Pyramid edges
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # lados
        [1, 2], [2, 3], [3, 4], [4, 1]   # base
    ]

    colors = [color for _ in lines]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    return line_set


def create_camera_axes(R_wc, C, axis_length=0.2, colors=None):
    """
    Draw the camera local axes (X, Y, Z) as a colored LineSet.

    Convention:
        - R_wc maps camera coordinates -> world coordinates
        - C is the camera center in world coordinates (t_wc)
        - Camera axes in camera frame:
            X = [1,0,0], Y = [0,1,0], Z = [0,0,1]

    Parameters
    ----------
    R_wc : (3,3) ndarray
        Camera-to-world rotation.
    C : (3,) ndarray
        Camera center in world coordinates.
    axis_length : float
        Length of each axis segment in world units.
    colors : dict or None
        Optional custom colors in [0,1]. Example:
            {"x":[1,0,0], "y":[0,1,0], "z":[0,0,1]}
        If None, uses X=red, Y=green, Z=blue.

    Returns
    -------
    axes_ls : open3d.geometry.LineSet
        LineSet with 3 segments: origin->X, origin->Y, origin->Z.
    """
    if colors is None:
        colors = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}

    C = np.asarray(C, dtype=float).reshape(3)
    R_wc = np.asarray(R_wc, dtype=float).reshape(3, 3)

    # Endpoints in camera frame
    p0_cam = np.array([0.0, 0.0, 0.0])
    px_cam = np.array([axis_length, 0.0, 0.0])
    py_cam = np.array([0.0, axis_length, 0.0])
    pz_cam = np.array([0.0, 0.0, axis_length])

    # Transform to world: p_w = R_wc * p_c + C
    p0_w = R_wc @ p0_cam + C
    px_w = R_wc @ px_cam + C
    py_w = R_wc @ py_cam + C
    pz_w = R_wc @ pz_cam + C

    points = np.vstack([p0_w, px_w, py_w, pz_w])  # 0=origin, 1=X, 2=Y, 3=Z
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=int)

    line_colors = np.array([colors["x"], colors["y"], colors["z"]], dtype=float)

    axes_ls = o3d.geometry.LineSet()
    axes_ls.points = o3d.utility.Vector3dVector(points)
    axes_ls.lines = o3d.utility.Vector2iVector(lines)
    axes_ls.colors = o3d.utility.Vector3dVector(line_colors)

    return axes_ls


def cylinder_between(p1, p2, radius=0.01, color=None):
    if color is None:
        color = [0, 1, 0]
    if color is None:
        color = [0, 1, 0]
    p1 = np.array(p1)
    p2 = np.array(p2)
    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length < 1e-6:
        return None

    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length)
    cyl.paint_uniform_color(color)

    cyl.translate([0, 0, length/2])

    z = np.array([0, 0, 1])
    axis = np.cross(z, direction)
    angle = np.arccos(np.dot(z, direction) / length)

    if np.linalg.norm(axis) > 1e-6:
        axis = axis / np.linalg.norm(axis)
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
        cyl.rotate(R, center=[0, 0, 0])

    cyl.translate(p1)

    return cyl


def compute_pose_pair_errors(
        R_colmap_wc, C_colmap_wc, ids_colmap,
        R_slam_wc, C_slam_wc, ids_slam,
        color=None,  # line color
        radius=0.02,      # cylinder radius
):
    """
    Automatically matches COLMAP <-> SLAM poses using numeric IDs
    and computes the error for each matched pair.

    Instead of returning a LineSet, it returns a LIST of cylinders
    (o3d TriangleMesh) that connect each pair of centers C_colmap <-> C_slam.

    Inputs:
        R_colmap_wc : (Nc, 3,3)  reference rotations (cam->world)
        C_colmap_wc : (Nc, 3)    reference camera centers in world coordinates
        ids_colmap : (Nc,)    reference IDs
        R_slam_wc : (Ns,3,3)     estimated rotations (cam->world)
        C_slam_wc : (Ns,3)       estimated camera centers in world coordinates
        ids_slam : (Ns,)      estimated IDs
        color : [3]           RGB color in [0,1] for the cylinders
        radius : float        cylinder radius

    Returns:
        common_ids          : array (N,)          matched IDs
        errors_translation  : array (N,)          translational error per pair
        errors_rotation_deg : array (N,)          angular error (degrees) per pair
        cylinders_error     : list[TriangleMesh]  cylinders between C_colmap and C_slam
    """
    if color is None:
        color = [1, 0, 0]
    if radius <= 0:
        radius = 0.02

    # --- Maps: id -> idx ---
    dict_col = {int(id): k for k, id in enumerate(ids_colmap)}
    dict_slam = {int(id): k for k, id in enumerate(ids_slam)}

    common_ids = sorted(set(dict_col.keys()) & set(dict_slam.keys()))
    if len(common_ids) == 0:
        print("There are not common ids between COLMAP and SLAM.")
        return None, None, None, None

    trans_errors = []
    rot_errors_deg = []
    cylinders = []

    for li, img_id in enumerate(common_ids):
        k_col = dict_col[img_id]
        k_slam = dict_slam[img_id]

        Cc = C_colmap_wc[k_col]
        Cs = C_slam_wc[k_slam]
        Rc = R_colmap_wc[k_col].T
        Rs = R_slam_wc[k_slam].T

        e_trans = np.linalg.norm(Cs - Cc)
        trans_errors.append(e_trans)

        R_err = Rs @ Rc.T
        angle = np.arccos(np.clip((np.trace(R_err) - 1) / 2, -1.0, 1.0))
        angle_deg = np.degrees(angle)
        rot_errors_deg.append(angle_deg)

        cyl = cylinder_between(Cc, Cs, radius=radius, color=color)
        if cyl is not None:
            cylinders.append(cyl)

    return (
        np.array(common_ids),
        np.array(trans_errors),
        np.array(rot_errors_deg),
        cylinders
    )


def compute_rpe(
    R_ref_wc, C_ref_wc, ids_ref,
    R_est_wc, C_est_wc, ids_est,
    delta=1,
):
    """
    Computes the Relative Pose Error (RPE) between a reference trajectory (ref)
    and an estimated one (est), using a 'delta' frame interval.

    Inputs:
        R_ref_wc  : (N_ref, 3, 3)  reference rotations, format R_wc (cam -> world)
        C_ref_wc  : (N_ref, 3)     reference camera centers in world coordinates
        ids_ref: (N_ref,)       numeric IDs of the reference poses

        R_est_wc  : (N_est, 3, 3)  estimated rotations, format R_wc (cam -> world)
        C_est_wc  : (N_est, 3)     estimated camera centers in world coordinates
        ids_est: (N_est,)       numeric IDs of the estimated poses

        delta  : index offset for the relative pose calculation (e.g., 1, 5, 10)

    Returns:
        pair_ids_ref      : (M, 2)  reference IDs used in each pair (id_k, id_{k+delta})
        trans_errors      : (M,)    relative translational error per pair
        rot_errors_deg    : (M,)    relative rotational error per pair (degrees)
    """

    # Maps id -> index
    ref_idx = {int(i): k for k, i in enumerate(ids_ref)}
    est_idx = {int(i): k for k, i in enumerate(ids_est)}

    # Common IDs
    common_ids = sorted(set(ref_idx.keys()) & set(est_idx.keys()))
    if len(common_ids) == 0:
        raise ValueError("There are not common IDs between reference and estimated poses.")

    pair_ids_ref = []
    trans_errors = []
    rot_errors_deg = []

    def build_Twc(R_wc, C):
        """
        Build T_wc (cam -> world):
            R_wc
            t_wc = C
        """
        t_wc = C
        T = np.eye(4)
        T[:3, :3] = R_wc
        T[:3, 3] = t_wc
        return T

    for i_id in common_ids:
        j_id = i_id + delta
        if j_id not in common_ids:
            continue

        k_ref_i = ref_idx[i_id]
        k_ref_j = ref_idx[j_id]
        k_est_i = est_idx[i_id]
        k_est_j = est_idx[j_id]

        T_ref_i = build_Twc(R_ref_wc[k_ref_i], C_ref_wc[k_ref_i])
        T_ref_j = build_Twc(R_ref_wc[k_ref_j], C_ref_wc[k_ref_j])

        T_est_i = build_Twc(R_est_wc[k_est_i], C_est_wc[k_est_i])
        T_est_j = build_Twc(R_est_wc[k_est_j], C_est_wc[k_est_j])

        T_ref_rel = np.linalg.inv(T_ref_i) @ T_ref_j
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j

        T_err = np.linalg.inv(T_ref_rel) @ T_est_rel

        R_err = T_err[:3, :3]
        t_err = T_err[:3, 3]

        e_trans = np.linalg.norm(t_err)

        cos_angle = (np.trace(R_err) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        e_rot_deg = np.degrees(angle)

        pair_ids_ref.append((i_id, j_id))
        trans_errors.append(e_trans)
        rot_errors_deg.append(e_rot_deg)

    if len(pair_ids_ref) == 0:
        # raise ValueError("No matching (id, id+delta) pairs were found for the RPE.")
        print("No matching (id, id+delta) pairs were found for the RPE.")
        return (
            np.array([0], dtype=int),
            np.array([0], dtype=float),
            np.array([0], dtype=float),
        )

    return (
        np.array(pair_ids_ref, dtype=int),
        np.array(trans_errors, dtype=float),
        np.array(rot_errors_deg, dtype=float),
    )


def compute_rpe_metrics(gt_rot_wc, gt_center_wc, gt_ids,
                        slam_rot_wc, slam_pose_wc, slam_ids,
                        deltas=(1, 10, 20, 40), verbose=False):
    out = {}
    for delta in deltas:
        pair_ids, trans_rpe, rot_rpe_deg = compute_rpe(
            gt_rot_wc, gt_center_wc, gt_ids,
            slam_rot_wc, slam_pose_wc, slam_ids,
            delta=delta
        )
        out[delta] = {
            "pair_ids": pair_ids,
            "trans_mean": float(trans_rpe.mean()),
            "trans_std": float(trans_rpe.std()),
            "rot_mean": float(rot_rpe_deg.mean()),
            "rot_std": float(rot_rpe_deg.std()),
        }
        if verbose:
            print(f"Number of elements for RPE (with {delta} frames):", len(pair_ids))
            print(f"Mean translational RPE (with {delta} frames): {out[delta]['trans_mean']:.4f}")
            print(f"Mean rotational RPE (deg) (with {delta} frames): {out[delta]['rot_mean']:.4f}")
    return out


def merge_linesets(lineset_list):
    """Merge a list of LineSet into a single LineSet."""
    merged = o3d.geometry.LineSet()

    all_points = []
    all_lines = []
    all_colors = []

    p_offset = 0
    for ls in lineset_list:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        if ls.has_colors():
            cols = np.asarray(ls.colors)
        else:
            cols = np.tile(np.array([[1.0, 1.0, 1.0]]), (lines.shape[0], 1))

        all_points.append(pts)
        all_lines.append(lines + p_offset)
        all_colors.append(cols)

        p_offset += pts.shape[0]

    merged.points = o3d.utility.Vector3dVector(
        np.vstack(all_points) if all_points else np.zeros((0, 3))
    )
    merged.lines = o3d.utility.Vector2iVector(
        np.vstack(all_lines) if all_lines else np.zeros((0, 2), dtype=np.int32)
    )
    merged.colors = o3d.utility.Vector3dVector(
        np.vstack(all_colors) if all_colors else np.zeros((0, 3))
    )

    return merged


def merge_triangle_meshes(mesh_list):
    """Merge the list of TriangleMesh into a single triangle mesh."""
    merged = o3d.geometry.TriangleMesh()

    all_vertices = []
    all_triangles = []
    all_vertex_colors = []

    v_offset = 0
    for m in mesh_list:
        if m is None:
            continue

        verts = np.asarray(m.vertices)
        tris  = np.asarray(m.triangles)

        # opcional: saltar meshes vacíos
        if verts.size == 0 or tris.size == 0:
            continue

        all_vertices.append(verts)
        all_triangles.append(tris + v_offset)

        if m.has_vertex_colors():
            all_vertex_colors.append(np.asarray(m.vertex_colors))

        v_offset += verts.shape[0]

    merged.vertices = o3d.utility.Vector3dVector(
        np.vstack(all_vertices) if all_vertices else np.zeros((0, 3))
    )
    merged.triangles = o3d.utility.Vector3iVector(
        np.vstack(all_triangles) if all_triangles else np.zeros((0, 3), dtype=np.int32)
    )

    if all_vertex_colors:
        merged.vertex_colors = o3d.utility.Vector3dVector(np.vstack(all_vertex_colors))

    return merged


def lineset_to_cylinders(lineset, radius=0.02, color=None):
    """
    Convert an Open3D LineSet into a list of cylinder TriangleMeshes.

    Parameters
    ----------
    lineset : o3d.geometry.LineSet
        Input LineSet.
    radius : float
        Cylinder radius.
    color : list[float] or None
        If provided, use this RGB color in [0,1] for all cylinders.
        If None, use per-line colors from `lineset.colors` when available.
        If LineSet has no colors, falls back to blue [0,0,1].

    Returns
    -------
    cylinders : list[o3d.geometry.TriangleMesh]
        One cylinder per line segment.
    """
    pts = np.asarray(lineset.points)
    lines = np.asarray(lineset.lines)

    # Optional per-line colors from LineSet
    ls_colors = None
    if color is None:
        try:
            c = np.asarray(lineset.colors)
            if c.size > 0:
                ls_colors = c
        except Exception:
            ls_colors = None

    cylinders = []
    for li, (i0, i1) in enumerate(lines):
        p0 = pts[i0]
        p1 = pts[i1]

        if color is not None:
            c_use = color
        elif ls_colors is not None:
            # If colors are per-line, pick matching index; if shorter, clamp
            idx = li if li < len(ls_colors) else -1
            c_use = ls_colors[idx].tolist()
        else:
            c_use = [0, 0, 1]  # fallback

        cyl = cylinder_between(p0, p1, radius=radius, color=c_use)
        if cyl is not None:
            cylinders.append(cyl)

    return cylinders


def get_file_num_frames_per_video(folder_path: str,
                                  file_name: str ="climb_seq_frames.csv"):
    basic_path = os.path.abspath(folder_path)
    file_path = os.path.join(basic_path, file_name)
    if os.path.exists(file_path) and  os.path.isfile(file_path):
        return file_path

    return None


def get_file_sequence_scales(folder_path: str,
                             file_name: str ="scales.csv"):
    basic_path = os.path.abspath(folder_path)
    file_path = os.path.join(basic_path, file_name)
    if os.path.exists(file_path) and  os.path.isfile(file_path):
        return file_path

    return None


def load_video_frame_counts(csv_file: str) -> Dict[str, int]:
    """
    Read a CSV file with columns:
        video_name,num_frames

    Returns
    -------
    frame_counts : dict[str, int]
        Dictionary mapping video_name -> num_frames
    """
    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(f"File not found: {csv_file}")

    frame_counts = {}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "video_name" not in reader.fieldnames or "num_frames" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'video_name' and 'num_frames' columns. "
                f"Found: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader):
            name = row["video_name"].strip()
            try:
                nframes = int(row["num_frames"])
            except ValueError as e:
                raise ValueError(
                    f"Invalid num_frames at row {row_idx + 2}: {row['num_frames']}"
                ) from e

            frame_counts[name] = nframes

    return frame_counts


def load_sequence_scales(csv_file: str) -> Dict[str, int]:
    """
    Read a CSV file with columns:
        sequence,points_ply,idx0,idx1,dist_units,target_mm,scale_to_target,center_mode,out_points_scaled,out_picked_points_scaled,out_picked_line_scaled

    Returns
    -------
    seq_scales : dict[str, float]
        Dictionary mapping sequence -> scale_to_target
    """
    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(f"File not found: {csv_file}")

    seq_scales = {}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "sequence" not in reader.fieldnames or "scale_to_target" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'sequence' and 'scale_to_target' columns. "
                f"Found: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader):
            seq_name = row["sequence"].strip()
            try:
                scale = float(row["scale_to_target"])
            except ValueError as e:
                raise ValueError(
                    f"Invalid 'scale_to_target' at row {row_idx + 2}: {row['scale_to_target']}"
                ) from e

            seq_scales[seq_name] = scale

    return seq_scales


def load_sequence_trajectory_lengths(csv_file: str) -> Dict[str, float]:
    """
    Read a CSV file with columns:
        seq_name,total_trajectory_mm,displacement_mm

    Returns
    -------
    seq_traj_lengths : dict[str, float]
        Dictionary mapping seq_name -> total_trajectory_mm
    """
    csv_file = Path(csv_file)
    if not csv_file.is_file():
        raise FileNotFoundError(f"File not found: {csv_file}")

    seq_traj_lengths = {}

    with open(csv_file, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "seq_name" not in reader.fieldnames or "total_trajectory_mm" not in reader.fieldnames:
            raise ValueError(
                f"CSV must contain 'seq_name' and 'total_trajectory_mm' columns. "
                f"Found: {reader.fieldnames}"
            )

        for row_idx, row in enumerate(reader):
            seq_name = row["seq_name"].strip()
            try:
                total_trajectory_mm = float(row["total_trajectory_mm"])
            except ValueError as e:
                raise ValueError(
                    f"Invalid 'total_trajectory_mm' at row {row_idx + 2}: {row['total_trajectory_mm']}"
                ) from e

            try:
                displacement_mm = float(row["displacement_mm"])
            except ValueError as e:
                raise ValueError(
                    f"Invalid 'displacement_mm' at row {row_idx + 2}: {row['displacement_mm']}"
                ) from e

            seq_traj_lengths[seq_name] = (
                {
                    'total_trajectory_mm': total_trajectory_mm,
                    'displacement_mm': displacement_mm,
                }
            )

    return seq_traj_lengths


def ensure_ply_extension(filename: str) -> str:
    """
    Ensures that the filename has .ply extension.
    - If no extension → adds .ply
    - If different extension → replaces with .ply
    """
    p = Path(filename)
    return str(p.with_suffix(".ply"))


def draw_in_files_ply(rot_wc, center_wc, ids, points_3D=None, points_color=None,
                      out_dir=None, traj_color=None, pyramid_color=None,
                      traj_filename=None, pyramid_filename=None, cam_axis_filename=None,
                      points_filename=None):
    if not _SAVE_PLY:
        return
    if out_dir is None:
        print(f"out_dir cannot be None")
        return
    if not Path(out_dir).is_dir():
        print(f"out_dir is not a directory: {out_dir}")
        return

    if traj_filename:
        # --Trajectory --
        trajectory = []
        if traj_color is None:
            traj_color = [0, 0, 0]
        for i in range(len(ids) - 1):
            id_i = int(ids[i])
            id_j = int(ids[i + 1])

            p_i = center_wc[i]
            p_j = center_wc[i + 1]

            if id_j == id_i + 1:
                cyl = cylinder_between(p_i, p_j, radius=0.1, color=traj_color)
                trajectory.append(cyl)
            else:
                # Put marker at the end of the segment
                marker_pos = p_i

                sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
                sph.compute_vertex_normals()
                sph.paint_uniform_color(traj_color)

                # translate by vector (not using relative=True to avoid surprises)
                sph.translate(marker_pos, relative=False)

                trajectory.append(sph)
        merged_trajectory = merge_triangle_meshes(trajectory)
        merged_trajectory.compute_vertex_normals()
        traj_filename = ensure_ply_extension(traj_filename)
        traj_out_file = os.path.join(out_dir, traj_filename)
        o3d.io.write_triangle_mesh(traj_out_file, merged_trajectory)

    if pyramid_filename:
        # -- Pyramid poses --
        pyramid = []
        if pyramid_color is None:
            pyramid_color = [1, 0, 0]
        for R, C in zip(rot_wc, center_wc):
            cam_lines = create_camera_wireframe(R, C,
                                                scale=1.0, aspect=1.5, fov_deg=60,
                                                color=pyramid_color)
            pyramid.append(cam_lines)
        merged_pyramids = merge_linesets(pyramid)
        pyramid_cylinders = lineset_to_cylinders(merged_pyramids, radius=0.02,
                                                        color=pyramid_color)
        pyramid_mesh = merge_triangle_meshes(pyramid_cylinders)
        pyramid_filename = ensure_ply_extension(pyramid_filename)
        pyramid_out_file = os.path.join(out_dir, pyramid_filename)
        o3d.io.write_triangle_mesh(pyramid_out_file, pyramid_mesh)

    if cam_axis_filename:
        # -- Cameras axis --
        cameras_axis = []
        for R, C in zip(rot_wc, center_wc):
            cam_lines = create_camera_axes(R, C, axis_length=0.5)
            cameras_axis.append(cam_lines)
        merged_axis = merge_linesets(cameras_axis)
        axis_cylinders = lineset_to_cylinders(merged_axis, radius=0.02)
        axis_mesh = merge_triangle_meshes(axis_cylinders)
        camera_axis_filename = ensure_ply_extension(cam_axis_filename)
        axis_out_file = os.path.join(out_dir, cam_axis_filename)
        o3d.io.write_triangle_mesh(axis_out_file, axis_mesh)

    if points_filename and points_3D is not None:

        if isinstance(points_3D, o3d.geometry.PointCloud):
            # Already a point cloud
            points_filename = ensure_ply_extension(points_filename)
            points_out_file = os.path.join(out_dir, points_filename)
            o3d.io.write_point_cloud(points_out_file, points_3D)

        elif points_color is not None:
            # Raw arrays → build point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points_3D)
            points_color = points_color.astype(float) / 255.0
            pcd.colors = o3d.utility.Vector3dVector(points_color)
            points_filename = ensure_ply_extension(points_filename)
            points_out_file = os.path.join(out_dir, points_filename)
            o3d.io.write_point_cloud(points_out_file, pcd)

        else:
            print("points_3D provided but points_color is missing")


def _to_builtin(obj):
    """Convert numpy/scalar objects into JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_results_json(results_file, ref_path, slam_path, all_results, all_results_mean, global_mean=None):
    if not results_file:
        return

    out_dir = os.path.dirname(os.path.abspath(results_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "colmap_path": ref_path,
        "slam_path": slam_path,
        "per_experiment": _to_builtin(all_results),
        "per_sequence_mean": _to_builtin(all_results_mean),
        "global_mean": _to_builtin(global_mean) if global_mean else {}
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved evaluation results to: {results_file}")