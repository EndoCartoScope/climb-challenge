import os
import numpy as np

from utils import qvec2rotmat


def read_colmap_images(images_file, initial_id=0):
    if not os.path.isfile(images_file):
        raise FileNotFoundError(f"The file {images_file} does not exist.")

    image_poses_wc = []
    image_rotations_wc = []
    image_ids = []
    image_names = []

    with open(images_file) as file:
        lines = file.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line or line[0] == "#":
            i += 1
            continue

        tokens = line.split()
        if len(tokens) < 10:
            i += 1
            continue

        qvec = np.array([float(tokens[1]), float(tokens[2]),
                         float(tokens[3]), float(tokens[4])])
        t_cw = np.array([float(tokens[5]), float(tokens[6]), float(tokens[7])])

        R_cw = qvec2rotmat(qvec)
        R_wc = R_cw.T
        C_w = -R_wc @ t_cw # camera center in world, C_w = t_wc

        name = tokens[9]  # Example: "23.jpg" or "23"
        # Extract number
        num_str = ""
        for ch in name:
            if ch.isdigit():
                num_str += ch

        try:
            img_id = int(num_str)
        except ValueError:
            # if it cannot be converted into int, it is avoided
            print(f"The name {name} is not a valid image ID.")
            i += 2
            continue

        image_rotations_wc.append(R_wc)
        image_poses_wc.append(C_w)
        image_ids.append(img_id + initial_id)
        image_names.append(name)

        i += 2  # avoid line with points

    return (np.stack(image_rotations_wc),
            np.stack(image_poses_wc),
            np.array(image_ids, dtype=int),
            image_names)


def read_colmap_points(points_file, filter_points=False):
    if not os.path.isfile(points_file):
        raise FileNotFoundError(f"The file {points_file} does not exist.")

    points_3D_list = []
    points_3D_color = []
    points_bad = []
    keys_list = []
    track_lengths_kept = []

    with open(points_file) as file:
        for line in file:
            if not line or line[0] == '#':
                continue

            tokens = line.split()

            key = int(tokens[0])
            point = np.array([float(tokens[i]) for i in range(1, 4)])

            r = int(tokens[4])
            g = int(tokens[5])
            b = int(tokens[6])
            color = np.array([r, g, b])
            if filter_points:

                if (r < 120 and g < 120 and b < 120) or (r > 220 and g > 220 and b > 220):
                    points_bad.append(point)
                    continue

                error = float(tokens[7])

                # Compute track length without repeated IMAGE_IDs
                # Remaining tokens are alternating IMAGE_ID, POINT2D_IDX
                track_tokens = tokens[8:]
                unique_image_ids = set(int(track_tokens[i]) for i in range(0, len(track_tokens), 2))
                track_length = len(unique_image_ids)

                if track_length < 4:
                    points_bad.append(point)

                if error > 2:
                    points_bad.append(point)
                    continue

                track_lengths_kept.append(track_length)

            points_3D_list.append(point)
            points_3D_color.append(color)
            keys_list.append(key)

    return points_3D_list, keys_list, track_lengths_kept, points_3D_color


def normalize_image_name(name):
    """
    Takes an image name like '23.jpg', '0005.png', '18', '18.jpeg'
    and returns the integer ID (e.g., 23, 5, 18).
    """
    base = os.path.splitext(name)[0]  # remove extension
    # Extract numeric
    num_str = ""
    for ch in base:
        if ch.isdigit():
            num_str += ch

    try:
        return int(num_str)
    except ValueError:
        raise ValueError(f"Image name '{name}' cannot be converted to int.")


def build_colmap_image_name_set(images_file, initial_id=0):
    """
    Reads a COLMAP-format images.txt file and returns the set of image names.
    """
    _, _, _, image_names = read_colmap_images(images_file, initial_id=initial_id)
    return { normalize_image_name(n) for n in image_names }


def list_colmap_multimaps_for_sequence(colmap_seq_path):
    """
    Discover COLMAP maps inside a given COLMAP sequence folder.

    Expected structure:
        <colmap_seq_path>/<map_id>/images.txt
        <colmap_seq_path>/<map_id>/points3D.txt

    Returns:
        colmap_maps: list of dicts with keys:
            - 'map_id'
            - 'images_file'
            - 'points_file'
    """
    colmap_maps = []

    if not os.path.isdir(colmap_seq_path):
        print(f"[WARN] COLMAP sequence folder not found: {colmap_seq_path}")
        return colmap_maps

    print(f"[INFO] COLMAP multimap folder found: {colmap_seq_path}")

    for map_name in sorted(os.listdir(colmap_seq_path)):
        map_dir = os.path.join(colmap_seq_path, map_name)
        if not os.path.isdir(map_dir):
            continue

        try:
            map_id = int(map_name)
        except ValueError:
            print(f"[WARN] Map folder '{map_name}' is not a valid integer ID, skipping.")
            continue

        images_file = os.path.join(map_dir, "images.txt")
        points_file = os.path.join(map_dir, "points3D.txt")

        if not os.path.isfile(images_file):
            print(f"[WARN] COLMAP images.txt not found for map_id={map_id}: {images_file}")
            continue
        if not os.path.isfile(points_file):
            print(f"[WARN] COLMAP points3D.txt not found for map_id={map_id}: {points_file}")
            continue

        colmap_maps.append(
            {
                "map_id": map_id,
                "images_file": images_file,
                "points_file": points_file,
            }
        )

    return colmap_maps


def list_colmap_maps_for_sequence(colmap_seq_path):
    """
    Discover COLMAP maps inside a given COLMAP sequence folder.

    Expected structure:
        <colmap_seq>/results_txt/images.txt
        <colmap_seq>/results_txt/points3D.txt

    Returns:
        colmap_maps: list of dicts with keys:
            - 'map_id'
            - 'images_file'
            - 'points_file'
    """
    colmap_maps = []

    if not os.path.isdir(colmap_seq_path):
        print(f"[WARN] COLMAP sequence folder not found: {colmap_seq_path}")
        return colmap_maps

    print(f"[INFO] Colmap folder found: {colmap_seq_path}")

    name_folder = "results_txt"
    colmap_seq_map = os.path.join(colmap_seq_path, name_folder)
    if not os.path.isdir(colmap_seq_map):
        print(f"[WARN] COLMAP map sequence folder not found: {colmap_seq_map}")
        return colmap_maps

    images_file = os.path.join(colmap_seq_map, "images.txt")
    points_file = os.path.join(colmap_seq_map, "points3D.txt")

    if not os.path.isfile(images_file):
        print(f"[WARN] COLMAP images.txt not found: {images_file}")
        return colmap_maps

    if not os.path.isfile(points_file):
        print(f"[WARN] COLMAP points3D.txt not found: {points_file}")
        return colmap_maps

    colmap_maps.append(
        {
            "map_id": 0,
            "images_file": images_file,
            "points_file": points_file,
        }
    )
    return colmap_maps


def read_colmap_data(colmap_map):
    colmap_images_file = colmap_map['images_file']
    colmap_rot_wc, colmap_center_wc, colmap_ids, colmap_image_names = read_colmap_images(colmap_images_file)

    colmap_points_file = colmap_map['points_file']
    filter_points = True
    colmap_points, _, _, colmap_points_color = read_colmap_points(colmap_points_file, filter_points)
    colmap_points_3D_array = np.copy(np.array(colmap_points))
    colmap_points_color_array = np.copy(np.array(colmap_points_color))

    return (colmap_rot_wc, colmap_center_wc, colmap_ids, colmap_image_names,
            colmap_points_3D_array, colmap_points_color_array)


def read_colmap_data_as_slam(colmap_map):
    (colmap_rot_wc, colmap_center_wc, colmap_ids, colmap_image_names,
     colmap_points_3D_array, colmap_points_color_array) = read_colmap_data(colmap_map)

    colmap_images_file = colmap_map['images_file']
    out_dir = os.path.dirname(os.path.normpath(colmap_images_file))
    return (colmap_center_wc, colmap_rot_wc, colmap_ids,
            colmap_center_wc, colmap_rot_wc, colmap_ids, colmap_image_names,
            colmap_points_3D_array, colmap_points_color_array, out_dir)


def match_with_colmap_maps(colmap_maps, ref_ids_set):
    matched_colmap_exp = {}
    matched_colmap_maps = {}
    for map_dict in colmap_maps:
        map_id = map_dict["map_id"]

        f_ids_set = build_colmap_image_name_set(map_dict["images_file"])

        # Match if there is at least one common image ID
        if len(ref_ids_set.intersection(f_ids_set)) > 3:
            matched_colmap_maps[map_id] = map_dict

    if matched_colmap_maps:
        matched_colmap_exp[0] = matched_colmap_maps
    return matched_colmap_exp
