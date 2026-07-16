from builtins import len

from utils import get_file_num_frames_per_video, get_file_sequence_scales, save_results_json
from evaluation_utils import match_and_align_sequences, match_sequences_by_folder
from slam_utils import get_slam_seq_directories, parse_endomapper_args


if __name__ == '__main__':
    args = parse_endomapper_args()

    colmap_sequences = get_slam_seq_directories(args.colmap_path)
    slam_sequences = get_slam_seq_directories(args.slam_path)

    colmap_match_seqs, slam_match_seqs = match_sequences_by_folder(colmap_sequences, slam_sequences)

    num_frames_per_video_file = get_file_num_frames_per_video(args.colmap_path)
    sequence_scales_file = get_file_sequence_scales(args.colmap_path)
    trajectory_lengths_file = get_file_sequence_scales(args.colmap_path, "traj_lengths_mm.csv")

    if args.verbose:
        print("Colmap sequences:", colmap_sequences)
        print("SLAM sequences:", slam_sequences)
        print("Number of matched sequences:", len(colmap_match_seqs))
        print("Colmap matched sequences:", colmap_match_seqs)
        print("SLAM matched sequences:", slam_match_seqs)
        if num_frames_per_video_file:
            print("Video frames file:", num_frames_per_video_file)
        if sequence_scales_file:
            print("Sequence scales file:", sequence_scales_file)
        if trajectory_lengths_file:
            print("Sequence trajectory lengths file:", trajectory_lengths_file)

    assert len(colmap_match_seqs) == len(slam_match_seqs), \
        "Colmap sequences and SLAM sequences must have the same length."

    ref_type = "COLMAP"
    slam_type = "SLAM"
    all_results, all_results_mean, global_mean = match_and_align_sequences(
        ref_seqs=colmap_match_seqs,
        slam_seqs=slam_match_seqs,
        ref_type=ref_type,
        slam_type=slam_type,
        verbose=args.verbose,
        file_num_frames=num_frames_per_video_file,
        file_scales=sequence_scales_file,
        file_traj_lengths=trajectory_lengths_file,
        save_ply=args.save_ply
    )

    save_results_json(
        args.results_file,
        args.colmap_path,
        args.slam_path,
        all_results,
        all_results_mean,
        global_mean
    )
