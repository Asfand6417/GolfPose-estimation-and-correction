# utils/detect_keypoints.py

from keypoint_extraction.openpose_keypoints import run_openpose, parse_openpose_json
import os

def extract_2d_keypoints(back_video_path, side_video_path, output_base_dir, openpose_bin_path):
    back_json_dir = os.path.join(output_base_dir, 'json_back')
    side_json_dir = os.path.join(output_base_dir, 'json_side')
    back_vid_output = os.path.join(output_base_dir, 'rendered_back')
    side_vid_output = os.path.join(output_base_dir, 'rendered_side')

    print("📹 Running OpenPose on back view...")
    run_openpose(back_video_path, back_json_dir, back_vid_output, openpose_bin_path)

    print("📹 Running OpenPose on side view...")
    run_openpose(side_video_path, side_json_dir, side_vid_output, openpose_bin_path)

    print("🧠 Parsing keypoints from JSON...")
    keypoints_back = parse_openpose_json(back_json_dir)
    keypoints_side = parse_openpose_json(side_json_dir)

    # Make sure lengths match
    min_len = min(len(keypoints_back), len(keypoints_side))
    return list(zip(keypoints_back[:min_len], keypoints_side[:min_len]))
