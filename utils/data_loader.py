import os

def get_nested_video_pairs(base_path="data"):
    """
    Returns a list of tuples: (swing_quality, swing_type, back_view_video, side_view_video)
    """
    pairs = []

    for swing_quality in ["Bad Swings", "Good Swings"]:
        back_quality_path = os.path.join(base_path, "Back View", swing_quality)
        side_quality_path = os.path.join(base_path, "Side View", swing_quality)

        if not os.path.exists(back_quality_path) or not os.path.exists(side_quality_path):
            print(f"Missing directory: {back_quality_path} or {side_quality_path}")
            continue

        for swing_type in os.listdir(back_quality_path):
            back_swing_path = os.path.join(back_quality_path, swing_type)
            side_swing_path = os.path.join(side_quality_path, swing_type)

            if not os.path.exists(side_swing_path):
                print(f"Warning: Side view folder missing for {swing_type} in {swing_quality}")
                continue

            # ✅ FIXED: Proper brackets and parentheses
            back_videos = sorted([f for f in os.listdir(back_swing_path) if f.endswith(".mp4")])
            side_videos = sorted([f for f in os.listdir(side_swing_path) if f.endswith(".mp4")])

            if len(back_videos) != len(side_videos):
                print(f"⚠️ Mismatch in number of videos for {swing_type}: {len(back_videos)} back vs {len(side_videos)} side")

            for back_vid, side_vid in zip(back_videos, side_videos):
                back_full = os.path.join(back_swing_path, back_vid)
                side_full = os.path.join(side_swing_path, side_vid)
                pairs.append((swing_quality, swing_type, back_full, side_full))

    return pairs
