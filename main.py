import os
import argparse
from src.detector import detect_players
from src.tracker import track_players
from src.feature_extractor import extract_features
from src.matcher import match_players
from src.utils import save_annotated_video


def main(broadcast_path, tacticam_path, model_path):
    print("[INFO] Running detection...")
    broadcast_dets = detect_players(broadcast_path, model_path)
    tacticam_dets = detect_players(tacticam_path, model_path)

    print("[INFO] Tracking players...")
    broadcast_tracks = track_players(broadcast_dets)
    tacticam_tracks = track_players(tacticam_dets)

    print("[INFO] Extracting features...")
    broadcast_features = extract_features(broadcast_path, broadcast_tracks)
    tacticam_features = extract_features(tacticam_path, tacticam_tracks)

    print("[INFO] Matching players across views...")
    id_map = match_players(broadcast_features, tacticam_features)

    print("[INFO] Annotating videos with consistent IDs...")
    save_annotated_video(broadcast_path, broadcast_tracks, 'output/broadcast_annotated.mp4', global_ids=id_map)
    save_annotated_video(tacticam_path, tacticam_tracks, 'output/tacticam_annotated.mp4', global_ids=id_map, reverse=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--broadcast', required=True, help='Path to broadcast video')
    parser.add_argument('--tacticam', required=True, help='Path to tacticam video')
    parser.add_argument('--model', required=True, help='Path to YOLOv8 model')
    args = parser.parse_args()

    os.makedirs('output', exist_ok=True)
    main(args.broadcast, args.tacticam, args.model)
