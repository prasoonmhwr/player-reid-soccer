import cv2
from norfair import Detection, Tracker, detection_to_xy
import numpy as np

tracker = Tracker(distance_function="euclidean", distance_threshold=30)
def track_players(detections):
    def euclidean_distance(detection, tracked_object):
        return np.linalg.norm(detection.points[0] - tracked_object.estimate[0])

    tracker = Tracker(distance_function=euclidean_distance, distance_threshold=30)
    tracked = []
    for frame_id, dets in detections:
        norfair_dets = [Detection(points=np.array([[ (x1 + x2) / 2, (y1 + y2) / 2 ]])) for x1, y1, x2, y2, conf in dets]
        tracked_objects = tracker.update(norfair_dets)
        frame_tracks = []
        for obj in tracked_objects:
            x, y = obj.estimate[0]
            frame_tracks.append((obj.id, x, y))
        tracked.append((frame_id, frame_tracks))
    return tracked