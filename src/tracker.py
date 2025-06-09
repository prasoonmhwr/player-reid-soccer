import numpy as np
from filterpy.kalman import KalmanFilter

class Track:
    def __init__(self, bbox, track_id):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                              [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])

        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0]])

        self.kf.R[2:,2:] *= 10.  
        self.kf.P[4:,4:] *= 1000.  
        self.kf.P *= 10.

        self.kf.x[:4] = np.array(bbox).reshape((4, 1))
        self.id = track_id
        self.time_since_update = 0

    def predict(self):
        self.kf.predict()
        self.time_since_update += 1
        return self.kf.x[:4].reshape(-1)

    def update(self, bbox):
        self.kf.update(np.array(bbox))
        self.time_since_update = 0

def iou(bb1, bb2):
    x1 = max(bb1[0], bb2[0])
    y1 = max(bb1[1], bb2[1])
    x2 = min(bb1[2], bb2[2])
    y2 = min(bb1[3], bb2[3])
    w = max(0., x2 - x1)
    h = max(0., y2 - y1)
    inter = w * h
    area1 = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    area2 = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])
    return inter / (area1 + area2 - inter + 1e-6)

def track_players(detections, iou_threshold=0.7, max_lost=10):
    next_id = 0
    tracks = []
    active_tracks = []

    for frame_id, dets in detections:
        updated_ids = set()
        matched_tracks = []

        
        for trk in active_tracks:
            trk.predict()

        
        for det in dets:
            best_iou = 0
            best_track = None
            for trk in active_tracks:
                i = iou(det[:4], trk.kf.x[:4].reshape(-1))
                if i > best_iou:
                    best_iou = i
                    best_track = trk
            if best_iou > iou_threshold:
                best_track.update(det[:4])
                matched_tracks.append((best_track.id, best_track.kf.x[:4]))
                updated_ids.add(best_track.id)
            else:
                
                new_trk = Track(det[:4], next_id)
                matched_tracks.append((next_id, det[:4]))
                active_tracks.append(new_trk)
                next_id += 1

        
        active_tracks = [trk for trk in active_tracks if trk.time_since_update <= max_lost]

        tracks.append((frame_id, [(tid, x1, y1, x2, y2) for tid, (x1, y1, x2, y2) in matched_tracks]))
    return tracks
