import cv2
from ultralytics import YOLO

def detect_players(video_path, model_path):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    detections = []
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame_dets = []
        for result in results:
            for box in result.boxes.data:
                x1, y1, x2, y2, conf, cls = box[:6].tolist()
                if int(cls) == 0 and conf > 0.3:  # class 0 = player
                    frame_dets.append([x1, y1, x2, y2, conf])
        detections.append((frame_id, frame_dets))
        frame_id += 1
    cap.release()
    return detections