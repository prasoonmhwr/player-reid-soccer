import cv2

def save_annotated_video(video_path, tracks, out_path, global_ids=None, reverse=False):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    track_dict = {fid: objs for fid, objs in tracks}
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in track_dict:
            for pid, x, y in track_dict[frame_id]:
                gid = global_ids.get(pid, pid) if not reverse else next((k for k, v in global_ids.items() if v == pid), pid)
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)
                cv2.putText(frame, f'ID: {gid}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out.write(frame)
        frame_id += 1
    cap.release()
    out.release()