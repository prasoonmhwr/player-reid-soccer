import torch
import torchvision.transforms as T
from torchvision.models import resnet50
from PIL import Image
import cv2

model = resnet50(pretrained=True)
model.eval()
transform = T.Compose([
    T.Resize((128, 64)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def extract_features(video_path, tracks):
    cap = cv2.VideoCapture(video_path)
    frame_dict = {}
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        frame_dict[frame_id] = frame
    cap.release()

    features = {}
    for frame_id, objs in tracks:
        frame = frame_dict.get(frame_id)
        for pid, x1, y1, x2, y2 in objs:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                feat = model(transform(pil_img).unsqueeze(0)).squeeze().numpy()
            features[pid] = feat
    return features