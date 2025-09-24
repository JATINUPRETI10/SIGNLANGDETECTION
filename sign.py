# ----------------------------
# sign.py  (with confidence score, fixed model loading)
# ----------------------------

import warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype")

import cv2
import mediapipe as mp
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F  # for softmax

# Import your CNN model
from model import SimpleCNN

# Load class names
with open("classes.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

num_classes = len(class_names)

# Load model
model = SimpleCNN(num_classes=num_classes)

# ---- FIX: partial load to avoid fc size mismatch ----
checkpoint = torch.load("model.pth", map_location="cpu")
model_dict = model.state_dict()
# only load matching layers
pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.size() == model_dict[k].size()}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)
# -----------------------------------------------------

model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    label = "No Hand"
    conf = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, c = frame.shape
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))

            margin = 40
            x_min, y_min = max(0, x_min - margin), max(0, y_min - margin)
            x_max, y_max = min(w, x_max + margin), min(h, y_max + margin)

            hand_img = frame[y_min:y_max, x_min:x_max]

            if hand_img.size != 0:
                hand_pil = Image.fromarray(cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)).convert("RGB")
                input_tensor = transform(hand_pil).unsqueeze(0)

                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = F.softmax(outputs, dim=1)
                    conf_val, predicted = torch.max(probs, 1)
                    label = class_names[predicted.item()]
                    conf = conf_val.item()

                    print(f"Pred: {label}, Conf: {conf:.2f}")

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
