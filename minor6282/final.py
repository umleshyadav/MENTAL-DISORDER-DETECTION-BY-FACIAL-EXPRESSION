import cv2
import torch
import timm
import torch.nn as nn
import numpy as np
from torchvision import transforms
from collections import deque, Counter
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
EMOTIONS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
STATES = ["Stable", "Depression-like", "Anxiety-like"]

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

# =====================
# LOAD VIT MODEL
# =====================
vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)

vit.head = nn.Sequential(
    nn.Linear(vit.head.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 7)
)

vit.load_state_dict(torch.load("best_vit.pth", map_location=device))
vit.to(device)
vit.eval()

# =====================
# LOAD LSTM MODEL
# =====================
class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(7, 64, batch_first=True)
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm = LSTMModel()
lstm.load_state_dict(torch.load("lstm_model.pth", map_location=device))
lstm.to(device)
lstm.eval()

# =====================
# TRANSFORM
# =====================
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3,[0.5]*3)
])

# =====================
# BUFFERS
# =====================
sequence_buffer = deque(maxlen=20)
emotion_history = deque(maxlen=20)

# =====================
# GRAPH SETUP
# =====================
plt.ion()
fig, ax = plt.subplots()

# =====================
# FACE DETECTOR
# =====================
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]

        input_tensor = transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            out = vit(input_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        # store sequence
        sequence_buffer.append(probs)

        # predicted emotion
        idx = np.argmax(probs)
        emotion = EMOTIONS[idx]
        confidence = probs[idx]

        # store emotion history
        emotion_history.append(emotion)

        # -------- SHOW TOP 3 EMOTIONS --------
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_text = ", ".join([f"{EMOTIONS[i]}:{probs[i]:.2f}" for i in top3_idx])

        # draw face box
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        # show emotion
        cv2.putText(frame,f"{emotion} ({confidence:.2f})",
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,(0,255,0),2)

        # show top 3
        cv2.putText(frame,top3_text,
                    (x,y+h+20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,(255,255,0),1)

    # =====================
    # LSTM PREDICTION
    # =====================
    state = "Analyzing..."

    if len(sequence_buffer) == 20:
        seq = np.array(sequence_buffer)
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = lstm(seq_tensor)
            state = STATES[torch.argmax(out).item()]

    # =====================
    # SMOOTH EMOTION (MAJORITY)
    # =====================
    if len(emotion_history) > 5:
        most_common = Counter(emotion_history).most_common(1)[0][0]
    else:
        most_common = "..."

    # =====================
    # DISPLAY RESULTS
    # =====================
    cv2.putText(frame,f"Final Emotion: {most_common}",
                (20,40), cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,255),2)

    cv2.putText(frame,f"Mental State: {state}",
                (20,80), cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,0,255),2)

    # =====================
    # GRAPH (Emotion over time)
    # =====================
    if len(sequence_buffer) > 5:
        ax.clear()
        seq_np = np.array(sequence_buffer)

        for i, emo in enumerate(EMOTIONS):
            ax.plot(seq_np[:, i], label=emo)

        ax.legend(loc='upper right')
        ax.set_title("Emotion Over Time")
        plt.pause(0.001)

    cv2.imshow("AI Mental Health System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()