import streamlit as st
import cv2
import torch
import numpy as np
import timm
import matplotlib.pyplot as plt
import time
from torchvision import transforms
import base64

st.set_page_config(layout="wide")

def get_base64(img_file):
    with open(img_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64("b.jpg")

EMOTIONS = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]
STATES = ["Stable","Depression-like","Anxiety-like"]
device = "cuda" if torch.cuda.is_available() else "cpu"

vit = timm.create_model("vit_tiny_patch16_224", pretrained=False)

vit.head = torch.nn.Sequential(
    torch.nn.Linear(vit.head.in_features, 256),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.5),
    torch.nn.Linear(256, 7)
)

vit.load_state_dict(torch.load("best_vit.pth", map_location=device))
vit.to(device)
vit.eval()

class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(7, 64, batch_first=True)
        self.fc = torch.nn.Linear(64, 3)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

lstm = LSTMModel()
lstm.load_state_dict(torch.load("lstm_model.pth", map_location=device))
lstm.to(device)
lstm.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = transform(frame).unsqueeze(0)
    return img.to(device)

if "running" not in st.session_state:
    st.session_state.running = False

if "data" not in st.session_state:
    st.session_state.data = []

if "cap" not in st.session_state:
    st.session_state.cap = None

if "final_done" not in st.session_state:
    st.session_state.final_done = False

st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{img}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.stApp::before {{
    content: "";
    position: fixed;
    width: 100%;
    height: 100%;
    background: rgba(10,10,20,0.8);
    z-index: -1;
}}

h1 {{
    text-align: center;
    color: white;
}}

div.stButton > button {{
    width: 100%;
    border-radius: 12px;
    height: 50px;
    font-size: 16px;
    font-weight: 600;
    background: linear-gradient(135deg, #00c6ff, #0072ff);
    color: white;
    border: none;
}}

div.stButton > button:hover {{
    transform: scale(1.05);
}}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Mental Disorder Detection System")

col1, col2, col3 = st.columns(3)

if col1.button("📷 Start Capture"):
    st.session_state.running = True
    st.session_state.data = []
    st.session_state.final_done = False

    if st.session_state.cap:
        st.session_state.cap.release()

    st.session_state.cap = cv2.VideoCapture(0)

if col2.button("⏹ Stop & Analyze"):
    st.session_state.running = False
    st.session_state.final_done = True

    if st.session_state.cap:
        st.session_state.cap.release()

if col3.button("↩ Reset"):
    st.session_state.running = False
    st.session_state.data = []
    st.session_state.final_done = False

    if st.session_state.cap:
        st.session_state.cap.release()

left_col, right_col = st.columns([1,1])

with left_col:
    frame_placeholder = st.empty()

with right_col:
    graph_placeholder = st.empty()

if st.session_state.running and st.session_state.cap:

    ret, frame = st.session_state.cap.read()

    if ret:
        input_tensor = preprocess(frame)

        with torch.no_grad():
            out = vit(input_tensor)
            probs = torch.softmax(out, dim=1).cpu().numpy()[0]

        emotion = EMOTIONS[np.argmax(probs)]
        st.session_state.data.append(probs)

        cv2.putText(frame, emotion, (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        frame_placeholder.image(frame, channels="BGR")

        st.write(f"Frames collected: {len(st.session_state.data)}")

        data = np.array(st.session_state.data)

        if len(data) > 5:
            fig, ax = plt.subplots()

            for i, emo in enumerate(EMOTIONS):
                ax.plot(data[:, i], label=emo)

            ax.set_title("Live Emotion Graph", fontsize=16)
            ax.legend(fontsize=8)

            graph_placeholder.pyplot(fig)

    time.sleep(0.03)
    st.rerun()

if st.session_state.final_done:

    data_len = len(st.session_state.data)
    st.write(f"📊 Frames captured: {data_len}")

    if data_len < 10:
        st.warning("⚠️ Capture for 3–5 seconds")

    if data_len >= 10:
        seq = np.array(st.session_state.data[-20:])
        seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            out = lstm(seq_tensor)
            state = STATES[torch.argmax(out).item()]

        st.markdown("## 🧠 Final Mental State")

        st.markdown(f"""
        <div style="
            background: rgba(0,255,150,0.15);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            color: #00ffcc;
            font-weight: bold;">
            {state}
        </div>
        """, unsafe_allow_html=True)

    if data_len > 0:
        data = np.array(st.session_state.data)

        fig, ax = plt.subplots()

        for i, emo in enumerate(EMOTIONS):
            ax.plot(data[:, i], label=emo)

        ax.set_title("Emotion Trend Over Time", fontsize=16)
        ax.set_xlabel("Time")
        ax.set_ylabel("Probability")
        ax.legend(fontsize=8)

        st.pyplot(fig)

        st.markdown("## 📊 Emotion Distribution")

        avg_probs = np.mean(data, axis=0)

        fig2, ax2 = plt.subplots()
        ax2.bar(EMOTIONS, avg_probs)
        ax2.set_ylabel("Average Probability")

        st.pyplot(fig2)

    st.session_state.final_done = False