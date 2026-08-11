import streamlit as st
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import base64
from vit_model import load_vit, load_lstm, EMOTIONS, LSTMModel
from utils import get_inference_transform, predict_emotion, predict_mental_state, plot_emotion_sequence, plot_emotion_bar, STATES

st.set_page_config(layout="wide")

def get_base64(img_file):
    with open(img_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_base64("b.jpg")

device = "cuda" if torch.cuda.is_available() else "cpu"

vit       = load_vit("best_vit.pth", device=device)
lstm      = load_lstm("lstm_model.pth", device=device)
transform = get_inference_transform()

def preprocess(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return transform(rgb).unsqueeze(0).to(device)

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