import streamlit as st
import cv2
import time
import os
from Emotion import SmoothedEmotionDetector
from knowledge_engine import KnowledgeBase
from tutor_engine import GeminiSocraticTutor

st.set_page_config(page_title="Socratic AI Tutor", layout="wide")

API_KEY = "AIzaSyCKWa7bNmc17ZZAyGwVEwm09d5E_Q9lxdc"

if 'detector' not in st.session_state:
    st.session_state.detector = SmoothedEmotionDetector()
if 'tutor' not in st.session_state:
    st.session_state.tutor = GeminiSocraticTutor(API_KEY)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = "neutral"

with st.sidebar:
    st.title("⚙️ Tutor Settings")
    
    uploaded_pdf = st.file_uploader("Upload Study Material (PDF)", type="pdf")
    if uploaded_pdf and 'kb' not in st.session_state:
        with st.spinner("Indexing PDF..."):
            with open("temp_kb.pdf", "wb") as f:
                f.write(uploaded_pdf.getbuffer())
            st.session_state.kb = KnowledgeBase("temp_kb.pdf", API_KEY)
        st.success("Textbook Indexed!")

    st.divider()
    
    uploaded_img = st.file_uploader("Upload Screenshot", type=["png", "jpg", "jpeg"])
    
    st.divider()
    
    run_cam = st.toggle("Live Emotion Tracking", value=True)
    video_placeholder = st.empty()
    st.session_state.emotion_metric = st.empty()

st.title("🎓 Socratic AI Tutor")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("How can I help you study today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = st.session_state.kb.search_context(prompt) if 'kb' in st.session_state else ""
    
    img_path = None
    if uploaded_img:
        img_path = "current_problem.png"
        with open(img_path, "wb") as f:
            f.write(uploaded_img.getbuffer())

    with st.chat_message("assistant"):
        with st.spinner("Tutor is thinking..."):
            response = st.session_state.tutor.get_response(
                prompt, 
                st.session_state.last_emotion, 
                context, 
                img_path
            )
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if run_cam:
    cap = cv2.VideoCapture(0)
    while run_cam:
        ret, frame = cap.read()
        if not ret: break
        current_state = st.session_state.detector.get_smoothed_emotion(frame)
        confidence = 100
        st.session_state.last_emotion = current_state
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", width=250)
        st.session_state.emotion_metric.metric("Current State", current_state.upper())
        
        time.sleep(0.05)
    cap.release()