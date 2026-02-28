import streamlit as st
import cv2
import time
from Emotion import EmotionDetector
from tutor_engine import GeminiSocraticTutor

st.set_page_config(page_title="Sentio AI Tutor", layout="wide")

st.markdown("""
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main {
        padding-top: 2rem;
    }
    /* Fixed height for chat to prevent page scrolling */
    .stChatFloatingInputContainer {
        padding-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

API_KEY = "YOUR_GOOGLE_API_KEY_HERE"  # Replace with your actual API key

if 'tutor' not in st.session_state:
    st.session_state.tutor = GeminiSocraticTutor(API_KEY)
if 'detector' not in st.session_state:
    st.session_state.detector = EmotionDetector()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = "neutral"

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Monitoring & Assets")
    
    video_feed = st.empty()
    status_text = st.empty()
    
    st.divider()
    
    st.write("### Uploads")
    up_pdf = st.file_uploader("Textbook (PDF)", type="pdf")
    up_img = st.file_uploader("Problem Screenshot", type=["png", "jpg", "jpeg"])

    if up_pdf and 'kb' not in st.session_state:
        from knowledge_engine import KnowledgeBase
        with st.spinner("Processing PDF..."):
            with open("temp_study.pdf", "wb") as f: f.write(up_pdf.getbuffer())
            st.session_state.kb = KnowledgeBase("temp_study.pdf", API_KEY)
        st.success("Knowledge Base Loaded")

with col_right:
    st.title("Sentio Socratic Tutor")
    
    chat_box = st.container(height=550)
    
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)

        context = st.session_state.kb.search_context(prompt) if 'kb' in st.session_state else ""
        img_p = "current_query.png" if up_img else None
        if up_img:
            with open(img_p, "wb") as f: f.write(up_img.getbuffer())

        with chat_box:
            with st.chat_message("assistant"):
                response = st.session_state.tutor.get_response(
                    prompt, 
                    st.session_state.last_emotion, 
                    context, 
                    img_p
                )
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        st.session_state.last_emotion, _ = st.session_state.detector.analyze_frame(frame)
        fr = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_feed.image(fr, channels="RGB", use_container_width=True)
        status_text.info(f"Current State: {st.session_state.last_emotion.upper()}")
        
    time.sleep(0.05)