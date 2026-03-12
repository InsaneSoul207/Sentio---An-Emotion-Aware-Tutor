import streamlit as st
import cv2
import time
import os
from Emotion import EmotionDetector
from tutor_engine import GeminiSocraticTutor

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Sentio | Socratic AI", layout="wide", initial_sidebar_state="collapsed")

# Minimalist Professional CSS
st.markdown("""
    <style>
    /* Force GPU Rendering to stop background glitches */
    html, body, [data-testid="stAppViewContainer"] {
        overflow: hidden; 
        height: 100vh; 
        background: #0b0e14;
        -webkit-backface-visibility: hidden;
        -webkit-transform: translate3d(0, 0, 0);
    }
    
    /* Smooth out the video display */
    [data-testid="stImage"] {
        border-radius: 12px;
        transition: none;
    }
    </style>
    """, unsafe_allow_html=True)

# --- CONFIG & INITIALIZATION ---
API_KEY = "AIzaSyA_xjoY-XoACHqT_CT-lefRJO3t1EvxTlU"  # Replace with your actual API key

if 'tutor' not in st.session_state:
    st.session_state.tutor = GeminiSocraticTutor(API_KEY)
if 'detector' not in st.session_state:
    st.session_state.detector = EmotionDetector()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_emotion' not in st.session_state:
    st.session_state.last_emotion = "neutral"

# --- MAIN LAYOUT ---
left_col, right_col = st.columns([1, 2.5], gap="large")

with left_col:
    st.title("SENTIO")
    
    # 1. Vision Feed (Top Left)
    video_placeholder = st.empty()
    emotion_display = st.empty()
    
    st.divider()
    
    # 2. Asset Management (Below Video)
    st.subheader("Study Assets")
    up_pdf = st.file_uploader("Textbook PDF", type="pdf", label_visibility="collapsed")
    up_img = st.file_uploader("Visual Problem (IMG)", type=["png", "jpg"], label_visibility="collapsed")

    # Knowledge Base Logic
    if up_pdf and 'kb' not in st.session_state:
        from knowledge_engine import KnowledgeBase
        with st.spinner("Indexing PDF..."):
            with open("active_study.pdf", "wb") as f: f.write(up_pdf.getbuffer())
            st.session_state.kb = KnowledgeBase("active_study.pdf", API_KEY)
        st.toast("Knowledge Base Linked!")

with right_col:
    # 3. Chat Space (Fixed Height)
    chat_box = st.container(height=600, border=False)
    
    # Render Message History
    with chat_box:
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    # 4. Chat Input Logic
    if prompt := st.chat_input("Ask me about the material..."):
        # Store user text
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_box:
            with st.chat_message("user"):
                st.markdown(prompt)

        # Gather context
        rag_context = st.session_state.kb.search_context(prompt) if 'kb' in st.session_state else ""
        
        # Handle temporary image storage for multimodal query
        img_path = None
        if up_img:
            img_path = "query_visual.png"
            with open(img_path, "wb") as f: f.write(up_img.getbuffer())

        # Generate Adaptive Response
        with chat_box:
            with st.chat_message("assistant"):
                with st.spinner("Thinking Socraticly..."):
                    response = st.session_state.tutor.get_response(
                        user_text=prompt,
                        emotion=st.session_state.last_emotion,
                        context=rag_context,
                        image_path=img_path
                    )
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

# --- LIVE CAMERA THREAD ---
# Note: Streamlit reruns the script, but this loop keeps the feed live
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret:
        # Detect state via Vision Engine
        state, confidence = st.session_state.detector.analyze_frame(frame)
        st.session_state.last_emotion = state
        
        # Convert for Streamlit display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
        
        # Update Status UI
        emotion_display.markdown(f"""
            <div class="status-box">
                AFFECTIVE_STATE: <span style="color:#007bff;">{state.upper()}</span>
            </div>
            """, unsafe_allow_html=True)
            
    time.sleep(0.04)