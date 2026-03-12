# Sentio: The Emotion-Aware Socratic Mentor 🎓🧠

**Sentio** is an intelligent, multimodal AI tutor designed to replicate the Socratic method through the lens of Affective Computing. By analyzing real-time facial expressions and classifying linguistic intent, Sentio adapts its pedagogical strategy to move a student from frustration to "Eureka!" moments without ever giving away the final answer.

---

## ✨ Key Innovations

### 🧬 Affective State Mapping
Sentio doesn't just "detect emotions"; it maps raw expressions (via DeepFace) to **Cognitive Learning States**:
* **Focused:** Sustained neutral affect during problem-solving.
* **Struggling:** Persistent high-intensity frustration or sadness detected over a 15-frame temporal buffer.
* **Confused:** A specific mix of neutral and subtle surprise/fear cues.

### 🎯 Semantic Intent Engine (NLP)
Instead of a simple chatbot, Sentio uses a **Meta-Prompting Gatekeeper**:
* **Intent Matcher:** Uses **Sentence Embeddings (gemini-embedding-001)** to categorize user queries into one of 60 pre-defined Socratic archetypes (e.g., *Direct Code Request*, *Verification*, *Emotional Venting*).
* **Instruction Synthesis:** Automatically wraps the student's question in a pedagogical instruction *before* it reaches the LLM.

### 📚 Multimodal RAG
Integrates **Gemini 2.5 Flash Lite** with a **FAISS Vector DB** to provide context-aware tutoring based on uploaded PDFs and image screenshots.

---

## 🏗️ System Architecture

Sentio is built on a modular pipeline designed for low-latency feedback:

1. **Vision Engine (`Emotion.py`):** DeepFace + OpenCV with a 15-frame smoothing buffer for state stability.
2. **Intent Engine (`tutor_engine.py`):** Pre-calculated semantic vectors saved in `.npy` format for zero-latency classification.
3. **Knowledge Base (`knowledge_engine.py`):** LangChain-powered RAG for document retrieval.
4. **Interface (`app.py`):** A minimalist, single-pane Streamlit dashboard.



---

## 🚀 Installation & Quick Start

### 1. Prerequisites
- Python 3.11+
- A Google AI Studio API Key

### 2. Setup
```bash
# Clone the repository
git clone [https://github.com/InsaneSoul207/Sentio---An-Emotion-Aware-Tutor.git](https://github.com/InsaneSoul207/Sentio---An-Emotion-Aware-Tutor.git)
cd Sentio

# Install dependencies in one line
pip install streamlit opencv-python deepface google-generativeai langchain langchain-community langchain-google-genai pypdf faiss-cpu pillow scikit-learn

# Run the application
streamlit run app.py
