# Sentio: Emotion-Aware Socratic AI Tutor 🎓🧠

**Sentio** is an advanced, multimodal AI tutoring platform that utilizes **Affective Computing** and the **Socratic Method** to provide a personalized learning experience. By analyzing real-time facial expressions and mapping them to complex cognitive states (like struggling, focused, or confused), Sentio adapts its pedagogical strategy to meet the student's immediate needs.

## 🚀 Key Features

* **Affective State Mapping:** Moves beyond basic emotion detection to identify complex learning states: *Focused, Struggling, Confused, and Distracted*.
* **Socratic Pedagogical Engine:** Built on the **OpenEdu framework**, the tutor uses inquiry-based learning—asking guiding questions rather than providing direct answers.
* **Multimodal RAG (Retrieval-Augmented Generation):**
    * **Textbook Context:** Upload PDFs to provide the AI with specific domain knowledge via a **FAISS** vector database.
    * **Visual Reasoning:** Uses **Gemini 1.5 Flash** to analyze uploaded screenshots of complex problems (e.g., Peterson's Solution).
* **Persistent Context Memory:** Maintains a full conversation history using Gemini’s chat session management to ensure continuity in complex topics.

---

## 🏗️ System Architecture

Sentio is built with a modular architecture to ensure scalability and clear separation of concerns:

1.  **Vision Engine (`Emotion.py`):** Leverages **DeepFace** and OpenCV with a temporal smoothing buffer to stabilize emotion detection and map raw data to cognitive states.
2.  **Knowledge Engine (`knowledge_engine.py`):** Handles document processing, text splitting, and vector embeddings using **LangChain** and **Google Generative AI Embeddings**.
3.  **Tutor Engine (`tutor_engine.py`):** The "Brain" of the project, managing the **Gemini 2.5 Flash lite** API, multimodal inputs, and Socratic system prompting.
4.  **Interface (`app.py`):** A high-performance **Streamlit** dashboard that synchronizes the webcam feed, PDF indexing, and the chat interface.

---

## 🛠️ Tech Stack

* **Language:** Python
* **AI/ML:** Gemini 2.5 Flash lite, DeepFace, TensorFlow
* **NLP:** LangChain, FAISS Vector DB
* **Computer Vision:** OpenCV
* **Web Framework:** Streamlit

---

## 🚦 Getting Started

### Prerequisites
- Python 3.11+
- A Google AI Studio API Key (for Gemini)

### Installation
1. Clone the repository:
   ```bash
   git clone [https://github.com/InsaneSoul207/Sentio---An-Emotion-Aware-Tutor.git](https://github.com/InsaneSoul207/Sentio---An-Emotion-Aware-Tutor.git)
   cd Sentio

2. Install dependencies:
    ```bash
    pip install streamlit opencv-python deepface google-generativeai langchain-community pypdf faiss-cpu langchain-google-genai

3. Run the application:
    ```bash
    streamlit run app.py


### 📖 Educational Philosophy: The Socratic Method
Sentio does not simply "solve" problems for the user. Based on the detected state:

If **Struggling**: It provides "Micro-Hints" to scaffold the learning process.

If **Focused**: It stays concise to maintain the user's flow.

If **Confused**: It pivots to analogies or simpler textbook definitions.

Developed by **Eshaan Mishra** as part of a specialization in **AI/ML and System Design**.