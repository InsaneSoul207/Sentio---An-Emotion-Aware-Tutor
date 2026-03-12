import google.generativeai as genai
import numpy as np
import json
import os
from PIL import Image
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

class GeminiSocraticTutor:
    def __init__(self, api_key):
        """
        Initializes the Sentio Socratic Brain.
        Integrates persistent memory and a semantic intent matching engine.
        """
        genai.configure(api_key=api_key)
        
        # 1. Initialize Gemini 1.5 Flash and Persistent Chat Session
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.chat = self.model.start_chat(history=[])
        
        # 2. Setup Intent Matching Engine
        self.embeddings_model = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001", 
            google_api_key=api_key
        )
        self.intent_cache_path = "intent_vectors.npy"
        self.prompts_path = "prompts.json"
        
        # Load and "Save" the intent model during initialization
        self.intent_data, self.vectors = self._initialize_intent_engine()

    def _initialize_intent_engine(self):
        """
        Loads the 60-prompt dataset and handles vector caching (.npy).
       
        """
        if not os.path.exists(self.prompts_path):
            raise FileNotFoundError("prompts.json not found in the project root.")

        with open(self.prompts_path, 'r') as f:
            data = json.load(f)
        
        # Flatten metadata for easy lookup
        flat_metadata = []
        texts_to_embed = []
        for cat in data:
            for s in cat['scenarios']:
                flat_metadata.append({"intent": cat['intent'], "output": s['output']})
                texts_to_embed.append(s['input'])
        
        # Logic for "Saving" the model: Load from cache if it exists
        if os.path.exists(self.intent_cache_path):
            vectors = np.load(self.intent_cache_path)
        else:
            # First-time run: Generate and save embeddings locally
            vectors = np.array(self.embeddings_model.embed_documents(texts_to_embed))
            np.save(self.intent_cache_path, vectors)
        
        return flat_metadata, vectors

    def get_intent_instruction(self, user_query):
        """
        Semantic similarity search to find the best Socratic strategy.
       
        """
        query_vec = np.array(self.embeddings_model.embed_query(user_query)).reshape(1, -1)
        sims = cosine_similarity(query_vec, self.vectors)
        best_match_idx = np.argmax(sims)
        return self.intent_data[best_match_idx]['output']

    def get_response(self, user_text, emotion, context, image_path=None):
        """
        The main pipeline: Detect Intent -> Merge with Emotion/Context -> Chat.
       
        """
        # 1. Semantic Intent Retrieval
        socratic_strategy = self.get_intent_instruction(user_text)
        
        # 2. Meta-Prompt Construction
        # This wrapper guides Gemini to stay in character based on real-time data
        meta_instruction = f"""
        SYSTEM_PEDAGOGY: {socratic_strategy}
        STUDENT_EMOTION: {emotion}
        PDF_CONTEXT: {context}
        
        SOCRATIC_RULE: Guide the student to the answer. Never provide direct code or solutions first.
        """
        
        full_query = f"{meta_instruction}\n\nStudent Question: {user_text}"
        
        # 3. Multimodal Input Handling
        content_payload = [full_query]
        if image_path and os.path.exists(image_path):
            img = Image.open(image_path)
            content_payload.append(img)
            
        # 4. Generate response using persistent history
        response = self.chat.send_message(content_payload)
        return response.text