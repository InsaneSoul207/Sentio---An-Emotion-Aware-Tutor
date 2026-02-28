import google.generativeai as genai
from PIL import Image

class GeminiSocraticTutor:
    def __init__(self, api_key):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        self.chat = self.model.start_chat(history=[])

    def get_response(self, user_text, emotion, context, image_path=None):
        system_prefix = f"""
        [STUDENT STATE: {emotion}]
        [TEXTBOOK CONTEXT: {context}]
        
        INSTRUCTIONS: 
        - Follow Socratic/OpenEdu teaching principles.
        - Use the context and state to guide your response. 
        - Never give direct code answers immediately.
        """
        
        full_message = f"{system_prefix}\n\nStudent: {user_text}"
        
        content_parts = [full_message]
        
        if image_path:
            img = Image.open(image_path)
            content_parts.append(img)
            
        response = self.chat.send_message(content_parts)
        return response.text