import cv2
from deepface import DeepFace

class EmotionDetector:
    def __init__(self):
        print("Loading Emotion Detection Model...")
        
    def analyze_frame(self, frame):
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            if results:
                dominant_emotion = results[0]['dominant_emotion']
                emotion_score = results[0]['emotion'][dominant_emotion]
                return dominant_emotion, emotion_score
        except Exception as e:
            return "neutral", 0
        return "neutral", 0