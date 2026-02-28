import cv2
from deepface import DeepFace
from collections import Counter

class EmotionDetector:
    def __init__(self, buffer_size=15):
        """
        Initializes the Sentio Vision Engine.
        :param buffer_size: Number of frames to average for state stability.
        """
        print("Sentio Vision Engine: Loading Affective Models...")
        self.emotion_buffer = []
        self.buffer_size = buffer_size
        
    def analyze_frame(self, frame):
        """
        Analyzes a single frame and maps basic emotions to complex cognitive states.
       
        """
        try:
            results = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if results:
                raw_dominant = results[0]['dominant_emotion']
                scores = results[0]['emotion']
                confidence = scores.get(raw_dominant, 0)
                
                complex_state = raw_dominant
                
                if raw_dominant == 'neutral' and scores['neutral'] > 85:
                    complex_state = "focused"
                
                elif raw_dominant == 'neutral' and (scores['sad'] > 5 or scores['fear'] > 5):
                    complex_state = "confused"
                
                elif (raw_dominant == 'angry' or raw_dominant == 'sad') and scores[raw_dominant] > 40:
                    complex_state = "frustrated"

                self.emotion_buffer.append(complex_state)
                if len(self.emotion_buffer) > self.buffer_size:
                    self.emotion_buffer.pop(0)
                
                smoothed_state = Counter(self.emotion_buffer).most_common(1)[0][0]
                
                if self.emotion_buffer.count('frustrated') > (self.buffer_size * 0.6):
                    smoothed_state = "struggling"
                
                return smoothed_state, confidence

        except Exception as e:
            return "distracted", 0
        
        return "neutral", 0