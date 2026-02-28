from collections import Counter
from deepface import DeepFace

class SmoothedEmotionDetector:
    def __init__(self, buffer_size=15):
        self.emotion_buffer = []
        self.buffer_size = buffer_size

    def get_smoothed_emotion(self, frame):
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emo_scores = results[0]['emotion']
            dominant = results[0]['dominant_emotion']
            
            complex_state = dominant
            
            if dominant == 'neutral' and emo_scores['neutral'] > 80:
                complex_state = 'focused'
                
            if dominant == 'neutral' and (emo_scores['sad'] > 5 or emo_scores['fear'] > 5):
                complex_state = 'confused'
                
            if (dominant == 'angry' or dominant == 'sad') and emo_scores[dominant] > 30:
                complex_state = 'frustrated'

            self.emotion_buffer.append(complex_state)
            if len(self.emotion_buffer) > self.buffer_size:
                self.emotion_buffer.pop(0)
            
            final_state = Counter(self.emotion_buffer).most_common(1)[0][0]
            
            if self.emotion_buffer.count('frustrated') > (self.buffer_size * 0.7):
                return 'struggling'
                
            return final_state

        except Exception:
            return "distracted"