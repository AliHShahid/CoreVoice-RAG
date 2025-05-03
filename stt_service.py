import whisper
import numpy as np

class SpeechToTextService:
    def __init__(self, model_name="base"):
        self.model = whisper.load_model(model_name)

    def transcribe(self, audio_np: np.ndarray) -> str:
        result = self.model.transcribe(audio_np, fp16=False)
        return result["text"].strip()
