import time
import numpy as np
import sounddevice as sd
from queue import Queue

def record_audio(stop_event, data_queue):
    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        data_queue.put(bytes(indata))

    with sd.RawInputStream(samplerate=16000, dtype="int16", channels=1, callback=callback):
        while not stop_event.is_set():
            time.sleep(0.1)

def get_audio_np(data_queue: Queue) -> np.ndarray:
    audio_data = b"".join(list(data_queue.queue))
    return np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

def play_audio(sample_rate, audio_array):
    sd.play(audio_array, sample_rate)
    sd.wait()
