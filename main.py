import threading
from queue import Queue
from rich.console import Console

from tts_service import TextToSpeechService
from stt_service import SpeechToTextService
from llm_service import LLMService
from audio_utils import record_audio, get_audio_np, play_audio

console = Console()
stt = SpeechToTextService()
tts = TextToSpeechService()
llm = LLMService()

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    try:
        while True:
            console.input("Press Enter to start recording, then press Enter again to stop.")
            data_queue = Queue()
            stop_event = threading.Event()

            thread = threading.Thread(target=record_audio, args=(stop_event, data_queue))
            thread.start()

            input()  # wait for second enter
            stop_event.set()
            thread.join()

            audio_np = get_audio_np(data_queue)

            if audio_np.size > 0:
                with console.status("Transcribing...", spinner="earth"):
                    user_text = stt.transcribe(audio_np)
                console.print(f"[yellow]You: {user_text}")

                with console.status("Generating response...", spinner="earth"):
                    reply = llm.get_response(user_text)
                    sr, audio = tts.long_form_synthesize(reply)

                console.print(f"[cyan]Assistant: {reply}")
                play_audio(sr, audio)
            else:
                console.print("[red]No audio recorded. Please ensure your microphone is working.")
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...")

    console.print("[blue]Session ended.")
