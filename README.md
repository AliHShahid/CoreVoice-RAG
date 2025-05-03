# RAIVA
## RAIVA (Retrieval-Augmented Intelligent Voice Assistant)
### MODULE 1: Streamlit Integration
Objective: Deploy the entire system locally using a responsive, intuitive interface.

Components:

Voice recording button.

Real-time STT and response rendering.

TTS playback in browser.

Implementation:

Frontend built using Streamlit widgets (st.button, st.audio, st.text_area).

Backend integrations for microphone input, Whisper decoding, and Bark TTS.

Contribution:

Developed a clean interface with clear state management.

Ensured minimal latency and high responsiveness.

### MODULE 2: Retrieval-Augmented Generation (RAG) with Ollama
Objective: Enable smart, personalized conversations using a lightweight LLM served via Ollama and enhanced by RAG.

Components:

Vector DB (Chroma).

Ollama’s model (LLaMA3).

Contextual embedding + prompt management.

Implementation:

Documents embedded using sentence-transformers.

Custom retriever pipeline for context-aware responses.

Ollama queried via local REST API.

Contribution:

Built scalable document ingestion and embedding pipeline.

Designed and integrated the RAG framework using LangChain.

### MODULE 3: Text-to-Speech (TTS) with Bark
Objective: Convert assistant’s responses to natural-sounding speech.

Components:

Bark model for multilingual and expressive TTS.

Output in WAV/MP3 format.

Implementation:

Process LLM responses and pass to Bark.

Handle audio preprocessing (padding/sampling).

Integrate output playback via Streamlit.

Contribution:

Optimized inference time for real-time speech.

Custom voice fine-tuning (optional extension).

### MODULE 4: Speech-to-Text (STT) with Whisper
Objective: Transcribe user’s voice commands accurately.

Components:

OpenAI’s Whisper (tiny/base model).

Audio processing pipeline.

Implementation:

Capture and process audio via WebRTC or Streamlit Recorder.

Pass to Whisper for transcription.

Feed result to RAG module.

Contribution:

Robust transcription for multiple accents and noise conditions.

Low-latency STT response for conversational flow.
