import whisper

def extract_audio_text(file):
    """Convert audio to text using OpenAI Whisper for formats: mp3, wav, m4a, flac, mp4, etc."""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(file)
        segments = [segment["text"].strip() for segment in result["segments"]]
        return "\n\n".join(segments)
    except Exception as e:
        print(f"Error transcribing audio: {e}")
        return []