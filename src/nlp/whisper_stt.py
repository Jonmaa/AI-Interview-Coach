"""
Speech-to-Text module using OpenAI Whisper (free, local).
Transcribes audio responses for hands-free interview practice.

Uses the open-source Whisper model (runs locally, no API costs).
Demonstrates NLP and Whisper experience as required by the role.
"""

import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False


# Whisper model sizes (all free, local):
# tiny   - ~39M params, fastest, lower accuracy
# base   - ~74M params, good balance
# small  - ~244M params, better accuracy
# medium - ~769M params, high accuracy
# large  - ~1550M params, best accuracy (needs GPU)
DEFAULT_MODEL = "base"


class SpeechToText:
    """
    Transcribes spoken audio to text using the open-source Whisper model.
    Runs entirely locally - no API calls, no costs.

    Usage:
        stt = SpeechToText()
        text = stt.transcribe_file("answer.wav")
        text = stt.record_and_transcribe(duration=30)
    """

    def __init__(self, model_size: str = DEFAULT_MODEL, device: str = "cpu"):
        """
        Initialize the Whisper speech-to-text engine.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: "cpu" or "cuda" for GPU acceleration
        """
        if not WHISPER_AVAILABLE:
            raise ImportError(
                "Whisper is not installed. Install with: pip install openai-whisper\n"
                "Note: this is the FREE open-source model, not the paid API."
            )
        self.model_size = model_size
        self.device = device
        self.model = None  # Lazy loading

    def _load_model(self):
        """Lazy-load the Whisper model (downloads on first use)."""
        if self.model is None:
            print(f"   🎙️ Loading Whisper '{self.model_size}' model (first time may download)...")
            self.model = whisper.load_model(self.model_size, device=self.device)
            print(f"   ✅ Whisper model loaded on {self.device}")
        return self.model

    def transcribe_file(self, audio_path: str, language: str = "en") -> dict:
        """
        Transcribe an audio file to text.

        Args:
            audio_path: Path to audio file (wav, mp3, m4a, etc.)
            language: Language code (e.g. "en", "es", "fr")

        Returns:
            Dict with 'text' (full transcription) and 'segments' (timestamped parts)
        """
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self._load_model()
        result = model.transcribe(
            str(path),
            language=language,
            fp16=(self.device == "cuda"),
        )

        return {
            "text": result["text"].strip(),
            "segments": [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ],
            "language": result.get("language", language),
        }

    def record_and_transcribe(
        self, duration: int = 30, sample_rate: int = 16000, language: str = "en"
    ) -> str:
        """
        Record audio from microphone and transcribe it.

        Args:
            duration: Maximum recording duration in seconds
            sample_rate: Audio sample rate (16000 recommended for Whisper)
            language: Language code

        Returns:
            Transcribed text
        """
        if not AUDIO_AVAILABLE:
            raise ImportError(
                "Audio recording requires: pip install sounddevice numpy"
            )

        print(f"🎙️ Recording for up to {duration} seconds... (press Ctrl+C to stop early)")

        try:
            audio = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
        except KeyboardInterrupt:
            sd.stop()
            print("⏹️ Recording stopped early.")

        print("🔄 Transcribing...")

        # Save to temp file for Whisper
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            import scipy.io.wavfile as wavfile

            audio_int16 = (audio * 32767).astype(np.int16)
            wavfile.write(f.name, sample_rate, audio_int16)
            result = self.transcribe_file(f.name, language=language)

        Path(f.name).unlink(missing_ok=True)
        return result["text"]

    def detect_language(self, audio_path: str) -> str:
        """
        Detect the spoken language in an audio file.

        Args:
            audio_path: Path to audio file

        Returns:
            Detected language code (e.g. "en", "es")
        """
        model = self._load_model()
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.device)
        _, probs = model.detect_language(mel)
        detected = max(probs, key=probs.get)
        return detected


if __name__ == "__main__":
    print("=" * 50)
    print("WHISPER SPEECH-TO-TEXT TEST (Free - Local)")
    print("=" * 50)

    if not WHISPER_AVAILABLE:
        print("❌ Whisper not installed. Run: pip install openai-whisper")
        print("   This is the FREE open-source model, NOT the paid API.")
    else:
        stt = SpeechToText(model_size="base")
        print(f"✅ SpeechToText engine initialized")
        print(f"   Model: whisper-{stt.model_size}")
        print(f"   Device: {stt.device}")
        print(f"   💰 Cost: $0.00 (runs locally)")
        print(f"\n   Use stt.transcribe_file('audio.wav') to transcribe audio")
        print(f"   Use stt.record_and_transcribe(duration=30) for live recording")
