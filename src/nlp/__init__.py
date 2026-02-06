"""
NLP module - Natural Language Processing utilities.
Includes Whisper speech-to-text for hands-free interview practice.
"""

from .whisper_stt import SpeechToText, WHISPER_AVAILABLE

__all__ = ["SpeechToText", "WHISPER_AVAILABLE"]
