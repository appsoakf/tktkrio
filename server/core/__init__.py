"""
AI Companion Core Services

This package contains the core backend services for the AI companion:
- ASR: Speech recognition (听觉)
- LLM: Language model and emotion parsing (大脑)
- TTS: Text-to-speech synthesis (表达)
- Memory: Long-term memory and RAG (记忆)
- Config: Global configuration settings
"""

from server.core.config import settings
from server.core.asr import ASRInterface, FunASRService
from server.core.llm import LLMInterface, QwenLLMService, EmotionParser
from server.core.tts import TTSInterface, GPTSoVITSService
from server.core.memory import MemoryInterface, ChromaMemoryService

__all__ = [
    "settings",
    "ASRInterface",
    "FunASRService",
    "LLMInterface",
    "QwenLLMService",
    "EmotionParser",
    "TTSInterface",
    "GPTSoVITSService",
    "MemoryInterface",
    "ChromaMemoryService",
]
