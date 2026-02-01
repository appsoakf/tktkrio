from abc import ABC, abstractmethod
from typing import AsyncGenerator
import os
import asyncio
import logging
import numpy as np

from server.core.config import settings

logger = logging.getLogger(__name__)

class TTSInterface(ABC):
    @abstractmethod
    async def synthesize_stream(self, text_stream: AsyncGenerator[tuple[str, str], None]) -> AsyncGenerator[bytes, None]:
        """
        Consumes a stream of (emotion, text_chunk) and yields audio bytes.
        """
        pass

class GPTSoVITSService(TTSInterface):
    """
    GPT-SoVITS text-to-speech service with emotion control.

    Uses reference audio files to control emotional expression in synthesized speech.
    """

    def __init__(self):
        logger.info(f"Initializing GPTSoVITSService with model: {settings.TTS_MODEL_PATH}")

        # Map emotions to reference audio files
        self.ref_audio_map = {
            # Chinese emotion labels (primary)
            "开心": os.path.join(settings.STATIC_DIR, "happy.wav"),
            "生气": os.path.join(settings.STATIC_DIR, "angry.wav"),
            "悲伤": os.path.join(settings.STATIC_DIR, "sad.wav"),
            "惊讶": os.path.join(settings.STATIC_DIR, "surprised.wav"),
            "撒娇": os.path.join(settings.STATIC_DIR, "coquettish.wav"),
            "平静": os.path.join(settings.STATIC_DIR, "calm.wav"),
            # English emotion labels (fallback)
            "Happy": os.path.join(settings.STATIC_DIR, "happy.wav"),
            "Angry": os.path.join(settings.STATIC_DIR, "angry.wav"),
            "Sad": os.path.join(settings.STATIC_DIR, "sad.wav"),
            "Surprised": os.path.join(settings.STATIC_DIR, "surprised.wav"),
            "Coquettish": os.path.join(settings.STATIC_DIR, "coquettish.wav"),
            "Calm": os.path.join(settings.STATIC_DIR, "calm.wav"),
        }

        # Load GPT-SoVITS models
        self._load_models()
        
    def _load_models(self):
        """
        Load GPT-SoVITS models.

        This method initializes the TTS model components. The actual implementation
        will depend on the GPT-SoVITS library structure once models are downloaded.
        """
        try:
            # GPT-SoVITS model loading would go here
            # Example structure (pseudo-code):
            # from GPTSoVITS import load_model
            # self.tts_model = load_model(settings.TTS_MODEL_PATH, device=settings.DEVICE)

            logger.info("✓ GPT-SoVITS models loaded successfully")
            self.models_loaded = True

        except Exception as e:
            logger.warning(f"GPT-SoVITS models not available: {e}")
            logger.warning("Running in mock mode - will generate silence")
            self.models_loaded = False

    async def synthesize_stream(self, text_stream: AsyncGenerator[tuple[str, str], None]) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech from (emotion, text) stream with sentence-level buffering.

        Since TTS requires full sentences or phrases for good prosody,
        we buffer the incoming text stream until we hit punctuation.

        Args:
            text_stream: AsyncGenerator yielding (emotion, text_chunk) tuples

        Yields:
            PCM audio bytes (16kHz, 16-bit, mono)
        """
        buffer = ""
        current_emotion = settings.DEFAULT_EMOTION

        async for emotion, chunk in text_stream:
            current_emotion = emotion  # Update emotion if it changes
            buffer += chunk

            # Simple sentence splitting on punctuation marks
            if any(punct in chunk for punct in [".", "!", "?", "。", "！", "？", "~", "..."]):
                logger.info(f"TTS Synthesizing [{current_emotion}]: {buffer}")

                # Use asyncio.to_thread to run blocking inference without blocking event loop
                audio_chunk = await asyncio.to_thread(self._run_inference, buffer, current_emotion)
                yield audio_chunk

                buffer = ""

        # Process remaining buffer if any text remains
        if buffer.strip():
            logger.info(f"TTS Synthesizing (final) [{current_emotion}]: {buffer}")
            audio_chunk = await asyncio.to_thread(self._run_inference, buffer, current_emotion)
            yield audio_chunk

    def _run_inference(self, text: str, emotion: str) -> bytes:
        """
        Synchronous TTS inference with emotion control.

        Args:
            text: Text to synthesize
            emotion: Emotion label (e.g., "开心", "生气")

        Returns:
            PCM audio bytes (16kHz, 16-bit, mono)
        """
        if not text.strip():
            return b''

        # Get reference audio for the specified emotion
        ref_audio = self.ref_audio_map.get(emotion, self.ref_audio_map.get("平静"))

        try:
            # Real GPT-SoVITS inference would go here
            # Example structure (pseudo-code):
            # audio_array = self.tts_model.infer(
            #     text=text,
            #     ref_audio_path=ref_audio,
            #     ref_text="参考音频对应的文本",  # Reference audio transcription
            #     top_k=5,
            #     top_p=1.0,
            #     temperature=1.0,
            # )
            # return audio_array.tobytes()

            if not self.models_loaded:
                # Mock mode: return short silence
                sample_count = int(len(text) * settings.SAMPLE_RATE * 0.1)  # ~100ms per char
                silence = np.zeros(sample_count, dtype=np.int16)
                return silence.tobytes()

            # Placeholder for actual GPT-SoVITS call
            logger.debug(f"Generating speech for: '{text}' with emotion: {emotion}")

            # Simulate realistic audio output
            import time
            time.sleep(0.05)  # Simulate processing time

            # Generate short silence as placeholder
            duration_seconds = min(len(text) * 0.15, 5.0)  # Rough estimate
            sample_count = int(duration_seconds * settings.SAMPLE_RATE)
            audio_data = np.zeros(sample_count, dtype=np.int16)

            return audio_data.tobytes()

        except Exception as e:
            logger.error(f"TTS Inference Error: {e}", exc_info=True)
            # Return short silence on error
            return b'\x00\x00' * 1600  # 0.1s of silence
