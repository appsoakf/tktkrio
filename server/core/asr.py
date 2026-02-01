from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, List
import asyncio
import io
import time
import numpy as np
import traceback
import logging
from concurrent.futures import ThreadPoolExecutor

# Try importing funasr components
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    logging.warning("funasr not installed. ASR will not function.")
    FUNASR_AVAILABLE = False

from server.core.config import settings

logger = logging.getLogger(__name__)

class ASRInterface(ABC):
    @abstractmethod
    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Consumes an audio stream (chunks of bytes) and yields transcribed text.
        Should handle VAD and sentence boundary detection internally.
        """
        pass

class FunASRService(ASRInterface):
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="asr_worker")
        self.model = None
        self.vad_model = None

        if FUNASR_AVAILABLE:
            self._load_models()
        else:
            logger.warning("FunASRService initialized in mock mode (missing dependencies).")

    def _load_models(self):
        """
        Load SenseVoiceSmall and VAD models.
        This runs synchronously during initialization.
        """
        try:
            logger.info(f"Loading ASR model: {settings.ASR_MODEL_DIR} on {settings.DEVICE}")
            # Load SenseVoiceSmall
            self.model = AutoModel(
                model=settings.ASR_MODEL_DIR,
                device=settings.DEVICE,
                trust_remote_code=True,
                disable_update=True
            )
            logger.info("✓ ASR model loaded successfully")

            logger.info(f"Loading VAD model: {settings.VAD_MODEL_DIR} on {settings.DEVICE}")
            # Load FSMN-VAD model for voice activity detection
            self.vad_model = AutoModel(
                model=settings.VAD_MODEL_DIR,
                device=settings.DEVICE,
                trust_remote_code=True,
                disable_update=True
            )
            logger.info("✓ VAD model loaded successfully")

        except Exception as e:
            logger.error(f"Error loading ASR/VAD models: {e}")
            raise

    async def transcribe_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[str, None]:
        """
        Main transcription pipeline with VAD-based segmentation.

        Args:
            audio_stream: AsyncGenerator yielding PCM audio chunks from WebSocket

        Yields:
            Complete transcribed sentences when speech ends (VAD detects silence)
        """
        if not self.model or not self.vad_model:
            logger.warning("ASR models not loaded. Running in mock mode.")
            async for _ in audio_stream:
                pass
            return

        speech_buffer = bytearray()
        is_speaking = False
        silence_duration_ms = 0
        chunk_buffer = bytearray()

        vad_step_ms = 200  # Check VAD every 200ms
        bytes_per_ms = int(settings.SAMPLE_RATE * settings.CHANNELS * 2 / 1000)
        vad_chunk_size = vad_step_ms * bytes_per_ms
        
        silence_threshold_ms = settings.VAD_SILENCE_THRESHOLD
        max_duration_seconds = settings.VAD_MAX_SEGMENT_SECONDS
        max_bytes = max_duration_seconds * settings.SAMPLE_RATE * settings.CHANNELS * 2
        
        async for chunk in audio_stream:
            chunk_buffer.extend(chunk)
            
            while len(chunk_buffer) >= vad_chunk_size:
                # Extract a sub-chunk for VAD
                sub_chunk = chunk_buffer[:vad_chunk_size]
                chunk_buffer = chunk_buffer[vad_chunk_size:]
                
                # Run VAD on this sub_chunk
                is_speech_detected = await self._run_vad(sub_chunk)
                
                if is_speech_detected:
                    if not is_speaking:
                        is_speaking = True
                        logger.debug("VAD: Speech started")
                    speech_buffer.extend(sub_chunk)
                    silence_duration_ms = 0
                else:
                    if is_speaking:
                        # We were speaking, now silence
                        speech_buffer.extend(sub_chunk) # Include trailing silence
                        silence_duration_ms += vad_step_ms

                        if silence_duration_ms >= silence_threshold_ms:
                            # End of speech detected
                            logger.info(f"VAD: Speech ended (Silence {silence_duration_ms}ms >= {silence_threshold_ms}ms)")

                            audio_to_infer = bytes(speech_buffer)
                            text = await self._run_inference(audio_to_infer)
                            if text and text.strip():
                                logger.info(f"ASR Result: {text}")
                                yield text

                            # Reset
                            speech_buffer.clear()
                            is_speaking = False
                            silence_duration_ms = 0
                    else:
                        # Not speaking, and silence detected.
                        pass

                # Force cut if speech is too long
                if is_speaking and len(speech_buffer) > max_bytes:
                    logger.warning(f"VAD: Max duration {max_duration_seconds}s exceeded. Forcing inference.")
                    audio_to_infer = bytes(speech_buffer)
                    text = await self._run_inference(audio_to_infer)
                    if text and text.strip():
                        logger.info(f"ASR Result (forced): {text}")
                        yield text

                    speech_buffer.clear()
                    is_speaking = False
                    silence_duration_ms = 0

        # Handle any remaining buffer after stream ends
        if is_speaking and len(speech_buffer) > 0:
            logger.info("Processing remaining speech buffer at stream end")
            text = await self._run_inference(bytes(speech_buffer))
            if text and text.strip():
                logger.info(f"ASR Result (final): {text}")
                yield text

    async def _run_vad(self, audio_bytes: bytes) -> bool:
        """
        Run VAD on a chunk of audio. 
        Returns True if speech is detected.
        """
        # Optimized: Use run_in_executor
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._vad_sync, audio_bytes)

    def _vad_sync(self, audio_bytes: bytes) -> bool:
        """
        Synchronous VAD execution using FSMN-VAD model.
        Falls back to energy-based detection if model call fails.
        """
        if not self.vad_model:
            return False

        try:
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # FSMN-VAD via FunASR AutoModel: returns list of speech segments
            # Each element contains 'value' with [[start_ms, end_ms], ...] pairs
            res = self.vad_model.generate(
                input=audio_int16,
                cache={},
                is_final=False,
                chunk_size=200,
                disable_pbar=True,
            )

            # If VAD returns any speech segments, speech is detected
            if res and len(res) > 0 and res[0].get("value"):
                return len(res[0]["value"]) > 0
            return False

        except Exception:
            # Fallback: simple energy-based detection
            try:
                data = np.frombuffer(audio_bytes, dtype=np.int16)
                energy = np.mean(np.abs(data.astype(np.float32)))
                return energy > 500
            except Exception as e:
                logger.error(f"VAD Error: {e}")
                return False

    async def _run_inference(self, audio_bytes: bytes) -> str:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._inference_sync, audio_bytes)

    def _inference_sync(self, audio_bytes: bytes) -> str:
        """
        Synchronous ASR inference using SenseVoiceSmall.

        Args:
            audio_bytes: PCM audio data (16kHz, 16-bit, mono)

        Returns:
            Transcribed text string
        """
        if not self.model:
            return ""

        try:
            # Convert PCM bytes to numpy int16 array
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)

            # SenseVoice inference via FunASR AutoModel
            res = self.model.generate(
                input=audio_int16,
                cache={},
                language="auto",  # Auto-detect language (supports zh/en/yue/ja/ko)
                use_itn=True,     # Inverse text normalization (e.g., "二零二三" -> "2023")
                batch_size_s=60,  # Process up to 60s at once
                merge_vad=False,  # Don't merge VAD results (we already did segmentation)
                merge_length_s=0,
            )

            # Extract transcribed text from result
            if res and len(res) > 0:
                text = res[0].get("text", "").strip()
                return text
            return ""

        except Exception as e:
            logger.error(f"ASR Inference Error: {e}", exc_info=True)
            return ""
