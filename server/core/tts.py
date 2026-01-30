from abc import ABC, abstractmethod
from typing import AsyncGenerator
import os
import asyncio
from server.core.config import settings

class TTSInterface(ABC):
    @abstractmethod
    async def synthesize_stream(self, text_stream: AsyncGenerator[tuple[str, str], None]) -> AsyncGenerator[bytes, None]:
        """
        Consumes a stream of (emotion, text_chunk) and yields audio bytes.
        """
        pass

class GPTSoVITSService(TTSInterface):
    def __init__(self):
        # Placeholder for GPT-SoVITS initialization
        print(f"Initializing GPTSoVITSService with model: {settings.TTS_MODEL_PATH}")
        self.ref_audio_map = {
            "Happy": os.path.join(settings.STATIC_DIR, "happy.wav"),
            "Sad": os.path.join(settings.STATIC_DIR, "sad.wav"),
            "Angry": os.path.join(settings.STATIC_DIR, "angry.wav"),
            # ... others
        }
        
    async def synthesize_stream(self, text_stream: AsyncGenerator[tuple[str, str], None]) -> AsyncGenerator[bytes, None]:
        """
        Since TTS usually requires full sentences or at least phrases for good prosody,
        we need to buffer the incoming text stream until we hit punctuation.
        """
        buffer = ""
        current_emotion = "Calm"
        
        async for emotion, chunk in text_stream:
            current_emotion = emotion # Update emotion if it changes (though usually one per response)
            buffer += chunk
            
            # Simple sentence splitting
            if any(punct in chunk for punct in [".", "!", "?", "。", "！", "？"]):
                print(f"Synthesizing [{current_emotion}]: {buffer}")
                
                # Use asyncio.to_thread to run the blocking inference in a separate thread
                # This ensures the main event loop is not blocked during TTS generation
                audio_chunk = await asyncio.to_thread(self._run_inference, buffer, current_emotion)
                yield audio_chunk
                
                buffer = ""
        
        # Process remaining buffer
        if buffer:
            print(f"Synthesizing [{current_emotion}]: {buffer}")
            audio_chunk = await asyncio.to_thread(self._run_inference, buffer, current_emotion)
            yield audio_chunk

    def _run_inference(self, text: str, emotion: str) -> bytes:
        # This function simulates a blocking CPU/GPU bound operation
        # Call GPT-SoVITS inference here
        # ref_audio = self.ref_audio_map.get(emotion, self.ref_audio_map["Calm"])
        
        # Simulate processing time
        import time
        time.sleep(0.1) 
        
        return b'\x00\x00' * 1024
