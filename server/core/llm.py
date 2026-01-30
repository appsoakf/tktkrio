from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict
from server.core.config import settings
import re

class LLMInterface(ABC):
    @abstractmethod
    async def generate_stream(self, prompt: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass

    @abstractmethod
    def build_prompt(self, user_input: str, memories: List[str], history: List[Dict[str, str]]) -> str:
        """Construct the full prompt with RAG context."""
        pass

class QwenLLMService(LLMInterface):
    def __init__(self):
        # Placeholder for model loading
        # from transformers import AutoTokenizer, AutoModelForCausalLM
        # self.tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_PATH)
        # self.model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL_PATH, device_map="auto")
        print(f"Initializing QwenLLMService with model: {settings.LLM_MODEL_PATH}")
        pass

    def build_prompt(self, user_input: str, memories: List[str], history: List[Dict[str, str]]) -> str:
        # Construct system prompt
        system_prompt = (
            "You are an AI companion. Your response must include an emotion tag at the beginning, "
            "like [Happy], [Sad], [Angry], [Surprised], [Coquettish], [Calm].
"
        )
        
        # Add long-term memory context
        if memories:
            system_prompt += f"Relevant Memories:
" + "
".join(memories) + "
"

        # Note: In a real implementation using chat templates (tokenizer.apply_chat_template), 
        # you would structure this as a list of messages rather than a single string.
        # This is a simplified representation.
        return f"{system_prompt}
User: {user_input}
Assistant:"

    async def generate_stream(self, prompt: str, history: List[Dict[str, str]] = None) -> AsyncGenerator[str, None]:
        # In a real implementation with transformers:
        # 1. Setup TextIteratorStreamer
        # 2. Run model.generate in a separate thread (to avoid blocking event loop)
        # 3. Yield from streamer
        
        # from transformers import TextIteratorStreamer
        # from threading import Thread
        # streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)
        # generation_kwargs = dict(inputs=inputs, streamer=streamer, ...)
        # thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        # thread.start()
        # for new_text in streamer:
        #     yield new_text

        # Simulation with non-blocking sleep (asyncio.sleep already yields control)
        # But if we were doing heavy computation here, we MUST use a thread.
        # Here we simulate the thread pattern for architectural correctness.
        
        import asyncio
        from queue import Queue, Empty
        from threading import Thread
        import time

        # Queue to communicate between generation thread and async generator
        token_queue = Queue()
        
        def _generation_task():
            simulated_response = "[Happy] Hello! I am your AI companion. How can I help you today?"
            # Simulate heavy computation blocking a thread, NOT the loop
            for char in simulated_response:
                time.sleep(0.02) # Simulate compute time per token
                token_queue.put(char)
            token_queue.put(None) # Signal end
        
        # Start generation in a separate thread
        thread = Thread(target=_generation_task)
        thread.start()
        
        # Consume from queue asynchronously
        while True:
            # Check queue non-blocking
            try:
                char = token_queue.get_nowait()
                if char is None:
                    break
                yield char
            except Empty:
                # If empty, yield control back to event loop for a bit
                await asyncio.sleep(0.01)


# EmotionParser接收的是llm的输出，并假设text总是形如: [emotion tag1]text1[emotion tag2]text2...
class EmotionParser:
    @staticmethod
    def parse(text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[tuple[str, str], None]:
        """
        Parses the text stream to extract emotion tags.
        Yields (emotion, text_chunk).
        
        Logic:
        1. Buffer text until an emotion tag [...] is found or rule out its presence at start.
        2. Once emotion is determined (or defaulted), yield subsequent text.
        """
        # This is a complex logic to implement perfectly in a stream.
        # Simplified version: Assume the tag is always at the very start.
        
        buffer = ""
        emotion = "Calm" # Default
        emotion_detected = False
        
        # async for可以非阻塞等待前一个模块的输出，当等待前一个模块还未输出时，释放线程处理其他连接
        async for chunk in text_stream:
            if not emotion_detected:
                buffer += chunk
                # Check if buffer contains a complete tag like [Happy]
                match = re.match(r"\[(.*?)\]", buffer)
                if match:
                    emotion = match.group(1)
                    # Yield the part after the tag
                    remaining_text = buffer[match.end():]
                    if remaining_text:
                        yield (emotion, remaining_text)
                    emotion_detected = True
                    buffer = "" # Clear buffer
                elif len(buffer) > 20: # If buffer gets too long without tag, assume no tag
                    emotion_detected = True
                    yield (emotion, buffer)
                    buffer = ""
            else:
                yield (emotion, chunk)
