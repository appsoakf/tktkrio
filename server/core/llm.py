import asyncio
import logging
import re
from abc import ABC, abstractmethod
from typing import AsyncGenerator, List, Dict, Optional
from threading import Thread
from queue import Queue, Empty

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer

from server.core.config import settings

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """Generate a streaming response from the LLM."""
        pass

    @abstractmethod
    def build_prompt(
        self,
        user_input: str,
        memories: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        """Construct the full prompt with RAG context."""
        pass


class QwenLLMService(LLMInterface):
    """
    Qwen-2.5-14B-Instruct LLM Service with streaming generation and emotion control.

    Hardware Requirements:
    - Minimum VRAM: 10GB (quantized model)
    - Recommended: RTX 4090 (24GB) for headroom

    Context Window: 32,768 tokens (Qwen-2.5 native)
    Max Generation: 512 tokens per response (configurable)
    """

    def __init__(self):
        """Initialize the Qwen LLM service with model loading and VRAM management."""
        try:
            self.tokenizer = None
            self.model = None
            self.fallback_mode = False
            self._load_model()
            logger.info("✓ QwenLLMService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize QwenLLMService: {e}")
            raise

    def _load_model(self):
        """Load the Qwen model with GPTQ quantization."""
        logger.info(f"Loading model: {settings.LLM_MODEL_PATH}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_MODEL_PATH,
                trust_remote_code=True,
            )

            # Load model with automatic device mapping
            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_MODEL_PATH,
                device_map="auto",
                torch_dtype=torch.float16, # 使用 FP16 提升效率
                trust_remote_code=True,
            )

            logger.info(
                f"✓ Loaded {settings.LLM_MODEL_PATH} successfully"
            )
            self.fallback_mode = False
            self._log_vram_usage()

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                logger.warning(
                    f"CUDA OOM during model load, trying fallback model: {settings.LLM_FALLBACK_MODEL}"
                )
                self._load_fallback_model()
            else:
                raise
        except Exception as e:
            logger.error(f"Unexpected error loading model: {e}")
            raise

    # 若发生 OOM，则加载 Qwen-2.5-7B 作为备用
    def _load_fallback_model(self):
        """Load a smaller fallback model if the main model fails to load."""
        try:
            logger.info(f"Loading fallback model: {settings.LLM_FALLBACK_MODEL}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                settings.LLM_FALLBACK_MODEL,
                trust_remote_code=True,
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                settings.LLM_FALLBACK_MODEL,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )

            logger.info(f"✓ Loaded fallback model {settings.LLM_FALLBACK_MODEL}")
            self.fallback_mode = True
            self._log_vram_usage()

        except Exception as e:
            logger.error(f"Fallback model loading failed: {e}")
            raise

    def _log_vram_usage(self):
        """Log current VRAM usage for debugging."""
        try:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(
                f"VRAM Usage: {allocated:.2f}GB / {total:.2f}GB "
                f"({allocated/total*100:.1f}%)"
            )
        except Exception:
            pass

    def _truncate_history(
        self,
        history: List[Dict[str, str]],
        max_tokens: int = None
    ) -> List[Dict[str, str]]:
        """
        Truncate chat history to fit within token budget.
        Always keeps at least 1 exchange for coherence.
        """
        if not history or max_tokens is None:
            return history if history else []

        # Estimate tokens (1 char ≈ 1.3 tokens for Chinese/English mix)
        def estimate_tokens(text: str) -> int:
            return max(1, int(len(text) * 1.3))

        total_tokens = sum(estimate_tokens(msg["content"]) for msg in history)

        # If within budget, return all
        if total_tokens <= max_tokens:
            return history

        # Truncate from oldest, keeping at least 1 pair
        truncated = []
        remaining_budget = max_tokens

        # Iterate from newest to oldest
        for exchange in reversed(history):
            exchange_tokens = estimate_tokens(exchange["content"])

            if remaining_budget >= exchange_tokens:
                truncated.insert(0, exchange)
                remaining_budget -= exchange_tokens
            else:
                break

        return truncated if truncated else (history[-1:] if history else [])

    def build_prompt(
        self,
        user_input: str,
        memories: List[str],
        history: List[Dict[str, str]]
    ) -> str:
        """
        Construct the full prompt with system prompt, memories, history, and user input.

        Args:
            user_input: Current user message
            memories: List of relevant memory fragments from ChromaDB
            history: Chat history (list of messages with 'role' and 'content')

        Returns:
            Formatted prompt string ready for tokenization
        """
        # Start with system prompt
        system_prompt = settings.SYSTEM_PROMPT_TEMPLATE

        # Add memory context if available
        # 结构如下：
        """
        相关记忆：
        - xxx
        - xxx
        ... 
        """
        memory_section = ""
        if memories:
            memory_section = "\n\n相关记忆:\n" + "\n".join(
                f"- {m}" for m in memories[:settings.MAX_MEMORY_FRAGMENTS]
            )

        # Truncate history to fit within token budget
        # Reserve space: system (500) + memories (500) + generation (2000)
        history_token_budget = settings.MAX_CONTEXT_TOKENS - 500 - 500 - 2000
        truncated_history = self._truncate_history(history, history_token_budget)

        # Build history section
        # 结构如下：
        """
        [History Section]
        最近的对话:
        用户: 你好
        助手: [开心] 你好！
        用户: 你是谁？
        助手: [平静] 我是你的AI伴侣。
        ...
        """
        history_section = ""
        if truncated_history:
            history_lines = []
            for msg in truncated_history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_lines.append(f"用户: {content}")
                else:
                    history_lines.append(f"助手: {content}")
            history_section = "\n\n最近的对话:\n" + "\n".join(history_lines)

        # Assemble full prompt
        full_prompt = (
            f"{system_prompt}"
            f"{memory_section}"
            f"{history_section}"
            f"\n\n用户: {user_input}\n助手:"
        )

        return full_prompt

    async def generate_stream(
        self,
        prompt: str,
        history: List[Dict[str, str]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream token-by-token generation without blocking the async event loop.

        Uses TextIteratorStreamer with threading to handle blocking model.generate()
        calls while keeping the async loop responsive.

        Args:
            prompt: Full prompt string
            history: Chat history (unused but kept for interface compatibility)

        Yields:
            Text chunks as they're generated token-by-token
        """
        try:
            # Tokenize the prompt
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)

            # Create streamer for token-by-token output
            streamer = TextIteratorStreamer(
                self.tokenizer,
                skip_prompt=True,  # Don't repeat input prompt
                skip_special_tokens=True
            )

            # Prepare generation kwargs
            generation_kwargs = {
                "inputs": inputs,
                "streamer": streamer,
                "max_new_tokens": settings.LLM_MAX_NEW_TOKENS,
                "temperature": settings.LLM_TEMPERATURE,
                "top_p": settings.LLM_TOP_P,
                "top_k": settings.LLM_TOP_K,
                "do_sample": True,
                "use_cache": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            # Start generation in separate thread
            generation_thread = Thread(
                target=self.model.generate,
                kwargs=generation_kwargs,
                daemon=False
            )
            generation_thread.start()

            # Consume streamer output asynchronously
            try:
                # 增量式的一个个token进行输出
                for text_chunk in streamer:
                    yield text_chunk
                    # 将控制权归还给eventloop，避免协程“饿死”其他任务
                    await asyncio.sleep(0)
            finally:
                # Ensure thread completes
                # 超时熔断，如果线程因为生成线程因模型卡死、死循环等异常未在超时内结束
                # 则join() 会主动返回（不再阻塞）
                generation_thread.join(timeout=settings.GENERATION_TIMEOUT_SECONDS)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("CUDA OOM during generation, attempting recovery")
                # Try with reduced tokens
                yield await self._generate_with_reduced_tokens(prompt)
            else:
                logger.error(f"Generation error: {e}")
                yield f"[{settings.DEFAULT_EMOTION}] 抱歉，生成回复时出错了。"

        except GeneratorExit:
            logger.info("Generation stream interrupted by client")

        except Exception as e:
            logger.error(f"Unexpected error during generation: {e}")
            yield f"[{settings.DEFAULT_EMOTION}] 抱歉，出了点问题。"

    async def _generate_with_reduced_tokens(self, prompt: str) -> str:
        """
        Fallback generation with reduced max_new_tokens for OOM recovery.
        """
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.model.device)

            # Reduce tokens significantly
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=256,  # Reduced from 512
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract only the response part (after prompt)
            if "助手:" in response:
                response = response.split("助手:")[-1].strip()

            return response

        except Exception as e:
            logger.error(f"Even reduced generation failed: {e}")
            return f"[{settings.DEFAULT_EMOTION}] 抱歉，我目前资源紧张，无法生成长回复。"


class EmotionParser:
    """
    Parses LLM output stream to extract emotion tags.

    Expected format: [Emotion]text...
    If no emotion tag found in first 20 characters, defaults to configured default emotion.
    """

    EMOTION_TAG_PATTERN = re.compile(r"\[(.*?)\]")

    @staticmethod
    async def parse(
        text_stream: AsyncGenerator[str, None]
    ) -> AsyncGenerator[tuple[str, str], None]:
        """
        Parse text stream to extract emotion tags.

        Args:
            text_stream: AsyncGenerator yielding text chunks

        Yields:
            Tuples of (emotion, text_chunk) where emotion is detected or defaults
        """
        buffer = ""
        emotion = settings.DEFAULT_EMOTION
        emotion_detected = False

        async for chunk in text_stream:
            if not emotion_detected:
                buffer += chunk

                # Try to find emotion tag
                match = EmotionParser.EMOTION_TAG_PATTERN.match(buffer)

                if match:
                    detected_emotion = match.group(1)

                    # Validate emotion is in allowed list
                    if detected_emotion in settings.ALLOWED_EMOTIONS:
                        emotion = detected_emotion
                    else:
                        # Map to closest match or use default
                        logger.warning(
                            f"Unknown emotion: {detected_emotion}, using default"
                        )
                        emotion = settings.DEFAULT_EMOTION

                    # Yield remaining text after tag
                    remaining = buffer[match.end():]
                    if remaining:
                        yield (emotion, remaining)

                    emotion_detected = True
                    buffer = ""

                elif len(buffer) > 20:
                    # No tag found after 20 chars, use default
                    emotion_detected = True
                    yield (emotion, buffer)
                    buffer = ""
            else:
                # Emotion already detected, just forward chunks
                yield (emotion, chunk)
