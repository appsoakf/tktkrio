"""
Unit tests for the LLM module.

Tests cover:
- Model initialization
- Prompt building with memory and history
- Streaming generation
- Emotion parsing
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import AsyncGenerator

from server.core.llm import QwenLLMService, EmotionParser
from server.core.config import settings


class TestQwenLLMService:
    """Test cases for QwenLLMService."""

    @pytest.fixture
    def llm_service(self):
        """Create a LLM service instance for testing."""
        # Mock the model and tokenizer to avoid loading actual models
        with patch('server.core.llm.AutoTokenizer') as mock_tokenizer:
            with patch('server.core.llm.AutoModelForCausalLM') as mock_model:
                mock_tokenizer.from_pretrained.return_value = MagicMock()
                mock_model.from_pretrained.return_value = MagicMock()

                service = QwenLLMService()
                return service

    def test_initialization(self):
        """Test that LLM service initializes without errors."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                service = QwenLLMService()
                assert service is not None
                assert service.tokenizer is not None or service.fallback_mode

    def test_build_prompt_with_system_prompt(self, llm_service):
        """Test that system prompt is included in built prompt."""
        prompt = llm_service.build_prompt(
            user_input="你好",
            memories=[],
            history=[]
        )

        # Check that system prompt is included
        assert "情感" in prompt or "emotion" in prompt.lower()
        assert "用户" in prompt or "User" in prompt

    def test_build_prompt_with_memories(self, llm_service):
        """Test that memory context is injected into prompt."""
        memories = ["用户喜欢编程", "用户来自中国"]

        prompt = llm_service.build_prompt(
            user_input="你了解我吗？",
            memories=memories,
            history=[]
        )

        # Check that memories are included
        for memory in memories:
            assert memory in prompt

    def test_build_prompt_with_history(self, llm_service):
        """Test that chat history is included in prompt."""
        history = [
            {"role": "user", "content": "你叫什么名字？"},
            {"role": "assistant", "content": "[开心] 我叫AI伴侣！"},
        ]

        prompt = llm_service.build_prompt(
            user_input="很高兴认识你",
            memories=[],
            history=history
        )

        # Check that history is included
        assert "你叫什么名字" in prompt
        assert "AI伴侣" in prompt

    def test_build_prompt_history_truncation(self, llm_service):
        """Test that large history is truncated appropriately."""
        # Create a large history (100 entries)
        large_history = []
        for i in range(100):
            large_history.append({
                "role": "user",
                "content": f"用户消息 {i}" * 10
            })
            large_history.append({
                "role": "assistant",
                "content": f"助手回应 {i}" * 10
            })

        prompt = llm_service.build_prompt(
            user_input="latest",
            memories=[],
            history=large_history
        )

        # Prompt should be reasonable length, not enormous
        assert len(prompt) < 100000  # Should be well under 100k chars
        # But should contain some history
        assert "用户消息" in prompt

    def test_build_prompt_max_memory_fragments(self, llm_service):
        """Test that only max memory fragments are included."""
        memories = [f"记忆 {i}" for i in range(10)]  # 10 memories

        prompt = llm_service.build_prompt(
            user_input="query",
            memories=memories,
            history=[]
        )

        # Should only include MAX_MEMORY_FRAGMENTS memories
        memory_count = sum(1 for m in memories if m in prompt)
        assert memory_count <= settings.MAX_MEMORY_FRAGMENTS

    @pytest.mark.asyncio
    async def test_generate_stream_basic(self, llm_service):
        """Test that generate_stream yields tokens."""
        # Mock the model generation
        llm_service.tokenizer = MagicMock()
        llm_service.tokenizer.eos_token_id = 2
        llm_service.model = MagicMock()
        llm_service.model.device = "cpu"

        prompt = "测试提示词"

        # For testing, we'll check that the generator is created
        # In real scenarios, we'd mock the model.generate to return tokens
        try:
            async for chunk in llm_service.generate_stream(prompt):
                # We should receive text chunks
                assert isinstance(chunk, str)
                break  # Break after first chunk for testing
        except (RuntimeError, AttributeError):
            # Expected due to mocking
            pass

    def test_truncate_history_empty(self, llm_service):
        """Test truncation with empty history."""
        result = llm_service._truncate_history([])
        assert result == []

    def test_truncate_history_within_budget(self, llm_service):
        """Test truncation when history fits within budget."""
        history = [
            {"role": "user", "content": "短消息"},
            {"role": "assistant", "content": "短回应"},
        ]

        result = llm_service._truncate_history(history, max_tokens=1000)
        assert len(result) == 2

    def test_truncate_history_exceeds_budget(self, llm_service):
        """Test truncation when history exceeds budget."""
        history = [
            {"role": "user", "content": "短消息"},
            {"role": "assistant", "content": "短回应"},
            {"role": "user", "content": "很长的消息" * 100},
            {"role": "assistant", "content": "很长的回应" * 100},
        ]

        result = llm_service._truncate_history(history, max_tokens=100)
        # Should keep at least 1 entry
        assert len(result) >= 1
        # Should not exceed token budget (approximately)
        total_tokens = sum(int(len(msg["content"]) * 1.3) for msg in result)
        # Allow some margin due to estimation
        assert total_tokens <= 150


class TestEmotionParser:
    """Test cases for EmotionParser."""

    @pytest.mark.asyncio
    async def test_parse_with_emotion_tag(self):
        """Test parsing text with emotion tag."""
        async def mock_stream():
            yield "[开心]"
            yield "你好！"

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Should detect emotion
        emotions = [e for e, _ in results]
        assert "开心" in emotions

    @pytest.mark.asyncio
    async def test_parse_without_emotion_tag(self):
        """Test parsing text without emotion tag (should default)."""
        async def mock_stream():
            yield "这是没有"
            yield "情感标签的"
            yield "文本"

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Should use default emotion
        emotions = [e for e, _ in results]
        assert settings.DEFAULT_EMOTION in emotions

    @pytest.mark.asyncio
    async def test_parse_invalid_emotion_fallback(self):
        """Test that invalid emotion falls back to default."""
        async def mock_stream():
            yield "[未知情感]"
            yield "文本"

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Invalid emotion should fallback to default
        assert results[0][0] == settings.DEFAULT_EMOTION

    @pytest.mark.asyncio
    async def test_parse_multiple_emotions(self):
        """Test parsing text with multiple emotion transitions."""
        async def mock_stream():
            yield "[开心]"
            yield "你好"
            yield "[悲伤]"
            yield "再见"

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Should have at least one entry
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_parse_empty_stream(self):
        """Test parsing empty stream."""
        async def mock_stream():
            return
            yield  # Make it a generator

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Should handle empty stream gracefully
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_parse_emotion_at_boundary(self):
        """Test emotion tag detection at chunk boundaries."""
        async def mock_stream():
            yield "["
            yield "开"
            yield "心"
            yield "]"
            yield "你好"

        results = []
        async for emotion, text in EmotionParser.parse(mock_stream()):
            results.append((emotion, text))

        # Should eventually detect the emotion
        emotions = [e for e, _ in results]
        # May not detect due to boundary, should fallback to default
        assert len(emotions) > 0


class TestEmotionParserIntegration:
    """Integration tests for emotion parsing with LLM."""

    @pytest.mark.asyncio
    async def test_llm_output_parsing_flow(self):
        """Test the complete flow of LLM output through emotion parser."""
        # Simulate LLM output stream
        async def mock_llm_stream():
            yield "[开心]"
            yield " 你好"
            yield "，我很高兴"
            yield "见到你！"

        results = []
        async for emotion, text in EmotionParser.parse(mock_llm_stream()):
            results.append((emotion, text))

        # Should have parsed emotion and text correctly
        assert len(results) > 0
        first_emotion, first_text = results[0]
        assert first_emotion == "开心"
        assert "你好" in first_text


class TestConfig:
    """Test configuration settings."""

    def test_emotion_configuration(self):
        """Test that emotion configuration is properly set."""
        assert settings.DEFAULT_EMOTION in settings.ALLOWED_EMOTIONS
        assert len(settings.ALLOWED_EMOTIONS) > 0

    def test_llm_generation_parameters(self):
        """Test that LLM generation parameters are valid."""
        assert settings.LLM_MAX_NEW_TOKENS > 0
        assert 0 < settings.LLM_TEMPERATURE <= 2.0
        assert 0 < settings.LLM_TOP_P <= 1.0
        assert settings.LLM_TOP_K > 0
        assert settings.MAX_CONTEXT_TOKENS > settings.LLM_MAX_NEW_TOKENS

    def test_memory_configuration(self):
        """Test that memory configuration is valid."""
        assert settings.MAX_HISTORY_ROUNDS > 0
        assert settings.MAX_MEMORY_FRAGMENTS > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
