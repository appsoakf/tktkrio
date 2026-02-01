"""
Integration tests for the complete LLM pipeline.

Tests the interaction between multiple modules:
- LLM generation
- Emotion parsing
- Memory integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import AsyncGenerator

from server.core.llm import QwenLLMService, EmotionParser
from server.core.memory import ChromaMemoryService
from server.core.config import settings


class TestLLMMemoryIntegration:
    """Integration tests between LLM and Memory services."""

    @pytest.mark.asyncio
    async def test_memory_injection_in_prompt(self):
        """Test that retrieved memories are properly injected into prompt."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        # Mock memories
        memories = [
            "用户的名字是张三",
            "用户喜欢编程",
            "用户最喜欢的编程语言是Python"
        ]

        # Build prompt with memories
        prompt = llm.build_prompt(
            user_input="你还记得我吗？",
            memories=memories,
            history=[]
        )

        # Verify memories are in prompt
        assert "张三" in prompt
        assert "编程" in prompt
        assert "Python" in prompt

    def test_full_prompt_structure(self):
        """Test that full prompt has correct structure."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        memories = ["用户来自北京"]
        history = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "[开心] 你好！"}
        ]

        prompt = llm.build_prompt(
            user_input="今天天气如何？",
            memories=memories,
            history=history
        )

        # Check structure elements
        assert "情感" in prompt  # System prompt
        assert "北京" in prompt  # Memory
        assert "你好" in prompt  # History
        assert "今天天气如何" in prompt  # User input
        assert "助手:" in prompt  # Prompt for model output

    def test_history_with_emotion_tags(self):
        """Test that history with emotion tags is handled correctly."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        # History includes emotion tags
        history = [
            {"role": "user", "content": "我很开心"},
            {"role": "assistant", "content": "[开心] 我也很高兴看到你开心！"},
            {"role": "user", "content": "昨天怎么样？"},
            {"role": "assistant", "content": "[平静] 昨天还不错。"},
        ]

        prompt = llm.build_prompt(
            user_input="明天呢？",
            memories=[],
            history=history
        )

        # Emotion tags should be preserved in history
        assert "[开心]" in prompt or "开心" in prompt
        assert "[平静]" in prompt or "平静" in prompt


class TestEmotionParsingWithLLMOutput:
    """Integration tests for emotion parsing with realistic LLM output."""

    @pytest.mark.asyncio
    async def test_realistic_emotional_response(self):
        """Test parsing a realistic emotional LLM response."""
        async def simulate_llm_output():
            # Simulate token-by-token output from LLM
            response = "[开心] 很高兴见到你！今天天气真好。"
            for char in response:
                yield char

        results = []
        async for emotion, text in EmotionParser.parse(simulate_llm_output()):
            results.append((emotion, text))

        # Should detect emotion and capture text
        assert len(results) > 0
        assert results[0][0] == "开心"
        full_text = "".join(text for _, text in results)
        assert "很高兴见到你" in full_text

    @pytest.mark.asyncio
    async def test_streaming_with_multiple_emotions(self):
        """Test streaming response with emotion changes."""
        async def simulate_varied_response():
            # Mix of emotions in one response
            parts = ["[开心]", "是的，", "[惊讶]", "这太棒了！"]
            for part in parts:
                for char in part:
                    yield char

        results = []
        async for emotion, text in EmotionParser.parse(simulate_varied_response()):
            results.append((emotion, text))

        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_long_streaming_response(self):
        """Test parsing a long streaming response."""
        long_response = "[平静]" + "这是一个很长的回应。" * 50

        async def simulate_long_output():
            for char in long_response:
                yield char

        count = 0
        async for emotion, text in EmotionParser.parse(simulate_long_output()):
            count += 1

        # Should parse without issues
        assert count > 0

    @pytest.mark.asyncio
    async def test_malformed_emotion_recovery(self):
        """Test recovery from malformed emotion tags."""
        async def simulate_malformed():
            yield "【开心】这是错误的括号"  # Chinese brackets instead of English
            yield "\n"
            yield "[悲伤]"
            yield "这是正确的括号"

        results = []
        async for emotion, text in EmotionParser.parse(simulate_malformed()):
            results.append((emotion, text))

        assert len(results) > 0


class TestHistoryManagement:
    """Test chat history management."""

    def test_growing_history(self):
        """Test that history grows correctly with new messages."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        history = []

        # Simulate multiple conversation turns
        for i in range(5):
            user_msg = f"用户消息 {i}"
            assistant_msg = f"[开心] 助手回应 {i}"

            history.append({"role": "user", "content": user_msg})
            history.append({"role": "assistant", "content": assistant_msg})

        # Check history is built correctly
        assert len(history) == 10  # 5 turns * 2 messages each
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_history_truncation_with_max_rounds(self):
        """Test that history respects MAX_HISTORY_ROUNDS setting."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        # Create history with more than MAX_HISTORY_ROUNDS
        history = []
        for i in range(settings.MAX_HISTORY_ROUNDS + 5):
            history.append({"role": "user", "content": f"消息 {i}"})
            history.append({"role": "assistant", "content": f"回应 {i}"})

        prompt = llm.build_prompt(
            user_input="test",
            memories=[],
            history=history
        )

        # Prompt should not be excessively long
        assert len(prompt) < settings.MAX_CONTEXT_TOKENS * 4  # Reasonable upper bound


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_empty_input_handling(self):
        """Test handling of empty user input."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        prompt = llm.build_prompt(
            user_input="",
            memories=[],
            history=[]
        )

        # Should still create valid prompt
        assert "用户: " in prompt
        assert "助手:" in prompt

    def test_very_long_input(self):
        """Test handling of very long user input."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        long_input = "这是一个非常长的输入。" * 100

        prompt = llm.build_prompt(
            user_input=long_input,
            memories=[],
            history=[]
        )

        # Prompt should still be reasonable
        assert len(prompt) < settings.MAX_CONTEXT_TOKENS * 2

    def test_special_characters_in_input(self):
        """Test handling of special characters."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        special_input = "测试@#$%^&*()[]{}你好"

        prompt = llm.build_prompt(
            user_input=special_input,
            memories=[],
            history=[]
        )

        # Should handle special chars without errors
        assert "测试" in prompt
        assert "你好" in prompt


class TestPerformanceCharacteristics:
    """Test performance-related aspects."""

    def test_memory_fragments_limit(self):
        """Test that memory injection respects MAX_MEMORY_FRAGMENTS."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        # Create more memories than MAX_MEMORY_FRAGMENTS
        memories = [f"记忆 {i}" for i in range(10)]

        prompt = llm.build_prompt(
            user_input="query",
            memories=memories,
            history=[]
        )

        # Count how many memories appear
        count = sum(1 for m in memories if m in prompt)
        assert count <= settings.MAX_MEMORY_FRAGMENTS

    def test_prompt_token_estimation_accuracy(self):
        """Test token estimation for history truncation."""
        with patch('server.core.llm.AutoTokenizer'):
            with patch('server.core.llm.AutoModelForCausalLM'):
                llm = QwenLLMService()

        # Create history with known character count
        history = [
            {"role": "user", "content": "短"},  # 1 char
            {"role": "assistant", "content": "回应" * 100},  # 200 chars
        ]

        # Estimate
        total_tokens = sum(int(len(msg["content"]) * 1.3) for msg in history)

        # Should estimate reasonably (1*1.3 + 200*1.3 = 261.3)
        assert 250 < total_tokens < 300


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
