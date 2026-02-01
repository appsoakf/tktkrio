# Static Reference Audio Files for TTS Emotion Control

This directory contains reference audio files used by GPT-SoVITS to control emotional expression in synthesized speech.

## Required Files

You need to provide **6 reference audio files** corresponding to the 6 supported emotions:

| Filename | Emotion (Chinese) | Emotion (English) | Description |
|----------|-------------------|-------------------|-------------|
| `happy.wav` | 开心 | Happy | Joyful, excited, cheerful tone |
| `angry.wav` | 生气 | Angry | Frustrated, irritated, stern tone |
| `sad.wav` | 悲伤 | Sad | Melancholic, sympathetic, gentle tone |
| `surprised.wav` | 惊讶 | Surprised | Shocked, amazed, curious tone |
| `coquettish.wav` | 撒娇 | Coquettish | Playful, cute, affectionate tone |
| `calm.wav` | 平静 | Calm | Neutral, composed, relaxed tone |

## Audio Specifications

- **Format**: WAV (recommended)
- **Sample Rate**: 16kHz or higher
- **Channels**: Mono (1 channel preferred)
- **Duration**: 3-10 seconds of clean speech
- **Quality**: High-quality recording without background noise

## How to Create Reference Audio

1. **Record or source audio clips** with the target emotion
2. **Ensure consistent speaker** (same voice across all emotions)
3. **Clean audio**: Remove noise, normalize volume
4. **Name files correctly** as listed above
5. **Place files in this directory** (`server/static/`)

## Example Usage

When the LLM generates a response like:
```
[开心] 你好！很高兴见到你！
```

The TTS service will use `happy.wav` as the reference audio to synthesize the text with a happy emotional tone.

## Testing

Once files are in place, you can test TTS with:
```python
from server.core.tts import GPTSoVITSService

tts = GPTSoVITSService()
audio = tts._run_inference("你好！", "开心")
```

## Notes

- Reference audio should contain natural speech from your target voice actor
- The text content of the reference audio doesn't matter (prosody is what's being cloned)
- Higher quality references = better emotion control in output
