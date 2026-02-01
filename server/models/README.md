# AI Models Directory

This directory should contain all downloaded AI model files and checkpoints.

## Required Models

### 1. LLM (Large Language Model)

**Primary Model:**
- **Name**: Qwen2.5-14B-Instruct-GPTQ-Int4
- **Path**: `Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4/`
- **Size**: ~10 GB
- **Source**: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
- **VRAM**: ~10 GB

**Fallback Model (optional but recommended):**
- **Name**: Qwen2.5-7B-Instruct-GPTQ-Int4
- **Path**: `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4/`
- **Size**: ~5 GB
- **Source**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4
- **VRAM**: ~5 GB

### 2. ASR (Automatic Speech Recognition)

**Model:**
- **Name**: SenseVoiceSmall
- **Path**: `iic/SenseVoiceSmall/`
- **Size**: ~1-2 GB
- **Source**: ModelScope or HuggingFace
- **Command**:
  ```bash
  # Using modelscope
  pip install modelscope
  from modelscope import snapshot_download
  snapshot_download('iic/SenseVoiceSmall', cache_dir='./models')
  ```

### 3. VAD (Voice Activity Detection)

**Model:**
- **Name**: FSMN-VAD
- **Path**: `iic/speech_fsmn_vad_jc_84000-20k-pytorch/`
- **Size**: ~100 MB
- **Source**: ModelScope
- **Command**:
  ```bash
  from modelscope import snapshot_download
  snapshot_download('iic/speech_fsmn_vad_jc_84000-20k-pytorch', cache_dir='./models')
  ```

### 4. TTS (Text-to-Speech)

**Model:**
- **Name**: GPT-SoVITS
- **Path**: `GPT-SoVITS/models/`
- **Size**: ~4 GB
- **Source**: https://github.com/RVC-Boss/GPT-SoVITS
- **Notes**: Follow GPT-SoVITS setup instructions for model checkpoints

### 5. Embedding Model (for RAG)

**Model:**
- **Name**: BGE-M3
- **Path**: `BAAI/bge-m3/`
- **Size**: ~1 GB
- **Source**: https://huggingface.co/BAAI/bge-m3
- **Command**:
  ```bash
  from huggingface_hub import snapshot_download
  snapshot_download('BAAI/bge-m3', cache_dir='./models')
  ```

## Directory Structure

After downloading all models, your directory should look like:

```
models/
├── Qwen/
│   ├── Qwen2.5-14B-Instruct-GPTQ-Int4/
│   │   ├── config.json
│   │   ├── tokenizer.json
│   │   ├── model.safetensors
│   │   └── ...
│   └── Qwen2.5-7B-Instruct-GPTQ-Int4/
│       └── ...
├── iic/
│   ├── SenseVoiceSmall/
│   │   └── ...
│   └── speech_fsmn_vad_jc_84000-20k-pytorch/
│       └── ...
├── BAAI/
│   └── bge-m3/
│       └── ...
└── GPT-SoVITS/
    └── models/
        └── ...
```

## Download Scripts

### Quick Download All Models (Python)

```python
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_download

# ASR & VAD (ModelScope)
snapshot_download('iic/SenseVoiceSmall', cache_dir='./models')
snapshot_download('iic/speech_fsmn_vad_jc_84000-20k-pytorch', cache_dir='./models')

# LLM (HuggingFace)
hf_download('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', cache_dir='./models')
hf_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', cache_dir='./models')

# Embedding (HuggingFace)
hf_download('BAAI/bge-m3', cache_dir='./models')
```

## Total Storage Required

- **Minimum**: ~18 GB (without fallback LLM)
- **Recommended**: ~25 GB (with fallback LLM)

## VRAM Budget (RTX 4090 - 24GB)

| Model | VRAM Usage |
|-------|------------|
| LLM (14B) | ~10 GB |
| TTS | ~4 GB |
| ASR | ~1-2 GB |
| Embedding | ~1 GB |
| System | ~2 GB |
| **Total** | **~18-19 GB** |

Safe operating range with headroom for inference.
