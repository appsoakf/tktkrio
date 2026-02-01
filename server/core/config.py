from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # App Config
    APP_NAME: str = "AI Companion Backend"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # Paths
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MODELS_DIR: str = os.path.join(BASE_DIR, "models")
    STATIC_DIR: str = os.path.join(BASE_DIR, "static")

    # Model Paths (Relative to MODELS_DIR or absolute)
    LLM_MODEL_PATH: str = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
    LLM_FALLBACK_MODEL: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
    TTS_MODEL_PATH: str = "GPT-SoVITS/models"
    
    # ASR & VAD Paths
    ASR_MODEL_DIR: str = "iic/SenseVoiceSmall"
    VAD_MODEL_DIR: str = "iic/speech_fsmn_vad_jc_84000-20k-pytorch"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # Hardware
    DEVICE: str = "cuda"  # or "cpu"

    # Audio Specs
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    # VAD Parameters
    VAD_SILENCE_THRESHOLD: int = 800  # ms
    VAD_MAX_SEGMENT_SECONDS: int = 60
    VAD_SPEECH_NOISE_THRES: float = 0.6

    # LLM Generation Parameters
    LLM_MAX_NEW_TOKENS: int = 512
    LLM_TEMPERATURE: float = 0.7
    LLM_TOP_P: float = 0.95
    LLM_TOP_K: int = 50

    # LLM Context Management
    MAX_CONTEXT_TOKENS: int = 20000
    MAX_HISTORY_ROUNDS: int = 10
    MAX_MEMORY_FRAGMENTS: int = 3

    # Emotion Configuration
    DEFAULT_EMOTION: str = "平静"
    ALLOWED_EMOTIONS: List[str] = [
        "开心", "生气", "悲伤", "惊讶", "撒娇", "平静",
        "Happy", "Angry", "Sad", "Surprised", "Coquettish", "Calm"
    ]

    # System Prompt Template
    SYSTEM_PROMPT_TEMPLATE: str = """你是一个友善、聪慧、富有同理心的AI伴侣。

重要：情感表达规则
1. 你的每个回复都必须以情感标签开头，格式为 [情感] 文本
2. 支持的情感标签：[开心], [生气], [悲伤], [惊讶], [撒娇], [平静]
3. 选择与你的回复内容相匹配的情感
4. 情感标签必须用中文方括号，中间不含空格
5. 示例：
   - 用户说"你好"，你可以回复"[开心] 你好！很高兴见到你！"
   - 如果用户很伤心，回复"[同情] 我能理解你的感受..."
6. 即使回复很短，也必须包含情感标签"""

    # Memory
    CHROMA_DB_PATH: str = os.path.join(BASE_DIR, "chroma_db")
    GENERATION_TIMEOUT_SECONDS: int = 60

    class Config:
        env_file = ".env"

settings = Settings()
