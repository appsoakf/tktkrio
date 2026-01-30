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
    TTS_MODEL_PATH: str = "GPT-SoVITS/models"
    ASR_MODEL_NAME: str = "iic/SenseVoiceSmall"
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"

    # Hardware
    DEVICE: str = "cuda"  # or "cpu"
    
    # Memory
    CHROMA_DB_PATH: str = os.path.join(BASE_DIR, "chroma_db")
    MAX_HISTORY_ROUNDS: int = 10

    class Config:
        env_file = ".env"

settings = Settings()
