from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
import logging
from typing import AsyncGenerator
from datetime import datetime

from server.core.config import settings
from server.core.asr import FunASRService
from server.core.llm import QwenLLMService, EmotionParser
from server.core.tts import GPTSoVITSService
from server.core.memory import ChromaMemoryService

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global services (initialized on startup)
services = {}

@app.on_event("startup")
async def startup_event():
    """Initialize all AI services on startup."""
    logger.info("=" * 60)
    logger.info(f"Starting {settings.APP_NAME}")
    logger.info("=" * 60)

    try:
        logger.info("Initializing services...")

        services["memory"] = ChromaMemoryService()
        services["llm"] = QwenLLMService()
        services["asr"] = FunASRService()
        services["tts"] = GPTSoVITSService()

        logger.info("✓ All services initialized successfully")
        logger.info(f"Server ready at http://{settings.HOST}:{settings.PORT}")
        logger.info(f"WebSocket endpoint: ws://{settings.HOST}:{settings.PORT}/ws/audio")

    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Companion Backend is running"}

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    Main WebSocket endpoint for real-time audio conversation.

    Pipeline:
    1. Client sends PCM audio chunks
    2. ASR transcribes to text with VAD
    3. Query long-term memory for relevant context
    4. LLM generates response with emotion tags
    5. TTS synthesizes emotional speech
    6. Stream audio back to client
    7. Save interaction to memory
    """
    await websocket.accept()
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"[Session {session_id}] WebSocket connection established")

    # Retrieve service instances
    asr = services["asr"]
    llm = services["llm"]
    tts = services["tts"]
    memory = services["memory"]

    # Session state - short-term memory for this conversation
    chat_history = []

    try:
        # Define the pipeline: audio stream from client
        async def receive_audio_stream():
            """Generator that yields audio chunks from WebSocket."""
            try:
                while True:
                    data = await websocket.receive_bytes()
                    yield data
            except WebSocketDisconnect:
                logger.info(f"[Session {session_id}] Client disconnected during audio reception")
            except Exception as e:
                logger.error(f"[Session {session_id}] Error receiving audio: {e}")

        # Step 1: ASR Pipeline - Audio -> Text
        asr_stream = asr.transcribe_stream(receive_audio_stream())

        async for user_text in asr_stream:
            logger.info(f"[Session {session_id}] User: {user_text}")

            # Step 2: RAG - Query relevant memories
            memories = await memory.query_memory(user_text, n_results=settings.MAX_MEMORY_FRAGMENTS)
            if memories:
                logger.debug(f"[Session {session_id}] Retrieved {len(memories)} memory fragments")
            
            # Step 3: Build LLM prompt with context
            prompt = llm.build_prompt(user_text, memories, chat_history)

            # Add user message to short-term history
            chat_history.append({"role": "user", "content": user_text})

            # Truncate history if too long
            if len(chat_history) > settings.MAX_HISTORY_ROUNDS * 2:
                chat_history = chat_history[-(settings.MAX_HISTORY_ROUNDS * 2):]

            # Step 4: LLM generates streaming response
            llm_text_stream = llm.generate_stream(prompt, chat_history)

            # Step 5: Parse emotion tags from LLM output
            emotion_stream = EmotionParser.parse(llm_text_stream)

            # Track full response for history and memory
            full_response = ""

            async def observing_emotion_stream(stream):
                """Wrapper to accumulate full response while streaming."""
                nonlocal full_response
                async for emotion, text in stream:
                    full_response += text
                    yield emotion, text

            # Step 6: TTS synthesis and streaming
            audio_stream = tts.synthesize_stream(observing_emotion_stream(emotion_stream))

            async for audio_chunk in audio_stream:
                await websocket.send_bytes(audio_chunk)

            logger.info(f"[Session {session_id}] Assistant: {full_response}")

            # Update short-term history
            chat_history.append({"role": "assistant", "content": full_response})

            # Step 7: Save to long-term memory (async background task)
            asyncio.create_task(memory.add_memory(
                text=f"User: {user_text}\nAssistant: {full_response}",
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "session_id": session_id
                }
            ))

    except WebSocketDisconnect:
        logger.info(f"[Session {session_id}] Client disconnected")
    except Exception as e:
        logger.error(f"[Session {session_id}] WebSocket error: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # 第一个参数是web模块实例，“模块路径：模块实例名”
    uvicorn.run("server.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
