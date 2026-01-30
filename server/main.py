from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import asyncio
from typing import AsyncGenerator

from server.core.config import settings
from server.core.asr import FunASRService
from server.core.llm import QwenLLMService, EmotionParser
from server.core.tts import GPTSoVITSService
from server.core.memory import ChromaMemoryService

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
    # Initialize services
    # In production, you might want to lazy load or manage via dependency injection
    services["asr"] = FunASRService()
    services["llm"] = QwenLLMService()
    services["tts"] = GPTSoVITSService()
    services["memory"] = ChromaMemoryService()
    print("All services started.")

@app.get("/")
async def root():
    return {"status": "ok", "message": "AI Companion Backend is running"}

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Retrieve service instances
    asr = services["asr"]
    llm = services["llm"]
    tts = services["tts"]
    memory = services["memory"]
    
    # Session state
    chat_history = [] # Short-term memory for this session
    
    try:
        # Define the pipeline generator
        # 由于websocket必须异步处理，因此此处返回异步生成器
        async def receive_audio_stream():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    yield data
            except WebSocketDisconnect:
                pass
            except Exception as e:
                print(f"Error receiving audio: {e}")

        # 1. ASR Pipeline: Audio Bytes -> Text
        # Note: In a real bidirectional stream, we need to handle "User finished speaking" events.
        # Here we assume the ASR service yields text segments.
        # For simplicity in this structure, we'll assume one turn per connection or a continuous loop handled by ASR.
        
        # We need a way to detect when a full user query is ready. 
        # For now, let's assume ASR yields a complete sentence string when silence is detected.
        
        asr_stream = asr.transcribe_stream(receive_audio_stream())
        
        async for user_text in asr_stream:
            print(f"User said: {user_text}")
            
            # 2. RAG Retrieval
            memories = await memory.query_memory(user_text)
            
            # 3. LLM Generation
            prompt = llm.build_prompt(user_text, memories, chat_history)
            
            # Store user msg in history
            chat_history.append({"role": "user", "content": user_text})
            
            llm_text_stream = llm.generate_stream(prompt, chat_history)
            
            # 4. Emotion Parsing
            emotion_stream = EmotionParser.parse(llm_text_stream)
            
            # 5. TTS Synthesis & Streaming back
            # We need to capture the full response for history while streaming audio
            full_response = ""
            
            # We duplicate the stream: one for TTS, one for history/logging? 
            # Or just accumulate inside the synth loop if possible, but TTS consumes (emotion, text).
            
            # Let's wrap the emotion stream to accumulate text
            async def observing_emotion_stream(stream):
                nonlocal full_response
                async for emotion, text in stream:
                    full_response += text
                    yield emotion, text
            
            audio_stream = tts.synthesize_stream(observing_emotion_stream(emotion_stream))
            
            async for audio_chunk in audio_stream:
                await websocket.send_bytes(audio_chunk)
            
            # Update history with full response
            chat_history.append({"role": "assistant", "content": full_response})
            
            # Optionally save interaction to long-term memory asynchronously
            # Use create_task to run in background without blocking the loop
            asyncio.create_task(memory.add_memory(
                text=f"User: {user_text}Assistant: {full_response}",
                metadata={"timestamp": "now"}
            ))
            
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.close()

if __name__ == "__main__":
    # 第一个参数是web模块实例，“模块路径：模块实例名”
    uvicorn.run("server.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
