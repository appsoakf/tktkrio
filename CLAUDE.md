# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此仓库中工作时提供指导。

## 项目概览

AI Companion Backend - 使用 FastAPI + WebSocket 流式传输的实时情感感知语音助手。系统执行 ASR → RAG → LLM → TTS 流水线，具备情感控制的语音合成功能，专为 RTX 4090 (24GB VRAM) 设计。

## 开发命令

### 环境设置
```bash
# 创建并激活虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r server/requirements.txt
```

### 运行服务器
```bash
# 开发模式（启用自动重载）
cd server
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload

# 生产模式
python server/main.py
```

### 测试
```bash
# 使用 pytest 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_llm.py -v

# 运行特定测试类
pytest tests/test_llm.py::TestQwenLLMService -v

# 运行并生成覆盖率报告
pytest --cov=server/core --cov-report=html
```

### 模型下载
首次运行前必须下载模型。详细说明请参阅 `server/models/README.md`。

```python
# 快速下载脚本（从项目根目录运行）
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_download

snapshot_download('iic/SenseVoiceSmall', cache_dir='./server/models')
snapshot_download('iic/speech_fsmn_vad_jc_84000-20k-pytorch', cache_dir='./server/models')
hf_download('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', cache_dir='./server/models')
hf_download('BAAI/bge-m3', cache_dir='./server/models')
```

## 架构设计

### 核心流水线（7个步骤）

整个系统是一个**流式异步流水线**。数据流经以下阶段：

```
客户端音频 → ASR → RAG记忆 → LLM → 情感解析 → TTS → 客户端音频
```

**实现位置：** `server/main.py:websocket_audio_endpoint()`

1. **音频接收** - WebSocket 接收 PCM 音频块（16kHz, 16-bit, mono）
2. **ASR 转录** - FunASR + FSMN-VAD 检测语音并转录
3. **RAG 检索** - ChromaDB 通过 BGE-M3 嵌入向量查询语义记忆
4. **LLM 生成** - Qwen 生成带有情感标签的回复，如 `[开心] 文本`
5. **情感解析** - 从流式 LLM 输出中提取情感标签
6. **TTS 合成** - GPT-SoVITS 使用特定情感参考音频进行合成
7. **记忆存储** - 后台任务将对话保存到 ChromaDB

### 关键架构模式

#### 1. 基于接口的设计
所有服务都实现抽象接口以实现可替换性：
- `ASRInterface` → `FunASRService`
- `LLMInterface` → `QwenLLMService`
- `TTSInterface` → `GPTSoVITSService`
- `MemoryInterface` → `ChromaMemoryService`

添加新实现时（例如 WhisperASR），实现接口并在 `main.py:startup_event()` 中更新服务初始化。

#### 2. 异步生成器用于流式传输
每个阶段都使用 `AsyncGenerator` 以实现低延迟的逐 token 流式传输：

```python
# 通用模式
async def process_stream(input_stream: AsyncGenerator) -> AsyncGenerator:
    async for chunk in input_stream:
        processed = await process(chunk)
        yield processed
```

**关键：** 永远不要在内存中累积完整响应 - 始终流式传输。

#### 3. ThreadPoolExecutor 用于 GPU 推理
GPU 模型（ASR、LLM、TTS）使用同步 API。为避免阻塞 asyncio 事件循环：

```python
# 在 asr.py, tts.py 中的模式
self.executor = ThreadPoolExecutor(max_workers=4)

async def async_inference(data):
    return await asyncio.to_thread(self._sync_inference, data)
```

#### 4. CUDA OOM 恢复
LLM 服务（`llm.py`）在 14B 模型 OOM 时自动回退到 7B 模型：

```python
try:
    model = load_model(settings.LLM_MODEL_PATH)
except torch.cuda.OutOfMemoryError:
    logger.warning("OOM，回退到 7B 模型")
    model = load_model(settings.LLM_FALLBACK_MODEL)
```

### 配置系统

**所有配置：** `server/core/config.py`

使用 Pydantic Settings，支持 `.env` 文件。关键配置：

- **模型路径：** 相对于 `MODELS_DIR` 或绝对路径
- **VRAM 管理：** `MAX_CONTEXT_TOKENS`, `LLM_MAX_NEW_TOKENS`
- **VAD 调优：** `VAD_SILENCE_THRESHOLD`（默认 800ms）, `VAD_MAX_SEGMENT_SECONDS`
- **情感系统：** `ALLOWED_EMOTIONS`, `DEFAULT_EMOTION`

**重要：** 模型路径通过 `os.path.join(MODELS_DIR, LLM_MODEL_PATH)` 构建 - 确保下载的模型与这些路径匹配。

### 情感系统

LLM 输出带有情感标签的文本：`[开心] 你好！` → emotion="开心", text="你好！"

**情感解析器：** `server/core/llm.py:EmotionParser`
- 有状态的解析器，缓冲流式 token 以检测 `[emotion]` 模式
- 如果未检测到标签，则回退到 `DEFAULT_EMOTION`（"平静"）
- 针对 `ALLOWED_EMOTIONS` 列表进行验证

**TTS 集成：** `server/core/tts.py`
- 将情感映射到 `server/static/` 中的参考音频文件
- 示例："开心" → `server/static/happy.wav`
- GPT-SoVITS 使用参考音频控制韵律

### 记忆与 RAG

**ChromaDB 设置：** `server/core/memory.py`
- 持久化存储位于 `server/chroma_db/`
- 使用 BGE-M3 嵌入向量进行多语言（中文+英文）语义搜索
- 每条记忆包含元数据：`timestamp`, `session_id`

**RAG 流程：**
1. 用户查询 → `query_memory(text, n=3)` → 返回前 3 条相似记忆
2. 记忆通过 `build_prompt()` 注入到 LLM 提示词中
3. 新对话通过后台 `asyncio.create_task(add_memory())` 保存

**提示词结构：** `llm.py:build_prompt()`
```
SYSTEM_PROMPT_TEMPLATE
---
相关记忆：
{retrieved_memories}
---
对话历史：
{truncated_history}
---
用户: {user_input}
助手:
```

### 会话管理

**短期记忆：** WebSocket 处理器中的 `chat_history` 列表
- 每个连接独立存储在 `main.py:websocket_audio_endpoint()` 中
- 截断为 `MAX_HISTORY_ROUNDS * 2` 条消息（默认：20 条消息）
- 结构：`[{"role": "user/assistant", "content": "text"}, ...]`

**长期记忆：** ChromaDB 持久化存储
- 服务器重启后仍然保留
- 通过语义相似度查询用于 RAG 上下文

## 常见开发任务

### 添加新模型

1. **在 `config.py` 中定义路径**：
   ```python
   NEW_MODEL_PATH: str = "org/model-name"
   ```

2. **更新服务初始化**（例如 `llm.py:__init__`）：
   ```python
   model_path = os.path.join(settings.MODELS_DIR, settings.NEW_MODEL_PATH)
   self.model = load_model(model_path, device=settings.DEVICE)
   ```

3. **添加到下载文档** `server/models/README.md` 中

### 修改流水线

所有流水线阶段都在 `main.py:websocket_audio_endpoint()` 中。添加新阶段：

1. 将其创建为异步生成器
2. 将其链接到现有阶段之间
3. 确保保持流式传输（增量 yield）

示例 - 添加内容过滤器：
```python
async def content_filter(text_stream):
    async for text in text_stream:
        if is_safe(text):
            yield text

# 插入到流水线中
llm_stream = llm.generate_stream(prompt)
filtered_stream = content_filter(llm_stream)
emotion_stream = EmotionParser.parse(filtered_stream)
```

### 在不下载模型的情况下测试

大多数服务都有**模拟模式**，在模型未加载时激活：

- **TTS：** `tts.py` 在 `models_loaded=False` 时返回静音
- **LLM：** 在测试中通过 `@patch('server.core.llm.AutoModelForCausalLM')` 模拟
- **ASR：** 如果 FSMN-VAD 不可用，则使用基于能量的 VAD 回退

对于没有 GPU 的本地测试，在 `.env` 中设置 `DEVICE="cpu"`。

### 调试 WebSocket 问题

1. **检查日志：** 所有模块都使用带有 session ID 的结构化日志
   ```
   logger.info(f"[Session {session_id}] 事件详情")
   ```

2. **测试客户端：** 使用 `wscat` 进行快速测试
   ```bash
   npm install -g wscat
   wscat -c ws://localhost:8000/ws/audio
   # 发送二进制音频数据
   ```

3. **监控 VRAM：** 在推理期间监视 GPU 内存
   ```bash
   watch -n 1 nvidia-smi
   ```

## 重要实现说明

### VAD（语音活动检测）

**`asr.py:_vad_sync()` 中的双重策略**：
1. **主要：** FSMN-VAD 模型（200ms 窗口）
2. **回退：** 基于能量的检测（阈值：500）

**片段检测：**
- 累积音频直到检测到静音（800ms 阈值）
- 在 60 秒最大值时强制刷新以防止内存问题
- 向 ASR 返回完整的话语

### 历史截断算法

**`llm.py:_truncate_history()`** 防止上下文溢出：

1. 估计 token：`len(text) * 1.3`（中英文启发式）
2. 持续删除最旧的消息，直到低于预算
3. 始终保留至少 1 次最近的交互
4. 预算：`MAX_CONTEXT_TOKENS - LLM_MAX_NEW_TOKENS`

### TTS 句子缓冲

**`tts.py:synthesize_stream()`** 缓冲直到遇到标点符号：

```python
buffer = ""
async for emotion, chunk in text_stream:
    buffer += chunk
    if any(punct in chunk for punct in ["。", "！", "？", ".", "!", "?"]):
        audio = await synthesize(buffer, emotion)
        yield audio
        buffer = ""
```

**原因：** TTS 需要完整的句子才能获得正确的韵律。逐词合成听起来像机器人。

## 文件结构重要性

```
server/
├── core/              # 所有 AI 服务（谨慎修改 - 接口契约）
│   ├── config.py      # 所有设置的单一真实来源
│   ├── asr.py         # VAD + ASR 逻辑，ThreadPoolExecutor 模式
│   ├── llm.py         # 提示词构建，历史截断，情感解析
│   ├── tts.py         # 情感到音频映射，句子缓冲
│   └── memory.py      # ChromaDB + BGE-M3 嵌入向量
├── main.py            # 流水线编排，WebSocket 处理器
├── models/            # AI 模型权重（已 gitignore，用户下载）
├── static/            # 情感参考音频（已 gitignore，用户提供）
└── chroma_db/         # 持久化向量数据库（已 gitignore）
```

**核心洞察：** `main.py` 是编排器。核心服务是可替换的实现。配置是集中化的。一切都是流式的。
