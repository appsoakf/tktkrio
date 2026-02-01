# AI Companion Backend

一个具备情感表达、语音交互、长期记忆的 AI 伴侣后端服务。

## 项目愿景

* 基于大语言模型的自然语言对话，上下文理解和人机聊天
* 识别用户麦克风语音输入内容，理解用户的问题，并用语音生成回复
* AI的语音回复具备情感模拟功能，比如说话时有和人一样的喜怒哀乐等情绪
* 基于向量数据库的长期记忆存储与提取
* 具备运行时上下文短期记忆
* AI拥有一个Live2D形象，可在桌面自由交流

## 系统架构

```
┌──────────────────────────────────────────────────────────────┐
│                       远程服务器 (RTX 4090)                    │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────────┐  │
│  │   ASR   │ → │   LLM   │ → │   TTS   │ → │  WebSocket  │  │
│  │ 语音识别 │   │ 大脑推理 │   │ 语音合成 │   │   流式传输   │  │
│  └─────────┘   └────┬────┘   └─────────┘   └─────────────┘  │
│                     │                                        │
│              ┌──────┴──────┐                                 │
│              │   Memory    │                                 │
│              │ ChromaDB    │                                 │
│              │ (RAG检索)   │                                 │
│              └─────────────┘                                 │
└──────────────────────────────────────────────────────────────┘
                              ↕ WebSocket
┌──────────────────────────────────────────────────────────────┐
│                       本地客户端                              │
│           Live2D 渲染 + 音频采集/播放 + 口型同步               │
└──────────────────────────────────────────────────────────────┘
```

## 技术栈

| 模块 | 技术选型 | 说明 |
|------|----------|------|
| **LLM** | Qwen-2.5-14B-GPTQ-Int4 | 中文对话，情感控制 |
| **ASR** | FunASR (SenseVoiceSmall) | 多语言语音识别 |
| **VAD** | FSMN-VAD | 语音活动检测 |
| **TTS** | GPT-SoVITS | 情感语音合成 |
| **Memory** | ChromaDB + BGE-M3 | 向量存储与RAG |
| **Framework** | FastAPI + WebSocket | 异步流式传输 |

## 项目结构

```
tktkrio/
├── server/
│   ├── core/
│   │   ├── __init__.py      # 模块导出
│   │   ├── config.py        # 配置管理
│   │   ├── asr.py           # 语音识别服务
│   │   ├── llm.py           # 大语言模型服务
│   │   ├── tts.py           # 语音合成服务
│   │   └── memory.py        # 长期记忆服务
│   ├── main.py              # FastAPI 入口
│   ├── requirements.txt     # 依赖列表
│   ├── models/              # 模型权重目录
│   └── static/              # TTS参考音频
├── docs/
│   ├── BACKEND_DESIGN.md    # 后端设计文档
│   ├── LLM_DESIGN.md        # LLM模块详细设计
│   └── ASR_DESIGN.md        # ASR模块详细设计
└── tests/                   # 测试用例
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r server/requirements.txt
```

### 2. 下载模型

参考 `server/models/README.md` 下载所需的 AI 模型：

```python
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_download

# ASR & VAD
snapshot_download('iic/SenseVoiceSmall', cache_dir='./server/models')
snapshot_download('iic/speech_fsmn_vad_jc_84000-20k-pytorch', cache_dir='./server/models')

# LLM
hf_download('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', cache_dir='./server/models')

# Embedding
hf_download('BAAI/bge-m3', cache_dir='./server/models')
```

### 3. 准备参考音频

在 `server/static/` 目录下放置6个情感参考音频文件：
- `happy.wav` (开心)
- `angry.wav` (生气)
- `sad.wav` (悲伤)
- `surprised.wav` (惊讶)
- `coquettish.wav` (撒娇)
- `calm.wav` (平静)

### 4. 启动服务

```bash
cd server
python main.py
# 或使用 uvicorn
uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. WebSocket 连接

```javascript
const ws = new WebSocket('ws://your-server:8000/ws/audio');

// 发送音频数据 (PCM 16kHz/16bit/mono)
ws.send(audioChunkBytes);

// 接收合成的语音
ws.onmessage = (event) => {
    const audioData = event.data;
    // 播放音频 & 驱动Live2D口型
};
```

## 数据流

```
用户语音 → [ASR] 文本 → [Memory] RAG检索 → [LLM] 生成带情感标签的回复
                                                    ↓
               用户 ← [WebSocket] 音频流 ← [TTS] 情感语音合成
```

**情感标签格式：**
```
[开心] 你好！很高兴见到你！
[悲伤] 我能理解你的感受...
[撒娇] 哎呀，人家也想要嘛~
```

## 硬件要求

推荐配置：**RTX 4090 (24GB VRAM)**

| 模块 | VRAM占用 |
|------|----------|
| LLM (14B) | ~10 GB |
| TTS | ~4 GB |
| ASR | ~1-2 GB |
| Embedding | ~1 GB |
| 系统预留 | ~2 GB |
| **总计** | **~18-19 GB** |

## API 接口

### HTTP 健康检查

```bash
GET /
# 返回: {"status": "ok", "message": "AI Companion Backend is running"}
```

### WebSocket 音频流

```
WebSocket /ws/audio
# 发送: PCM 音频字节流 (16kHz, 16bit, mono)
# 接收: PCM 合成语音字节流
```

## 开发说明

- 所有模块采用抽象接口设计，方便替换实现
- 异步流式处理，低延迟响应
- 自动显存管理，OOM时切换小模型
- ChromaDB持久化存储对话记忆

## License

MIT