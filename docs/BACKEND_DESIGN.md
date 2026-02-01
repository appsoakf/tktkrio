# AI Companion Backend Design

## 1. 系统架构总览

本项目的后端核心目标是构建一个低延迟、高智能、具备情感表达能力和长期记忆的 AI 伴侣。
利用 RTX 4090 (24GB VRAM) 的强大算力，我们将实现**全本地化部署**，确保隐私安全与极致响应速度。

### 1.1 技术栈

*   **语言**: Python 3.10+
*   **Web 框架**: FastAPI (支持 WebSocket 异步并发)
*   **AI 模型**:
    *   **LLM (大脑)**: Qwen-2.5-14B-Instruct (GPTQ-Int4 量化版) 或 Llama-3-8B-Instruct
    *   **TTS (语音合成)**: GPT-SoVITS (具备情感控制能力)
    *   **ASR (语音识别)**: FunASR (SenseVoiceSmall 或 Paraformer-Streaming)
    *   **Embedding**: text-embedding-3-small (如调用API) 或 BGE-M3 (本地)
*   **数据库**:
    *   **向量库**: ChromaDB (本地持久化，用于长期记忆)
    *   **KV 存储**: Python 内存 / Redis (用于会话级短期记忆)

---

## 2. 核心模块设计

### 2.1 ASR 服务 (听觉层)

*   **职责**: 接收客户端发来的音频流，实时转换为文本。
*   **实现方案**: 
    *   采用 **FunASR** 的流式识别管道。
    *   **输入**: WebSocket 接收 PCM 音频流 (16k/16bit/mono)。
    *   **处理**: VAD (语音活动检测) -> ASR 推理。
    *   **输出**: 实时文本流。当检测到句尾停顿时，触发“用户发言结束”信号，将完整文本推送至 Brain 模块。

### 2.2 Brain 服务 (大脑层)

*   **职责**: 理解意图，检索记忆，生成带情感标签的回复。
*   **Memory Pipeline (RAG)**:
    1.  **Query Embedding**: 将用户输入转为向量。
    2.  **Retrieve**: 在 ChromaDB 中搜索 Top-k 相关历史片段。
    3.  **Context Assembly**: 
        ```
        System Prompt: 你是[名字]，一个[性格]的AI伴侣...
        Long-term Memory: [检索到的历史片段]
        Short-term History: [最近10轮对话]
        User Input: [当前问题]
        ```
*   **Emotion Protocol**:
    *   强制 LLM 在输出内容中包含情感标记。
    *   格式示例: `[开心] 真的吗？我也这么觉得！`
    *   支持情感: `[开心]`, `[生气]`, `[悲伤]`, `[惊讶]`, `[撒娇]`, `[平静]`

### 2.3 TTS 服务 (表达层)

*   **职责**: 将文本转换为带情感的语音流。
*   **实现方案**: **GPT-SoVITS**
*   **情感控制**:
    *   预加载多组 Reference Audio (参考音频)，对应不同的情感状态。
    *   当接收到 Brain 发来的 `[开心]` 标签时，切换至 `happy.wav` 作为参考音频进行合成。
    *   **流式输出**: 生成的音频数据通过 WebSocket 实时推流回客户端，实现“首字低延迟”播放。

---

## 3. 数据流转 (Streaming Pipeline)

1.  **Client** -> `WebSocket` -> **Gateway**
2.  **Gateway** -> `Audio Stream` -> **ASR**
3.  **ASR** -> `Text` -> **Brain**
4.  **Brain** -> `Retrieval` -> **ChromaDB**
5.  **Brain** -> `Prompt` -> **LLM**
6.  **LLM** -> `Stream Text` -> **Emotion Parser**
    *   解析到 `[开心]` -> 锁定 TTS 情感 = Happy
    *   解析到文本 "你好" -> 发送给 TTS
7.  **TTS** -> `Audio Stream` -> **Client**
8.  **Client** -> 播放音频 & 驱动 Live2D 口型

---

## 4. 硬件资源预估 (RTX 4090 - 24GB)

| 模块 | 模型选择 | 显存占用 (VRAM) | 备注 |
| :--- | :--- | :--- | :--- |
| **LLM** | Qwen-2.5-14B-GPTQ-Int4 | ~10 GB | 核心推理，保留足够的 Context Window |
| **TTS** | GPT-SoVITS | ~4 GB | 包含 VITS 模型与参考音频编码 |
| **ASR** | FunASR (SenseVoice) | ~1-2 GB | 极低占用 |
| **Embed** | BGE-M3 (Optional) | ~1 GB | 若本地跑 RAG embedding |
| **System** | CUDA Overhead | ~2 GB | 系统预留 |
| **Total** | | **~18-19 GB** | **安全范围内** |

---

## 5. 项目结构

```
server/
├── core/
│   ├── config.py           # 全局配置
│   ├── llm.py              # LLM 封装 (加载模型、生成流)
│   ├── memory.py           # ChromaDB 封装 (RAG)
│   ├── asr.py              # FunASR 接口
│   └── tts.py              # GPT-SoVITS 接口
├── models/                 # 存放本地模型权重文件
├── static/                 # 存放 TTS 参考音频
├── main.py                 # FastAPI 入口 & WebSocket 路由
└── requirements.txt        # 依赖列表
client/                     # 前端代码 (待定)
```
