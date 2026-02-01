

# LLM 模块实现指南

## 概述

本指南涵盖了 AI Companion 后端中 **Qwen-2.5-14B-Instruct** LLM 模块的完整实现。

- **模型**: Qwen-2.5-14B-Instruct-GPTQ-Int4 (需 10GB VRAM)
- **框架**: FastAPI + WebSocket + Streaming
- **目标硬件**: RTX 4090 (24GB VRAM)

---

## 模块架构

### 核心组件

#### 1. QwenLLMService (`server/core/llm.py:38-339`)

主 LLM 服务类，实现了 `LLMInterface` 接口。

**关键方法**:
- `__init__()` - 加载模型并管理 VRAM
- `build_prompt()` - 组装系统提示词 + 记忆 + 历史对话 + 用户输入
- `generate_stream()` - 逐 token 流式生成
- `_load_model()` - 加载主模型 Qwen-2.5-14B
- `_load_fallback_model()` - 若发生 OOM，则加载 Qwen-2.5-7B 作为备用
- `_truncate_history()` - 在 token 预算内管理聊天历史
- `_log_vram_usage()` - 监控 GPU 显存使用情况

**核心特性**:
- 自动设备映射 (`device_map="auto"`)
- CUDA OOM 恢复机制（自动切换至小模型）
- 使用线程实现非阻塞异步流式生成
- 上下文窗口内的 token 预算管理
- 从 ChromaDB 注入记忆信息

#### 2. EmotionParser (`server/core/llm.py:342-405`)

从 LLM 输出流中提取情感标签。

**关键方法**:
- `parse()` - 异步生成器，逐段产出 `(emotion, text)` 元组

**特性**:
- 检测格式为 `[emotion]text` 的情感标签
- 验证情感是否在允许列表中
- 若未检测到有效标签，则使用配置的 `DEFAULT_EMOTION`
- 线程安全且非阻塞

---

## 配置

### 配置文件: `server/core/config.py`

**关键 LLM 相关设置**:

```python
# 模型加载
LLM_MODEL_PATH = "Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4"
LLM_FALLBACK_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"

# 生成参数
LLM_MAX_NEW_TOKENS = 512    # 回复长度
LLM_TEMPERATURE = 0.7       # 创造性 vs. 一致性
LLM_TOP_P = 0.95            # Nucleus 采样
LLM_TOP_K = 50              # Top-k 采样

# 上下文管理
MAX_CONTEXT_TOKENS = 20000  # 总 token 预算
MAX_HISTORY_ROUNDS = 10     # 最大对话轮数
MAX_MEMORY_FRAGMENTS = 3    # 最多注入的记忆片段数

# 情感配置
DEFAULT_EMOTION = "平静"    # 未检测到时的默认情感
ALLOWED_EMOTIONS = [
    "开心", "生气", "悲伤", "惊讶", "撒娇", "平静",
    "Happy", "Angry", "Sad", "Surprised", "Coquettish", "Calm"
]

# 系统提示词
SYSTEM_PROMPT_TEMPLATE = """
你是一个友善、聪慧、富有同理心的AI伴侣。

重要：情感表达规则
1. 你的每个回复都必须以情感标签开头，格式为 [情感] 文本
2. 支持的情感标签：[开心], [生气], [悲伤], [惊讶], [撒娇], [平静]
3. 选择与你的回复内容相匹配的情感
...
"""

# 生成超时
GENERATION_TIMEOUT_SECONDS = 60
```

### 环境变量 (`.env`)

```ini
# 如需覆盖默认值可在此设置
LLM_MODEL_PATH=Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
LLM_MAX_NEW_TOKENS=512
LLM_TEMPERATURE=0.7
DEVICE=cuda
```

---

## 数据流

### 完整流水线

```
User Audio Input (WebSocket)
        ↓
ASR Service (Audio → Text)
        ↓
User: "你好"
        ↓
Memory Service (ChromaDB Query)
        ↓
memories = ["用户喜欢编程", ...]
        ↓
LLM Service (build_prompt)
        ↓
prompt = system + memories + history + input
        ↓
QwenLLM (generate_stream)
        ↓
yields tokens: "[开心] 你好..." (逐 token)
        ↓
EmotionParser (parse)
        ↓
yields: (emotion="开心", text=" 你好...")
        ↓
TTS Service (synthesize_stream)
        ↓
yields audio bytes with emotion-matched voice
        ↓
WebSocket (send_bytes)
        ↓
Client Audio Output
```

### Prompt 结构示例

```
[System Prompt]
你是一个友善、聪慧、富有同理心的AI伴侣。
重要：情感表达规则...

[Memory Section]
相关记忆:
- 用户喜欢编程
- 用户来自中国

[History Section]
最近的对话:
用户: 你好
助手: [开心] 你好！
用户: 你是谁？
助手: [平静] 我是你的AI伴侣。

[Current Input]
用户: 今天天气如何？
助手:
```

---

## 关键实现细节

### 1. 模型加载与 VRAM 管理

**初始化代码**:
```python
# In QwenLLMService.__init__()
self.tokenizer = AutoTokenizer.from_pretrained(
    settings.LLM_MODEL_PATH,
    trust_remote_code=True,
)
self.model = AutoModelForCausalLM.from_pretrained(
    settings.LLM_MODEL_PATH,
    device_map="auto",      # 自动张量并行
    torch_dtype=torch.float16,  # 使用 FP16 提升效率
    trust_remote_code=True,
)
```

**VRAM 预算 (RTX 4090 - 24GB)**:
- 模型权重: 9.5GB
- KV Cache + 生成: 3.5GB
- TTS & ASR 预留: 4GB
- 系统开销: 2GB
- **安全总用量**: ~19GB

### 2. 非阻塞流式生成

**线程模式**:
```python
# 创建 streamer 用于输出 token
streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True)

# 在独立线程中运行 model.generate
thread = Thread(
    target=self.model.generate,
    kwargs={"streamer": streamer, ...}
)
thread.start()

# 在事件循环中异步消费 streamer
for text_chunk in streamer:
    yield text_chunk
    await asyncio.sleep(0)  # 交出控制权
```

**为何需要线程**:
- `model.generate()` 是阻塞的 CPU 密集型操作
- FastAPI 事件循环不能被阻塞
- 线程允许阻塞操作在后台运行
- `TextIteratorStreamer` 通过内部 Queue 实现线程安全

### 3. 聊天历史截断

**Token 预算分配**:
```
MAX_CONTEXT_TOKENS = 20000
├── System Prompt: 500 tokens
├── Memory Context: 500 tokens
├── Chat History: varies
└── Generation Space: 2000 tokens (max_new_tokens * 4 安全余量)
↓
可用于 History = 20000 - 500 - 500 - 2000 = 17000 tokens
```

**截断算法**:
- 估算 token 数: `len(text) * 1.3`（中英混合）
- 从最新消息开始向前累积
- 超出预算时停止
- 始终保留至少 1 轮对话以保证连贯性

### 4. 情感控制

**系统提示词指令**:
> 重要：情感表达规则  
> 1. 你的每个回复都必须以情感标签开头，格式为 [情感] 文本  
> 2. 支持的情感标签：[开心], [生气], [悲伤], [惊讶], [撒娇], [平静]  
> 3. 选择与你的回复内容相匹配的情感  
> 4. 情感标签必须用中文方括号，中间不含空格  
> 5. 示例：[开心] 你好！很高兴见到你！

**为何有效**:
- LLM 更擅长遵循明确的格式规则
- 多重强化（列表格式 + 示例）
- 清晰分隔符（中文方括号）
- 少样本示例有助于微调行为

**降级机制**:
- 若前 20 个字符未找到标签 → 使用 `DEFAULT_EMOTION`
- 若标签无效 → 对照 `ALLOWED_EMOTIONS` 验证
- 始终有兜底情感

### 5. 错误处理

**CUDA OOM 恢复**:
```python
try:
    yield from self.generate_stream(prompt)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # 减少 token 并重试
        yield await self._generate_with_reduced_tokens(prompt)
```

**模型加载失败**:
```python
try:
    self._load_model()  # 尝试主模型
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        self._load_fallback_model()  # 加载更小模型
```

---

## 集成点

### 1. 与 ASR (语音转文本)
- **输入**: WebSocket 的音频流
- **输出**: `FunASRService.transcribe_stream()` 返回的文本
- **集成**: 直接作为 `build_prompt(user_input=...)` 的输入

### 2. 与 Memory (RAG)
- **输入**: 用户文本
- **查询**: `await memory.query_memory(user_text, n_results=3)`
- **输出**: 记忆片段列表
- **集成**: 传入 `build_prompt(..., memories=...)`

### 3. 与 EmotionParser
- **输入**: `generate_stream()` 的 token 流
- **处理**: 使用正则 `$$(.*?)$$` 提取情感标签
- **输出**: `(emotion, text)` 元组
- **集成**: 被 TTS 服务消费

### 4. 与 TTS (文本转语音)
- **输入**: EmotionParser 输出的 `(emotion, text)` 对
- **处理**: 将情感映射到参考音频
  ```python
  emotion_map = {
      "开心": "happy.wav",
      "生气": "angry.wav",
      "悲伤": "sad.wav",
      "惊讶": "surprised.wav",
      "撒娇": "coquettish.wav",
      "平静": "calm.wav",
  }
  ```
- **输出**: 通过 WebSocket 流式返回的音频字节

### 5. 与 Session 管理
- **位置**: `main.py` 中的 WebSocket 处理器
- **数据**: 每个连接维护 `chat_history = []` 列表
- **用法**:
  ```python
  # 存储用户消息
  chat_history.append({"role": "user", "content": user_text})
  
  # 构建 LLM prompt
  prompt = llm.build_prompt(user_text, memories, chat_history)
  
  # 存储助手回复
  chat_history.append({"role": "assistant", "content": full_response})
  ```

---

## 性能指标

| 指标                | 目标     | 说明          |
| ------------------- | -------- | ------------- |
| 模型加载时间        | <5s      | 仅启动时      |
| 首 token 延迟       | <200ms   | 用户体验关键  |
| 单 token 延迟       | 50-100ms | 取决于 GPU    |
| 总延迟 (100 tokens) | 5-10s    | 典型响应时间  |
| VRAM 峰值使用       | <22GB    | 预留 2GB 余量 |
| 并发连接数          | 1 (排队) | 单 GPU 限制   |
| 记忆注入延迟        | <50ms    | ChromaDB 查询 |
| 历史截断延迟        | <10ms    | 快速估算      |

---

## 测试

### 单元测试
运行基础功能测试:
```bash
pytest tests/test_llm.py -v
```
覆盖范围:
- 模型初始化
- 带记忆/历史的 prompt 构建
- 流式生成（模拟）
- 情感解析
- 历史截断
- 配置验证

### 集成测试
运行集成测试:
```bash
pytest tests/test_integration.py -v
```
覆盖范围:
- LLM + Memory 集成
- 真实输出的情感解析
- 完整 prompt 结构
- 错误恢复
- 性能特征

### 手动测试

**模型加载**:
```python
python -c "from server.core.llm import QwenLLMService; llm = QwenLLMService()"
```
预期: 日志显示模型加载状态和 VRAM 使用情况

**Prompt 构建**:
```python
prompt = llm.build_prompt(
    user_input="你好",
    memories=["用户来自中国"],
    history=[]
)
print(prompt)
```

**流式生成 (使用实际模型)**:
```python
async for chunk in llm.generate_stream(prompt):
    print(chunk, end="", flush=True)
```

**完整流水线 (含音频)**:
1. 启动服务: `python server/main.py`
2. 连接 WebSocket 客户端
3. 发送音频字节
4. 验证响应包含情感标签

---

## 故障排除

### 问题: CUDA 显存不足 (Out of Memory)

**症状**: `RuntimeError: CUDA out of memory`

**解决方案 (按顺序尝试)**:
1. 将 `LLM_MAX_NEW_TOKENS` 从 512 降至 256
2. 将 `MAX_HISTORY_ROUNDS` 从 10 降至 5
3. 将 `MAX_CONTEXT_TOKENS` 从 20000 降至 15000
4. 切换至备用模型 (Qwen-2.5-7B)
5. 设置 `DEVICE=cpu`（仅用于调试，极慢）

**诊断命令**:
```python
import torch
print(f"Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.1f}GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.1f}GB")
```

### 问题: 未检测到情感标签

**症状**: 所有回复均使用 `DEFAULT_EMOTION` 而非模型生成的情感

**根本原因**:
- 系统提示词格式不正确
- 模型未遵循指令
- 正则表达式不匹配

**解决方案**:
- 验证 `SYSTEM_PROMPT_TEMPLATE` 包含情感指令
- 检查 EmotionParser 正则: `$$(.*?)$$`
- 手动测试: `EmotionParser.parse(["[开心]", "你好"])`
- 提高模型 temperature (0.7 → 0.8)

### 问题: 首 token 延迟过高 (>500ms)

**症状**: 首个 token 出现前长时间延迟

**根本原因**:
- 模型未加载到 GPU
- VRAM 碎片化
- Prompt 过长导致分词耗时

**解决方案**:
- 检查模型设备: `self.model.device`
- 清理缓存: `torch.cuda.empty_cache()`
- 缩短 prompt（减少记忆或历史）
- 启用 KV cache: `use_cache=True`（默认已启用）

### 问题: 记忆未注入回复

**症状**: LLM 输出未引用检索到的记忆

**根本原因**:
- ChromaDB 为空或无匹配结果
- 记忆 token 预算过小
- LLM 忽略指令

**解决方案**:
- 测试记忆查询: `await memory.query_memory("test")`
- 将 `MAX_MEMORY_FRAGMENTS` 从 3 增至 5
- 减少历史对话以释放 token 空间给记忆
- 检查 memory collection 是否包含文档

---

## 高级配置

### 生成配置文件

**快速响应 (游戏/实时场景)**:
```python
LLM_MAX_NEW_TOKENS = 256
LLM_TEMPERATURE = 0.5
LLM_TOP_P = 0.9
# 预期延迟: 256 tokens 约 2-3 秒
```

**平衡模式 (默认)**:
```python
LLM_MAX_NEW_TOKENS = 512
LLM_TEMPERATURE = 0.7
LLM_TOP_P = 0.95
# 预期延迟: 512 tokens 约 5-8 秒
```

**高质量 (创意回复)**:
```python
LLM_MAX_NEW_TOKENS = 768
LLM_TEMPERATURE = 0.8
LLM_TOP_P = 0.98
# 预期延迟: 768 tokens 约 8-12 秒
```

### 自定义情感处理

**添加/修改情感**:
1. 在 `config.py` 的 `ALLOWED_EMOTIONS` 中添加:
   ```python
   ALLOWED_EMOTIONS = [ ..., "紧张", "兴奋"  # 新增情感]
   ```
2. 更新系统提示词:
   ```python
   SYSTEM_PROMPT_TEMPLATE = """...
   支持的情感标签：[开心], [生气], ..., [紧张], [兴奋]"""
   ```
3. 在 TTS 中添加参考音频:
   ```python
   self.ref_audio_map = {
       ...,
       "紧张": os.path.join(settings.STATIC_DIR, "nervous.wav"),
       "兴奋": os.path.join(settings.STATIC_DIR, "excited.wav"),
   }
   ```

---

## 部署清单

- [ ] 模型权重已下载并置于 `models/` 目录
- [ ] 参考音频文件置于 `static/` 目录 (happy.wav, sad.wav 等)
- [ ] ChromaDB 目录已创建: `server/chroma_db/`
- [ ] `.env` 文件已配置正确的模型路径
- [ ] CUDA/GPU 可用且驱动已安装
- [ ] 所有依赖已安装: `pip install -r requirements.txt`
- [ ] 测试通过: `pytest tests/`
- [ ] 首 token 延迟 <200ms 已验证
- [ ] 启动时已监控 VRAM 使用
- [ ] 日志已配置用于生产环境监控
- [ ] 已实现速率限制（如需要）
- [ ] 错误处理已测试（OOM、断连等）

---

## 参考资料

- Qwen-2.5 文档: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
- Transformers 流式生成: https://huggingface.co/docs/transformers/generation_strategies/streaming
- TextIteratorStreamer: https://huggingface.co/docs/transformers/main/en/main_classes/text_generation#transformers.TextIteratorStreamer
- ChromaDB: https://docs.trychroma.com/
- FastAPI WebSocket: https://fastapi.tiangolo.com/advanced/websockets/

---
**最后更新**: 2025年1月31日  
**版本**: 1.0  
**状态**: 生产就绪
```