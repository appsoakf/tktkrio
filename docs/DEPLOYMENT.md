# 部署指南

本指南将帮助你在 RTX 4090 服务器上部署 AI Companion 后端。

## 系统要求

### 硬件
- **GPU**: NVIDIA RTX 4090 (24GB VRAM) 或同等性能显卡
- **RAM**: 32GB+ 系统内存
- **存储**: 50GB+ 可用空间（用于模型存储）
- **网络**: 稳定的互联网连接（下载模型），低延迟内网（与客户端通信）

### 软件
- **操作系统**: Ubuntu 20.04+ / Windows Server 2019+
- **CUDA**: 11.8+ 或 12.0+
- **Python**: 3.10+
- **驱动**: NVIDIA Driver 525+

## 部署步骤

### 1. 环境准备

```bash
# 克隆项目
git clone <your-repo-url>
cd tktkrio

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux
# 或 venv\Scripts\activate  # Windows

# 升级 pip
pip install --upgrade pip
```

### 2. 安装依赖

```bash
cd server
pip install -r requirements.txt

# 验证 PyTorch CUDA 支持
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
```

### 3. 下载模型文件

#### 方法 A: 使用下载脚本

创建 `download_models.py`:

```python
import os
from modelscope import snapshot_download
from huggingface_hub import snapshot_download as hf_download

os.makedirs("./models", exist_ok=True)

print("Downloading ASR models from ModelScope...")
snapshot_download('iic/SenseVoiceSmall', cache_dir='./models')
snapshot_download('iic/speech_fsmn_vad_jc_84000-20k-pytorch', cache_dir='./models')

print("Downloading LLM models from HuggingFace...")
hf_download('Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4', cache_dir='./models')
# hf_download('Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4', cache_dir='./models')  # Fallback (optional)

print("Downloading Embedding model...")
hf_download('BAAI/bge-m3', cache_dir='./models')

print("✓ All models downloaded successfully!")
```

运行下载脚本：
```bash
python download_models.py
```

#### 方法 B: 手动下载

使用 `git-lfs` 和 `huggingface-cli`:

```bash
# 安装 git-lfs
git lfs install

# 下载 LLM
cd models
huggingface-cli download Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 --local-dir Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4

# 下载 Embedding
huggingface-cli download BAAI/bge-m3 --local-dir BAAI/bge-m3
```

### 4. 配置参考音频

将准备好的情感参考音频放入 `server/static/` 目录：

```bash
cd server/static
# 放置以下文件：
# - happy.wav
# - angry.wav
# - sad.wav
# - surprised.wav
# - coquettish.wav
# - calm.wav
```

### 5. 配置环境变量（可选）

创建 `.env` 文件：

```bash
cd server
cat > .env << EOF
DEBUG=False
HOST=0.0.0.0
PORT=8000
DEVICE=cuda
EOF
```

### 6. 测试启动

```bash
# 测试模式启动
python main.py

# 或使用 uvicorn（生产环境推荐）
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

查看启动日志：
```
============================================================
Starting AI Companion Backend
============================================================
Initializing services...
Loading ASR model: iic/SenseVoiceSmall on cuda
✓ ASR model loaded successfully
Loading VAD model: iic/speech_fsmn_vad_jc_84000-20k-pytorch on cuda
✓ VAD model loaded successfully
Loading model: Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4
✓ Loaded Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 successfully
VRAM Usage: 10.23GB / 24.00GB (42.6%)
✓ All services initialized successfully
Server ready at http://0.0.0.0:8000
WebSocket endpoint: ws://0.0.0.0:8000/ws/audio
```

### 7. 健康检查

```bash
# 测试 HTTP 接口
curl http://localhost:8000/
# 预期返回: {"status":"ok","message":"AI Companion Backend is running"}
```

## 生产环境部署

### 使用 Systemd (Linux)

创建服务文件 `/etc/systemd/system/ai-companion.service`:

```ini
[Unit]
Description=AI Companion Backend Service
After=network.target

[Service]
Type=simple
User=your-username
WorkingDirectory=/path/to/tktkrio/server
Environment="PATH=/path/to/tktkrio/venv/bin"
ExecStart=/path/to/tktkrio/venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable ai-companion
sudo systemctl start ai-companion
sudo systemctl status ai-companion
```

### 使用 Docker

创建 `Dockerfile`:

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y python3.10 python3-pip git

WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

构建并运行：
```bash
docker build -t ai-companion .
docker run --gpus all -p 8000:8000 -v ./models:/app/models ai-companion
```

## 反向代理（可选）

### Nginx 配置

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /ws/audio {
        proxy_pass http://localhost:8000/ws/audio;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400;
    }
}
```

## 性能优化

### 1. 模型量化
- 已使用 GPTQ-Int4 量化降低 VRAM 占用
- 如需更低显存，可使用 7B 模型

### 2. 批处理优化
- 当前配置为单用户设计
- 多用户场景需调整 `max_workers` 和批处理策略

### 3. 缓存优化
```python
# 启用 KV Cache
LLM_USE_CACHE = True

# 调整 ChromaDB 缓存
CHROMA_CACHE_SIZE = 1000
```

## 故障排查

### 问题 1: CUDA Out of Memory
**解决方案:**
1. 检查是否有其他进程占用 GPU
2. 降低 `LLM_MAX_NEW_TOKENS` (默认 512)
3. 使用 7B 模型作为主模型

### 问题 2: 模型加载失败
**解决方案:**
1. 检查模型路径是否正确
2. 确认模型文件完整（MD5校验）
3. 查看日志中的详细错误信息

### 问题 3: WebSocket 连接失败
**解决方案:**
1. 检查防火墙设置
2. 确认端口 8000 未被占用
3. 测试网络连通性

## 监控与日志

### 查看实时日志
```bash
# Systemd 服务
sudo journalctl -u ai-companion -f

# 直接运行
tail -f logs/ai-companion.log
```

### GPU 监控
```bash
# 安装 nvidia-smi
watch -n 1 nvidia-smi
```

## 安全建议

1. **不要暴露公网**: 仅在内网使用，或通过 VPN 访问
2. **定期备份**: 备份 ChromaDB 数据库（`server/chroma_db/`）
3. **更新依赖**: 定期检查安全更新
4. **限制访问**: 使用防火墙限制来源 IP

## 下一步

- [ ] 完成客户端开发（Live2D + 音频采集）
- [ ] 优化 TTS 实时性
- [ ] 添加用户认证
- [ ] 实现多用户支持

## 支持

如遇问题，请查看：
- `docs/` 目录中的详细设计文档
- GitHub Issues
- 项目 Wiki
