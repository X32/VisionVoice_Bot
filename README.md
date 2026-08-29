# learnPartner - 智能视觉交互机器人

一个集成了**语音交互**、**视觉感知**和**大语言模型**的可交互AI机器人系统。支持自然语音对话、摄像头实时视觉分析、function calling等功能。

## 🌟 主要特性

- **🎤 智能语音对话**
  - 自动语音活动检测（VAD），无需按键触发
  - 高精度中文语音识别（SenseVoice ASR）
  - 支持多轮对话上下文记忆
  - 自然流畅的语音合成播报（Matcha TTS）
  - 回声门控技术，防止机器人"自问自答"

- **👁️ 视觉感知能力**
  - 摄像头实时画面采集与分析
  - 视觉大模型（VLM）场景理解
  - 定时监控和状态识别
  - 可配置的采集间隔和分析参数

- **🧠 多模态大模型支持**
  - 统一的LLM接口，支持多种后端
  - llama.cpp / Ollama / OpenAI兼容API
  - Function calling工具调用能力
  - 可扩展的工具系统

- **🔧 完善的工程架构**
  - 线程安全的队列+worker模型
  - 优雅的进程退出机制（SIGINT看门狗）
  - 配置化管理和命令行参数
  - 调试模式和详细日志

## 📋 系统要求

### 硬件要求
- **麦克风** - 用于语音输入
- **扬声器** - 用于语音输出  
- **摄像头** (可选) - 用于视觉功能
- **CPU** - 支持AVX2指令集（推荐4核心以上）
- **内存** - 建议8GB以上

### 软件依赖
```bash
# Python 3.10+
# 系统依赖
sudo apt install python3-opencv ffmpeg

# Python依赖（重要库）
pip install sherpa-onnx
pip install sounddevice
pip install opencv-python
pip install numpy
```

### 外部服务
需要运行本地大模型服务，选择其一：
- **llama.cpp** (推荐): `llama-server --model qwen2.5-vl-3b.gguf`
- **Ollama**: `ollama run qwen2.5:7b`

## 🚀 快速开始

### 1. 智能语音对话

```bash
# 使用默认Ollama后端
python3 src/chat/voice_chat.py

# 使用远程llama.cpp服务
python3 src/chat/voice_chat.py --backend llama --url http://192.168.4.52:8080

# 仅文字模式，不播报语音
python3 src/chat/voice_chat.py --no-tts

# 自定义系统提示词
python3 src/chat/voice_chat.py --system "你是编程助手，回答要专业详细"
```

**使用技巧**：
- 直接说话即可，无需按键
- 说"现在几点了"测试时间查询功能
- 说"帮我打开空调"测试工具调用功能
- 按`Ctrl+C`退出

### 2. 视觉监控分析

```bash
# 使用默认配置启动监控
python3 src/camera_vlm_monitor.py

# 自定义采集间隔（10秒）
python3 src/camera_vlm_monitor.py --interval 10

# 自定义分析提示词
python3 src/camera_vlm_monitor.py --prompt "描述图片中的文字内容"

# 只分析一次（测试用）
python3 src/camera_vlm_monitor.py --once
```

### 3. 摄像头测试

```bash
# 列出可用摄像头
python3 src/camera_test.py --list

# 实时预览摄像头画面
python3 src/camera_test.py

# 捕获单张图片
python3 src/camera_test.py --capture
```

## 📁 项目结构

```
learnPartner/
├── src/
│   ├── chat/
│   │   └── voice_chat.py          # 智能语音对话主程序
│   ├── voice_input/
│   │   └── vad_asr.py             # VAD+ASR语音识别
│   ├── llm/
│   │   └── unified_llm.py         # 统一大模型接口
│   ├── tts/
│   │   └── test_tts.py            # TTS语音合成类
│   ├── camera_vlm_monitor.py      # 视觉监控分析
│   └── camera_test.py             # 摄像头测试工具
├── models/                         # 模型文件目录（需自行下载）
└── README.md
```

## 🔧 配置说明

### LLM后端配置

**使用llama.cpp（推荐用于VLM）：**
```bash
python3 src/chat/voice_chat.py \
  --backend llama \
  --url http://localhost:8080 \
  --model qwen2.5-vl-3b
```

**使用Ollama：**
```bash
# 启动Ollama服务
ollama serve

# 运行语音助手
python3 src/chat/voice_chat.py --backend ollama --url http://localhost:11434
```

**使用OpenAI兼容API：**
```bash
python3 src/chat/voice_chat.py \
  --backend openai \
  --url https://api.openai.com/v1 \
  --api-key YOUR_KEY \
  --model gpt-4
```

### Function Calling工具

当前支持的工具调用：
- `get_local_time` - 获取本地日期时间
- `turn_on_air_conditioner` - 打开空调（示例）

添加新工具需三步：
1. 在`voice_chat.py`中编写函数
2. 添加到`TOOL_SCHEMAS`定义
3. 注册到`TOOL_FUNCS`字典

### 回声门控调整

不同声学环境可能需要调整静音窗口参数：

```bash
# 默认0.8秒（适合普通房间）
python3 src/chat/voice_chat.py --mute-tail 0.8

# 强混响环境加大窗口
python3 src/chat/voice_chat.py --mute-tail 1.2
```

## 🎯 应用场景

### 1. 智能助手
- 语音控制家居设备
- 信息查询（天气、时间、新闻）
- 闲聊对话

### 2. 视觉监控
- 工作状态监控（工作/摸鱼/离开）
- 场景内容描述
- 定时画面分析

### 3. 多模态交互
- "看看现在几点了" → 摄像头获取时间显示
- "帮我读一下屏幕上的文字" → OCR+翻译
- 视觉+语音的复合任务

## 🔍 调试技巧

### 开启调试模式

```bash
# VAD调试（显示回声门控状态）
python3 src/chat/voice_chat.py --vad-debug

# 查看详细的时间戳和状态变化
# 输出包含静音窗口、VAD重置、段落弹出等详细信息
```

### 常见问题

**问题：机器人一直播报，停不下来**
- 解决：已修复SIGINT处理，直接`Ctrl+C`退出

**问题：出现自问自答死循环**  
- 解决：增加`--mute-tail`参数到1.0秒以上

**问题：语音识别不准确**
- 调整VAD阈值：`--vad-threshold 0.6`（越高越严格）
- 调整静音断句：`--vad-silence 0.3`（更快断句）

**问题：TTS播报卡顿**
- 检查音频设备：`python3 src/camera_test.py --list`
- 使用更快的模型或降低采样率

## 📊 性能参数

### 推荐配置

| 组件 | 推荐模型 | 延迟 | 备注 |
|------|----------|------|------|
| ASR | SenseVoice | ~100ms | 中文识别效果好 |
| TTS | Matcha | ~500ms | 音质自然 |
| LLM | Qwen2.5:7B | ~2s | 本地运行 |
| VLM | Qwen2.5-VL-3B | ~3s | 视觉理解 |

### 系统资源占用

- **内存**: ~4GB（LLM 7B模型）
- **CPU**: 单核心30-50%（对话期间）
- **存储**: 模型文件约10GB

## 🔐 安全与隐私

- 所有AI处理均在本地进行，数据不上云
- 摄像头采集的画面默认保存在本地`captures/`目录
- 可配置图片保留时长和数量上限
- 建议定期清理旧的图片和日志文件

## 🛠️ 扩展开发

### 添加新的语音指令

在系统提示词中添加指令描述：

```bash
python3 src/chat/voice_chat.py --system "你是机器人助手，
可以回答问题、控制设备、讲笑话等。"
```

### 集成新的传感器

参考`vad_asr.py`的接口设计，添加：
1. 新的数据采集线程
2. 队列传递机制  
3. Worker线程处理逻辑

### 自定义工具调用

在`voice_chat.py`的`TOOL_SCHEMAS`中添加：

```python
{
    "type": "function",
    "function": {
        "name": "my_custom_tool",
        "description": "工具描述...",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}
```

## 📝 待实现功能

- [ ] 英文OCR识别与翻译
- [ ] 多语言支持
- [ ] 情感识别与响应
- [ ] 更丰富的工具库（天气查询、音乐控制等）
- [ ] Web界面管理
- [ ] 移动端App支持

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

项目遵循以下开发原则：
1. 代码可读性优于性能
2. 优先考虑用户体验
3. 安全第一（特别是进程退出）
4. 完善的错误处理和日志

## 📜 技术架构细节

### 线程模型
- **主线程**: 麦克风采集 + VAD检测
- **Worker线程**: LLM处理 + TTS播报（单线程，避免并发冲突）
- **看门狗线程**: SIGINT信号处理（保证进程可被Ctrl+C杀死）

### 回声门控方案
采用ROS2验证的门控策略：
1. TTS播报前开启门控（`mute()`）
2. 播放结束后保留静音窗口（`unmute(tail)`）
3. 窗口内音频块直接丢弃，不进VAD
4. 窗口到期后重置VAD状态

### 时序关键点
- 回声段落约在T_end+1.2s弹出（设备缓冲+VAD延迟）
- 默认静音窗口0.8s，确保覆盖回声弹出时间
- VAD reset清除内部状态，防止残留影响下次识别

## 📄 许可证

MIT License

## 🙏 致谢

本项目基于以下优秀的开源项目：
- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 语音识别与合成
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 大模型推理
- [Ollama](https://ollama.com) - 本地大模型运行
- [OpenCV](https://opencv.org) - 计算机视觉
- [sounddevice](https://python-sounddevice.readthedocs.io/) - 音频I/O

---

**享受你的智能机器人之旅！** 🤖✨