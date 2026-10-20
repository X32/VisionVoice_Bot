# learnPartner - 智能视觉交互机器人

一个集成了**语音交互**、**视觉感知**和**大语言模型**的可交互AI机器人系统。支持唤醒词、自然语音对话、智能打断（barge-in）、流式响应、摄像头实时视觉分析、function calling等功能。全链路本地推理，可离线运行。

## 🌟 主要特性

- **🎤 完整语音对话链路**

  - 唤醒词状态机（IDLE/ACTIVE，本地KWS零依赖训练换词）
  - 自动语音活动检测（VAD）断句，无需按键触发
  - 高精度中文语音识别（SenseVoice ASR）
  - 多轮对话上下文记忆 + function calling 工具调用
  - 自然流畅的语音合成播报（Matcha TTS），启动时模型预热
- **✋ 智能打断（barge-in，R2/R2.1）**

  - WebRTC AEC3 回声消除：播报期间持续收音并区分"回声"与"真人语音"
  - **语义化判定**："嗯/对/好的"等听话反馈不打断；说停止词（不要说了/闭嘴/别说了）或唤醒词立即停播；中途说新问题（长语音>1.2s）自动停播转入收听
  - 流式分块播放器（ChunkedPlayer），停播实测毫秒级
  - 三层 ALSA 卡死防御（读自愈/异步abort/有界返回），打断不会冻结系统
- **🔇 回声门控**（与打断并存）

  - TTS播报前 mute、播完后带尾巴静音窗口 + VAD 重置
  - 播报前置牺牲静音对抗声卡省电唤醒吃首字
- **⚡ 流式响应（R3）**

  - LLM 流式输出（ollama/llama.cpp/OpenAI 三端点统一 `chat_stream`）→ 按句切分 → 逐句 TTS → 边合成边播放
  - 30 字回答首次出声实测 **~1.0s**（首token 0.13s + 首句合成 0.4s + 0.5s牺牲静音），预算 2s
  - 句间零停顿（预合成下一句 + 段落队列背压）；打断即取消上游 LLM 生成（连接级取消）
  - `--no-stream` 可回退全量合成路径
- **⏱️ 倒计时提醒**（function calling）

  - "倒计时30分钟"即生效：独立线程计时，对话不被阻塞
  - 到点前 10 秒逐秒读秒（对齐墙钟整秒），到点后播"xx分钟时间已到"三遍
  - 播报走回声门控+打断链，可随时说停止词取消；重复设定=新替旧
- **👁️ 视觉感知能力**

  - 摄像头实时画面采集与分析（`camera_vlm_monitor.py`）
  - 视觉大模型（Qwen2.5-VL via llama.cpp）场景理解
  - 定时监控和状态识别（摸鱼检测等）
  - **摸鱼判定服务化**：HTTP+WebSocket 接口，其他程序可查询状态/订阅状态翻转事件/下发采集命令（`vlm_monitor_client.py` 提供对话侧客户端）
- **🧠 多后端大模型支持**

  - 统一 LLM 接口：llama.cpp / Ollama / OpenAI兼容API
  - 启动时后台预热（消除 ollama 首请求的模型加载延迟）
- **🔧 工程健壮性**

  - 线程安全的队列+worker模型（单消费者串行播报，播报会话互斥锁防倒计时/主对话互踩）
  - SIGINT 看门狗硬退出（Ctrl+C 必死）
  - 麦克风读停滞看门狗自愈（ALSA 间歇性问题运行时恢复）

## 📋 系统要求

### 硬件要求

- **麦克风 + 扬声器**（同一声卡即可，AEC 会处理回声）
- **摄像头** (可选) - 用于视觉功能
- **CPU** - 支持AVX2指令集（推荐4核心以上）
- **内存** - 建议8GB以上（本地7B模型约4GB）

### 软件依赖

```bash
# Python 3.10+，系统依赖
sudo apt install python3-opencv ffmpeg

# Python依赖
pip install sherpa-onnx sounddevice opencv-python numpy
pip install pywebrtc-audio        # WebRTC AEC（打断功能）
pip install pypinyin sentencepiece  # 唤醒词/停止词转拼音音素
pip install aiohttp               # 摸鱼判定服务/客户端（HTTP+WS）
```

### 模型文件

```
models/
├── tts/matcha/                    # TTS（matcha两段式）
├── asr/                           # ASR+VAD（sense-voice + silero_vad.onnx）
└── kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/     # 唤醒词/停止词KWS
```

KWS模型下载：

```bash
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2
```

### 外部服务

需要运行本地大模型服务，选择其一：

- **Ollama** (默认): `ollama serve` + `ollama pull qwen2.5:7b`
- **llama.cpp**: `llama-server --model qwen2.5-vl-3b.gguf`（视觉分析走此服务）

## 🚀 快速开始

### 1. 智能语音对话（主程序）

```bash
# 默认：唤醒词"你好小迪"，Ollama后端
python3 src/chat/voice_chat.py

# 自定义唤醒词
python3 src/chat/voice_chat.py --wake-word 你好伙伴

# 远程llama.cpp后端
python3 src/chat/voice_chat.py --backend llama --url http://192.168.4.52:8080

# 关闭唤醒（开放麦克风，任何语音直接触发）
python3 src/chat/voice_chat.py --no-wake

# 调试模式（时间戳/门控/打断状态全可见）
python3 src/chat/voice_chat.py --vad-debug
```

**交互方式**：

1. 说唤醒词（默认"你好小迪"）→ 听到"哎，我在"
2. 直接说话提问 → 语音回答（支持"今天天气怎么样"、"倒计时30分钟"等工具指令）
3. 播报中：
   - 说"嗯/对/好的" → 反馈，**不会**打断
   - 说"不要说了/闭嘴/别说了" → 立即停播
   - 直接开始问新问题 → ~1.3s 后停播并开始听
4. 说"**进入待机模式**" → 播确认音后立即休眠（不经过大模型），需重新唤醒
5. 30秒无语音自动休眠，再说唤醒词唤醒
6. `Ctrl+C` 退出

**注意**：停止词请用三音节以上（"停"等单词KWS检不出）；唤醒词需连续不停顿地说。

### 2. 单独联调各模块

```bash
# 唤醒+ASR（不启LLM/TTS）
python3 src/voice_input/vad_asr.py --wake-word 你好小迪 --debug

# 唤醒词检测离线回放测试
python3 src/voice_input/kws.py --keyword 你好小迪 --wav 某段录音.wav

# AEC打断离线验证 / 打断语义五场景验证
python3 src/voice_input/test_aec.py --pipeline
python3 src/voice_input/test_barge.py

# 流式管线离线验证（句切分/延迟/打断演练）
python3 src/chat/test_stream.py

# TTS / 播放器自测
python3 src/tts/test_tts.py --text "你好"
python3 src/tts/player.py
```

### 3. 视觉监控分析（摸鱼判定服务）

```bash
python3 src/camera_vlm_monitor.py                # 服务模式（默认 :8300，HTTP+WS）
python3 src/camera_vlm_monitor.py --port 8301    # 换端口
python3 src/camera_vlm_monitor.py --interval 10  # 采集间隔
python3 src/camera_vlm_monitor.py --no-serve     # 纯CLI循环（原行为，无服务）
python3 src/camera_vlm_monitor.py --once         # 单次测试
python3 src/camera_test.py --list                # 摄像头列表
```

服务配置在 `src/monitor_config.json`（或 `--config` 指定）：采集间隔、VLM 服务地址、`server.host/port`。

服务接口：

| 接口 | 说明 |
|---|---|
| `GET /api/status` | 当前状态快照：状态/持续时长/今日统计/最近记录/是否暂停 |
| `GET /api/history?limit=20` | 最近N条分析记录（新→旧） |
| `GET /healthz` | 存活探针 |
| `GET /ws` | WebSocket 双向通道 |

WS 协议——服务端推：`hello`(连接即推快照) / `analysis`(每轮) / `status_change`(工作↔摸鱼↔离开 翻转) / `error` / `pong`；客户端发：`capture_now` / `set_interval{seconds}` / `pause` / `resume` / `ping`。

对话程序侧客户端（`src/vlm_monitor_client.py`，自动重连）：

```python
from vlm_monitor_client import MonitorClient

def on_event(ev):
    if ev["type"] == "status_change" and ev["to"] == "摸鱼":
        ...   # 摸鱼提醒（R4接入对话管线）

c = MonitorClient(on_event=on_event)   # ws://127.0.0.1:8300/ws
c.start()
c.get_status()        # REST快照
c.send_command("capture_now")
```

## 📁 项目结构

```
learnPartner/
├── src/
│   ├── chat/
│   │   ├── voice_chat.py          # 语音对话主程序（状态机编排/工具调用/流式管线/
│   │   │                          #   打断标记/倒计时/待机口令）
│   │   └── test_stream.py         # 流式管线离线验证（切分/延迟/打断/真实LLM探针）
│   ├── voice_input/
│   │   ├── vad_asr.py             # ASR核心：读流自愈mic_loop/唤醒状态机/
│   │   │                          #   回声门控/_barge_check打断判定
│   │   ├── kws.py                 # 唤醒词/停止词检测器（WakeWordDetector，
│   │   │                          #   中文→拼音音素，换词无需训练）
│   │   ├── test_aec.py            # AEC回声消除离线验证（go/no-go工具）
│   │   ├── test_barge.py          # 打断语义五场景离线验证
│   │   ├── test_vad_asr.py        # VAD断句+ASR联调
│   │   └── test_asr.py            # ASR单独测试
│   ├── llm/
│   │   ├── unified_llm.py         # 统一LLM接口（三后端/流式chat_stream/工具调用/预热）
│   │   └── demo_llm.py            # LLM联调demo
│   ├── tts/
│   │   ├── test_tts.py            # TTS合成类（matcha/melo）
│   │   ├── player.py              # ChunkedPlayer流式播放器（play/play_segments/
│   │   │                          #   可打断/far-end参考回调/有界返回）
│   │   └── experiments.py         # TTS实验脚本
│   ├── camera_vlm_monitor.py      # 定时抓拍+VLM分析（摸鱼判定，HTTP+WS服务）
│   ├── vlm_monitor_client.py      # 上述服务的客户端（对话程序用，自动重连）
│   ├── monitor_config.json        # 监控服务配置（间隔/VLM地址/服务端口）
│   ├── camera_test.py             # 摄像头测试
│   └── detect_camera.py / test_vlm.py / run_vlm_test.sh   # VLM/摄像头联调脚本
├── models/                         # 模型（见"系统要求"：asr/kws/tts）
├── DOC/                            # 需求/技术方案/实施报告/架构图（见文末文档索引）
└── README.md
```

### 线程与数据流架构

```
mic_loop线程（唯一麦克风流，读流在_reader线程+2s自愈）
  ├─ IDLE: KWS唤醒词检测 ──命中──→ ACTIVE + 播"哎，我在"
  ├─ ACTIVE:
  │   ├─ 播报中(mute窗口): _barge_check
  │   │     far参考+mic → AEC3 → 停止词KWS ──命中──→ on_barge_in
  │   │                              ↘ 长语音>1.2s ──→ on_barge_in
  │   │     （"嗯/对"等backchannel两通道都不触发，播报继续）
  │   └─ 空闲: silero VAD断句 → SenseVoice ASR → text_q
  └─ 30s无语音 → IDLE

worker线程（单消费者，串行播报；持_gate_lock播报会话）
  text_q → LLM流式(+function calling) → 按句切分 → 逐句TTS合成线程
    → 段落队列(背压) → ChunkedPlayer边合成边播
    （每块回调far-end参考给AEC；被打断→历史记"（被用户打断）"；
     播完→unmute恢复收音）

countdown线程（start_countdown工具，同样持_gate_lock）
  独立计时(不阻塞对话) → 前10s墙钟对齐读秒 → 到点提醒×3 → 可打断/可取消

monitor client线程（可选，R4接入点）
  WS收 status_change/analysis 事件 → on_event回调（断线自动重连）
```

## 🔧 配置说明

### 常用参数

| 参数                        | 默认                 | 说明                               |
| --------------------------- | -------------------- | ---------------------------------- |
| `--wake-word/-w`          | 你好小迪             | 唤醒词（可多次指定多个）           |
| `--no-wake`               | -                    | 关闭唤醒，开放麦克风               |
| `--wake-threshold`        | 0.25                 | 唤醒灵敏度（低=灵敏，误唤醒↑）    |
| `--active-timeout`        | 30                   | 唤醒后无语音休眠秒数               |
| `--standby-phrase`        | 进入待机模式         | 待机口令（可多次指定；识别文本包含即休眠） |
| `--stop-words`            | 不要说了/闭嘴/别说了 | 停止词（须三音节以上）             |
| `--barge-long-speech`     | 1.2                  | 连续语音视为新问题的秒数           |
| `--barge-threshold`       | 0.7                  | 打断长语音通道VAD阈值              |
| `--no-barge-in`           | -                    | 关闭打断（回到纯门控）             |
| `--mute-tail`             | 0.8                  | 播完后静音窗口秒数（混响强则加大） |
| `--no-stream`             | -                    | 关闭流式，回退全量合成后播放       |
| `--backend/--url/--model` | ollama               | LLM后端                            |

### Function Calling工具

当前工具：`get_local_time`（时间查询）、`get_weather`（天气查询，Open-Meteo 免Key，支持中文城市名，含今明预报）、`start_countdown`（倒计时提醒）、`turn_on_air_conditioner`（占位）。
新增三步：写函数 → 加 `TOOL_SCHEMAS` → 注册 `TOOL_FUNCS`。
天气工具依赖外网（open-meteo.com，实测延迟 <1s）；断网时返回错误 JSON，模型会口语告知查询失败。
倒计时：独立线程计时播报，对话不受阻塞；到点前 10 秒逐秒读秒（对齐墙钟整秒），到点后"xx分钟时间已到"提醒三遍；播报走回声门控+打断链（说停止词/唤醒词/插话即停）；重复设定=新替旧。

### 声学环境调参

```bash
--mute-tail 1.2          # 混响强的房间加大静音尾巴
--wake-threshold 0.2     # 唤不醒时降低（更灵敏）
--barge-threshold 0.8    # 误打断多时提高
```

## 🔍 调试与常见问题

```bash
python3 src/chat/voice_chat.py --vad-debug   # 全链路时间戳与状态
```

| 问题                             | 处理                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| 唤不醒                           | `--wake-threshold 0.2`；确认唤醒词连续不停顿地说                                                                |
| 环境音误唤醒                     | `--wake-threshold 0.3`                                                                                          |
| 播报首字丢失                     | 声卡省电所致，代码已垫0.5s牺牲静音；可根治：`echo 0 \| sudo tee /sys/module/snd_hda_intel/parameters/power_save` |
| 说"停"无法打断                   | 预期行为（单词KWS检不出），请说"不要说了"                                                                         |
| 打断后卡顿≤2s并见"⚠️停滞"告警 | ALSA间歇问题，已自愈，属最坏正常路径                                                                              |
| 识别不准                         | `--vad-threshold 0.6`（更严格）/ `--vad-silence 0.3`（更快断句）                                              |
| LLM首答慢                        | 已内置启动预热；检查 ollama 服务与 `OLLAMA_KEEP_ALIVE`                                                          |

## 📊 性能参考（本机实测）

| 环节                  | 耗时          | 备注               |
| --------------------- | ------------- | ------------------ |
| 唤醒响应              | ~0.2s         | KWS流式+预合成应答 |
| 打断停播（停止词）    | 词长+~0.15s   | KWS短语说完即停    |
| 打断停播（新问题）    | 开口后~1.3s   | 长语音通道         |
| 首次出声（30字回答）  | ~1.0s         | R3流式：首token+首句合成+primer |
| 句间停顿              | 0ms           | 预合成下一句       |
| 回声消除              | 残留-25~-37dB | 误触发0            |
| ASR识别               | ~100ms        | SenseVoice int8    |
| LLM（本地qwen2.5:7b） | ~2s           | 预热后首请求0.25s  |

## 🗺️ 路线图

- [X] R1 唤醒词+会话状态机（sherpa-onnx KWS）
- [X] R2 打断barge-in（WebRTC AEC3 + 流式播放器）
- [X] R2.1 打断语义化（backchannel不打断/停止词/长语音）
- [X] LLM启动预热 / TTS首字修复 / ALSA卡死三层防御
- [X] R3 流式响应（LLM流式+按句TTS+边合成边播，首次出声~1.0s）
- [X] 工具扩展：天气查询（Open-Meteo）/ 倒计时提醒 / 摸鱼判定服务化（HTTP+WS）
- [ ] R4 视觉交互（VLM工具接入对话管线 + 摸鱼监督主线；接入点已就绪：`vlm_monitor_client` 事件回调）
- [ ] R5 真实设备控制

## 📖 术语表

| 缩写 | 全称 | 项目里对应 |
|------|------|------------|
| VAD | Voice Activity Detection（语音活动检测） | silero VAD，区分"有人说话/静音噪声"，不识别内容 |
| ASR | Automatic Speech Recognition | SenseVoice，语音→文字 |
| TTS | Text To Speech | matcha，文字→语音 |
| KWS | Keyword Spotting | sherpa-onnx 唤醒词/停止词检测 |
| AEC | Acoustic Echo Cancellation | pywebrtc-audio 回声消除 |
| NS | Noise Suppression | WebRTC 管线里的降噪级 |
| LLM | Large Language Model | qwen2.5（ollama） |
| VLM | Vision Language Model | Qwen2.5-VL（视觉） |
| RTF | Real-Time Factor | TTS指标：合成耗时/音频时长，<1即快于实时 |
| barge-in | （术语）打断 | 播报中识别用户插话并停播 |

**为什么有两个 VAD 实例？** 主 VAD `min_speech_duration=0.25s` 是为断句质量设计的（防短噪声误弹），直接拿来做打断检测会慢 0.19s；打断通道 `_barge_vad` 单独建了低阈值实例（0.06s）——同一参数服务两个冲突目标时拆实例（踩坑记录见 `DOC/报告-R2打断功能实施过程.md`）。

## 📜 技术架构细节

### 三层ALSA卡死防御（实测踩坑总结）

1. **读侧**：`_reader`线程+队列解耦，mic_loop 2s超时，停滞即重建InputStream
2. **停止**：`player.stop()` 只置事件，`abort()` 在daemon线程执行
3. **写侧**：`play()` 有界返回（写完/stop/超时=音频时长+2s三态）

原则：任何线程不得无限期阻塞在PortAudio/ALSA的C调用上；unmute/收音恢复等关键路径必须有超时逃生门。

### 回声处理双机制

- **回声门控**（不打断时）：mute→播→unmute(tail)+VAD重置，回声从源头丢弃
- **AEC**（打断检测）：播报期间far-end参考消回声，clean只用于打断判定、绝不进主VAD/ASR

### 流式管线关键设计（R3）

- **三线程流水线**：worker（LLM流式+攒句）→ 合成线程（逐句TTS）→ 播放writer线程（边取边写），两个有界队列背压
- **结束信号用 Event 不用哨兵**：`put(None, timeout)` 满队会丢哨兵导致消费方永远等不到结束——改用 producer_done/synth_done/play_done 三个 Event
- **首段就绪才开播**：播放线程等首段事件，避免下游满队逐级上游阻塞成环；开播后边合成边播，句间零停顿

### 关键时序

- 唤醒后无语音30s休眠（`last_voice_time`在检测到语音/断句弹出时刷新，机器人回答时长不计入）
- 被打断的回复记入历史并加"（被用户打断）"标记（模型能答"刚才说到哪"）
- 播报会话互斥：主对话/倒计时播报共享 `_gate_lock`，整段 mute→set_playing→play→unmute 串行执行，防止互踩门控状态

## 📚 文档索引（DOC/）

| 文件 | 内容 |
|------|------|
| `需求文档-语音助手交互升级.md` | R1~R5 需求、验收标准与各阶段勾选记录 |
| `架构图-系统架构与脚本清单.md` | 系统分层架构图 + 全部脚本说明 |
| `技术调研-R1唤醒词选型.md` / `报告-R1唤醒词功能.md` | KWS 选型与 R1 实施报告 |
| `技术方案-R2打断barge-in-AEC.md` / `报告-R2打断功能实施过程.md` / `报告-R2.1打断语义化.md` | 打断功能全链路（方案/实施/语义化） |
| `报告-打断卡死问题排查.md` | 打断后 ALSA 卡死的排查与三层防御 |
| `技术方案-R3流式响应.md` / `报告-R3流式响应.md` | 流式管线设计、死锁排查与实测数据 |
| `pro.md` | 项目最初的需求备忘（摸鱼检测原始想法） |

## 📄 许可证

MIT License

## 🙏 致谢

- [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - ASR/TTS/KWS
- [pywebrtc-audio](https://github.com/strands-labs/pywebrtc-audio) - WebRTC AEC3
- [llama.cpp](https://github.com/ggerganov/llama.cpp) / [Ollama](https://ollama.com) - 大模型推理
- [OpenCV](https://opencv.org) - 计算机视觉
- [sounddevice](https://python-sounddevice.readthedocs.io/) - 音频I/O

---

**享受你的智能机器人之旅！** 🤖✨
