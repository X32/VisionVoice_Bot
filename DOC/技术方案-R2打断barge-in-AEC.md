# R2 打断（barge-in）技术方案 —— WebRTC AEC3

> 项目：learnPartner ｜ 关联：`DOC/需求文档-语音助手交互升级.md` R2、NFR2/3/4
> 已定决策：AEC 库用 **pywebrtc-audio**；被打断回复记**全文+打断标记**；范围**仅 R2**（ChunkedPlayer 做成 R3 可复用的通用件，但不改 LLM/TTS 流式）

## Context

现状：TTS 播报期间回声门控（`mute_until`）丢弃所有麦克风音频块，用户无法打断。直接放开音频会把自己播报的回声当指令。方案：播报期间不再丢弃，改为 **AEC 回声消除 + VAD**——用"我们正在播放的音频"作为 far-end 参考，从 mic 信号中减掉回声，剩下的语音就是真人插话 → 停播进入正常识别。

选型：`pywebrtc-audio`（strands-labs，pip 预编译 wheel，Linux x86_64/py3.10 命中）——WebRTC AEC3 内置延迟估计器（容忍 ALSA 输出缓冲延迟不确定，这是 speexdsp 方案的最大短板）、自带 `VoiceDetector`（speech_probability）、C++ 释放 GIL、Apache 2.0。锁版本安装。

## 架构

```
                              ┌─ far-end 参考（播放块, 16k）─┐
worker线程                    │                            ▼
  LLM → TTS合成(play=False) → ChunkedPlayer.play(samples, on_chunk=...)
                                │  分块写 OutputStream(16k)    AEC3.process(near=mic块, far=参考)
                                │  stop()→abort立即停               │ clean（回声已消）
                                ▼                                  ▼
mic_loop（主线程，唯一InputStream）──────────────→ 播报期间: VoiceDetector(clean)
                                                     p>阈值连续≥3帧 → on_barge_in()
未播放时: 照旧 VAD→ASR 管线（不动）                    │
                                                     ▼
                                              player.stop() + interrupted标记
                                              unmute(短tail) + vad.reset()
                                              用户后续语音走正常管线
```

不变量（NFR）：SIGINT 看门狗不动；不打断时门控行为与现在完全一致（播完仍 unmute(tail) + vad.reset()）；单消费者串行播报保留；播放路径不再用 `sd.play/sd.wait`（规避已知 C 层互锁坑），统一走 ChunkedPlayer。

## 实施步骤

### Phase A：离线 AEC 可行性验证（go/no-go 门，不动现有代码）

1. `pip install --user pywebrtc-audio`（记入需求文档依赖）
2. 新增 `src/voice_input/test_aec.py`：
   - far = matcha TTS 合成回答（22050→16k 重采样，复用 `kws.py` 的 np.interp 逻辑）
   - near = far 经模拟回声通道（延迟 100/200/400ms 扫描 × 衰减 0.3/0.6 × 简单混响 comb filter）+ 用户语音（用 KWS 模型 test_wavs 里的真人语音）
   - `EchoCanceller(16000, stream_delay_ms=...).process(near, far)` → 统计 clean 中回声残留 RMS（相对用户语音）+ `VoiceDetector` 概率
3. **通过标准**：回声残留比用户语音低 ≥20dB、无人声时 VoiceDetector 概率 <0.3、用户插话时 >0.7、延迟容差 ≥400ms。不达标则回到评审（speexdsp 或能量启发式）

### Phase B：ChunkedPlayer（流式播放器，R3 复用底座）

新增 `src/tts/player.py`：
```python
class ChunkedPlayer:
    def __init__(self, samplerate=16000, block=512): ...
    def play(self, samples_16k, on_chunk=None) -> bool   # 阻塞至播完/被打断；返回是否完整
    def stop(self)          # 线程安全：set event + stream.abort()，≤300ms
    def close(self): ...
```
- `sd.OutputStream(samplerate=16000, blocksize=512)`，play 循环逐块写、每块检查 stop event、通过 `on_chunk(block)` 回调吐 far-end 参考
- 音频统一 16k 播放（TTS 22050 样本进 play 前重采样一次）——**参考信号=实际播出信号**，消除率最高
- wake prompt（`voice_chat.on_wake`）同样改走 ChunkedPlayer

### Phase C：AEC 打断管线集成

`src/voice_input/vad_asr.py`（SmartSpeakerASR）：
- 新成员：`aec`（EchoCanceller）、`barge_vad`（VoiceDetector）、`ref_q`（deque 累积 far-end 样本）、`on_barge_in` 回调、`playing` 标志
- 新方法 `feed_reference(block)`：player 的 on_chunk 回调（worker 线程调用），append 到 ref_q
- mic_loop 改造点（现 `vad_asr.py` 回声门控 `continue` 处）：
  ```python
  if time.time() < self.mute_until:
      if self.playing and ref_q 有足够样本:
          ref = 按512对齐drain（不足补零=无参考直通，不误消）
          clean = self.aec.process(data[:,0], ref)
          if self.barge_vad.process(clean) > 0.8:   # 连续≥3帧才确认（~60ms防抖）
              self.on_barge_in()
      continue   # clean 不进主VAD——不打断时残留绝不进ASR
  ```
- 每次播报开始 `aec.reset()`（清滤波器残留）
- 休眠/唤醒切换时 `barge_vad`/`aec` reset（与现有 wake.reset 同位置）

`src/chat/voice_chat.py`：
- `__init__`：创建 ChunkedPlayer；`self.interrupted = False`
- worker 播报段改为：`result = tts.speak(text, play=False)` → 重采样16k → `complete = player.play(samples, on_chunk=asr.feed_reference)`（前后 mute()/unmute(tail) 逻辑不变）
- `on_barge_in()`：`player.stop()`；worker 侧 `play()` 返回 False → 历史记 `resp.text + "（被用户打断）"`，继续消费队列下一条（FR2.3）
- 播报起止处设 `asr.playing` 标志 + `aec.reset()`
- 新 CLI：`--no-barge-in`（关闭 AEC 路径，回退纯门控=现行为）、`--barge-threshold`（默认0.8）
- 打断后 unmute 用短 tail（0.3s，覆盖 abort 后残余回声，比正常播完的 0.8s 短——用户在等着继续说）

### Phase D：实测验证（验收）

1. **打断延迟**：播报长回答中途说"停"，日志时间戳测 从发音→停播 ≤500ms（FR2.2 300ms 停 + 检测余量）
2. **误打断**：机器人连播 10 条回答无人说话 → 零打断
3. **打断后流程**：打断 → 追问 → 正常回答；"刚才说到哪"能基于打断标记回答
4. **回归**：唤醒/对话/休眠全流程正常；Ctrl+C 硬退出正常（NFR2）；`--no-barge-in` 与旧行为一致
5. 调参记录：实测 stream_delay_ms（player 写入时刻 vs 回声出现时刻）

## 关键风险与对策

| 风险 | 对策 |
|------|------|
| ALSA 播放→回声延迟未知且漂移 | AEC3 延迟估计器 + stream_delay_ms 实测提示；Phase A 先扫延迟容差 |
| 双设备时钟漂移（长回答参考/近端失配） | 每次播报 aec.reset()；>60s 播放专项测试；ref 按样本数对齐非墙钟 |
| Pa_ReadStream 播放期间间歇停滞（已知坑） | ref_q 为样本数队列，停滞恢复后自然追齐；不足时补零直通不误消 |
| pywebrtc-audio 项目新（4 commits） | 锁版本；API 面极小（process/reset/2属性），替换成本低 |
| AEC 残留进 ASR 造成幻听指令 | 打断判定与 ASR 输入完全隔离：clean 只喂 barge VAD，从不进主 VAD/ASR |

## 验证方式（端到端）

```bash
# Phase A 离线
python3 src/voice_input/test_aec.py --delay 200 --att 0.5
# Phase D 真机
cd src/chat && python3 voice_chat.py --debug        # 唤醒后问长问题，中途说"停"
python3 voice_chat.py --no-barge-in                 # 回退对照
```
