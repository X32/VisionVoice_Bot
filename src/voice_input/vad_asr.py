#!/usr/bin/env python3
"""
实时语音识别脚本 - 使用SenseVoice VAD自动检测人声
参考ROS2 voice_input_node.py的实现方式
自动检测语音段落，识别后立即显示结果并继续下一轮

用法:
    python3 vad_asr.py                          # 使用默认参数
    python3 vad_asr.py --duration 8             # 最大录音时长8秒
    python3 vad_asr.py --silence 0.5            # 静音0.5秒后断句
    python3 vad_asr.py --threshold 0.5         # VAD阈值0.5
    python3 vad_asr.py --debug                 # 显示调试信息
"""
import argparse
import queue
import re
import sys
import time
import threading
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd
import sherpa_onnx

SAMPLE_RATE = 16000     # SenseVoice标准采样率
VAD_BLOCK = 512         # silero VAD窗长（16kHz下用512）

# 唤醒词检测器同目录（延迟导入，仅启用唤醒时加载）
sys.path.insert(0, str(Path(__file__).parent))


class SmartSpeakerASR:
    """智能音响式ASR - 使用VAD自动检测语音段落"""

    def __init__(
        self,
        sensevoice_model_dir: str = "/mnt/apfs/workplace/ROS2/models/asr/sense-voice",
        vad_model_path: str = "/mnt/apfs/workplace/ROS2/models/asr/silero_vad.onnx",
        max_duration: float = 8.0,
        silence_duration: float = 0.5,
        vad_threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        cooldown: float = 1.0,
        on_text: Optional[Callable[[str], None]] = None,
        wake_words: Optional[List[str]] = None,
        kws_model_dir: str = None,
        wake_threshold: float = 0.25,
        active_timeout: float = 30.0,
        on_wake: Optional[Callable[[str], None]] = None,
        on_sleep: Optional[Callable[[], None]] = None,
        barge_in: bool = True,
        on_barge_in: Optional[Callable[[str], None]] = None,
        barge_threshold: float = 0.7,
        stop_words: Optional[List[str]] = None,
        barge_long_speech: float = 1.2,
        barge_kws_threshold: float = 0.25,
        debug: bool = False
    ):
        """
        初始化智能音响ASR系统

        Args:
            sensevoice_model_dir: SenseVoice模型目录
            vad_model_path: VAD模型路径
            max_duration: 单段最大录音时长（秒）
            silence_duration: 静音多久后断句（秒）
            vad_threshold: VAD检测阈值（0-1）
            min_speech_duration: 最短语音时长（秒，短于此视为噪声）
            cooldown: 识别后冷却时间（秒，防尾音被切成第二段）
            on_text: 识别文本回调函数
            wake_words: 唤醒词列表（None=开放麦克风模式，无状态机）
            kws_model_dir: KWS模型目录（None用kws.py默认）
            wake_threshold: 唤醒词命中阈值（越低越灵敏）
            active_timeout: 唤醒后无有效语音多久回到待唤醒（秒）
            on_wake: 唤醒回调（mic_loop线程调用，应快速返回或只递纸条）
            on_sleep: 超时休眠回调
            barge_in: 启用打断检测（AEC回声消除+VAD，播报期间检测插话）
            on_barge_in: 检测到打断意图的回调，参数为原因（mic_loop线程调用，
                如player.stop）。原因: 'stop_word:停' / 'long_speech'
            barge_threshold: 长语音通道silero阈值（离线实测0.7误报最优）
            stop_words: 停止词表（命中即打断；唤醒词自动并入）
            barge_long_speech: 连续语音超过该秒数视为新问题打断（backchannel
                如"嗯/对"通常<0.5s，真新问题>1.5s，1.2取中间）
            barge_kws_threshold: 停止词KWS阈值（单词"停"易误报时可调高）
            debug: 是否显示调试信息
        """
        self.max_duration = max_duration
        self.silence_duration = silence_duration
        self.vad_threshold = vad_threshold
        self.min_speech_duration = min_speech_duration
        self.cooldown = cooldown
        self.on_text = on_text
        self.debug = debug
        self.in_speech = False
        self.last_recognize_time = 0.0
        self.round_num = 0

        # 唤醒词状态机：wake_words为None时保持原开放麦克风行为
        self.on_wake = on_wake
        self.on_sleep = on_sleep
        self.active_timeout = active_timeout
        self.last_voice_time = 0.0
        self.wake = None
        self.state = 'active'   # 无唤醒模式恒为active
        if wake_words:
            from kws import WakeWordDetector, DEFAULT_MODEL_DIR
            kws_kwargs = dict(
                keywords=wake_words,
                keywords_threshold=wake_threshold,
                debug=debug,
            )
            if kws_model_dir:
                kws_kwargs['model_dir'] = kws_model_dir
            elif not Path(DEFAULT_MODEL_DIR).exists():
                raise FileNotFoundError(
                    f"KWS默认模型目录不存在: {DEFAULT_MODEL_DIR}，"
                    f"请下载或用 kws_model_dir 指定")
            self.wake = WakeWordDetector(**kws_kwargs)
            self.state = 'idle'
            print(f"🔔 唤醒模式: {' / '.join(wake_words)}（阈值{wake_threshold}，"
                  f"{active_timeout:.0f}s无语音休眠）")

        # 回声抑制（参考ROS2 tts_mute设计）
        self.mute_until = 0.0
        self.vad_reset_requested = False   # 恢复收音时重置VAD（由mic_loop线程执行）
        self._in_mute_window = False       # 调试日志用：是否处于静音窗口

        # ==== 打断检测（R2 barge-in） ====
        # 播报期间麦克风音频不再直接丢弃：far-end参考(实际播出音频)经AEC
        # 从麦克风信号中消掉回声，剩余语音即用户插话→触发on_barge_in。
        # clean信号只用于打断判定，绝不进主VAD/ASR（残留不会变幻听指令）。
        self.on_barge_in = on_barge_in
        self.playing = False               # worker播报起止置位（feed_reference对齐前提）
        self._ref_buf = np.empty(0, dtype=np.float32)   # far-end参考累积
        self._ref_max = SAMPLE_RATE * 4    # 参考缓冲上限4s（mic停滞防膨胀）
        self._play_t0 = 0.0                # 本次播报开始时刻（收敛期豁免用）
        self._speech_run = 0               # 长语音通道：连续语音样本累计
        self._clean_ring = np.empty(0, dtype=np.float32)  # clean音频回补环（长语音打断用）
        self._long_speech = barge_long_speech
        self.aec_proc = None
        self.aec_kws = None
        self.barge_vad = None
        self.barge_kws = None
        if barge_in and on_barge_in:
            from pywebrtc_audio import AudioProcessor, EchoCanceller
            # 双分支AEC（R2.1实测踩坑）：
            #  - silero长语音通道用NS管线：谱底处理后能量检测更稳
            #  - KWS停止词通道用裸AEC：NS在双讲时的激进谱处理会破坏
            #    音素级识别（实测：管线clean上KWS全漏、裸AEC clean命中）
            self.aec_proc = AudioProcessor(
                sample_rate=SAMPLE_RATE, echo_cancellation=True,
                noise_suppression=True, ns_level=2, stream_delay_ms=0)
            self.aec_kws = EchoCanceller(sample_rate=SAMPLE_RATE,
                                         stream_delay_ms=0)
            barge_silero = sherpa_onnx.SileroVadModelConfig(
                model=vad_model_path,
                threshold=barge_threshold,
                min_silence_duration=0.3,
                min_speech_duration=0.06,   # 主VAD是0.25s(为ASR断句)，打断求快
                window_size=VAD_BLOCK,
            )
            self.barge_vad = sherpa_onnx.VoiceActivityDetector(
                sherpa_onnx.VadModelConfig(silero_vad=barge_silero,
                                           sample_rate=SAMPLE_RATE))
            # 停止词通道：复用R1的KWS做流式停止词检测（唤醒词自动并入）。
            # 词表实测可靠性（bare-AEC clean）：闭嘴/不要说了 高分2/2；
            # 别说了高分会诱发'对'误触发 → 逐词低分(1.0)保留（1/2召回）；
            # 单/双音节（停/停下）KWS无法可靠检出——不放词表。
            # 唤醒词KWS不中也无碍：长语音通道兜底触发
            from kws import WakeWordDetector
            words = list(dict.fromkeys(
                (stop_words or ['不要说了', '闭嘴', '别说了']) + (wake_words or [])))
            self.barge_kws = WakeWordDetector(
                keywords=words, keywords_threshold=barge_kws_threshold,
                keywords_score=2.0,
                keyword_scores={'别说了': 1.0})
            print(f"✋ 打断检测已启用（停止词:{'/'.join(stop_words or [])}"
                  f"+唤醒词，长语音>{barge_long_speech}s）")

        print(f"🎙️  初始化智能音响ASR系统")
        print(f"📍 ASR模型: {sensevoice_model_dir}")
        print(f"📍 VAD模型: {vad_model_path}")
        print(f"⚙️  参数: 采样率={SAMPLE_RATE}Hz, 最长录音={max_duration}s, 静音检测={silence_duration}s")

        # 初始化ASR
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=f'{sensevoice_model_dir}/model.int8.onnx',
            tokens=f'{sensevoice_model_dir}/tokens.txt',
            use_itn=True,
        )

        # 初始化VAD - 参考ROS2实现
        silero_vad_config = sherpa_onnx.SileroVadModelConfig(
            model=vad_model_path,
            threshold=vad_threshold,
            min_silence_duration=silence_duration,
            min_speech_duration=min_speech_duration,
            max_speech_duration=max_duration,
            window_size=VAD_BLOCK,
        )
        vad_config = sherpa_onnx.VadModelConfig(
            silero_vad=silero_vad_config,
            sample_rate=SAMPLE_RATE,
        )
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config)

        print("✅ 初始化完成")

    def set_playing(self, playing: bool):
        """播报起止置位（worker线程调用）。

        开始时彻底清态：AEC实例直接重建（构造仅0.1ms，实测reset()清不净
        语音后的滤波器/延迟估计器残留，backchannel场景后停止词会漏检——
        R2.1踩坑）；KWS换全新stream（解码状态归零，创建零成本）。
        """
        self.playing = playing
        self._ref_buf = np.empty(0, dtype=np.float32)
        self._speech_run = 0
        if playing:
            from pywebrtc_audio import AudioProcessor, EchoCanceller
            self._play_t0 = time.time()
            self._clean_ring = np.empty(0, dtype=np.float32)
            if self.aec_proc is not None:
                self.aec_proc = AudioProcessor(
                    sample_rate=SAMPLE_RATE, echo_cancellation=True,
                    noise_suppression=True, ns_level=2, stream_delay_ms=0)
            if self.aec_kws is not None:
                self.aec_kws = EchoCanceller(sample_rate=SAMPLE_RATE,
                                             stream_delay_ms=0)
            if self.barge_vad is not None:
                self.barge_vad.reset()
                while not self.barge_vad.empty():
                    self.barge_vad.pop()
            if self.barge_kws is not None:
                self.barge_kws.new_stream()

    def drain_interrupted_audio(self) -> np.ndarray:
        """取走长语音打断前积累的clean音频（worker在unmute后注入主VAD用）"""
        buf = self._clean_ring
        self._clean_ring = np.empty(0, dtype=np.float32)
        return buf

    def feed_reference(self, chunk: np.ndarray):
        """far-end参考入队（ChunkedPlayer的on_chunk回调，worker线程调用）

        每块512样本与mic_loop读取块同规格，按样本数对齐（非墙钟，
        避免双设备时钟漂移累积误差）。
        """
        self._ref_buf = np.concatenate([self._ref_buf, chunk])
        if len(self._ref_buf) > self._ref_max:
            self._ref_buf = self._ref_buf[-self._ref_max:]

    def _barge_check(self, mic_block: np.ndarray):
        """播报期间的打断意图检测（mic_loop线程调用，回声门控continue之前）

        双通道判定（R2.1语义化）：
          通道1 停止词KWS —— 检出停止词/唤醒词，短语说完即命中 → 立即停播
          通道2 长语音 —— silero连续语音累计超时 → 用户在说新问题，停播转收听
        短且无关键词（"嗯/对/好的"类backchannel反馈）两通道都不触发，
        播报继续。开头0.6s为AEC收敛期，只喂信号不判定。
        clean音频留环备份（≤4s），长语音触发后由unmute流程注入主VAD，
        ASR可拿到完整句子（R2.1 stretch项）。
        """
        if self.aec_proc is None or len(self._ref_buf) < VAD_BLOCK:
            return   # 打断未启用 / 参考未到（播报刚开始）——跳过不误判
        ref = self._ref_buf[:VAD_BLOCK]
        self._ref_buf = self._ref_buf[VAD_BLOCK:]

        clean = self.aec_proc.process(mic_block, ref)        # silero分支
        clean_kws = self.aec_kws.process(mic_block, ref)     # KWS分支（裸AEC）

        # clean环备份（回补用），超限丢最旧
        self._clean_ring = np.concatenate([self._clean_ring, clean])
        if len(self._clean_ring) > self._ref_max:
            self._clean_ring = self._clean_ring[-self._ref_max:]

        # 通道1：停止词（KWS流式，含唤醒词）。收敛期也喂（保持流状态
        # 连续，停止词跨豁免窗边界不漏检）但命中不触发
        hit = self.barge_kws.process_block(clean_kws)
        in_grace = time.time() - self._play_t0 < 0.6
        if hit and not in_grace:
            self._speech_run = 0
            self.barge_vad.reset()
            while not self.barge_vad.empty():
                self.barge_vad.pop()
            self._fire_barge_in(f'stop_word:{hit}')
            return

        # 通道2：长语音（连续语音累计，间歇<0.3s不断累计——短语间换气）
        self.barge_vad.accept_waveform(clean)
        if self.barge_vad.is_speech_detected() and not in_grace:
            self._speech_run += len(clean)
            if self._speech_run >= self._long_speech * SAMPLE_RATE:
                self._speech_run = 0
                self._fire_barge_in('long_speech')
        elif not self.barge_vad.is_speech_detected():
            self._speech_run = 0

    def _fire_barge_in(self, reason: str):
        if self.debug:
            print(f"[{time.time():.3f}] ✋ 打断意图({reason})，触发停播")
        try:
            self.on_barge_in(reason)
        except Exception as e:
            print(f"⚠️  on_barge_in回调出错: {e}")

    def mute(self):
        """开启回声抑制（TTS播报期间，麦克风音频整块丢弃，不进VAD）"""
        self.mute_until = float('inf')
        if self.debug:
            print(f"[{time.time():.3f}] 🔇 回声抑制开启")

    def unmute(self, tail: float = 0.5):
        """关闭回声抑制：tail秒后恢复收音，恢复时重置VAD

        实测（本机）：播报结束后回声段落约 +1.2s 才从VAD弹出（设备输出
        缓冲 + 输入延迟 + silero阈值平滑 + 混响），因此不能只靠"弹出时
        检查门控"——静音窗口内音频块根本不喂给VAD，到期后reset()清掉
        内部残留状态，回声从源头进不来。

        Args:
            tail: 播放结束后的总静音窗口（秒），需覆盖设备延迟+混响
        """
        self.mute_until = time.time() + tail
        self.vad_reset_requested = True   # 到期时由mic_loop线程重置VAD
        if self.debug:
            print(f"[{time.time():.3f}] 🔊 回声抑制关闭，{tail:.1f}s后恢复收音")

    def enter_standby(self):
        """语音口令主动待机：立即回到idle（区别于超时休眠的主动出口）

        在ASR识别回调线程（mic_loop）中调用。复位序列与超时休眠路径
        一致：清VAD残留段落防待机瞬间弹出旧音频，KWS复位重新监听唤醒词。
        开放麦克风模式（wake为None）不响应——待机后没有唤醒KWS，等于失聪。
        """
        if self.wake is None:
            return
        self.state = 'idle'
        self.vad.reset()
        while not self.vad.empty():
            self.vad.pop()
        self.in_speech = False
        self.wake.reset()
        if self.on_sleep:
            try:
                self.on_sleep()
            except Exception as e:
                print(f"⚠️  on_sleep回调出错: {e}")

    def recognize_and_display(self, samples: np.ndarray, t_emit: float = 0.0):
        """识别音频并显示结果（门控/冷却检查已在上游 mic_loop 完成）"""
        audio_duration = len(samples) / SAMPLE_RATE

        # ASR识别
        if not self.debug:
            print("🧠 正在识别..." + " " * 20 + "\r", end="", flush=True)
        else:
            print(f"🧠 识别音频片段: {audio_duration:.2f}s")

        t0 = time.perf_counter()
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self.recognizer.decode_stream(stream)
        dt_ms = (time.perf_counter() - t0) * 1000

        # 清理SenseVoice的控制标签
        text = re.sub(r'<\|[^|]*\|>', '', stream.result.text).strip()

        if not text:
            if self.debug:
                print("🚫 识别结果为空，跳过")
            return

        if not self.debug:
            print(f"💬 {text} ({dt_ms:.0f}ms)" + " " * 20 + "\r", end="", flush=True)
            print()  # 换行
        else:
            print(f'[{time.time():.3f}] 💬 识别结果: "{text}"   音频={audio_duration:.2f}s 识别={dt_ms:.0f}ms')

        self.last_recognize_time = time.time()

        # 调用回调（如果有）
        if self.on_text:
            self.on_text(text)

        return text

    @staticmethod
    def _reader(stream, audio_q: 'queue.Queue'):
        """麦克风读线程：阻塞读塞队列

        把 stream.read 从 mic_loop 剥离的原因：Pa_ReadStream 会间歇性
        停滞（播放/abort期间偶发，卡在C调用里永不返回——实测踩坑）。
        停滞时本线程被放弃（daemon），mic_loop 超时后重建InputStream自愈。
        """
        while True:
            try:
                data, _ = stream.read(VAD_BLOCK)
            except Exception as e:
                audio_q.put(('err', str(e)))
                return
            audio_q.put(('ok', data))

    def _reset_listeners(self):
        """流重建/状态切换后清各检测器残留（VAD/唤醒/打断全家）"""
        self.vad.reset()
        while not self.vad.empty():
            self.vad.pop()
        self.in_speech = False
        if self.wake is not None:
            self.wake.reset()
        if self.barge_vad is not None:
            self.barge_vad.reset()
            while not self.barge_vad.empty():
                self.barge_vad.pop()
        if self.barge_kws is not None:
            self.barge_kws.reset()

    def mic_loop(self):
        """智能音响模式 - 参考ROS2 mic_loop实现

        读流与处理解耦：reader线程读麦克风塞队列，本循环带2s超时消费。
        读停滞（ALSA间歇性bug，播放/abort期间偶发）→ 重建InputStream自愈，
        不再出现整个助手冻结。
        """
        self.round_num = 0

        print("\n🎯 进入智能音响模式 - 自动循环识别")
        print("💡 直接说话即可，系统会自动检测和识别")
        print("🛑 按 Ctrl+C 退出")
        print("=" * 60)
        print()

        while True:   # 流生命周期：停滞/出错 → 重建
            stream = None
            try:
                # 初始化音频流
                stream = sd.InputStream(
                    channels=1,
                    dtype='float32',
                    samplerate=SAMPLE_RATE,
                    blocksize=VAD_BLOCK
                )
                stream.start()
                audio_q = queue.Queue()
                threading.Thread(target=self._reader,
                                 args=(stream, audio_q), daemon=True).start()

                if self.debug:
                    print("🎤 麦克风已开（VAD自动断句），请说话...")

                while True:
                    try:
                        tag, data = audio_q.get(timeout=2.0)
                    except queue.Empty:
                        print("\n⚠️  麦克风读停滞>2s（ALSA间歇性问题），重建音频流...")
                        break
                    if tag == 'err':
                        print(f"\n⚠️  麦克风读失败: {data}，重建音频流...")
                        break

                    # 🔇 回声门控：静音窗口内音频块直接丢弃（回声不进VAD，
                    #    从源头消除回声段落）；到期后重置VAD恢复干净收音
                    if time.time() < self.mute_until:
                        if not self._in_mute_window:
                            self._in_mute_window = True
                            if self.debug:
                                print(f"[{time.time():.3f}] 🔇 静音窗口内，丢弃音频块")
                        # ✋ 打断检测（R2）：正在播报时用AEC消回声后判定插话，
                        #    命中→on_barge_in（停播）；clean不进主VAD
                        if self.playing:
                            self._barge_check(data[:, 0])
                        continue
                    if self._in_mute_window:
                        self._in_mute_window = False

                    # 窗口刚结束：重置VAD（清掉静音前的内部状态与残留段落）
                    if self.vad_reset_requested:
                        self.vad_reset_requested = False
                        self.vad.reset()
                        while not self.vad.empty():   # 防御性清空段落队列
                            self.vad.pop()
                        self.in_speech = False
                        if self.wake is not None:
                            self.wake.reset()
                        if self.debug:
                            print(f"[{time.time():.3f}] ♻️ VAD已重置，恢复干净收音")

                    # ==== 唤醒词状态机（wake_words启用时） ====
                    if self.wake is not None:
                        if self.state == 'idle':
                            # 待唤醒：音频只喂KWS，不进VAD/ASR（拦截在源头）
                            hit = self.wake.process_block(data[:, 0])
                            if hit:
                                self.state = 'active'
                                self.last_voice_time = time.time()
                                self.round_num = 0
                                print(f"\n🔔 唤醒: {hit}")
                                if self.on_wake:
                                    try:
                                        self.on_wake(hit)
                                    except Exception as e:
                                        print(f"⚠️  on_wake回调出错: {e}")
                            continue

                        # active：超时无有效语音 → 回到待唤醒
                        if time.time() - self.last_voice_time > self.active_timeout:
                            self.state = 'idle'
                            print("\n💤 超时无语音，回到待唤醒...")
                            # 清VAD残留，避免休眠瞬间弹出旧段落
                            self.vad.reset()
                            while not self.vad.empty():
                                self.vad.pop()
                            self.in_speech = False
                            self.wake.reset()
                            if self.on_sleep:
                                try:
                                    self.on_sleep()
                                except Exception as e:
                                    print(f"⚠️  on_sleep回调出错: {e}")
                            continue

                    self.vad.accept_waveform(data[:, 0])

                    # 检测到语音开始
                    if self.vad.is_speech_detected() and not self.in_speech:
                        self.in_speech = True
                        self.last_voice_time = time.time()   # 刷新活跃计时（防超时休眠）
                        if self.debug:
                            print("🗣️ 检测到语音 ...")
                        elif not self.debug:
                            # 简洁模式显示
                            print("🔊 正在录音..." + " " * 20 + "\r", end="", flush=True)

                    # VAD检测到语音段落结束
                    if not self.vad.empty():
                        segment = np.asarray(self.vad.front.samples, dtype='float32')
                        t_emit = time.time()  # 段落切出时刻
                        self.vad.pop()
                        self.in_speech = False
                        self.last_voice_time = t_emit   # 有对话输入，刷新活跃计时
                        if self.debug:
                            print(f"[{t_emit:.3f}] ✂️ 段落弹出（{len(segment)/SAMPLE_RATE:.2f}s）")

                        # 冷却期内丢弃（同一句话的尾音可能被VAD切成第二段）
                        if time.time() - self.last_recognize_time < self.cooldown:
                            if self.debug:
                                print(f"[{time.time():.3f}] ⏸️  冷却期内，丢弃本段音频")
                            continue

                        # 播报门控：机器人正在说话（或混响尾巴内）时丢弃本段——
                        # 这段音频是扬声器回声，不是用户指令（参考ROS2回声门控方案）
                        if time.time() < self.mute_until:
                            if self.debug:
                                print(f"[{time.time():.3f}] 🔇 播报中，丢弃本段音频（回声门控）")
                            continue

                        self.round_num += 1
                        if not self.debug:
                            print(f"--- 第 {self.round_num} 次识别 ---" + "\r", end="", flush=True)

                        self.recognize_and_display(segment, t_emit)

                        if not self.debug:
                            # 短暂间隔后继续
                            time.sleep(0.3)
                            print(f"🎤 监听中..." + " " * 20 + "\r", end="", flush=True)

            except KeyboardInterrupt:
                print("\n🛑 用户中断，退出")
                raise   # 重新抛出：让上层（voice_chat.run）做停止播报+硬退出；
                        # 独立运行时由 main() 的兜底 except 接住
            finally:
                # 关闭音频流。注意：SIGINT打断阻塞读后，PortAudio的stop/close
                # 可能死锁——放到daemon线程里关，1秒关不完就放弃（进程退出时
                # 系统会回收ALSA设备），保证退出不悬挂
                if stream is not None:
                    def _close_stream():
                        try:
                            stream.stop()
                            stream.close()
                        except Exception:
                            pass
                    closer = threading.Thread(target=_close_stream, daemon=True)
                    closer.start()
                    closer.join(timeout=1.0)
                    if self.debug:
                        print("✅ 麦克风流清理完成（或已放弃，由系统回收）")

            # 走到这里说明是停滞/读错误重建（KeyboardInterrupt已raise走）：
            # 清各检测器残留状态，稍等设备释放后重开流
            self._reset_listeners()
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(
        description='智能音响式实时语音识别',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 vad_asr.py                          # 使用默认参数
  python3 vad_asr.py --duration 8             # 最大录音时长8秒
  python3 vad_asr.py --silence 0.5            # 静音0.5秒后断句
  python3 vad_asr.py --threshold 0.5         # VAD阈值0.5
  python3 vad_asr.py --debug                 # 显示调试信息
        '''
    )

    parser.add_argument('--vad-model', type=str,
                       default='/mnt/apfs/workplace/ROS2/models/asr/silero_vad.onnx',
                       help='VAD模型路径')
    parser.add_argument('--asr-model', type=str,
                       default='/mnt/apfs/workplace/ROS2/models/asr/sense-voice',
                       help='SenseVoice模型目录')
    parser.add_argument('--duration', type=float, default=8.0,
                       help='最大录音时长（秒，默认8.0）')
    parser.add_argument('--silence', type=float, default=0.5,
                       help='静音多长时间后断句（秒，默认0.5）')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='VAD检测阈值（0-1，越高越严格，默认0.5）')
    parser.add_argument('--min-speech', type=float, default=0.25,
                       help='最短语音时长（秒，默认0.25）')
    parser.add_argument('--cooldown', type=float, default=1.0,
                       help='识别后冷却时间（秒，默认1.0）')
    parser.add_argument('--debug', action='store_true',
                       help='显示调试信息')
    parser.add_argument('--wake-word', type=str, action='append',
                       help='启用唤醒词模式（可多次指定）；默认开放麦克风')
    parser.add_argument('--wake-threshold', type=float, default=0.25,
                       help='唤醒词命中阈值（默认0.25）')
    parser.add_argument('--active-timeout', type=float, default=30.0,
                       help='唤醒后无语音休眠时间（秒，默认30）')

    args = parser.parse_args()

    print("=" * 60)
    print("🎤 智能音响式实时语音识别")
    print("=" * 60)

    # 创建智能音响ASR实例
    smart_speaker = SmartSpeakerASR(
        sensevoice_model_dir=args.asr_model,
        vad_model_path=args.vad_model,
        max_duration=args.duration,
        silence_duration=args.silence,
        vad_threshold=args.threshold,
        min_speech_duration=args.min_speech,
        cooldown=args.cooldown,
        wake_words=args.wake_word,
        wake_threshold=args.wake_threshold,
        active_timeout=args.active_timeout,
        on_wake=lambda word: print(f"🔔 (独立测试) 唤醒: {word}，开始聆听..."),
        debug=args.debug
    )

    # 启动麦克风循环（mic_loop中断后会重新抛出KeyboardInterrupt，这里兜底）
    try:
        smart_speaker.mic_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()