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
import re
import time
import threading
import numpy as np
import sounddevice as sd
import sherpa_onnx
from typing import Callable, Optional


SAMPLE_RATE = 16000     # SenseVoice标准采样率
VAD_BLOCK = 512         # silero VAD窗长（16kHz下用512）


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

        # 回声抑制（参考ROS2 tts_mute设计）
        self.mute_until = 0.0
        self.vad_reset_requested = False   # 恢复收音时重置VAD（由mic_loop线程执行）
        self._in_mute_window = False       # 调试日志用：是否处于静音窗口

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

    def mic_loop(self):
        """智能音响模式 - 参考ROS2 mic_loop实现"""
        self.round_num = 0

        print("\n🎯 进入智能音响模式 - 自动循环识别")
        print("💡 直接说话即可，系统会自动检测和识别")
        print("🛑 按 Ctrl+C 退出")
        print("=" * 60)
        print()

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

            if self.debug:
                print("🎤 麦克风已开（VAD自动断句），请说话...")

            while True:
                data, _ = stream.read(VAD_BLOCK)

                # 🔇 回声门控：静音窗口内音频块直接丢弃（回声不进VAD，
                #    从源头消除回声段落）；到期后重置VAD恢复干净收音
                if time.time() < self.mute_until:
                    if not self._in_mute_window:
                        self._in_mute_window = True
                        if self.debug:
                            print(f"[{time.time():.3f}] 🔇 静音窗口内，丢弃音频块")
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
                    if self.debug:
                        print(f"[{time.time():.3f}] ♻️ VAD已重置，恢复干净收音")

                self.vad.accept_waveform(data[:, 0])

                # 检测到语音开始
                if self.vad.is_speech_detected() and not self.in_speech:
                    self.in_speech = True
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
        debug=args.debug
    )

    # 启动麦克风循环（mic_loop中断后会重新抛出KeyboardInterrupt，这里兜底）
    try:
        smart_speaker.mic_loop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()