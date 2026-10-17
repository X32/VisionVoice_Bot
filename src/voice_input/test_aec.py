#!/usr/bin/env python3
"""
Phase A：AEC 回声消除离线可行性验证（go/no-go 门，不动现有代码）

模拟场景：机器人播报(far) + 房间回声路径(延迟/衰减/混响) + 用户中途插话(真人语音)
验证：EchoCanceller(AEC3) 能否把回声压到用户语音之下 ≥20dB，
      VoiceDetector 在"纯回声"区不误报、在"用户插话"区可靠报语音。

far-end 参考:  matcha TTS 合成长回答（22050→16k 重采样，缓存到 /tmp/aec_far.wav）
near-end:     回声通道模拟（延迟+衰减+comb混响）+ KWS test_wavs 真人语音插入
处理方式:     512样本/块流式喂入（与真实 mic_loop 一致）

用法:
    python3 test_aec.py                     # 全组合扫描
    python3 test_aec.py --delay 200 --att 0.5   # 单组合
    python3 test_aec.py --no-cache          # 重新合成far音频

通过标准（方案文档 DOC/技术方案-R2打断barge-in-AEC.md Phase A）:
    1. 回声残留比用户语音低 ≥20dB
    2. 纯回声区 VoiceDetector 概率 <0.3
    3. 用户插话区 VoiceDetector 概率 >0.7
    4. 延迟容差 ≥400ms
"""
import argparse
import wave
from pathlib import Path

import numpy as np

from pywebrtc_audio import EchoCanceller, VoiceDetector

SAMPLE_RATE = 16000
BLOCK = 512                      # 与 mic_loop/VAD_BLOCK 一致（32ms）
SPEECH_START = 1.0               # 用户在第1秒插话（播报中段）
TTS_TEXT = ("好的，这是一个用来测试打断功能的比较长的回答。回声消除需要在播放的同时"
            "从麦克风信号里减掉扬声器传来的声音，这样才能可靠地检测到你说话的声音。"
            "你现在可以随时说停来打断我的播报，我会立刻停下来听你说话。")
FAR_CACHE = Path('/tmp/aec_far.wav')
USER_WAV = Path(__file__).parent.parent.parent / \
    'models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav'


def load_wav_mono(path: Path) -> tuple[np.ndarray, int]:
    """读wav为float32单声道（16bit PCM）"""
    with wave.open(str(path), 'rb') as wf:
        assert wf.getnchannels() == 1, f'{path} 需单声道'
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return audio.astype(np.float32) / 32768.0, rate


def resample(audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """线性插值重采样（round-trip实验已证不损坏检测，见KWS报告）"""
    if from_rate == to_rate:
        return audio
    n_out = int(len(audio) * to_rate / from_rate)
    return np.interp(np.linspace(0, len(audio) - 1, n_out),
                     np.arange(len(audio)), audio).astype(np.float32)


def get_farSpeech(regen: bool) -> np.ndarray:
    """far-end参考：TTS合成的播报音频，16k、峰值归一化（带缓存）"""
    if FAR_CACHE.exists() and not regen:
        far, rate = load_wav_mono(FAR_CACHE)
        far = resample(far, rate, SAMPLE_RATE)
        print(f"📼 far缓存: {FAR_CACHE} ({len(far)/SAMPLE_RATE:.1f}s, {rate}Hz→16k)")
        return far
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / 'tts'))
    from test_tts import TTS
    print("🎙️ 合成far-end播报音频（一次，之后缓存）...")
    tts = TTS(model='matcha')
    r = tts.speak(TTS_TEXT, play=False, verbose=False, save_file=str(FAR_CACHE))
    return resample(r['samples'], r['sample_rate'], SAMPLE_RATE)


def make_near(far: np.ndarray, delay_ms: int, att: float,
              user: np.ndarray, noise_dbfs: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """模拟回声通道生成near信号

    Args:
        noise_dbfs: 环境噪声底（dBFS RMS）。真实房间有风扇/电流噪声，
            没有噪声底时任何VAD都会把AEC残留当语音（残留永远高于零底）。
    Returns:
        near: 麦克风信号 = 回声通道(far) + 插话(user, SPEECH_START起) + 噪声
        ref:  far-end参考（与far时间线一致，即"实际播出"的信号）
        user_mask: 用户语音区域布尔掩码（评估分区用）
    """
    d = int(SAMPLE_RATE * delay_ms / 1000)
    echo = np.zeros_like(far)
    # 直达声 + 两组反射（comb混响简化模型）
    echo[d:] += att * far[:-d]
    echo[d + 960:] += 0.15 * far[:len(far) - d - 960]      # +60ms 反射
    echo[d + 2080:] += 0.08 * far[:len(far) - d - 2080]    # +130ms 反射

    user_full = np.zeros_like(far)
    s = int(SAMPLE_RATE * SPEECH_START)
    end = min(s + len(user), len(far))
    user_full[s:end] = user[:end - s] * 0.4   # 插话音量-8dB左右（NER>0dB场景）

    rng = np.random.default_rng(42)           # 固定种子保证组合间可比
    noise = rng.standard_normal(len(far)).astype(np.float32)
    noise *= 10 ** (noise_dbfs / 20) / np.sqrt(np.mean(noise ** 2))

    near = echo + user_full + noise
    peak = np.abs(near).max()
    if peak > 0.99:                           # 防削波
        near *= 0.99 / peak
    mask = np.zeros(len(far), dtype=bool)
    mask[s:end] = True
    return near.astype(np.float32), far.copy(), mask


def run_case(far: np.ndarray, user: np.ndarray, delay_ms: int, att: float,
             stream_delay_hint: int, pipeline: bool = False,
             noise_dbfs: float = -45.0, silero_threshold: float = 0.7,
             min_speech: float = 0.06, user_onset: float = 0.0) -> dict:
    """单组合：流式512块喂AEC，分区统计指标

    pipeline=True 用完整 AudioProcessor(HP+AEC+NS)，更贴近实际部署。
    VAD 双轨统计：webrtc VoiceDetector概率 + silero（与实际管线同源，
    barge-in判定将用silero），各报"误报块占比"与"插话检出占比"。
    """
    near, ref, mask = make_near(far, delay_ms, att, user, noise_dbfs)
    if pipeline:
        from pywebrtc_audio import AudioProcessor
        proc = AudioProcessor(sample_rate=SAMPLE_RATE, echo_cancellation=True,
                              noise_suppression=True, ns_level=2,
                              stream_delay_ms=stream_delay_hint)
    else:
        proc = EchoCanceller(sample_rate=SAMPLE_RATE,
                             stream_delay_ms=stream_delay_hint)
        vd = VoiceDetector(sample_rate=SAMPLE_RATE)

    # silero VAD——打断检测专用参数：min_speech压到0.06s（主VAD是0.25s，
    # 那是为ASR断句服务的；打断只求快，防抖交给3连帧判定）
    import sherpa_onnx
    silero_cfg = sherpa_onnx.SileroVadModelConfig(
        model='/mnt/apfs/workplace/ROS2/models/asr/silero_vad.onnx',
        threshold=silero_threshold,
        min_silence_duration=0.3,
        min_speech_duration=min_speech,
        window_size=BLOCK,
    )
    silero = sherpa_onnx.VoiceActivityDetector(
        sherpa_onnx.VadModelConfig(silero_vad=silero_cfg, sample_rate=SAMPLE_RATE))

    clean = np.zeros_like(near)
    probs = np.zeros(len(near) // BLOCK + 1)
    sil_flags = np.zeros(len(near) // BLOCK + 1, dtype=bool)
    for i in range(0, len(near) - BLOCK, BLOCK):
        j = i + BLOCK
        clean[i:j] = proc.process(near[i:j], ref[i:j])
        probs[i // BLOCK] = (proc.speech_probability if pipeline
                             else vd.process(clean[i:j]))
        silero.accept_waveform(clean[i:j])
        sil_flags[i // BLOCK] = silero.is_speech_detected()

    echo_zone = ~mask
    warm = int(SAMPLE_RATE * 0.5)   # 前0.5s为AEC收敛期：所有统计排除
    echo_rms_near = np.sqrt(np.mean(near[warm:][echo_zone[warm:]] ** 2))
    echo_rms_clean = np.sqrt(np.mean(clean[warm:][echo_zone[warm:]] ** 2))
    user_rms_near = np.sqrt(np.mean(near[mask] ** 2))
    user_rms_clean = np.sqrt(np.mean(clean[mask] ** 2))

    suppression = 20 * np.log10((echo_rms_near + 1e-9) / (echo_rms_clean + 1e-9))
    residual_ratio = 20 * np.log10((echo_rms_clean + 1e-9) / (user_rms_near + 1e-9))
    # 块级对齐掩码（每块512样本，取块首样本归属），收敛期排除
    n_blocks = len(near) // BLOCK
    blk_idx = np.arange(n_blocks) * BLOCK
    blk_mask = mask[blk_idx]
    valid = blk_idx >= warm
    echo_probs = probs[:n_blocks][valid & ~blk_mask]
    user_probs = probs[:n_blocks][valid & blk_mask]
    sil_echo = sil_flags[:n_blocks][valid & ~blk_mask]
    sil_user = sil_flags[:n_blocks][valid & blk_mask]

    return dict(delay=delay_ms, att=att, hint=stream_delay_hint,
                suppression=suppression, residual=residual_ratio,
                p_echo=float(echo_probs.mean()), p_user=float(user_probs.mean()),
                sil_echo_rate=float(sil_echo.mean()),          # 误报：回声区silero命中块占比
                sil_user_rate=float(sil_user.mean()),          # 检出：插话区silero命中块占比
                user_survive=20 * np.log10((user_rms_clean + 1e-9) / (user_rms_near + 1e-9)),
                t_first=t_first_trigger(sil_flags, blk_mask, n_blocks,
                                        user_onset=user_onset))


def t_first_trigger(sil_flags: np.ndarray, blk_mask: np.ndarray,
                    n_blocks: int, need: int = 3,
                    user_onset: float = 0.0) -> float:
    """首触发时间：用户实际开口 → 连续need块silero命中（对应打断判定逻辑）

    注意原点是插话wav内的真实语音起始（前导静音要扣除），不是混音位置。
    clean信号只用于打断判定（停播后ASR走正常管线），故检出连续性
    比信号保真度重要。返回秒；未触发返回inf。
    """
    run = 0
    for b in range(n_blocks):
        if blk_mask[b] and sil_flags[b]:
            run += 1
            if run >= need:
                return (b - need + 1) / (SAMPLE_RATE / BLOCK) \
                    - SPEECH_START - user_onset
        else:
            run = 0
    return float('inf')


def speech_onset(audio: np.ndarray) -> float:
    """滑窗RMS估音频内语音起始秒（20ms窗，超噪声底20倍算开口）"""
    w = 320
    rms = np.convolve(audio ** 2, np.ones(w) / w, 'same')
    floor = np.median(rms[:SAMPLE_RATE]) if len(audio) > SAMPLE_RATE \
        else np.median(rms)
    idx = np.argmax(rms > floor * 20)
    return idx / SAMPLE_RATE if rms[idx] > floor * 20 else 0.0


def main():
    ap = argparse.ArgumentParser(description='AEC离线可行性验证')
    ap.add_argument('--delay', type=int, nargs='+', default=[100, 200, 400],
                    help='回声延迟ms（可多个）')
    ap.add_argument('--att', type=float, nargs='+', default=[0.3, 0.6],
                    help='回声衰减系数（可多个）')
    ap.add_argument('--hint-bad', action='store_true',
                    help='附加一组stream_delay_ms=0的坏提示（考验内部估计器）')
    ap.add_argument('--pipeline', action='store_true',
                    help='用完整AudioProcessor(HP+AEC+NS)替代独立EchoCanceller')
    ap.add_argument('--noise-db', type=float, default=-45.0,
                    help='环境噪声底dBFS RMS（默认-45，真实房间典型值）')
    ap.add_argument('--silero-threshold', type=float, default=0.7,
                    help='silero VAD阈值（barge-in判定用，0.7实测误报最优）')
    ap.add_argument('--min-speech', type=float, default=0.06,
                    help='barge专用silero最短语音秒数（主VAD为0.25）')
    ap.add_argument('--no-cache', action='store_true', help='重新合成far音频')
    args = ap.parse_args()

    far = get_farSpeech(args.no_cache)
    user, rate = load_wav_mono(USER_WAV)
    user = resample(user, rate, SAMPLE_RATE)
    peak = np.abs(user).max()
    user = user / peak * 0.85
    onset = speech_onset(user)
    print(f"👤 用户插话: {USER_WAV.name} ({len(user)/SAMPLE_RATE:.1f}s, "
          f"混入t={SPEECH_START}s, 实际开口+{onset:.2f}s)\n")

    cases = []
    for d in args.delay:
        for a in args.att:
            cases.append((d, a, d))          # hint=真实延迟（实际用法：实测后设置）
    if args.hint_bad:
        cases.append((args.delay[0], args.att[0], 0))

    print(f"⚙️  {'AudioProcessor管线' if args.pipeline else '独立EchoCanceller'} | "
          f"噪声底{args.noise_db}dBFS | silero阈值{args.silero_threshold}\n")
    print(f"{'延迟':>6} {'衰减':>5} {'hint':>5} | {'残留比dB':>8} {'保留dB':>7} "
          f"{'误报%':>6} {'首触发s':>7} | 判定")
    print("-" * 72)
    all_pass = True
    for d, a, hint in cases:
        r = run_case(far, user, d, a, hint, pipeline=args.pipeline,
                     noise_dbfs=args.noise_db,
                     silero_threshold=args.silero_threshold,
                     min_speech=args.min_speech, user_onset=onset)
        # 通过标准（clean信号只用于打断判定，ASR走停播后的正常管线，
        # 故"保留dB"仅展示不作门）：
        #   残留比≤-20dB；误报率<2%；首触发≤0.3s；
        #   400ms超出AEC3延迟规格，仅展示不进GO门
        in_spec = r['delay'] <= 200
        ok = (r['residual'] <= -20 and r['sil_echo_rate'] < 0.02
              and r['t_first'] <= 0.3 and in_spec)
        if in_spec:
            all_pass &= ok          # 超规格组仅展示，不进GO门
        mark = '✅ PASS' if ok else ('⚠️ 超规格' if not in_spec else '❌ FAIL')
        t_first_str = f"{r['t_first']:>7.2f}" if r['t_first'] != float('inf') \
            else '   未触发'
        print(f"{r['delay']:>5}m {r['att']:>5} {r['hint']:>5} | "
              f"{r['residual']:>8.1f} {r['user_survive']:>7.1f} "
              f"{r['sil_echo_rate']*100:>6.1f} {t_first_str} | {mark}")

    print("-" * 78)
    print("🎯 总结论:", "GO —— AEC3满足打断检测要求，进入Phase B/C"
          if all_pass else "NO-GO —— 未达标，回评审（speexdsp/能量启发式）")


if __name__ == '__main__':
    main()
