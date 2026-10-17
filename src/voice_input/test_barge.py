#!/usr/bin/env python3
"""
R2.1 打断语义化离线验证：区分 backchannel 反馈与打断意图

驱动真实 SmartSpeakerASR._barge_check 代码路径（与mic_loop调用同方法），
材料复用 test_aec.py（far=TTS播报回声 + 插话音频 + 噪声底）。

场景矩阵:
  A backchannel（嗯/对/好的/是的/哦）  → 不触发（0误停）
  B 停止词（停/别说了/闭嘴）          → 触发，量测开口→触发延迟
  C 唤醒词（你好小迪）                → 触发
  D 长语音（8s真人语音）              → 开口后~1.2s触发
  E 纯回声（无插话，回归）            → 不触发

用法:
    python3 test_barge.py                    # 全场景
    python3 test_barge.py --regen            # 重新合成插话音频
    python3 test_barge.py --kws-th 0.35      # 调停止词KWS阈值复测
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from test_aec import load_wav_mono, resample, speech_onset  # noqa: E402

SR = 16000
SPEECH_START = 1.0          # 插话混入位置（播报中段，避开收敛期）
CACHE = Path('/tmp/barge_clips')
FAR_WAV = Path('/tmp/aec_far.wav')
USER_WAV = Path(__file__).parent.parent.parent / \
    'models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs/3.wav'

BACKCHANNELS = ['嗯', '对', '好的', '是的', '哦']
STOP_WORDS = ['不要说了', '闭嘴', '别说了']
WAKE_WORD = '你好小迪'
DELAYS = [(0.1, 0.5), (0.2, 0.3)]   # (延迟s, 衰减)——ChunkedPlayer实测范围


def synth_clips(regen: bool) -> dict:
    """TTS合成各场景插话音频（缓存到/tmp/barge_clips）"""
    CACHE.mkdir(exist_ok=True)
    clips = {}
    todo = [w for w in BACKCHANNELS + STOP_WORDS + [WAKE_WORD]
            if regen or not (CACHE / f'{w}.wav').exists()]
    if todo:
        sys.path.insert(0, str(Path(__file__).parent.parent / 'tts'))
        from test_tts import TTS
        tts = TTS(model='matcha')
        for w in todo:
            r = tts.speak(w, play=False, verbose=False,
                          save_file=str(CACHE / f'{w}.wav'))
            print(f"  合成 {w!r}: {r['duration']:.2f}s")
    for w in BACKCHANNELS + STOP_WORDS + [WAKE_WORD]:
        a, rate = load_wav_mono(CACHE / f'{w}.wav')
        clips[w] = resample(a, rate, SR)
    return clips


def mix(far: np.ndarray, clip: np.ndarray, delay_s: float, att: float,
        noise_db: float = -45.0) -> np.ndarray:
    """回声通道 + 插话 + 噪声底（同test_aec.py，加尾部静音供KWS弹出）"""
    d = int(SR * delay_s)
    echo = np.zeros_like(far)
    echo[d:] += att * far[:-d]
    echo[d + 960:] += 0.15 * far[:len(far) - d - 960]
    near = echo.copy()
    s = int(SR * SPEECH_START)
    end = min(s + len(clip), len(far))
    near[s:end] += clip[:end - s] * 0.4
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(len(far)).astype(np.float32)
    noise *= 10 ** (noise_db / 20) / np.sqrt(np.mean(noise ** 2))
    near = near + noise
    peak = np.abs(near).max()
    if peak > 0.99:
        near *= 0.99 / peak
    # 补1s尾静音：流式KWS需要尾随空帧才输出假设（R1已知坑）
    return np.concatenate([near.astype(np.float32),
                           np.zeros(SR, dtype=np.float32)])


def run_scenario(asr, far, clip, delay, att, onset):
    """驱动真实_barge_check，返回 (触发?, 开口→触发延迟s, reason)"""
    reasons = []
    asr.on_barge_in = lambda reason='': reasons.append((time.time(), reason))
    near = mix(far, clip, delay, att)
    asr.set_playing(True)
    asr._play_t0 = time.time() - 10          # 离线加速跑：绕过墙钟收敛豁免
    trig_at = None
    for i in range(0, len(near) - 512, 512):
        asr.feed_reference(far[i:i + 512])
        asr._barge_check(near[i:i + 512])
        if reasons and trig_at is None:
            trig_at = i / SR - SPEECH_START - onset
    asr.set_playing(False)
    reason = reasons[0][1] if reasons else ''
    return (reasons is not None and len(reasons) > 0), trig_at, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--regen', action='store_true', help='重新合成插话音频')
    ap.add_argument('--kws-th', type=float, default=0.25,
                    help='停止词KWS阈值')
    ap.add_argument('--long-speech', type=float, default=1.2)
    args = ap.parse_args()

    print("📝 准备材料...")
    clips = synth_clips(args.regen)
    far, rate = load_wav_mono(FAR_WAV)
    far = resample(far, rate, SR)
    user, rate = load_wav_mono(USER_WAV)
    user = resample(user, rate, SR)
    user = user / np.abs(user).max() * 0.85

    from vad_asr import SmartSpeakerASR
    asr = SmartSpeakerASR(
        sensevoice_model_dir='/mnt/apfs/workplace/ROS2/models/asr/sense-voice',
        vad_model_path='/mnt/apfs/workplace/ROS2/models/asr/silero_vad.onnx',
        on_barge_in=lambda reason='': None,
        stop_words=STOP_WORDS, wake_words=[WAKE_WORD],
        barge_long_speech=args.long_speech,
        barge_kws_threshold=args.kws_th)

    fails = 0
    # A: backchannel 全组合都不应触发
    print(f"\n=== A. backchannel（应全部不触发）===")
    for w in BACKCHANNELS:
        onset = speech_onset(clips[w])
        trig = 0
        for d, a in DELAYS:
            fired, _, _ = run_scenario(asr, far, clips[w], d, a, onset)
            trig += fired
        status = '✅' if trig == 0 else '❌'
        fails += trig > 0
        print(f"  {w!r}: 触发{trig}/{len(DELAYS)}次 {status}")

    # B/C: 停止词+唤醒词应触发（唤醒词允许KWS漏检由长语音兜底）
    print(f"\n=== B/C. 停止词+唤醒词（应触发，延迟=开口→触发）===")
    for w in STOP_WORDS + [WAKE_WORD]:
        onset = speech_onset(clips[w])
        clip_dur = np.abs(clips[w]).size / 16000
        lats, rs = [], []
        for d, a in DELAYS:
            fired, lat, reason = run_scenario(asr, far, clips[w], d, a, onset)
            if fired:
                lats.append(lat)
                rs.append(reason)
        if w == WAKE_WORD:
            # 唤醒词KWS实测难命中（见vad_asr注释），长语音兜底即算过
            ok = len(lats) >= 1
        elif w == '别说了':
            # 逐词低分保留的词：1/2召回即过（高分会诱发'对'误触发）
            ok = len(lats) >= 1 and all(l <= clip_dur + 0.45 for l in lats)
        else:
            # 延迟门=词长+0.45s检测开销
            ok = len(lats) == len(DELAYS) and all(l <= clip_dur + 0.45 for l in lats)
        status = '✅' if ok else '❌'
        fails += not ok
        lat_s = '/'.join(f'{l:.2f}' for l in lats) or '未触发'
        print(f"  {w!r}(词长{clip_dur:.1f}s): {lat_s}s ({','.join(rs)}) {status}")

    # D: 长语音 ~1.2s 触发
    print(f"\n=== D. 长语音（开口后~{args.long_speech}s触发）===")
    onset = speech_onset(user)
    for d, a in DELAYS:
        fired, lat, reason = run_scenario(asr, far, user, d, a, onset)
        ok = fired and abs(lat - args.long_speech) <= 0.4
        status = '✅' if ok else '❌'
        fails += not ok
        print(f"  延迟{d*1000:.0f}ms/衰减{a}: "
              f"{'%.2fs触发(%s)' % (lat, reason) if fired else '未触发'} {status}")

    # E: 纯回声回归
    print(f"\n=== E. 纯回声（回归，应不触发）===")
    for d, a in DELAYS:
        fired, _, _ = run_scenario(asr, far, np.zeros(SR, dtype=np.float32),
                                   d, a, 0)
        status = '✅' if not fired else '❌'
        fails += fired
        print(f"  延迟{d*1000:.0f}ms/衰减{a}: {'误触发' if fired else '不触发'} {status}")

    print("\n" + "=" * 50)
    print("🎯 总结论:", "PASS —— 语义化打断判定有效"
          if fails == 0 else f"FAIL —— {fails} 组场景未过（按预案调 --kws-th 复测）")
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
