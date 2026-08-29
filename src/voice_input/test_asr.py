#!/usr/bin/env python3
"""Day8 Step2：ASR 裸测（无 ROS）。固定录 4 秒 → SenseVoice 识别 → 打印。
目的：在进 ROS 之前，把麦克风/采样率/模型路径这三件事单独验证掉。"""
import time
import numpy as np
import sounddevice as sd
import sherpa_onnx

MODEL_DIR = '/mnt/apfs/workplace/ROS2/models/asr/sense-voice'
SR = 16000   # 语音模型标准采样率

recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
    model=f'{MODEL_DIR}/model.int8.onnx',
    tokens=f'{MODEL_DIR}/tokens.txt',
    use_itn=True,   # 逆文本正则：倾向输出"90"而非"九十"——两种形态 Day7 都兼容
)

print('3 秒后开始录音 4 秒，请说：左转九十度')
time.sleep(3)
audio = sd.rec(4 * SR, samplerate=SR, channels=1, dtype='float32')
sd.wait()

t0 = time.perf_counter()
s = recognizer.create_stream()
s.accept_waveform(SR, audio[:, 0])
recognizer.decode_stream(s)
print(f'识别结果: "{s.result.text}"   耗时 {(time.perf_counter()-t0)*1000:.0f} ms')
