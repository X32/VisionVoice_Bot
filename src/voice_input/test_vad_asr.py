#!/usr/bin/env python3
"""VAD+ASR功能测试脚本 - 模拟录音测试"""
import sys
import time
import numpy as np
sys.path.insert(0, 'src/voice_input')
import vad_asr

def create_test_audio(duration=2.0, sample_rate=16000):
    """创建测试音频数据（模拟人声）"""
    # 生成模拟的人声信号（正弦波组合）
    t = np.linspace(0, duration, int(sample_rate * duration))
    # 混合几个频率来模拟人声
    signal = (np.sin(2 * np.pi * 440 * t) * 0.3 +  # 440Hz
              np.sin(2 * np.pi * 880 * t) * 0.2 +   # 880Hz
              np.sin(2 * np.pi * 1200 * t) * 0.1)   # 1200Hz
    # 添加一些噪声
    noise = np.random.normal(0, 0.02, len(signal))
    audio = signal + noise

    # 确保在合理范围内
    audio = np.clip(audio, -1.0, 1.0)
    return audio.reshape(-1, 1).astype(np.float32)

def test_vad_detection():
    """测试VAD语音检测"""
    print("🧪 测试1: VAD语音检测")
    print("=" * 40)

    vad = vad_asr.VAD_ASR(
        sensevoice_model_dir='/mnt/apfs/workplace/ROS2/models/asr/sense-voice',
        energy_threshold=0.01,
        debug=True
    )

    # 测试静音
    silence = np.zeros(int(16000 * 0.5), dtype=np.float32).reshape(-1, 1)
    result = vad._detect_speech(silence)
    print(f"静音检测: {result} (应该是False)")

    # 测试模拟人声
    speech = create_test_audio(duration=1.0)
    result = vad._detect_speech(speech)
    print(f"人声检测: {result} (应该是True)")

    print("✅ VAD检测测试完成\n")

def test_asr_recognition():
    """测试ASR识别"""
    print("🧪 测试2: ASR识别功能")
    print("=" * 40)

    vad = vad_asr.VAD_ASR(
        sensevoice_model_dir='/mnt/apfs/workplace/ROS2/models/asr/sense-voice',
        max_duration=5.0,
        silence_duration=1.0
    )

    print("📝 准备测试音频...")
    # 这里无法测试实际的麦克风录音，但可以验证ASR初始化
    print("✅ ASR识别器已就绪")
    print("💡 实际使用时需要麦克风进行实时测试")
    print()

def test_energy_threshold():
    """测试不同能量阈值"""
    print("🧪 测试3: 能量阈值设置")
    print("=" * 40)

    thresholds = [0.001, 0.01, 0.05, 0.1]
    test_audio = create_test_audio(duration=1.0)

    for threshold in thresholds:
        vad = vad_asr.VAD_ASR(energy_threshold=threshold)
        result = vad._detect_speech(test_audio)
        print(f"阈值 {threshold:.3f}: {'🗣️ 检测到人声' if result else '🤫 未检测到'}")

    print("✅ 阈值测试完成\n")

def main():
    print("🎯 VAD+ASR功能测试套件")
    print("=" * 40)
    print()

    try:
        test_vad_detection()
        test_asr_recognition()
        test_energy_threshold()

        print("🎉 所有测试完成!")
        print()
        print("📋 使用说明:")
        print("  # 运行完整的VAD+ASR系统（需要麦克风）")
        print("  python3 src/voice_input/vad_asr.py")
        print()
        print("  # 调试模式")
        print("  python3 src/voice_input/vad_asr.py --debug")
        print()
        print("  # 自定义参数")
        print("  python3 src/voice_input/vad_asr.py --duration 5 --silence 2.0")

    except Exception as e:
        print(f"❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()