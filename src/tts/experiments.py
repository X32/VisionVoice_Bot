#!/usr/bin/env python3
"""TTS 实验脚本 - 展示如何使用TTS类进行各种实验"""
from pathlib import Path
import time

from test_tts import TTS


def experiment_basic():
    """基础实验：简单的语音合成"""
    print("🔬 实验1: 基础语音合成")
    print("=" * 50)

    tts = TTS(model='matcha', speed=1.0)
    result = tts.speak("你好，这是基础实验", play=True)

    print(f"✅ 完成！RTF={result['rtf']:.3f}, 语速={result['char_speed']:.1f}字/秒")
    print()


def experiment_speed_comparison():
    """语速对比实验：测试不同语速的效果"""
    print("🔬 实验2: 语速对比实验")
    print("=" * 50)

    tts = TTS(model='matcha')
    text = "这是一个语速对比测试实验，我们来测试不同的语速设置对音频的影响。"

    speeds = [0.5, 0.8, 1.0, 1.2, 1.5]
    results = []

    for speed in speeds:
        print(f"\n📊 测试语速 {speed}x:")
        result = tts.speak(text, speed=speed, play=False, verbose=False)
        results.append(result)
        print(f"  音频时长: {result['duration']:.2f}s | "
              f"生成耗时: {result['gen_time']:.2f}s | "
              f"RTF: {result['rtf']:.3f}")

    # 总结
    print(f"\n📈 语速实验总结:")
    for i, result in enumerate(results):
        print(f"  {speeds[i]}x: {result['duration']:.2f}s音频, "
              f"{result['char_speed']:.1f}字/秒")
    print()


def experiment_model_comparison():
    """模型对比实验：对比matcha和melo模型"""
    print("🔬 实验3: 模型对比实验")
    print("=" * 50)

    text = "这是一个模型对比测试，我们来比较不同模型的效果。"

    # 测试matcha模型
    print("🎙️  测试 matcha 模型:")
    tts_matcha = TTS(model='matcha', speed=1.0)
    result_matcha = tts_matcha.speak(text, play=False, verbose=False)

    # 测试melo模型
    print("🎙️  测试 melo 模型:")
    tts_melo = TTS(model='melo', speed=0.8)  # melo需要较慢语速
    result_melo = tts_melo.speak(text, play=False, verbose=False)

    # 对比结果
    print(f"\n📊 模型对比结果:")
    print(f"matcha: RTF={result_matcha['rtf']:.3f}, "
          f"语速={result_matcha['char_speed']:.1f}字/秒, "
          f"时长={result_matcha['duration']:.2f}s")

    print(f"melo:   RTF={result_melo['rtf']:.3f}, "
          f"语速={result_melo['char_speed']:.1f}字/秒, "
          f"时长={result_melo['duration']:.2f}s")
    print()


def experiment_batch_processing():
    """批处理实验：批量生成多个语音文件"""
    print("🔬 实验4: 批处理实验")
    print("=" * 50)

    tts = TTS(model='matcha', speed=1.0)

    texts = [
        "第一个测试文本",
        "第二个测试文本",
        "第三个测试文本",
        "第四个测试文本",
        "第五个测试文本"
    ]

    output_dir = Path("tts_outputs")
    output_dir.mkdir(exist_ok=True)

    start_time = time.time()

    for i, text in enumerate(texts, 1):
        output_file = output_dir / f"output_{i}.wav"
        print(f"📝 生成第{i}个: {text[:10]}...")

        result = tts.speak(text, play=False, save_file=str(output_file), verbose=False)
        print(f"   ✅ 完成: {result['duration']:.2f}s音频 → {output_file}")

    total_time = time.time() - start_time
    print(f"\n📊 批处理完成: {len(texts)}个文件, 总耗时{total_time:.2f}s")
    print(f"📁 输出目录: {output_dir}")
    print()


def experiment_audio_analysis():
    """音频分析实验：分析音频数据特性"""
    print("🔬 实验5: 音频分析实验")
    print("=" * 50)

    tts = TTS(model='matcha', speed=1.0)
    result = tts.speak("音频分析实验", play=False, verbose=False)

    samples = result['samples']

    print(f"📊 音频数据分析:")
    print(f"  采样点数量: {len(samples)}")
    print(f"  采样率: {result['sample_rate']} Hz")
    print(f"  音频时长: {result['duration']:.2f}s")
    print(f"  最大值: {samples.max():.6f}")
    print(f"  最小值: {samples.min():.6f}")
    print(f"  平均值: {samples.mean():.6f}")
    print(f"  标准差: {samples.std():.6f}")
    print(f"  RMS值: {((samples**2).mean()**0.5):.6f}")

    # 频谱分析
    import numpy as np
    fft = np.fft.fft(samples[:4096])  # 取前4096个采样点
    magnitude = np.abs(fft[:len(fft)//2])
    dominant_freq = np.argmax(magnitude) * result['sample_rate'] / len(samples[:4096])

    print(f"  主频率: {dominant_freq:.1f} Hz")
    print()


def run_all_experiments():
    """运行所有实验"""
    print("🧪 TTS 实验套件")
    print("=" * 50)
    print()

    experiments = [
        experiment_basic,
        experiment_speed_comparison,
        experiment_model_comparison,
        experiment_batch_processing,
        experiment_audio_analysis
    ]

    for experiment in experiments:
        try:
            experiment()
            input("按回车继续下一个实验...")
            print("\n" * 2)
        except KeyboardInterrupt:
            print("\n🛑 实验被用户中断")
            break
        except Exception as e:
            print(f"❌ 实验出错: {e}")
            continue


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════╗
    ║           TTS 实验脚本 - 选择要运行的实验                 ║
    ╚════════════════════════════════════════════════════════════╝

    1. 基础语音合成实验
    2. 语速对比实验
    3. 模型对比实验
    4. 批处理实验
    5. 音频分析实验
    6. 运行所有实验
    0. 退出
    """)

    while True:
        try:
            choice = input("请选择实验 (0-6): ").strip()

            if choice == "0":
                print("👋 再见！")
                break
            elif choice == "1":
                experiment_basic()
            elif choice == "2":
                experiment_speed_comparison()
            elif choice == "3":
                experiment_model_comparison()
            elif choice == "4":
                experiment_batch_processing()
            elif choice == "5":
                experiment_audio_analysis()
            elif choice == "6":
                run_all_experiments()
                break
            else:
                print("❌ 无效选择，请输入0-6")

        except KeyboardInterrupt:
            print("\n🛑 实验被用户中断")
            break
        except Exception as e:
            print(f"❌ 出错: {e}")