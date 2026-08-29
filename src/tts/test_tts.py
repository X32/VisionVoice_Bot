#!/usr/bin/env python3
"""TTS 文本转语音类
提供便捷的TTS类接口，支持命令行使用和程序调用

用法：
  # 命令行使用
  python3 test_tts.py --text "你好" --output test.wav

  # 程序调用
  from test_tts import TTS
  tts = TTS()  # 默认使用matcha模型
  audio = tts.speak("你好世界", play=True, save_file="hello.wav")

  # 高级使用
  tts = TTS(model="melo", speed=0.8, normalize_peak=0.9)
  result = tts.speak("测试", play=False)
  print(f"生成耗时: {result['gen_time']:.2f}s")
  print(f"音频时长: {result['duration']:.2f}s")
  print(f"RTF: {result['rtf']:.3f}")
"""
import argparse
import time
import wave
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import sherpa_onnx
import sounddevice as sd


class TTS:
    """文本转语音类

    Args:
        model: 模型名称，默认 'matcha'（可选 'melo'）
        speed: 语速倍率，默认 1.0
        project_root: 项目根目录，默认自动检测
        num_threads: 生成线程数，默认 2
        normalize_peak: 归一化峰值，默认 0.85（0.0-1.0）

    模型说明:
        matcha - 标准普通话，音质好，语速正常
        melo   - 中英混说，实测语速快、音量低
    """

    def __init__(
        self,
        model: str = 'matcha',
        speed: float = 1.0,
        project_root: Optional[str] = None,
        num_threads: int = 2,
        normalize_peak: float = 0.85
    ):
        self.model = model
        self.speed = speed
        self.normalize_peak = normalize_peak
        self.num_threads = num_threads

        # 自动检测项目根目录
        if project_root is None:
            script_dir = Path(__file__).parent
            self.project_root = script_dir.parent.parent
        else:
            self.project_root = Path(project_root)

        # 模型路径
        self.model_dir = self.project_root / 'models' / 'tts' / model

        # 初始化TTS引擎
        self._init_tts()

    def _init_tts(self):
        """初始化TTS引擎"""
        model_dir = str(self.model_dir)

        if self.model == 'matcha':
            # matcha 两段式：声学模型（mel谱）+ vocoder（波形）
            # rule_fsts 关键：number.fst 把阿拉伯数字转成中文读法（"2米"→"两米"）
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    matcha=sherpa_onnx.OfflineTtsMatchaModelConfig(
                        acoustic_model=f'{model_dir}/model-steps-3.onnx',
                        vocoder=f'{model_dir}/hifigan_v2.onnx',
                        lexicon=f'{model_dir}/lexicon.txt',
                        tokens=f'{model_dir}/tokens.txt',
                        dict_dir=f'{model_dir}/dict',
                    ),
                    num_threads=self.num_threads,
                ),
                rule_fsts=f'{model_dir}/number.fst,{model_dir}/phone.fst',
            )
        else:  # melo（vits 通道）
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=f'{model_dir}/model.onnx',
                        lexicon=f'{model_dir}/lexicon.txt',
                        tokens=f'{model_dir}/tokens.txt',
                        data_dir=f'{model_dir}/dict',
                    ),
                    num_threads=self.num_threads,
                ),
            )

        self.tts = sherpa_onnx.OfflineTts(tts_config)

    def speak(
        self,
        text: str,
        speed: Optional[float] = None,
        play: bool = True,
        save_file: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """合成语音

        Args:
            text: 要合成的文本
            speed: 语速倍率（覆盖初始化时的设置）
            play: 是否播放音频
            save_file: 保存音频到文件（WAV格式）
            verbose: 是否打印详细信息

        Returns:
            包含生成结果的字典:
                - samples: 音频样本数组（numpy.ndarray）
                - sample_rate: 采样率
                - gen_time: 生成耗时（秒）
                - duration: 音频时长（秒）
                - rtf: 实时因子（gen_time / duration）
                - speed: 语速（字/秒）
                - text: 原始文本
        """
        if speed is None:
            speed = self.speed

        if verbose:
            print(f"🎙️  模型: {self.model}")
            print(f"📝 文本: {text}")
            print(f"⏱️  语速: {speed}x")
            print("-" * 40)

        # 生成音频
        if verbose:
            print("🔊 正在生成音频...")

        t0 = time.perf_counter()
        audio = self.tts.generate(text, speed=speed)
        gen_time = time.perf_counter() - t0

        # 处理音频样本
        samples = np.asarray(audio.samples, dtype=np.float32)

        # 峰值归一化
        peak = np.abs(samples).max()
        if peak > 1e-4:
            samples = samples / peak * self.normalize_peak

        # 计算统计信息
        duration = len(samples) / audio.sample_rate
        rtf = gen_time / duration if duration > 0 else 0
        n_chars = len([c for c in text if c.isalnum() or '一' <= c <= '鿿'])
        char_speed = n_chars / duration if duration > 0 else 0

        if verbose:
            print(f'✅ [{self.model}] {gen_time:.2f}s 生成 → {duration:.2f}s 音频 | '
                  f'RTF={rtf:.3f} | 语速 {char_speed:.1f} 字/秒 | 采样率 {audio.sample_rate}')

        # 保存到文件
        if save_file:
            self._save_wav(samples, audio.sample_rate, save_file)
            if verbose:
                print(f'💾 音频已保存到: {save_file}')

        # 播放音频
        if play:
            if verbose:
                print('🔈 正在播放音频...')
            sd.play(samples, samplerate=audio.sample_rate)
            sd.wait()

        return {
            'samples': samples,
            'sample_rate': audio.sample_rate,
            'gen_time': gen_time,
            'duration': duration,
            'rtf': rtf,
            'char_speed': char_speed,
            'text': text
        }

    def _save_wav(self, samples: np.ndarray, sample_rate: int, filename: str):
        """保存音频为WAV文件"""
        with wave.open(filename, 'w') as wf:
            wf.setnchannels(1)  # 单声道
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # 转换为16-bit PCM
            int16_samples = (samples * 32767).astype(np.int16)
            wf.writeframes(int16_samples.tobytes())


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='TTS 文本转语音测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  python3 test_tts.py                                    # 使用默认文本
  python3 test_tts.py --text "你好世界"                  # 指定文本
  python3 test_tts.py --text "测试" --speed 1.2         # 指定文本和语速
  python3 test_tts.py --text "Hello" --model melo       # 使用melo模型
  python3 test_tts.py --text "测试" --output test.wav   # 保存到文件
  python3 test_tts.py --no-play --text "静默测试"       # 不播放音频

编程示例:
  from test_tts import TTS
  tts = TTS(model="matcha", speed=1.0)
  result = tts.speak("你好", play=False, save_file="test.wav")
  print(f"RTF: {result['rtf']:.3f}")
        '''
    )

    parser.add_argument('--text', '-t', type=str,
                       default='你好，前方零点八七米处有障碍物，请注意避让。',
                       help='要合成的文本')
    parser.add_argument('--speed', '-s', type=float, default=1.0,
                       help='语速倍率 (1.0=原始语速)')
    parser.add_argument('--model', '-m', type=str, default='matcha',
                       choices=['matcha', 'melo'],
                       help='TTS模型选择')
    parser.add_argument('--no-play', action='store_true',
                       help='不播放音频，只生成并显示信息')
    parser.add_argument('--output', '-o', type=str,
                       help='保存音频到WAV文件')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式，减少输出信息')

    return parser.parse_args()


def main():
    """命令行主函数"""
    args = parse_args()

    # 创建TTS实例
    tts = TTS(
        model=args.model,
        speed=args.speed
    )

    # 执行语音合成
    tts.speak(
        text=args.text,
        play=not args.no_play,
        save_file=args.output,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
