#!/usr/bin/env python3
"""
唤醒词检测（Keyword Spotting）封装 - sherpa-onnx zipformer KWS

模型: sherpa-onnx-kws-zipformer-wenetspeech-3.3M（纯中文，拼音声母+带调韵母建模）
关键词自定义无需训练：中文词经 sherpa-onnx-cli text2token 转成音素序列写入
keywords.txt 即可（换词秒级生效）。

用法:
    from kws import WakeWordDetector
    det = WakeWordDetector(model_dir=..., keywords=["你好伙伴"])
    hit = det.process_block(audio_float32_block)   # 返回命中的词或 None

    # 离线测试:
    python3 kws.py --keyword 小爱同学 --wav <模型目录>/test_wavs/0.wav
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import sherpa_onnx
import wave

# 与 mic_loop 一致的块大小（16kHz / 512样本 = 32ms）
VAD_BLOCK = 512
SAMPLE_RATE = 16000

# 模型默认目录（与 sense-voice 同级）
DEFAULT_MODEL_DIR = Path(__file__).parent.parent.parent / 'models' / 'kws' / \
    'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01'

TEXT2TOKEN = 'sherpa-onnx-cli'   # 依赖本机CLI做中文→拼音音素转换


class WakeWordDetector:
    """唤醒词检测器（有状态，逐块喂音频；命中后内部自动重置继续检测）"""

    def __init__(
        self,
        model_dir: str = str(DEFAULT_MODEL_DIR),
        keywords: List[str] = None,
        keywords_threshold: float = 0.25,
        keywords_score: float = 1.0,
        keyword_scores: dict = None,
        num_threads: int = 1,
        debug: bool = False,
    ):
        """
        Args:
            model_dir: KWS模型目录（含 encoder/decoder/joiner/tokens.txt）
            keywords: 唤醒词列表（中文，如 ["你好伙伴"]）
            keywords_threshold: 命中阈值，越低越灵敏（误唤醒↑），默认0.25
            keywords_score: 关键词路径全局加分，越高越容易压过普通路径
            keyword_scores: 逐词加分（{"词": 分数}，覆盖全局值）——用于
                混合词表：稳的词高分提召回，易误触发的词低分配平衡
            num_threads: 推理线程数（IDLE常驻，1线程够用）
            debug: 打印音素文件内容等调试信息
        """
        model_dir = Path(model_dir)
        if isinstance(keywords, str):
            # 字符串被迭代成单字，会生成 n ǐ @你 这样的单字关键词，
            # 误唤醒爆炸且行数校验恰好蒙混过关——必须在这里拦死
            raise TypeError(
                f'keywords 需为列表（如 ["你好小弟"]），不要传字符串: {keywords!r}')
        keywords = keywords or ["你好伙伴"]

        # 生成keywords.txt（音素序列 @原词；逐词加分写入":N"后缀）
        self.keywords_file = self._gen_keywords_file(
            model_dir, keywords, debug, keyword_scores or {})

        self.spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(model_dir / 'tokens.txt'),
            encoder=str(model_dir / 'encoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx'),
            decoder=str(model_dir / 'decoder-epoch-99-avg-1-chunk-16-left-64.int8.onnx'),
            joiner=str(model_dir / 'joiner-epoch-99-avg-1-chunk-16-left-64.int8.onnx'),
            keywords_file=self.keywords_file,
            keywords_threshold=keywords_threshold,
            keywords_score=keywords_score,
            num_threads=num_threads,
        )
        self.stream = self.spotter.create_stream()
        self.keywords = keywords

    @staticmethod
    def _gen_keywords_file(model_dir: Path, keywords: List[str], debug: bool,
                           keyword_scores: dict = None) -> str:
        """中文唤醒词 → 拼音音素keywords.txt（调用本机sherpa-onnx-cli text2token）

        生成到临时文件。wenetspeech模型用 --tokens-type ppinyin（无需lexicon）。
        逐词加分写为"词 :N @词"格式（text2token原样保留:N后缀）。
        """
        raw = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8')
        for kw in keywords:
            score = (keyword_scores or {}).get(kw)
            if score is not None:
                raw.write(f'{kw} :{score} @{kw}\n')
            else:
                raw.write(f'{kw} @{kw}\n')
        raw.close()

        out = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False, encoding='utf-8')
        out.close()

        cmd = [
            TEXT2TOKEN, 'text2token',
            '--tokens', str(model_dir / 'tokens.txt'),
            '--tokens-type', 'ppinyin',
            raw.name, out.name,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f'text2token 转换失败（需本机安装 {TEXT2TOKEN}）:\n{proc.stderr}')

        content = Path(out.name).read_text(encoding='utf-8').strip()
        if not content:
            raise RuntimeError(f'唤醒词转音素结果为空，请检查: {keywords}')
        if debug:
            print(f"📝 keywords.txt 内容:\n{content}")

        # 校验：空格分隔音素 + @原词
        lines = content.splitlines()
        if len(lines) != len(keywords):
            print(f"⚠️  转换行数({len(lines)})与唤醒词数({len(keywords)})不符，请人工核对")
        return out.name

    def process_block(self, block: np.ndarray) -> Optional[str]:
        """喂入一块音频（float32, 16kHz, 单声道），返回命中的唤醒词或None

        命中后stream已重置，可继续检测下一次唤醒。
        """
        self.stream.accept_waveform(SAMPLE_RATE, block)

        while self.spotter.is_ready(self.stream):
            self.spotter.decode_stream(self.stream)
            result = self.spotter.get_result(self.stream)
            if result:
                self.spotter.reset_stream(self.stream)
                return result
        return None

    def reset(self):
        """外部状态切换时清残留（静音窗口结束后调用，与VAD.reset同理）"""
        self.spotter.reset_stream(self.stream)

    def new_stream(self):
        """换全新stream（解码状态彻底归零）。实测某些语音后残留态会让
        reset_stream不够干净（R2.1踩坑：backchannel场景后停止词漏检），
        stream创建近乎零成本，每次播报开始时调用"""
        self.stream = self.spotter.create_stream()


def main():
    """离线测试：用wav文件回放验证检测器（含模型自带的test_wavs）"""
    parser = argparse.ArgumentParser(description='唤醒词检测离线测试')
    parser.add_argument('--model-dir', type=str, default=str(DEFAULT_MODEL_DIR))
    parser.add_argument('--keyword', action='append', default=None,
                        help='唤醒词（可多次指定）')
    parser.add_argument('--wav', type=str, required=True, help='测试音频文件')
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    keywords = args.keyword or ['小爱同学', '你好问问', '小艺小艺']
    print(f"🎯 唤醒词: {keywords}")
    print(f"🎧 测试音频: {args.wav}")

    det = WakeWordDetector(
        model_dir=args.model_dir,
        keywords=keywords,
        keywords_threshold=args.threshold,
        debug=args.debug,
    )

    # 读wav（单声道16bit；非16kHz自动重采样——方便用TTS生成的测试音频）
    with wave.open(args.wav, 'rb') as wf:
        assert wf.getnchannels() == 1, '需单声道'
        assert wf.getsampwidth() == 2, '需16-bit PCM'
        rate = wf.getframerate()
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    audio = audio.astype(np.float32) / 32768.0
    if rate != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / rate)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)), audio).astype(np.float32)
        print(f"🔄 已重采样 {rate}Hz → {SAMPLE_RATE}Hz")
    # 补1秒尾部静音：流式模型需要尾随空帧才输出假设（TTS生成的音频
    # 在末音素处截断，不补则永远不命中；真实麦克风流不存在此问题）
    audio = np.concatenate([audio, np.zeros(SAMPLE_RATE, dtype=np.float32)])
    print(f"📊 音频时长: {len(audio) / SAMPLE_RATE:.2f}s")

    # 逐块喂入（模拟实时流）
    hits = []
    for i in range(0, len(audio), VAD_BLOCK):
        hit = det.process_block(audio[i:i + VAD_BLOCK])
        if hit:
            t = i / SAMPLE_RATE
            print(f"✅ [{t:6.2f}s] 命中唤醒词: {hit}")
            hits.append((t, hit))

    if not hits:
        print("❌ 未检测到唤醒词（可试降低 --threshold）")
        sys.exit(1)


if __name__ == '__main__':
    main()
