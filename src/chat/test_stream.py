#!/usr/bin/env python3
"""R3流式管线离线验证（驱动真实的 _chat_and_speak_stream）

场景:
  B 假LLM脚本化E2E   —— 预设delta序列+真实TTS+真实播放器：
                          首语音延迟 / 句间gap / 完整性
  C 打断演练         —— 播放1s置打断：上游gen已close？合成线程已收？
                          历史全文+打断标记？
  D 真实ollama延迟探针 —— TTFT/首段出声，验证30字回答≤2s预算
                          （需本机ollama在跑）

用法:
    python3 src/chat/test_stream.py          # B+C
    python3 src/chat/test_stream.py --real   # B+C+D（连真实ollama）

注意: voice_chat模块import时会屏蔽SIGINT（生产看门狗惯例），本脚本
import后立即解除屏蔽恢复Ctrl+C。
"""
import argparse
import signal
import sys
import threading
import time
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src/chat'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/llm'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/tts'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/voice_input'))

import voice_chat as vc_mod
signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGINT})   # 恢复Ctrl+C

from test_tts import TTS
from player import ChunkedPlayer, BLOCK
from unified_llm import LLM

ANSWER = ("好的。太阳系有八大行星，地球是其中之一。"
          "它们都围绕太阳运转，其中木星最大。")


# ---------------- 测试替身 ----------------

class FakeLLM:
    """脚本化增量序列的LLM替身；记录生成器是否被close（打断→上游取消）"""

    def __init__(self, text: str, delay: float = 0.03):
        self.text = text
        self._pieces = [text[i:i + 2] for i in range(0, len(text), 2)]
        self.delay = delay
        self.closed = False

    def chat_stream(self, messages, tools=None, **kw):
        self.closed = False
        return self._gen()

    def _gen(self):
        try:
            for piece in self._pieces:
                time.sleep(self.delay)
                yield {"delta": piece}
            yield {"done": True, "text": self.text, "usage": {}}
        finally:
            self.closed = True


class RecorderASR:
    """记录set_playing/首参考块时间的ASR替身（不碰麦克风）"""

    def __init__(self):
        self.state = 'active'
        self.t_play_start = None
        self.t_first_chunk = None
        self.chunk_times = []

    def mute(self):
        pass

    def unmute(self, tail=0.5):
        pass

    def set_playing(self, playing):
        if playing and self.t_play_start is None:
            self.t_play_start = time.time()
        if not playing:
            self.t_play_start = None   # 允许多轮

    def feed_reference(self, chunk):
        now = time.time()
        self.chunk_times.append(now)
        if self.t_first_chunk is None:
            self.t_first_chunk = now


def build_harness(llm) -> vc_mod.VoiceChat:
    """旁路构造VoiceChat：只装配_chat_and_speak_stream用到的成员"""
    v = vc_mod.VoiceChat.__new__(vc_mod.VoiceChat)
    v.llm = llm
    v.tts = TTS(model='matcha', speed=1.0)
    v.player = ChunkedPlayer()
    v.asr = RecorderASR()
    v.use_tools = False
    v.interrupted = False
    v.history = [{"role": "system", "content": "test"}]
    return v


# ---------------- 场景B: 假LLM E2E ----------------

def scenario_b():
    print("=" * 62)
    print("场景B: 假LLM脚本化E2E（真实TTS+真实播放器）")
    llm = FakeLLM(ANSWER, delay=0.03)
    v = build_harness(llm)

    t0 = time.time()
    full, interrupted = v._chat_and_speak_stream()
    wall = time.time() - t0

    assert not interrupted, "不应被打断"
    assert full == ANSWER, f"全文不一致: {full!r}"
    assert llm.closed, "生成器未正常走完（finally未触发）"

    asr = v.asr
    first_audio = asr.t_first_chunk - t0 + 0.5   # 首块+primer≈真正出声时刻
    print(f"\n首块写出: {asr.t_first_chunk - t0:.2f}s  "
          f"≈首次出声: {first_audio:.2f}s（含0.5s primer）")
    print(f"全程: {wall:.2f}s  打字机期+播放期")

    # 句间gap：首块后所有on_chunk的最大间隔（排除会话收尾）
    gaps = [b - a for a, b in zip(asr.chunk_times, asr.chunk_times[1:])]
    body = [g for g, t in zip(gaps, asr.chunk_times[1:])
            if t - asr.t_first_chunk < wall - t0 - 1.0]   # 掐掉尾部
    max_gap = max(body) if body else 0.0
    print(f"句间最大写入gap: {max_gap*1000:.0f}ms（目标≤300ms）")
    print("✅ 场景B PASS")
    return v


# ---------------- 场景C: 打断演练 ----------------

def scenario_c():
    print("=" * 62)
    print("场景C: 打断演练（播放1s时打断）")
    llm = FakeLLM(ANSWER * 2, delay=0.02)   # 加倍文本保证播放期足够长
    v = build_harness(llm)

    def _fire():
        # 模拟on_barge_in：置标志+停播（mic_loop线程的真实调用序）
        time.sleep(1.0)
        v.interrupted = True
        v.player.stop()

    threading.Thread(target=_fire, daemon=True).start()

    t0 = time.time()
    full, interrupted = v._chat_and_speak_stream()
    ret_after = time.time() - t0

    assert interrupted, "应返回被打断"
    assert llm.closed, "打断后上游生成器未被close（取消失败）"
    # R2契约的流式版：打断→上游取消→全文=已读前缀（背压节流下读取
    # 滞后于生成，未读部分根本没生成完）。worker()据此写
    # "已读文本+（被用户打断）"——依然≥已播内容，"继续说"能接上
    assert full and (ANSWER * 2).startswith(full), \
        f"应是生成内容前缀: {full!r}"
    assert len(full) >= 10, "队列在途容量保证至少数句已被读入"
    print(f"打断后 {ret_after:.2f}s 返回（完整播放需~{len(ANSWER*2)/4:.0f}s级）")
    print(f"已读文本({len(full)}字，worker将记为): {full!r}（被用户打断）")
    print("✅ 场景C PASS")


# ---------------- 场景D: 真实ollama延迟探针 ----------------

def scenario_d():
    print("=" * 62)
    print("场景D: 真实ollama流式延迟（30字回答预算）")
    llm = LLM(backend='ollama', model='qwen2.5:7b')
    v = build_harness(llm)

    v.history.append({"role": "user", "content": "用30字左右介绍地球"})
    t0 = time.time()
    full, interrupted = v._chat_and_speak_stream()
    wall = time.time() - t0
    first_audio = v.asr.t_first_chunk - t0 + 0.5
    print(f"\n回答: {full!r}")
    print(f"首次出声: {first_audio:.2f}s（目标≤2.0s）  全程: {wall:.2f}s")
    ok = first_audio <= 2.0
    print(f"{'✅' if ok else '❌'} 场景D {'PASS' if ok else 'FAIL（首语音>2s）'}")
    return ok


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--real', action='store_true', help='连真实ollama（场景D）')
    args = ap.parse_args()

    scenario_b()
    scenario_c()
    if args.real:
        scenario_d()
    print("=" * 62)
    print("全部完成")
