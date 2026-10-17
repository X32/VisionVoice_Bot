#!/usr/bin/env python3
"""
智能语音对话助手
语音输入(VAD自动断句) → ASR识别 → LLM处理 → TTS语音播报

线程模型（参考ROS2回声门控方案）：
  - 主线程：mic_loop 持续读麦克风（VAD断句+ASR识别）
  - worker线程：单消费者队列，串行处理 LLM+TTS（绝不允许并发播报）
  - 回声门控：TTS播报前 mute()；播放结束后 unmute(tail) 开启总静音窗口，
    窗口内音频块不进VAD（回声从源头被丢弃），到期后 vad.reset() 清残留

用法:
    python3 voice_chat.py                          # 使用默认ollama
    python3 voice_chat.py --backend llama --url http://192.168.4.52:8080
    python3 voice_chat.py --no-tts                  # 仅文字显示，不播报
    python3 voice_chat.py --mute-tail 1.0           # 混响强的环境加大窗口
    python3 voice_chat.py --wake-word 你好伙伴     # 自定义唤醒词（默认已启用）
    python3 voice_chat.py --no-wake                 # 关闭唤醒，开放麦克风
"""
import signal

# ⚠️ 必须在一切重型库导入之前屏蔽SIGINT：sherpa_onnx/numpy/sounddevice在
# import阶段就会创建线程，晚于此的线程不继承掩码，会抢走SIGINT导致
# 看门狗（_start_exit_watchdog）收不到信号、进程杀不死（实测踩坑）
signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})

import argparse
import json
import os
import queue
import re
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

# 添加项目根目录到sys.path，方便导入其他模块
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src/llm'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/voice_input'))
sys.path.insert(0, str(PROJECT_ROOT / 'src/tts'))

from unified_llm import LLM
from vad_asr import SmartSpeakerASR
from test_tts import TTS
from player import ChunkedPlayer, resample_to_16k, trim_silence, with_primer

MUTE_TAIL = 0.8   # 播放结束后的总静音窗口（秒）：覆盖设备输出缓冲+混响+
                  # VAD静音检测；实测本机回声残留约0.7s，0.8留少量余量
MUTE_TAIL_INTERRUPTED = 0.3   # 被打断后的静音尾巴：短于正常值（用户等着继续说），
                              # abort()已清输出缓冲，只需覆盖残余混响
PRIMER_SILENCE = 0.5          # 播报前置牺牲静音：设备冷启动吃开头，让静音被吃
SAMPLE_RATE_16K = 16000       # 播放统一采样率（与mic/AEC一致）
MAX_TOOL_ROUNDS = 2   # 单次对话最多几轮工具调用（防模型无限要工具）
DEFAULT_WAKE_WORD = "你好小迪"   # 默认唤醒词（learnPartner）
WAKE_RESPONSE = "哎，我在"       # 唤醒应答语（启动时预合成，命中即播）


# ================= Function Calling 工具注册表 =================
# 新增工具三步：1)写函数 2)加schema 3)注册到TOOL_FUNCS（名字必须一致）

_CN_DIGITS = "零一二三四五六七八九"


def year_to_chinese(year: str) -> str:
    """年份逐位转汉字：2026 → 二零二六（TTS会把2026年读成"两千零二十六年"）"""
    return ''.join(_CN_DIGITS[int(c)] for c in year)


def tts_sanitize(text: str) -> str:
    """TTS播报前的文本清洗：4位数字年份转汉字读法"""
    return re.sub(r'(\d{4})年', lambda m: year_to_chinese(m.group(1)) + '年', text or '')


def get_local_time() -> str:
    """获取本地当前日期、时间和星期（返回JSON字符串供模型阅读）"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return json.dumps({
        # 年份直接给汉字读法，模型会跟随此格式（TTS不会读成"两千零二十六年"）
        "date": f"{year_to_chinese(now.strftime('%Y'))}年{now.strftime('%m月%d日')}",
        "time": now.strftime("%H:%M:%S"),
        "weekday": weekdays[now.weekday()],
    }, ensure_ascii=False)


def turn_on_air_conditioner() -> str:
    """打开空调（返回结果字符串，经模型转述后由TTS播报）"""
    # TODO: 在此接入真实设备控制（ROS2话题/红外编码/串口协议等）
    return "小度小度 --打开空调"


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_local_time",
            "description": "获取本地的当前日期、时间和星期几。凡用户询问时间、日期、星期都调用此工具，不要自己编造。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn_on_air_conditioner",
            "description": """打开空调。用户要求开空调时调用此工具。工具返回"小度小度  ，打开空调"，请直接使用这个文字回复用户，不要改写。""",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

TOOL_FUNCS = {
    "get_local_time": get_local_time,
    "turn_on_air_conditioner": turn_on_air_conditioner,
}


class VoiceChat:
    """智能语音对话助手"""

    def __init__(
        self,
        llm_config: dict,
        asr_config: dict,
        tts_config: dict,
        system_prompt: str,
        history_limit: int = 10,
        mute_tail: float = MUTE_TAIL,
        use_tools: bool = True,
        wake_words: list = None,
        wake_threshold: float = 0.25,
        active_timeout: float = 30.0,
        barge_in: bool = True,
        barge_threshold: float = 0.7,
        stop_words: list = None,
        barge_long_speech: float = 1.2,
        barge_kws_threshold: float = 0.25,
        debug: bool = False
    ):
        """
        初始化语音对话助手

        Args:
            llm_config: LLM配置参数
            asr_config: ASR配置参数
            tts_config: TTS配置参数
            system_prompt: 系统提示词
            history_limit: 对话历史保留轮数
            mute_tail: 混响尾巴时长（秒）
            use_tools: 是否启用function calling
            wake_words: 唤醒词列表（None=开放麦克风模式）
            wake_threshold: 唤醒词命中阈值
            active_timeout: 唤醒后无语音休眠时间（秒）
            barge_in: 启用打断检测（播报中插话→停播）
            barge_threshold: 长语音通道silero阈值
            stop_words: 停止词表（命中即打断；唤醒词自动并入）
            barge_long_speech: 连续语音超此秒数视为新问题打断
            barge_kws_threshold: 停止词KWS阈值
            debug: 调试模式
        """
        self.debug = debug
        self.mute_tail = mute_tail
        self.history_limit = history_limit
        self.use_tools = use_tools

        # 初始化LLM
        print("🧠 初始化大模型...")
        self.llm = LLM(**llm_config)

        # 模型预热（后台线程）：ollama懒加载，首次请求要数秒~十几秒把
        # 模型读进内存；趁ASR/TTS还在初始化时提前加载，首句对话不吃这笔
        # 延迟。失败不致命（服务没起时提示后继续，首个真实请求会再报错）
        def _warmup():
            try:
                t = self.llm.warmup()
                print(f"🔥 模型预热完成（{t:.1f}s，已常驻）")
            except Exception as e:
                print(f"⚠️  模型预热失败: {e}")
        threading.Thread(target=_warmup, daemon=True).start()

        # 初始化ASR（识别回调只递纸条，微秒级返回，不阻塞麦克风采集）
        print("🎤 初始化语音识别...")
        self.asr = SmartSpeakerASR(
            on_text=self.on_user_text,
            wake_words=wake_words,
            wake_threshold=wake_threshold,
            active_timeout=active_timeout,
            on_wake=self.on_wake,
            on_sleep=self.on_sleep,
            barge_in=barge_in,
            on_barge_in=self.on_barge_in,
            barge_threshold=barge_threshold,
            stop_words=stop_words,
            barge_long_speech=barge_long_speech,
            barge_kws_threshold=barge_kws_threshold,
            **asr_config,
            debug=debug
        )

        # 流式播放器（打断的停播执行者；R3流式TTS复用）
        self.player = ChunkedPlayer()
        self.interrupted = False

        # 初始化TTS（如果需要）
        self.tts = None
        if tts_config.get('enable', True):
            print("🔊 初始化语音合成...")
            try:
                self.tts = TTS(
                    model='matcha',
                    speed=tts_config.get('speed', 1.0),
                )
            except Exception as e:
                print(f"⚠️  TTS初始化失败: {e}")
                print("💡 将使用文字模式运行")
                self.tts = None

        # 唤醒应答音频预合成（命中唤醒词即播，避免现场合成的延迟）
        self.wake_audio = None
        if wake_words and self.tts:
            try:
                print("🔔 预合成唤醒应答...")
                result = self.tts.speak(WAKE_RESPONSE, play=False, verbose=False)
                self.wake_audio = (result['samples'], result['sample_rate'])
            except Exception as e:
                print(f"⚠️  唤醒应答预合成失败（唤醒后无提示音）: {e}")

        # 对话历史
        self.history = [
            {"role": "system", "content": system_prompt}
        ]

        # 文本队列：ASR线程 → worker线程（单消费者，串行播报）
        self.text_q = queue.Queue()

        print("✅ 语音助手初始化完成")

    def on_user_text(self, text: str):
        """用户语音识别回调（在ASR采集线程中调用）——只递纸条"""
        if text.strip():
            self.text_q.put(text)

    def on_wake(self, word: str):
        """唤醒回调（在mic_loop线程调用）：播预合成的应答短句

        与worker的TTS播报同样走回声门控（mute→播→unmute带尾巴），
        否则应答音会被自己识别成指令。走ChunkedPlayer统一播放路径。
        """
        if self.wake_audio is None:
            print("💡 请讲（无提示音）")
            return
        samples, sample_rate = self.wake_audio
        self.asr.mute()
        try:
            self.asr.set_playing(True)
            try:
                samples16 = with_primer(
                    trim_silence(resample_to_16k(samples, sample_rate),
                                 SAMPLE_RATE_16K),
                    SAMPLE_RATE_16K, PRIMER_SILENCE)
                self.player.play(samples16,
                                 on_chunk=self.asr.feed_reference)
            finally:
                self.asr.set_playing(False)
        finally:
            self.asr.unmute(tail=self.mute_tail)

    def on_sleep(self):
        """休眠回调（在mic_loop线程调用）：目前仅日志"""
        print("💤 说唤醒词可再次唤醒")

    def on_barge_in(self, reason: str):
        """打断回调（在mic_loop线程调用）：停播+置标记

        reason: 'stop_word:停'（停止词/唤醒词命中）或 'long_speech'
        （长语音=新问题）。player.stop()经abort()清输出缓冲立即静音
        （实测3ms内）；worker的play()随即返回False，据interrupted标记写历史。
        """
        print(f"\n[{time.time():.3f}] ✋ 打断({reason})，停止播报")
        self.interrupted = True
        self.player.stop()

    def worker(self):
        """工作线程：串行消费队列，处理 LLM+TTS（重活全在这里）"""
        while True:
            text = self.text_q.get()
            try:
                # LLM处理（机器人无声，不静音——用户此时插话可入队，下一轮处理）
                self.history.append({"role": "user", "content": text})
                print(f"👤 你说: {text}")
                print("🧠 大模型思考中...")
                resp = self._chat_with_tools()

                print(f"🤖 回答: {resp.text}")
                print(f"⏱️  耗时: {resp.elapsed:.2f}s")

                # TTS播报：门控开 → 合成 → 流式播放（可被打断）→ 门控关
                # 历史在播报后写入：被打断时加标记（模型能答"刚才说到哪"）
                assistant_text = resp.text
                interrupted = False
                if self.tts:
                    self.asr.mute()                       # ① 门控开（合成前）
                    try:
                        speak_text = tts_sanitize(resp.text)   # 年份等读法修正
                        print(f"[{time.time():.3f}] 🔊 语音播报中...")
                        result = self.tts.speak(speak_text, play=False,
                                                verbose=False)
                        # 垫头：0.5s牺牲静音——声卡power_save休眠后下次
                        # 播放开头~0.5s在唤醒codec，让静音被吃而非语音；
                        # 修剪：安全网，防个别合成结果带长静音
                        samples16 = resample_to_16k(result['samples'],
                                                    result['sample_rate'])
                        samples16 = with_primer(
                            trim_silence(samples16, SAMPLE_RATE_16K),
                            SAMPLE_RATE_16K, PRIMER_SILENCE)
                        self.interrupted = False
                        self.asr.set_playing(True)
                        try:
                            # ② 分块播放：每块回调far-end参考给AEC；
                            #    插话→on_barge_in→player.stop()→此处返回False
                            complete = self.player.play(
                                samples16, on_chunk=self.asr.feed_reference)
                        finally:
                            self.asr.set_playing(False)
                        if complete:
                            print(f"[{time.time():.3f}] 🔈 播放完成")
                        else:
                            interrupted = self.interrupted
                            print(f"[{time.time():.3f}] ✋ 播报被打断")
                    finally:
                        # ③ 门控关：保留静音窗口覆盖输出缓冲+混响。
                        #    被打断用短尾巴（abort已清缓冲，用户等着说话）
                        tail = MUTE_TAIL_INTERRUPTED if interrupted \
                            else self.mute_tail
                        self.asr.unmute(tail=tail)

                if interrupted:
                    assistant_text = resp.text + "（被用户打断）"
                self.history.append({"role": "assistant", "content": assistant_text})
                self._trim_history()

            except Exception as e:
                print(f"❌ 处理出错: {e}")
                if self.debug and hasattr(e, '__traceback__'):
                    import traceback
                    traceback.print_exc()

    def _chat_with_tools(self):
        """带function calling的对话回路

        模型请求工具 → 本地执行 → 结果以role=tool回传 → 模型生成最终回复。
        最终返回的一定是纯文本响应（tool_calls为空）。
        """
        tools = TOOL_SCHEMAS if self.use_tools else None
        for _ in range(MAX_TOOL_ROUNDS):
            # 工具轮用较低温度：高温下qwen偶发把工具模板碎片吐成乱码文本
            resp = self.llm.chat(self.history, tools=tools, temperature=0.3)

            # 乱码防御：极小概率采出无中文/字母/数字的碎片（如":';"），降温重试一次
            if not resp.tool_calls and self._looks_garbled(resp.text):
                print(f"⚠️  响应异常({resp.text!r})，降温重试...")
                resp = self.llm.chat(self.history, tools=tools, temperature=0.2)

            if not resp.tool_calls:
                return resp

            # 模型决定调用工具：把决议写回历史（可回放格式）
            replay = [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in resp.tool_calls
            ]
            self.history.append({"role": "assistant", "content": resp.text or "",
                                 "tool_calls": replay})

            # 逐个执行并回传结果
            for tc in resp.tool_calls:
                result = self._exec_tool(tc["name"], tc["arguments"])
                self.history.append({"role": "tool", "name": tc["name"],
                                     "content": result})

        # 达到轮数上限：最后一轮不给工具，强制文本收尾
        return self.llm.chat(self.history)

    @staticmethod
    def _looks_garbled(text: str) -> bool:
        """判断响应是否为乱码：不含任何中文/字母/数字视为乱码"""
        return not re.search(r'[一-鿿A-Za-z0-9]', text or '')

    def _exec_tool(self, name: str, arguments: dict) -> str:
        """执行工具并返回JSON字符串结果（异常也包装为JSON，模型可读）"""
        func = TOOL_FUNCS.get(name)
        if func is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = func(**(arguments or {}))
            if not isinstance(result, str):
                result = json.dumps(result, ensure_ascii=False)
            print(f"🔧 工具调用: {name}({arguments or {}}) → {result}")
            return result
        except Exception as e:
            print(f"⚠️  工具 {name} 执行失败: {e}")
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    def _trim_history(self):
        """修剪对话历史，保留system + 最近N轮"""
        non_system = [msg for msg in self.history if msg.get("role") != "system"]
        if len(non_system) > self.history_limit:
            for msg in self.history:
                if msg.get("role") != "system":
                    self.history.remove(msg)
                    if len([m for m in self.history if m.get("role") != "system"]) <= self.history_limit:
                        break

            if self.debug:
                print(f"📝 历史已修剪到 {len(self.history)} 条")

    def run(self):
        """启动语音对话循环"""
        print("\n" + "=" * 60)
        print("🎯 智能语音对话助手")
        print("=" * 60)
        if self.asr.wake is not None:
            print(f"🔔 说唤醒词唤醒: {' / '.join(self.asr.wake.keywords)}")
            print(f"💤 {self.asr.active_timeout:.0f}秒无语音自动休眠")
        else:
            print("💡 直接说话即可，助手会自动识别、思考并语音回复")
        print("🛑 按 Ctrl+C 退出")
        print("=" * 60)
        print()

        self._start_exit_watchdog()

        # worker线程：串行处理LLM+TTS（daemon，随进程退出）
        threading.Thread(target=self.worker, daemon=True).start()

        try:
            # 主线程：ASR麦克风循环
            self.asr.mic_loop()

        except KeyboardInterrupt:
            # 正常情况下到不了这里（SIGINT被watchdog接走），仅作兜底
            print("\n\n🛑 语音助手已停止")
            sys.stdout.flush()
            os._exit(0)

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            import traceback
            traceback.print_exc()

    def _start_exit_watchdog(self):
        """SIGINT看门狗：保证Ctrl+C必定能杀死进程

        实测（faulthandler栈）：播放期间输入流的阻塞读（Pa_ReadStream）
        会间歇性停滞，主线程卡在C调用里永不返回，KeyboardInterrupt
        根本没有抛出机会；且退出路径中调用sd.stop()会与worker的
        sd.wait()在C层互锁。因此：
        1) 主线程屏蔽SIGINT（后续创建的线程全部继承此掩码）
        2) 看门狗线程用sigwait原子接住SIGINT，直接os._exit(0)硬退出
        进程终止时ALSA设备由内核关闭，正在播放的音频立即停止。
        """
        def _watch():
            while True:
                signal.sigwait([signal.SIGINT])
                print("\n\n🛑 语音助手已停止")
                sys.stdout.flush()
                os._exit(0)

        threading.Thread(target=_watch, daemon=True).start()


def main():
    # SIGINT已在模块顶部屏蔽（早于所有会创建线程的import），此处无需重复
    parser = argparse.ArgumentParser(
        description='智能语音对话助手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 使用默认ollama
  python3 voice_chat.py

  # 使用远程llama.cpp
  python3 voice_chat.py --backend llama --url http://192.168.4.52:8080

  # 仅文字显示，不语音播报
  python3 voice_chat.py --no-tts

  # 混响强的环境加大回声门控尾巴
  python3 voice_chat.py --mute-tail 0.5

  # 自定义系统提示词和历史轮数
  python3 voice_chat.py --system "你是编程助手" --history 5
        '''
    )

    # LLM参数
    parser.add_argument('--backend', '-b', type=str, default='ollama',
                       choices=['llama', 'ollama', 'openai'],
                       help='大模型后端（默认: ollama）')
    parser.add_argument('--url', '-u', type=str,
                       help='服务地址')
    parser.add_argument('--model', '-m', type=str,
                       help='模型名称')
    parser.add_argument('--api-key', '-k', type=str,
                       help='API密钥（OpenAI兼容API）')
    parser.add_argument('--system', '-s', type=str,
                       default="你是一个语音助手，回答简洁口语化，控制在两三句话内，适合语音播报",
                       help='系统提示词')
    parser.add_argument('--max-tokens', '-n', type=int, default=256,
                       help='最大生成token数（默认256）')
    parser.add_argument('--temperature', '-t', type=float, default=0.7,
                       help='温度参数')
    parser.add_argument('--history', type=int, default=10,
                       help='对话历史保留轮数（默认10）')

    # TTS参数
    parser.add_argument('--no-tts', action='store_true',
                       help='禁用TTS语音播报，仅显示文字')
    parser.add_argument('--tts-speed', type=float, default=1.0,
                       help='TTS语速（默认1.0）')
    parser.add_argument('--mute-tail', type=float, default=MUTE_TAIL,
                       help=f'播放结束后总静音窗口（秒，默认{MUTE_TAIL}；混响强的房间可加大）')
    parser.add_argument('--no-tools', action='store_true',
                       help='禁用function calling（纯对话模式）')

    # 唤醒词参数
    parser.add_argument('--wake-word', '-w', type=str, action='append',
                       metavar='WORD',
                       help=f'唤醒词（可多次指定多个；默认 "{DEFAULT_WAKE_WORD}"）')
    parser.add_argument('--no-wake', action='store_true',
                       help='关闭唤醒词，开放麦克风模式（任何语音都触发识别）')
    parser.add_argument('--wake-threshold', type=float, default=0.25,
                       help='唤醒词命中阈值，越低越灵敏误唤醒越高（默认0.25）')
    parser.add_argument('--active-timeout', type=float, default=30.0,
                       help='唤醒后无语音自动休眠时间（秒，默认30）')

    # 打断参数
    parser.add_argument('--no-barge-in', action='store_true',
                       help='禁用打断检测（播报期间插话无效，回到纯门控）')
    parser.add_argument('--barge-threshold', type=float, default=0.7,
                       help='长语音通道silero阈值（默认0.7，离线实测误报最优）')
    parser.add_argument('--stop-words', type=str, action='append',
                       metavar='WORD',
                       help='停止词（可多次指定；默认 不要说了/闭嘴，唤醒词自动并入；'
                            '单词如"停"KWS检不出，请用三音节以上）')
    parser.add_argument('--barge-long-speech', type=float, default=1.2,
                       help='连续语音超过此秒数视为新问题打断（默认1.2）')
    parser.add_argument('--barge-kws-threshold', type=float, default=0.25,
                       help='停止词KWS阈值（"停"误报多时调高，默认0.25）')

    # ASR参数（透传）
    parser.add_argument('--vad-duration', type=float, default=8.0,
                       help='单段最大录音时长（秒）')
    parser.add_argument('--vad-silence', type=float, default=0.5,
                       help='静音多久后断句（秒）')
    parser.add_argument('--vad-threshold', type=float, default=0.5,
                       help='VAD检测阈值')
    parser.add_argument('--vad-debug', action='store_true',
                       help='ASR调试模式')

    args = parser.parse_args()

    # 准备配置
    llm_config = {
        "backend": args.backend,
        "base_url": args.url,
        "model": args.model,
        "api_key": args.api_key,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }

    asr_config = {
        "sensevoice_model_dir": "/mnt/apfs/workplace/ROS2/models/asr/sense-voice",
        "vad_model_path": "/mnt/apfs/workplace/ROS2/models/asr/silero_vad.onnx",
        "max_duration": args.vad_duration,
        "silence_duration": args.vad_silence,
        "vad_threshold": args.vad_threshold,
    }

    tts_config = {
        "enable": not args.no_tts,
        "speed": args.tts_speed,
    }

    print("=" * 60)
    print("🎙️ 智能语音对话助手")
    print("=" * 60)

    # 创建语音助手（--no-wake回到开放麦克风；否则默认启用唤醒词）
    wake_words = None if args.no_wake else (args.wake_word or [DEFAULT_WAKE_WORD])
    voice_chat = VoiceChat(
        llm_config=llm_config,
        asr_config=asr_config,
        tts_config=tts_config,
        system_prompt=args.system,
        history_limit=args.history,
        mute_tail=args.mute_tail,
        use_tools=not args.no_tools,
        wake_words=wake_words,
        wake_threshold=args.wake_threshold,
        active_timeout=args.active_timeout,
        barge_in=not args.no_barge_in,
        barge_threshold=args.barge_threshold,
        stop_words=args.stop_words,
        barge_long_speech=args.barge_long_speech,
        barge_kws_threshold=args.barge_kws_threshold,
        debug=args.vad_debug
    )

    # 启动对话循环
    voice_chat.run()


if __name__ == "__main__":
    main()
