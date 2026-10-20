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
# 待机口令：ACTIVE状态识别文本包含任一短语 → 不进LLM，直接回待机（需重新唤醒）。
# 短语要求"动词+宾语"完整（如"进入待机"），避免"什么是待机模式"类提问误触发
STANDBY_PHRASES = ("进入待机模式", "进入待机")
STANDBY_RESPONSE = "好的，进入待机模式"   # 待机确认语（启动时预合成）


# ================= Function Calling 工具注册表 =================
# 新增工具三步：1)写函数 2)加schema 3)注册到TOOL_FUNCS（名字必须一致）

# VoiceChat单例引用：模块级工具（需要实例状态者，如倒计时）的入口
_ACTIVE_CHAT = None

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


# Open-Meteo WMO天气码 → 中文（覆盖常见码，未知码给数值让模型自行描述）
WMO_WEATHER_ZH = {
    0: "晴", 1: "大致晴", 2: "局部多云", 3: "阴",
    45: "雾", 48: "冻雾",
    51: "毛毛雨", 53: "中等毛毛雨", 55: "浓毛毛雨",
    61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "强冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒",
    80: "小阵雨", 81: "阵雨", 82: "强阵雨",
    85: "小阵雪", 86: "阵雪",
    95: "雷雨", 96: "雷雨伴冰雹", 99: "强雷雨伴冰雹",
}


def get_weather(city: str) -> str:
    """查询城市实时天气与今明两天预报（Open-Meteo，免Key）

    两段请求：中文城市名→经纬度（geocoding API）→ 天气（forecast API）。
    网络异常/找不到城市时返回带error的JSON——模型会转为口语告知用户，
    而不是让worker的异常处理兜底（那样只有一句冷冰冰的"处理出错"）。
    """
    import urllib.request
    from urllib.parse import quote

    def _get_json(url: str, timeout: float = 6.0) -> dict:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    try:
        geo = _get_json(
            f"https://geocoding-api.open-meteo.com/v1/search"
            f"?name={quote(city)}&count=1&language=zh")
        results = geo.get("results") or []
        if not results:
            return json.dumps({"error": f"找不到城市: {city}"},
                              ensure_ascii=False)
        loc = results[0]

        w = _get_json(
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={loc['latitude']}&longitude={loc['longitude']}"
            f"&current=temperature_2m,weather_code,relative_humidity_2m,"
            f"wind_speed_10m&wind_speed_unit=ms"
            f"&daily=temperature_2m_max,temperature_2m_min,weather_code"
            f"&timezone=auto&forecast_days=2")
    except Exception as e:
        return json.dumps({"error": f"天气服务连接失败: {type(e).__name__}"},
                          ensure_ascii=False)

    c, d = w["current"], w["daily"]
    # 地名拼接去重：北京/北京市→"北京"；南山/广东→"广东南山"
    region = loc.get('admin1') or ''
    name = loc['name']
    label = f"{region}{name}" if region and name not in region else name
    return json.dumps({
        "city": label,
        "current": {
            "weather": WMO_WEATHER_ZH.get(c["weather_code"], "未知"),
            "temperature": round(c["temperature_2m"]),   # 取整：口语友好
            "humidity": c["relative_humidity_2m"],
            "wind_speed_ms": c["wind_speed_10m"],
        },
        "today": {
            "weather": WMO_WEATHER_ZH.get(d["weather_code"][0], "未知"),
            "temp_max": round(d["temperature_2m_max"][0]),
            "temp_min": round(d["temperature_2m_min"][0]),
        },
        "tomorrow": {
            "weather": WMO_WEATHER_ZH.get(d["weather_code"][1], "未知"),
            "temp_max": round(d["temperature_2m_max"][1]),
            "temp_min": round(d["temperature_2m_min"][1]),
        },
    }, ensure_ascii=False)


def start_countdown(minutes: float) -> str:
    """设置倒计时（独立线程计时播报，不阻塞对话；入口=VoiceChat单例）"""
    vc = _ACTIVE_CHAT
    if vc is None:
        return json.dumps({"error": "语音助手未就绪"}, ensure_ascii=False)
    return json.dumps(vc.start_countdown(minutes), ensure_ascii=False)


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
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的实时天气和今明两天预报（气温、天气现象、湿度、风力）。凡用户询问天气都调用此工具，不要自己编造。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，支持中文，如：北京、上海、深圳",
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_countdown",
            "description": "设置倒计时/定时提醒。用户说\"倒计时30分钟\"\"定时15分钟\"\"一分钟后提醒我\"时调用。时长统一换算为分钟后传入（90秒=1.5，半小时=30，1分30秒=1.5，2小时=120）。到点前10秒开始逐秒读秒，到点后语音提醒三遍；设定后对话照常进行，提醒声可被语音打断。",
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "number",
                        "description": "倒计时总时长（分钟），支持小数，如 30、1.5",
                    },
                },
                "required": ["minutes"],
            },
        },
    },
]

TOOL_FUNCS = {
    "get_local_time": get_local_time,
    "turn_on_air_conditioner": turn_on_air_conditioner,
    "get_weather": get_weather,
    "start_countdown": start_countdown,
}


class SentenceSplitter:
    """LLM增量文本 → 整句流出（R3按句增量合成）

    边界规则：句末标点（。！？!?；;\n）必切；逗号（，,）攒够缓冲再切
    （首句尽快出声，短逗号句不碎）；无标点攒满强切（列表/代码兜底）。
    纯符号/空白段丢弃（markdown列表符、分割线等不合成）。
    """

    HARD_PUNCT = '。！？!?；;\n'
    SOFT_PUNCT = '，,'
    COMMA_FLUSH = 10   # 逗号切分的最小缓冲字数
    MAX_FLUSH = 60     # 无标点强切阈值

    def __init__(self):
        self._buf = ''

    def feed(self, delta: str) -> list:
        """喂增量文本，返回切出的完整句列表"""
        self._buf += delta
        out = []
        while True:
            sent, self._buf = self._cut(self._buf)
            if sent is None:
                break
            sent = self._clean(sent)
            if sent:
                out.append(sent)
        return out

    def finish(self) -> list:
        """流结束：吐出剩余缓冲（尾句）"""
        rest, self._buf = self._buf, ''
        rest = self._clean(rest)
        return [rest] if rest else []

    def _cut(self, buf: str):
        """找下一个切点。返回(句子, 余buffer)；无切点返回(None, buf)"""
        for i, ch in enumerate(buf):            # 句末标点：最优先
            if ch in self.HARD_PUNCT:
                return buf[:i + 1], buf[i + 1:]
        if len(buf) >= self.COMMA_FLUSH:        # 逗号：攒够缓冲才切
            for i, ch in enumerate(buf):
                if ch in self.SOFT_PUNCT:
                    return buf[:i + 1], buf[i + 1:]
        if len(buf) >= self.MAX_FLUSH:          # 无标点强切
            return buf, ''
        return None, buf

    @staticmethod
    def _clean(s: str) -> str:
        """剥离markdown符号/多余空白；不含中文/字母/数字的段视为符号丢弃"""
        s = re.sub(r'[#*>`\[\]_~|]+', '', s)
        s = re.sub(r'\s+', ' ', s).strip(' 　·—-…')
        return s if re.search(r'[一-鿿A-Za-z0-9]', s) else ''


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
        standby_phrases: list = None,
        stream_tts: bool = True,
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
            standby_phrases: 待机口令列表（识别文本包含即回待机；None用默认）
            stream_tts: 启用R3流式管线（LLM增量→按句合成→边合成边播；
                False回退全量生成后播放）
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
        self.stream_tts = stream_tts
        self.standby_phrases = tuple(standby_phrases) if standby_phrases \
            else STANDBY_PHRASES

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

        # 唤醒应答/待机确认音频预合成（命中即播，避免现场合成的延迟）
        self.wake_audio = None
        self.standby_audio = None
        if wake_words and self.tts:
            try:
                print("🔔 预合成唤醒应答...")
                result = self.tts.speak(WAKE_RESPONSE, play=False, verbose=False)
                self.wake_audio = (result['samples'], result['sample_rate'])
            except Exception as e:
                print(f"⚠️  唤醒应答预合成失败（唤醒后无提示音）: {e}")
            try:
                print("🔇 预合成待机确认...")
                result = self.tts.speak(STANDBY_RESPONSE, play=False, verbose=False)
                self.standby_audio = (result['samples'], result['sample_rate'])
            except Exception as e:
                print(f"⚠️  待机确认预合成失败（待机时无提示音）: {e}")

        # 倒计时（工具设定，独立线程播报，可被语音打断）
        self._cd_cancel = threading.Event()
        # 门控会话互斥锁：worker播报 ⇄ 倒计时播报。两个"mute→播放→
        # unmute"会话交叠会互相拆set_playing/unmute状态（打断检测失效、
        # 回声漏进主VAD），必须整会话串行。_play_gated（mic_loop线程的
        # 唤醒/待机短句）不参与——mic_loop绝不能阻塞等锁
        self._gate_lock = threading.Lock()
        # 模块级工具（start_countdown）没有实例入口，注册单例引用
        global _ACTIVE_CHAT
        _ACTIVE_CHAT = self

        # 对话历史
        self.history = [
            {"role": "system", "content": system_prompt}
        ]

        # 文本队列：ASR线程 → worker线程（单消费者，串行播报）
        self.text_q = queue.Queue()

        print("✅ 语音助手初始化完成")

    def on_user_text(self, text: str):
        """用户语音识别回调（在ASR采集线程中调用）——只递纸条

        待机口令例外：不进LLM队列，直接回待机并播确认音。
        """
        if not text.strip():
            return
        # 仅唤醒模式有待机概念（开放麦克风无状态机，待机即失聪）
        if self.asr.wake is not None and self._is_standby_command(text):
            self._enter_standby_command(text)
            return
        self.text_q.put(text)

    def _is_standby_command(self, text: str) -> bool:
        """归一化后匹配待机口令（ASR可能带标点/空格）"""
        norm = re.sub(r'[，。！？、,.!?\s]', '', text)
        return any(p in norm for p in self.standby_phrases)

    def _enter_standby_command(self, text: str):
        """待机口令处理（在mic_loop线程调用）：立即回IDLE+播确认音

        先切状态再播放：确认音走回声门控（同唤醒应答），播完unmute
        后mic_loop已处于idle，直接监听唤醒词。与worker的竞态由worker
        侧的state=='idle'守卫兜底（LLM思考期间说口令→放弃播报）。
        """
        print(f"\n🔇 口令待机: {text}")
        self.asr.enter_standby()
        if self.standby_audio is not None:
            self._play_gated(self.standby_audio)

    def _play_gated(self, audio: tuple, primer: float = PRIMER_SILENCE) -> bool:
        """回声门控播放预合成短句（在mic_loop线程调用）

        唤醒应答/待机确认/倒计时读秒共用：与worker的TTS播报同样走回声
        门控（mute→垫静音播放→unmute带尾巴），否则提示音会被自己识别成
        指令。返回是否完整播完（False=被stop打断/写停滞——倒计时据此
        中止后续播报）。primer=0用于连续短句序列（倒计时读秒，只有首句
        需要牺牲静音防codec吃字）。
        """
        samples, sample_rate = audio
        self.asr.mute()
        try:
            self.asr.set_playing(True)
            try:
                samples16 = with_primer(
                    trim_silence(resample_to_16k(samples, sample_rate),
                                 SAMPLE_RATE_16K),
                    SAMPLE_RATE_16K, primer)
                return self.player.play(samples16,
                                        on_chunk=self.asr.feed_reference)
            finally:
                self.asr.set_playing(False)
        finally:
            self.asr.unmute(tail=self.mute_tail)

    # ================= 倒计时（工具设定，独立线程播报） =================

    def start_countdown(self, minutes: float) -> dict:
        """设置倒计时并立即返回（worker线程/工具调用中执行）

        语音片段在这里预合成（一次性约2~4s）：必须在worker线程做——
        sherpa OfflineTts单实例，与对话TTS并发调用不安全；这点阻塞只
        推迟模型的确认语，倒计时从设定时刻起算不受影响。
        重复设定=取消上一个（新替旧，旧播报在当前片段结束后停止）。
        """
        minutes = max(0.25, min(float(minutes), 1440))   # 15s ~ 24h
        total = round(minutes * 60)
        m, s = divmod(total, 60)
        phrase = f"{m}分{s}秒" if m and s else (f"{m}分钟" if m else f"{s}秒")

        clips = {}
        if self.tts:
            texts = {"lead": "倒计时最后10秒。10",
                     "done": f"{phrase}时间已到"}
            texts.update({str(sec): t for sec, t in
                          zip(range(9, 0, -1), "九八七六五四三二一")})
            try:
                for key, text in texts.items():
                    r = self.tts.speak(text, play=False, verbose=False)
                    clips[key] = (r['samples'], r['sample_rate'])
            except Exception as e:
                return {"error": f"倒计时语音合成失败: {e}"}

        self._cd_cancel.set()                 # 取消上一个
        self._cd_cancel = threading.Event()
        deadline = time.time() + total
        threading.Thread(target=self._countdown_run,
                         args=(clips, deadline, self._cd_cancel, phrase),
                         daemon=True, name="countdown").start()
        print(f"⏲️  倒计时已设定: {phrase}")
        return {"status": "已设定", "duration": phrase,
                "behavior": "到点前10秒读秒，到点后提醒三遍；播报可语音打断"}

    def _countdown_run(self, clips, deadline, cancel, phrase):
        """倒计时播报线程：静默等待→读秒→三遍提醒

        读秒对齐墙钟（每个数字卡在自己的整秒边界，播报晚了自己追上）；
        播报用一段连续的回声门控会话，每个片段play()返回False（被打断）
        即中止后续；与worker抢player由player._lock串行（极端时读秒稍晚，
        可接受）。cancel=被新设定替代。
        """
        def _wait_until(t: float) -> bool:
            """分片等待到时刻t；被取消返回False"""
            while True:
                if cancel.is_set():
                    return False
                now = time.time()
                if now >= t:
                    return True
                cancel.wait(timeout=min(1.0, t - now))

        if not _wait_until(deadline - 10):
            print("⏲️  倒计时已取消（被新设定替代）")
            return

        if not clips:   # 文字模式（无TTS）：到点打印三遍
            if _wait_until(deadline):
                for i in range(3):
                    print(f"⏰ {phrase}时间已到 ({i + 1}/3)")
                    time.sleep(1.0)
            return

        print(f"⏲️  倒计时读秒开始（{phrase}）")
        # gate_lock与worker播报整会话互斥：对话在播时到点，读秒等对话
        # 收尾后开始，数字按墙钟边界追平对齐（晚开始不乱节奏）
        with self._gate_lock:
            self.asr.mute()
            complete = True
            try:
                self.asr.set_playing(True)
                try:
                    def _play(clip, primer=0.0):
                        samples16 = trim_silence(
                            resample_to_16k(clip[0], clip[1]), SAMPLE_RATE_16K)
                        if primer > 0:
                            samples16 = with_primer(samples16, SAMPLE_RATE_16K,
                                                    primer)
                        return self.player.play(
                            samples16, on_chunk=self.asr.feed_reference)

                    complete = _play(clips["lead"], primer=PRIMER_SILENCE)
                    for sec in range(9, 0, -1):   # 九八…一，各卡整秒边界
                        if not complete:
                            break
                        if not _wait_until(deadline - sec):
                            return
                        complete = _play(clips[str(sec)])
                    for i in range(3):            # 到点提醒×3，间隔2s
                        if not complete:
                            break
                        if not _wait_until(deadline + 0.2 + i * 2.0):
                            return
                        complete = _play(clips["done"])
                    print(f"⏰ {'提醒完成' if complete else '提醒被打断'}（{phrase}）")
                finally:
                    self.asr.set_playing(False)
            finally:
                self.asr.unmute(tail=MUTE_TAIL_INTERRUPTED if not complete
                                else self.mute_tail)

    def on_wake(self, word: str):
        """唤醒回调（在mic_loop线程调用）：播预合成的应答短句"""
        if self.wake_audio is None:
            print("💡 请讲（无提示音）")
            return
        self._play_gated(self.wake_audio)

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
                self.history.append({"role": "user", "content": text})
                print(f"👤 你说: {text}")

                interrupted = False
                if self.stream_tts and self.tts:
                    # === R3流式路径：LLM增量→按句TTS→边合成边播 ===
                    # 待机竞态守卫（流式在LLM前判，省一次白跑的调用）
                    if self.asr.state == 'idle':
                        print("🔇 已进入待机，放弃处理本轮请求")
                        continue
                    # 门控跨LLM+TTS全程（首句即合成即播）；与倒计时播报
                    # 整会话互斥（交叠会互相拆set_playing/unmute状态）
                    with self._gate_lock:
                        self.asr.mute()
                        try:
                            assistant_text, interrupted = \
                                self._chat_and_speak_stream()
                        finally:
                            # 被打断用短尾巴（abort已清缓冲，用户等着说话）
                            self.asr.unmute(
                                tail=MUTE_TAIL_INTERRUPTED if interrupted
                                else self.mute_tail)
                else:
                    # === 旧路径（--no-stream回退）：全量生成→全量合成→播放 ===
                    print("🧠 大模型思考中...")
                    resp = self._chat_with_tools()
                    print(f"🤖 回答: {resp.text}")
                    print(f"⏱️  耗时: {resp.elapsed:.2f}s")
                    assistant_text = resp.text
                    # 待机竞态守卫：LLM思考期间用户说了待机口令 → 不再播
                    # （结果仍记入历史，重新唤醒后问"刚才说到哪"可以接上）
                    if self.asr.state == 'idle':
                        print("🔇 已进入待机，放弃播报本轮回复")
                    elif self.tts:
                      with self._gate_lock:   # 与倒计时播报互斥（同流式路径）
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
                            # ③ 门控关：保留静音窗口覆盖输出缓冲+混响
                            tail = MUTE_TAIL_INTERRUPTED if interrupted \
                                else self.mute_tail
                            self.asr.unmute(tail=tail)

                # 历史在播报后写入：被打断记全文+标记（模型能答/续"刚才说到哪"）
                if interrupted:
                    assistant_text = assistant_text + "（被用户打断）"
                self.history.append({"role": "assistant", "content": assistant_text})
                self._trim_history()

            except Exception as e:
                print(f"❌ 处理出错: {e}")
                if self.debug and hasattr(e, '__traceback__'):
                    import traceback
                    traceback.print_exc()

    def _chat_and_speak_stream(self):
        """R3流式管线：LLM增量→按句TTS→边合成边播。返回(全文, 是否被打断)

        管线（三个线程接力，队列背压防积压）：
          worker(LLM delta→Splitter攒句) → sent_q(4) → 合成线程(TTS→16k段落)
          → seg_q(2) → play_segments迭代器 → ChunkedPlayer writer逐块写

        工具调用：流中发现tool_calls即gen.close()取消上游（实测qwen工具轮
        首块即到、content为空，不会说出口）→ 本地执行 → 下一轮流式。
        打断：interrupted标志在delta循环/合成线程/播放迭代器三处生效；
        gen.close()关闭连接 → ollama取消生成（实测0.13s生效）。
        """
        tools = TOOL_SCHEMAS if self.use_tools else None
        self.interrupted = False
        t0 = time.time()

        sent_q = queue.Queue(maxsize=4)   # 句子：LLM读取 → 合成线程
        seg_q = queue.Queue(maxsize=2)    # 16k段落：合成线程 → 播放writer
        # maxsize即背压：LLM太快→sent_q满→停止读流；合成太快→seg_q满→
        # 停止合成。播放按实时消费，稳态下各级速度自然对齐
        synth_done = threading.Event()      # 合成线程已收（迭代器结束信号）
        producer_done = threading.Event()   # 生产方不再有新句（自查退出用）
        first_seg_evt = threading.Event()   # 首段就绪：播放线程此刻才开播
        play_done_evt = threading.Event()   # 播放线程收工：worker等它返回
        play_result = {'complete': None}    # 播放线程结果（None=从没开播）
        #
        # ⚠️结束信号不用哨兵（None）：sent_q满时put哨兵会超时丢失（实测
        # 踩坑——合成线程被seg_q满堵住时无人腾空间，哨兵丢了→合成线程
        # 永不退出→迭代器永不结束→30s无进展超时误报）。改用producer_done
        # 事件：合成线程排空队列后自查退出，无任何可丢句柄

        def _synth_worker():
            """合成线程：句子→TTS→重采样16k→修剪→入段队列"""
            try:
                while True:
                    try:
                        sent = sent_q.get(timeout=0.3)
                    except queue.Empty:
                        if producer_done.is_set() or self.interrupted:
                            return
                        continue   # 生产中短暂空窗
                    if self.interrupted:
                        return
                    try:
                        result = self.tts.speak(tts_sanitize(sent), play=False,
                                                verbose=False)
                        seg = trim_silence(
                            resample_to_16k(result['samples'],
                                            result['sample_rate']),
                            SAMPLE_RATE_16K)
                        if len(seg) > 0:
                            while True:   # 带超时put：打断后无人消费要能撤
                                try:
                                    seg_q.put(seg, timeout=0.3)
                                    first_seg_evt.set()   # 通知播放线程开播
                                    break
                                except queue.Full:
                                    if self.interrupted:
                                        return
                    except Exception as e:
                        print(f"\n⚠️  句子合成失败（跳过）: {e}")
            finally:
                synth_done.set()

        def _play_thread():
            """播放线程：首段就绪才开流开播（与LLM流并发，流式期就消
            费seg_q，防下游满→上游全堵死）。无可播内容则自然退出"""
            try:
                while not first_seg_evt.wait(timeout=0.2):
                    if self.interrupted:
                        return   # 打断在首段前：不出声
                    if producer_done.is_set() and synth_done.is_set():
                        return   # 生产已收且无段落：纯工具轮/全被跳过
                self.asr.set_playing(True)
                try:
                    play_result['complete'] = self.player.play_segments(
                        _segments(), on_chunk=self.asr.feed_reference,
                        primer=PRIMER_SILENCE)
                finally:
                    self.asr.set_playing(False)
            finally:
                play_done_evt.set()

        threading.Thread(target=_synth_worker, daemon=True).start()
        threading.Thread(target=_play_thread, daemon=True).start()

        def _segments():
            """播放迭代器（play_segments的writer线程内执行）：惰性取段落

            结束条件：打断 / 合成线程已收（synth_done=句子全部排空合成完）
            """
            while True:
                try:
                    seg = seg_q.get(timeout=0.5)
                except queue.Empty:
                    if self.interrupted or synth_done.is_set():
                        return
                    continue   # 合成中：短暂空窗正常
                if self.interrupted:
                    return
                yield seg

        def _enqueue(sentences):
            """句子入合成队列（乱码句跳过不合成）；返回是否有句子入队

            put带超时+打断退出：合成线程已死/被打断时不能无限阻塞
            （实测踩坑——流式期seg_q满→合成停→sent_q满→worker卡死在
            无超时put，打断也救不回）"""
            enqueued = False
            for sent in sentences:
                if self._looks_garbled(sent):
                    print(f"\n⚠️  跳过乱码句: {sent!r}")
                    continue
                while True:
                    try:
                        sent_q.put(sent, timeout=0.3)
                        break
                    except queue.Full:
                        if self.interrupted:
                            return enqueued   # 打断：剩余句全弃
                enqueued = True
            return enqueued

        full_parts = []      # 最终轮文本（历史记录用，含未说出口的部分）
        enqueued_any = False
        interrupted = False
        splitter = SentenceSplitter()
        temp = 0.3

        for round_i in range(MAX_TOOL_ROUNDS):
            if self.interrupted:
                break
            gen = self.llm.chat_stream(self.history, tools=tools,
                                       temperature=temp)
            round_text = []
            tool_calls = None
            final_text = None
            first_delta = True
            try:
                for ev in gen:
                    if self.interrupted:
                        break
                    if "delta" in ev:
                        if first_delta:
                            first_delta = False
                            print(f"🤖 （流式{time.time()-t0:.2f}s首token）",
                                  end="", flush=True)
                        round_text.append(ev["delta"])
                        print(ev["delta"], end="", flush=True)   # 打字机效果
                        if _enqueue(splitter.feed(ev["delta"])):
                            enqueued_any = True
                    elif "tool_calls" in ev:
                        tool_calls = ev["tool_calls"]
                        break   # close取消上游 → 执行工具 → 下一轮
                    if ev.get("done"):
                        final_text = ev.get("text") or ''
            finally:
                gen.close()   # 正常/break/异常都关连接（打断→服务端取消）

            if tool_calls:
                rt = ''.join(round_text)
                # 工具决议写回历史（可回放格式，同旧路径）
                self.history.append({
                    "role": "assistant", "content": rt,
                    "tool_calls": [
                        {"id": tc["id"], "type": "function",
                         "function": {"name": tc["name"],
                                      "arguments": tc["arguments"]}}
                        for tc in tool_calls
                    ]})
                for tc in tool_calls:
                    result = self._exec_tool(tc["name"], tc["arguments"])
                    self.history.append({"role": "tool", "name": tc["name"],
                                         "content": result})
                continue   # 下一轮：带工具结果的流式回答

            final_text = final_text if final_text is not None \
                else ''.join(round_text)

            # 整响应乱码且一句都没进入合成 → 降温重试一次（旧防御的流式版；
            # 已说出口的无法撤回，只能逐句跳过保底）
            if (round_i == 0 and not self.interrupted
                    and final_text and self._looks_garbled(final_text)
                    and not enqueued_any):
                print(f"\n⚠️  响应异常，降温重试...")
                splitter = SentenceSplitter()
                temp = 0.2
                continue

            full_parts.append(final_text)
            break

        # 收尾（打断与否共用）：吐尾句 → 置生产完成 → 等播放线程收工。
        # 打断时player已被on_barge_in stop，play_segments秒回
        if not self.interrupted and _enqueue(splitter.finish()):
            enqueued_any = True
        producer_done.set()
        play_done_evt.wait()   # 有界：play_segments三态必返回/首段前自查退出

        interrupted = self.interrupted
        if interrupted:
            print(f"\n[{time.time():.3f}] ✋ 播报被打断")
        elif play_result['complete']:
            print(f"\n[{time.time():.3f}] 🔈 播放完成")
        elif play_result['complete'] is None:
            pass   # 无可播内容（纯工具轮/首段前打断）：不出声属正常
        else:
            print(f"\n[{time.time():.3f}] ⚠️ 播放异常结束")

        print(f"⏱️  流式全程 {time.time()-t0:.2f}s")
        return ''.join(full_parts), interrupted

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
            print(f"💤 {self.asr.active_timeout:.0f}秒无语音自动休眠，"
                  f"说\"{' / '.join(self.standby_phrases)}\"立即待机")
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
    parser.add_argument('--no-stream', action='store_true',
                       help='关闭R3流式管线（回退：全量生成→全量合成→播放）')

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
    parser.add_argument('--standby-phrase', type=str, action='append',
                        metavar='PHRASE',
                        help='待机口令，可多次指定多个（识别文本包含即回待机；'
                             f'默认 "进入待机模式"/"进入待机"）')

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
        stream_tts=not args.no_stream,
        wake_words=wake_words,
        wake_threshold=args.wake_threshold,
        active_timeout=args.active_timeout,
        standby_phrases=args.standby_phrase,
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
