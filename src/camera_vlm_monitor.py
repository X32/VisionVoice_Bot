#!/usr/bin/env python3
"""
摄像头定时采集 + VLM 分析服务（摸鱼判定服务）

引擎：每隔一段时间从摄像头采集一张图片，发送到 llama-server (Qwen2.5-VL)
分析，判定 工作/摸鱼/离开。可作独立CLI跑，也可作为HTTP+WebSocket服务跑，
供对话程序等其他进程查询状态与接收实时事件。

用法:
    python camera_vlm_monitor.py                      # 服务模式（默认，:8300）
    python camera_vlm_monitor.py --port 8301          # 换端口
    python camera_vlm_monitor.py --no-serve           # 纯CLI循环（原行为）
    python camera_vlm_monitor.py --once               # 只采集分析一次(测试用)
    python camera_vlm_monitor.py --interval 5         # 命令行临时覆盖采集间隔(秒)
    python camera_vlm_monitor.py --config my.json     # 指定其他配置文件

HTTP接口（只读）:
    GET /api/status          当前状态快照（状态/持续/今日统计/最近一条）
    GET /api/history?limit=20  最近N条分析记录（新→旧）
    GET /healthz             存活探针

WebSocket /ws（双向）:
    连接即推 {"type":"hello", 快照}；此后服务端推送:
      {"type":"analysis",      ...}   每轮分析结果
      {"type":"status_change", ...}   状态翻转（工作↔摸鱼↔离开）
      {"type":"error",         ...}   采集/分析失败
      {"type":"pong"}                 应答ping
    客户端可发:
      {"type":"capture_now"}            立即采集分析一轮
      {"type":"set_interval","seconds":30}
      {"type":"pause"} / {"type":"resume"}
      {"type":"ping"}
"""
import argparse
import asyncio
import base64
import json
import queue
import re
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import urllib.request
import urllib.error
import http.client

try:
    import cv2
except ImportError:
    print("❌ 需要安装 OpenCV: pip install opencv-python")
    sys.exit(1)


DEFAULT_CONFIG = {
    "camera": {
        "device": 0,          # /dev/video 设备号
        "width": 1280,        # 采集分辨率
        "height": 720,
        "warmup_frames": 5,   # 采集前丢弃的帧数(等自动曝光稳定)
    },
    "capture": {
        "interval": 10,       # 采集间隔(秒)
        "save_image": True,   # 是否保存图片到本地
        "output_dir": "captures",
    },
    "storage": {
        "max_age_hours": 24,           # 图片保留时长，超过自动删除
        "max_count": 2000,             # 图片总数上限，超出删最旧
        "cleanup_interval_cycles": 30, # 每 N 轮循环执行一次清理
    },
    "vlm": {
        "url": "http://192.168.4.52:8080",
        "prompt": "描述这张图片的内容",
        "max_tokens": 256,
        "temperature": 0.7,
        "timeout": 300,       # 请求超时(秒)
        "max_side": 640,      # 发送前把长边缩到此像素，控制图片 token 数
        "retries": 2,         # 请求失败重试次数
        "retry_delay": 5,     # 首次重试等待秒数(之后指数翻倍, 上限30s)
    },
    "server": {
        "host": "0.0.0.0",    # 监听地址（局域网可访问）
        "port": 8300,
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    """递归合并配置，override 优先"""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | None) -> dict:
    """加载配置文件并与默认配置合并"""
    if path is None:
        return dict(DEFAULT_CONFIG)

    config_path = Path(path)
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path.absolute()}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        user_config = json.load(f)

    return deep_merge(DEFAULT_CONFIG, user_config)


def open_camera(cam_cfg: dict):
    """打开摄像头并设置分辨率"""
    cap = cv2.VideoCapture(cam_cfg["device"])
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 /dev/video{cam_cfg['device']}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg["height"])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"📹 摄像头已打开 (/dev/video{cam_cfg['device']}), 实际分辨率: {width}x{height}")

    return cap


def capture_frame(cap, warmup_frames: int):
    """采集一帧图像，先丢弃若干帧等自动曝光稳定"""
    for _ in range(warmup_frames):
        cap.read()
    ret, frame = cap.read()
    return frame if ret else None


STATUS_EMOJI = {"工作": "💼", "摸鱼": "🐟", "离开": "🚶"}


def parse_status(content: str) -> str | None:
    """从模型回答中解析状态分类；解析失败时按关键词兜底"""
    # 优先匹配 "状态: 工作" 格式
    match = re.search(r"状态\s*[:：]\s*(工作|摸鱼|离开)", content)
    if match:
        return match.group(1)

    # 兜底：统计关键词出现次数，取最高者
    keywords = {"工作": ["代码", "编辑器", "IDE", "终端", "文档", "编程"],
                "摸鱼": ["视频", "游戏", "娱乐", "播放"],
                "离开": ["黑屏", "锁屏", "屏保"]}
    scores = {status: sum(content.count(kw) for kw in kws)
              for status, kws in keywords.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def cleanup_images(output_dir: Path, storage_cfg: dict) -> int:
    """按时间+数量双上限清理旧图片，返回删除数量；只动 img_*.jpg 不动日志"""
    max_age_hours = storage_cfg.get("max_age_hours", 24)
    max_count = storage_cfg.get("max_count", 2000)

    images = sorted(output_dir.glob("img_*.jpg"), key=lambda p: p.stat().st_mtime)
    if not images:
        return 0

    now = time.time()
    deleted = 0

    # 第一遍: 删除超过保留时长的
    fresh = []
    for img in images:
        if now - img.stat().st_mtime > max_age_hours * 3600:
            img.unlink()
            deleted += 1
        else:
            fresh.append(img)

    # 第二遍: 仍超数量上限则删最旧的
    if len(fresh) > max_count:
        for img in fresh[:len(fresh) - max_count]:
            img.unlink()
            deleted += 1

    if deleted > 0:
        print(f"🧹 清理了 {deleted} 张旧图片 (剩余 {len(list(output_dir.glob('img_*.jpg')))})")
    return deleted


def print_summary(status_counts: dict, log_file: Path):
    """打印本次监控的统计汇总"""
    total = sum(status_counts.values())
    print("\n" + "=" * 60)
    print("📊 本次监控统计")
    print("=" * 60)
    if total == 0:
        print("   (无有效分类记录)")
        return
    for status, count in status_counts.items():
        pct = count / total * 100
        bar = "█" * int(pct / 5)
        emoji = STATUS_EMOJI.get(status, "")
        print(f"   {emoji} {status}: {count} 次 ({pct:.0f}%) {bar}")
    print("=" * 60)
    print(f"📁 详细日志: {log_file}")


def frame_to_data_url(frame, max_side: int, quality: int = 85) -> tuple[str, int]:
    """把 OpenCV 帧编码成 data URL；返回 (data_url, 缩放后的长边)"""
    h, w = frame.shape[:2]
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        actual_side = max_side
    else:
        actual_side = max(w, h)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG 编码失败")

    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", actual_side


class VLMError(RuntimeError):
    """VLM 请求错误；retryable=False 表示重试也无意义(如请求格式错误)"""
    def __init__(self, msg: str, retryable: bool = True):
        super().__init__(msg)
        self.retryable = retryable


def analyze(url: str, data_url: str, prompt: str,
            max_tokens: int, temperature: float, timeout: int) -> dict:
    """调用 llama-server 的 /v1/chat/completions 接口，返回带 _elapsed 的结果"""
    payload = {
        "model": "qwen2.5-vl-3b",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text", "text": prompt},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    req = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
        except Exception:
            err_body = "(无法读取错误响应体)"
        # 4xx 是请求本身的问题，重试无意义；5xx/429 是服务端临时问题，可重试
        retryable = not (400 <= e.code < 500 and e.code != 429)
        raise VLMError(f"HTTP {e.code}: {err_body[:200]}", retryable=retryable) from e
    except (urllib.error.URLError, TimeoutError, ConnectionError,
            http.client.HTTPException, OSError) as e:
        # 覆盖: 拒绝连接/断开连接(RemoteDisconnected)/超时/DNS失败 等
        raise VLMError(f"连接服务器失败: {type(e).__name__}: {e}") from e

    result["_elapsed"] = time.time() - start
    return result


def analyze_with_retry(data_url: str, prompt: str, vlm_cfg: dict) -> dict:
    """带重试的分析请求；连接类错误自动重试，直到次数用尽"""
    retries = vlm_cfg.get("retries", 2)
    delay = vlm_cfg.get("retry_delay", 5)
    attempts = 1 + retries

    for attempt in range(1, attempts + 1):
        try:
            return analyze(vlm_cfg["url"], data_url, prompt,
                           vlm_cfg["max_tokens"], vlm_cfg["temperature"],
                           vlm_cfg["timeout"])
        except VLMError as e:
            if not e.retryable or attempt == attempts:
                raise
            print(f"   ⚠️ 请求失败({attempt}/{attempts}): {e}")
            print(f"   😴 {delay}s 后重试...")
            time.sleep(delay)
            delay = min(delay * 2, 30)  # 重试间隔指数退避，上限30s


# ============================ 监控引擎 ============================

class MonitorEngine:
    """定时采集→VLM分析→状态判定（独立线程，可接收命令）

    线程模型：run()在自己的线程执行（cv2/urllib都是阻塞调用，不进asyncio
    事件循环）；命令经command_q提交（线程安全），_wake事件打断interval
    等待使命令立即生效。on_event回调在引擎线程调用——server侧负责跨线程
    投递进asyncio（loop.call_soon_threadsafe）。
    """

    def __init__(self, config: dict, on_event=None, once: bool = False):
        self.config = config
        self.cam_cfg = config["camera"]
        self.cap_cfg = config["capture"]
        self.vlm_cfg = config["vlm"]
        self.on_event = on_event
        self.once = once

        self.command_q: queue.Queue = queue.Queue()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._paused = False
        self._lock = threading.Lock()      # 保护下面的快照状态

        self._interval = float(self.cap_cfg["interval"])
        self._status: str | None = None    # 当前判定（None=尚无有效判定）
        self._status_since = time.time()   # 当前状态开始的时刻
        self._last_change: float | None = None
        self._counts = {"工作": 0, "摸鱼": 0, "离开": 0}
        self._records: deque = deque(maxlen=100)   # 内存历史（/api/history）
        self._cycle = 0
        self._fail_count = 0               # 连续采集失败计数
        self._camera_ok = False
        self._t_start: float | None = None

        self._cap = None
        self._log_file: Path | None = None

    # ---------- 对外接口（任意线程调用） ----------

    def start(self):
        threading.Thread(target=self.run, daemon=True,
                         name="monitor-engine").start()

    def stop(self):
        self._stop.set()
        self._wake.set()

    def submit(self, cmd: dict):
        """投递命令：capture_now / set_interval / pause / resume"""
        self.command_q.put(cmd)
        self._wake.set()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self._status,
                "status_since": self._status_since,
                "last_change": self._last_change,
                "duration": time.time() - self._status_since
                            if self._status else None,
                "today": dict(self._counts),
                "cycle": self._cycle,
                "camera_ok": self._camera_ok,
                "paused": self._paused,
                "interval": self._interval,
                "uptime": time.time() - self._t_start if self._t_start else 0,
                "last_record": self._records[-1] if self._records else None,
            }

    def history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            recs = list(self._records)
        return list(reversed(recs))[:max(1, limit)]

    # ---------- 引擎主循环（引擎线程） ----------

    def run(self):
        self._t_start = time.time()
        output_dir = Path(self.cap_cfg["output_dir"])
        # 无论是否存图都要建目录：JSONL日志写在同一处
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log_file = output_dir / \
            f"analysis_{datetime.now().strftime('%Y%m%d')}.jsonl"

        self._open_camera()
        try:
            while not self._stop.is_set():
                loop_start = time.time()
                self._cycle += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                print(f"\n--- 第 {self._cycle} 次采集 [{timestamp}] ---")

                if self._cap is None:
                    self._open_camera()   # 上轮重开失败，本轮再试
                frame = self._capture_frame(timestamp) if self._cap else None

                if frame is not None:
                    self._analyze_frame(frame, timestamp, output_dir)

                if self.once or self._stop.is_set():
                    break

                self._maybe_cleanup()
                self._sleep_until_next(loop_start)
        finally:
            if self._cap is not None:
                self._cap.release()
            print_summary(self._counts, self._log_file)

    def _open_camera(self):
        """打开摄像头；失败不退出进程（服务模式要活着报错误事件）"""
        try:
            cap = open_camera(self.cam_cfg)
        except SystemExit:
            cap = None
        if cap is None:
            self._camera_ok = False
            print("⚠️ 摄像头不可用，下一轮重试")
            return
        self._cap = cap
        self._camera_ok = True
        self._fail_count = 0

    def _capture_frame(self, timestamp: str):
        frame = capture_frame(self._cap, self.cam_cfg["warmup_frames"])
        if frame is None:
            self._fail_count += 1
            print(f"❌ 采集失败 (连续第 {self._fail_count} 次)")
            if self._fail_count >= 3:
                print("🔄 尝试重新打开摄像头...")
                self._cap.release()
                self._cap = None
                time.sleep(2)
                self._open_camera()
                self._fail_count = 0 if self._cap else 2
            self._record({"timestamp": timestamp, "cycle": self._cycle,
                          "status": None, "error": "采集失败"}, emit_error="采集失败")
        return frame

    def _analyze_frame(self, frame, timestamp: str, output_dir: Path):
        image_path = None
        if self.cap_cfg["save_image"]:
            image_path = output_dir / f"img_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            print(f"📸 已保存: {image_path}")

        try:
            data_url, side = frame_to_data_url(frame, self.vlm_cfg["max_side"])
            print(f"📤 发送分析请求 (长边 {side}px)...")
            result = analyze_with_retry(data_url, self.vlm_cfg["prompt"],
                                       self.vlm_cfg)

            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            elapsed = result["_elapsed"]
            tokens = usage.get("completion_tokens", 0)
            tps = tokens / elapsed if elapsed > 0 else 0

            status = parse_status(content)
            emoji = STATUS_EMOJI.get(status, "") if status else ""
            print(f"\n{emoji} 状态判定: 【{status}】" if status
                  else "\n⚠️  未能解析状态")
            print(f"💬 分析结果:\n{content}")
            print(f"⏱️  {elapsed:.2f}s | {tokens} tokens | {tps:.1f} tok/s")

            record = {
                "timestamp": timestamp,
                "cycle": self._cycle,
                "status": status,
                "image": str(image_path) if image_path else None,
                "prompt": self.vlm_cfg["prompt"],
                "response": content,
                "elapsed": round(elapsed, 2),
                "usage": usage,
            }
            self._record(record, new_status=status)
        except Exception as e:
            print(f"❌ 分析失败: {type(e).__name__}: {e}")
            print("   (继续下一轮采集)")
            self._record({"timestamp": timestamp, "cycle": self._cycle,
                          "status": None, "error": str(e)}, emit_error=str(e))

    def _record(self, record: dict, new_status: str | None = None,
                emit_error: str | None = None):
        """记录入内存+JSONL日志，按需发事件（引擎线程调用）"""
        if emit_error is None:
            record["ts"] = time.time()
        with self._lock:
            self._records.append(record)
            if record.get("status"):
                self._counts[record["status"]] += 1
            prev = self._status
            # 翻转=有旧状态且变了（首次判定不算，hello快照已覆盖）
            flip = (new_status is not None and prev is not None
                    and new_status != prev)
            if new_status and new_status != prev:
                self._last_change = time.time()
                self._status_since = self._last_change
                self._status = new_status
        if self._log_file is not None:
            try:
                with open(self._log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as e:
                print(f"⚠️ 日志写入失败(不影响监控): {e}")
        now = record.get("ts", time.time())
        if emit_error is not None:
            self._emit({"type": "error", "ts": now,
                        "cycle": self._cycle, "error": emit_error})
            return
        self._emit({"type": "analysis", "ts": now,
                    "cycle": self._cycle, "status": record.get("status"),
                    "response": record.get("response", "")[:500],
                    "elapsed": record.get("elapsed")})
        if flip:
            self._emit({"type": "status_change", "ts": now,
                        "from": prev, "to": new_status})

    def _emit(self, msg: dict):
        if self.on_event is not None:
            try:
                self.on_event(msg)
            except Exception as e:
                print(f"⚠️ 事件回调出错: {e}")

    def _maybe_cleanup(self):
        interval_cycles = max(1, self.config["storage"]["cleanup_interval_cycles"])
        if self._cycle % interval_cycles == 0:
            try:
                cleanup_images(Path(self.cap_cfg["output_dir"]),
                               self.config["storage"])
            except OSError as e:
                print(f"⚠️ 图片清理出错(不影响监控): {e}")

    def _sleep_until_next(self, loop_start: float):
        """等到下轮时刻；期间消费命令（capture_now立即返回）

        暂停时只等命令不长睡（resume命令能立刻生效）；resume后下轮
        锚点重定为当前时刻（否则会补等暂停前剩余的旧间隔，语义怪异）。
        """
        was_paused = self._paused
        while not self._stop.is_set():
            if self._paused:
                was_paused = True
                self._wake.wait(timeout=0.5)
                self._wake.clear()
                self._drain_commands()
                continue
            if was_paused:   # 刚恢复：锚点重定为现在，间隔从头计
                loop_start = time.time()
                was_paused = False
            remain = loop_start + self._interval - time.time()
            if remain <= 0:
                self._drain_commands()
                return
            self._wake.wait(timeout=min(remain, 0.5))
            self._wake.clear()
            if self._drain_commands() == "capture_now":
                return

    def _drain_commands(self) -> str | None:
        """应用所有待处理命令；返回'capture_now'表示要求立即跑一轮"""
        acted = None
        while True:
            try:
                cmd = self.command_q.get_nowait()
            except queue.Empty:
                return acted
            ctype = cmd.get("type")
            if ctype == "capture_now":
                print("⚡ 命令: 立即采集")
                acted = "capture_now"
            elif ctype == "set_interval":
                sec = max(1.0, float(cmd.get("seconds", self._interval)))
                with self._lock:
                    self._interval = sec
                print(f"⚡ 命令: 采集间隔 → {sec}s")
            elif ctype == "pause":
                self._paused = True
                print("⚡ 命令: 暂停采集")
            elif ctype == "resume":
                self._paused = False
                print("⚡ 命令: 恢复采集")
            elif ctype is not None:
                print(f"⚡ 未知命令: {cmd}")


# ============================ HTTP/WS 服务 ============================

def build_app(engine: MonitorEngine):
    """构建aiohttp应用：REST只读接口 + /ws双向通道

    跨线程事件投递：引擎线程on_event → loop.call_soon_threadsafe把消息
    放进每个客户端的asyncio.Queue（由各自的sender任务发出去）。
    """
    from aiohttp import web, WSMsgType

    app = web.Application()
    app["clients"] = set()          # 各ws客户端的asyncio.Queue

    def _dispatch(msg: dict):
        for q in list(app["clients"]):
            q.put_nowait(msg)

    def _emit(msg: dict):           # 引擎线程调用
        loop = app["loop"]
        loop.call_soon_threadsafe(_dispatch, msg)

    async def _on_startup(app):
        app["loop"] = asyncio.get_running_loop()

    async def _status(request):
        return web.json_response(engine.snapshot())

    async def _history(request):
        limit = int(request.query.get("limit", 20))
        return web.json_response(engine.history(limit))

    async def _healthz(request):
        return web.json_response({"ok": True})

    async def _index(request):
        return web.json_response({
            "service": "camera-vlm-monitor",
            "endpoints": ["GET /api/status", "GET /api/history?limit=20",
                          "GET /healthz", "GET /ws"],
        })

    async def _sender(ws, q: asyncio.Queue):
        """把队列里的引擎事件推给客户端（独立任务，与读循环并发）"""
        try:
            while True:
                msg = await q.get()
                await ws.send_json(msg)
        except asyncio.CancelledError:
            pass

    async def _ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=30)
        await ws.prepare(request)
        q: asyncio.Queue = asyncio.Queue()
        app["clients"].add(q)
        sender = asyncio.create_task(_sender(ws, q))
        peer = request.remote
        print(f"🔗 WS客户端接入: {peer} (共{len(app['clients'])})")
        try:
            await ws.send_json({"type": "hello", **engine.snapshot()})
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    continue
                try:
                    cmd = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "error": "非法JSON"})
                    continue
                ctype = cmd.get("type")
                if ctype == "ping":
                    await ws.send_json({"type": "pong"})
                elif ctype in ("capture_now", "set_interval", "pause", "resume"):
                    engine.submit(cmd)
                else:
                    await ws.send_json({"type": "error",
                                        "error": f"未知命令: {ctype}"})
        finally:
            app["clients"].discard(q)
            sender.cancel()
            print(f"🔌 WS客户端断开: {peer} (剩{len(app['clients'])})")
        return ws

    app.on_startup.append(_on_startup)
    app.router.add_get("/", _index)
    app.router.add_get("/api/status", _status)
    app.router.add_get("/api/history", _history)
    app.router.add_get("/healthz", _healthz)
    app.router.add_get("/ws", _ws_handler)
    app["engine"] = engine
    app["emit"] = _emit
    return app


def main():
    parser = argparse.ArgumentParser(description="摄像头定时采集 + VLM 分析服务")
    parser.add_argument("--config", "-c", default="monitor_config.json",
                        help="配置文件路径 (默认: monitor_config.json)")
    parser.add_argument("--interval", "-i", type=float, default=None,
                        help="覆盖配置的采集间隔(秒)")
    parser.add_argument("--prompt", "-p", default=None,
                        help="覆盖配置的分析 prompt")
    parser.add_argument("--url", "-u", default=None,
                        help="覆盖配置的服务器地址")
    parser.add_argument("--device", "-d", type=int, default=None,
                        help="覆盖配置的摄像头设备号")
    parser.add_argument("--once", action="store_true",
                        help="只采集分析一次后退出(测试用)")
    parser.add_argument("--no-serve", action="store_true",
                        help="不起HTTP/WS服务，纯CLI循环(原行为)")
    parser.add_argument("--host", default=None, help="覆盖监听地址")
    parser.add_argument("--port", type=int, default=None, help="覆盖监听端口")
    args = parser.parse_args()

    config = load_config(args.config)

    # 命令行参数覆盖配置文件
    if args.interval is not None:
        config["capture"]["interval"] = args.interval
    if args.prompt is not None:
        config["vlm"]["prompt"] = args.prompt
    if args.url is not None:
        config["vlm"]["url"] = args.url
    if args.device is not None:
        config["camera"]["device"] = args.device
    if args.host is not None:
        config["server"]["host"] = args.host
    if args.port is not None:
        config["server"]["port"] = args.port

    serve = not args.no_serve and not args.once
    engine = MonitorEngine(config, once=args.once)

    if not serve:
        engine.run()          # 纯CLI：主线程直接跑（原行为）
        return

    # 服务模式：引擎线程 + aiohttp主线程
    try:
        from aiohttp import web
    except ImportError:
        print("❌ 服务模式需要 aiohttp: pip install aiohttp")
        print("   （或用 --no-serve 跑纯CLI模式）")
        sys.exit(1)

    app = build_app(engine)
    engine.on_event = app["emit"]
    engine.start()

    srv_cfg = config["server"]
    print("=" * 60)
    print("🤖 摄像头定时采集 + VLM 分析服务")
    print("=" * 60)
    print(f"📍 VLM服务器: {config['vlm']['url']}")
    print(f"⏱️  采集间隔: {config['capture']['interval']}s")
    print(f"🌐 HTTP/WS: http://{srv_cfg['host']}:{srv_cfg['port']}"
          f" (/api/status /api/history /ws)")
    print("=" * 60)
    try:
        web.run_app(app, host=srv_cfg["host"], port=srv_cfg["port"],
                    print=None)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()


if __name__ == "__main__":
    main()
