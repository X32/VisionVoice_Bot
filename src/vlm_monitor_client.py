#!/usr/bin/env python3
"""
摸鱼判定服务客户端（供对话程序等集成方使用）

对应 camera_vlm_monitor.py 的服务模式：REST查快照 + WebSocket收实时事件
（analysis/status_change/error）+ 发命令（capture_now/set_interval/pause/
resume）。WS在独立线程里跑（aiohttp客户端），断线自动重连（服务可能比
本程序晚启动或中途重启），事件经on_event回调送出（客户端线程调用）。

用法:
    from vlm_monitor_client import MonitorClient

    def on_event(ev):
        if ev["type"] == "status_change" and ev["to"] == "摸鱼":
            ...  # 提醒逻辑（R4：接入对话/播报）

    client = MonitorClient("ws://127.0.0.1:8300/ws", on_event=on_event)
    client.start()
    client.get_status()                # REST快照（dict）
    client.send_command("capture_now")
    client.set_interval(30)
    client.stop()

依赖: aiohttp（与服务端同一依赖）
"""
import asyncio
import json
import threading
import time
import urllib.request

RECONNECT_DELAY = 3.0   # 断线重连间隔（秒）


class MonitorClient:
    """camera_vlm_monitor服务的线程化客户端"""

    def __init__(self, ws_url: str = "ws://127.0.0.1:8300/ws",
                 on_event=None, rest_base: str = "http://127.0.0.1:8300",
                 name: str = "monitor-client"):
        self.ws_url = ws_url
        self.rest_base = rest_base.rstrip('/')
        self.on_event = on_event
        self.name = name
        self.connected = False

        self._loop = None
        self._ws = None
        self._stop = threading.Event()
        self._thread = None

    # ---------- 生命周期 ----------

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name=self.name)
        self._thread.start()

    def stop(self):
        self._stop.set()
        # 唤醒阻塞中的WS会话（置stop_fut→会话收尾→线程退出）
        waker, loop = getattr(self, '_stop_waker', None), self._loop
        if waker is not None and loop is not None:
            loop.call_soon_threadsafe(waker)
        if self._thread is not None:
            self._thread.join(timeout=3)

    # ---------- 对外方法（任意线程） ----------

    def get_status(self, timeout: float = 3.0) -> dict:
        """REST快照：当前状态/持续时长/今日统计/最近记录"""
        with urllib.request.urlopen(f"{self.rest_base}/api/status",
                                    timeout=timeout) as resp:
            return json.loads(resp.read().decode('utf-8'))

    def send_command(self, ctype: str, **params):
        """发WS命令：capture_now / set_interval / pause / resume"""
        self._post_ws({"type": ctype, **params})

    def set_interval(self, seconds: float):
        self.send_command("set_interval", seconds=seconds)

    # ---------- 内部：WS线程 ----------

    def _post_ws(self, msg: dict):
        loop, ws = self._loop, self._ws
        if loop is None or ws is None or not self.connected:
            print(f"[{self.name}] ⚠️ 未连接，命令丢弃: {msg}")
            return
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(ws.send_json(msg)))

    def _run(self):
        """线程体：自带事件循环，断线重连"""
        while not self._stop.is_set():
            self._loop = asyncio.new_event_loop()
            try:
                self._loop.run_until_complete(self._ws_session())
            except Exception as e:
                print(f"[{self.name}] ⚠️ WS会话异常: {type(e).__name__}: {e}")
            finally:
                self.connected = False
                self._loop.close()
                self._loop = None
            if not self._stop.wait(RECONNECT_DELAY):
                continue   # 未收到停止：重连
            break

    async def _ws_session(self):
        import aiohttp

        stop_fut = self._loop.create_future()

        def _stop_from_other_thread():
            if not stop_fut.done():
                stop_fut.set_result(None)

        # 会话一开始就登记唤醒器（stop()从其他线程经它结束会话）
        self._stop_waker = _stop_from_other_thread
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(self.ws_url,
                                          heartbeat=25) as ws:
                self._ws = ws
                self.connected = True
                print(f"[{self.name}] ✅ 已连接 {self.ws_url}")

                recv_task = asyncio.ensure_future(self._recv_loop(ws))
                stop_task = asyncio.ensure_future(stop_fut)
                try:
                    await asyncio.wait([recv_task, stop_task],
                                       return_when=asyncio.FIRST_COMPLETED)
                finally:
                    for t in (recv_task, stop_task):
                        t.cancel()
                    self._ws = None
                    self.connected = False
                    print(f"[{self.name}] 🔌 断开，"
                          f"{RECONNECT_DELAY:.0f}s后重连")

    async def _recv_loop(self, ws):
        import aiohttp
        async for msg in ws:
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            try:
                ev = json.loads(msg.data)
            except json.JSONDecodeError:
                continue
            if self.on_event is not None:
                try:
                    self.on_event(ev)
                except Exception as e:
                    print(f"[{self.name}] ⚠️ on_event回调出错: {e}")


def main():
    """演示：连服务，收10秒事件，中途下一条capture_now命令"""
    def on_event(ev):
        t = ev.get("type")
        if t == "status_change":
            print(f"  🎯 状态翻转: {ev['from']} → {ev['to']}")
        elif t == "analysis":
            print(f"  📊 分析: {ev.get('status')} ({ev.get('elapsed')}s)")
        elif t == "error":
            print(f"  ⚠️ 错误: {ev.get('error')}")

    c = MonitorClient(on_event=on_event)
    c.start()
    try:
        time.sleep(4)
        print("… get_status():", json.dumps(c.get_status(),
                                            ensure_ascii=False)[:200])
        print("… 发送 capture_now")
        c.send_command("capture_now")
        time.sleep(6)
    finally:
        c.stop()
    print("演示结束")


if __name__ == "__main__":
    main()
