#!/usr/bin/env python3
"""
摄像头定时采集 + VLM 分析程序
每隔一段时间从摄像头采集一张图片，发送到 llama-server (Qwen2.5-VL) 进行分析

配置文件: monitor_config.json (采集间隔、prompt、服务器地址等都在里面改)

用法:
    python camera_vlm_monitor.py                      # 使用 monitor_config.json
    python camera_vlm_monitor.py --config my.json     # 指定其他配置文件
    python camera_vlm_monitor.py --interval 5         # 命令行临时覆盖采集间隔(秒)
    python camera_vlm_monitor.py --prompt "图里有什么" # 命令行临时覆盖 prompt
    python camera_vlm_monitor.py --once               # 只采集分析一次(测试用)
"""
import argparse
import base64
import json
import re
import sys
import time
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


def main():
    parser = argparse.ArgumentParser(description="摄像头定时采集 + VLM 分析")
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

    cam_cfg = config["camera"]
    cap_cfg = config["capture"]
    vlm_cfg = config["vlm"]

    output_dir = Path(cap_cfg["output_dir"])
    if cap_cfg["save_image"]:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("🤖 摄像头定时采集 + VLM 分析")
    print("=" * 60)
    print(f"📍 服务器: {vlm_cfg['url']}")
    print(f"⏱️  采集间隔: {cap_cfg['interval']}s")
    print(f"💭 Prompt: {vlm_cfg['prompt']}")
    print(f"📁 图片保存: {output_dir if cap_cfg['save_image'] else '(不保存)'}")
    print("=" * 60)

    cap = open_camera(cam_cfg)
    log_file = output_dir / f"analysis_{datetime.now().strftime('%Y%m%d')}.jsonl"
    cycle = 0
    fail_count = 0
    status_counts = {"工作": 0, "摸鱼": 0, "离开": 0}

    try:
        while True:
            cycle += 1
            loop_start = time.time()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            print(f"\n--- 第 {cycle} 次采集 [{timestamp}] ---")

            # 1. 采集
            frame = capture_frame(cap, cam_cfg["warmup_frames"])
            if frame is None:
                fail_count += 1
                print(f"❌ 采集失败 (连续第 {fail_count} 次)")
                # 连续失败 3 次则重开摄像头(应对USB松动/设备占用)
                if fail_count >= 3:
                    print("🔄 尝试重新打开摄像头...")
                    cap.release()
                    time.sleep(2)
                    try:
                        cap = open_camera(cam_cfg)
                        fail_count = 0
                    except SystemExit:
                        print("   重开失败，等待下一轮再试")
                        fail_count = 2
                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"timestamp": timestamp, "cycle": cycle,
                                        "status": None, "error": "采集失败"},
                                       ensure_ascii=False) + "\n")
            else:
                fail_count = 0
                image_path = None
                if cap_cfg["save_image"]:
                    image_path = output_dir / f"img_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), frame)
                    print(f"📸 已保存: {image_path}")

                # 2. 分析
                try:
                    data_url, side = frame_to_data_url(frame, vlm_cfg["max_side"])
                    print(f"📤 发送分析请求 (长边 {side}px)...")
                    result = analyze_with_retry(data_url, vlm_cfg["prompt"], vlm_cfg)

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    elapsed = result["_elapsed"]
                    tokens = usage.get("completion_tokens", 0)
                    tps = tokens / elapsed if elapsed > 0 else 0

                    # 解析工作状态
                    status = parse_status(content)
                    if status:
                        status_counts[status] += 1
                        emoji = STATUS_EMOJI.get(status, "")
                        print(f"\n{emoji} 状态判定: 【{status}】")
                    else:
                        print("\n⚠️  未能解析状态")

                    print(f"💬 分析结果:\n{content}")
                    print(f"⏱️  {elapsed:.2f}s | {tokens} tokens | {tps:.1f} tok/s")

                    # 3. 记录到日志
                    record = {
                        "timestamp": timestamp,
                        "cycle": cycle,
                        "status": status,
                        "image": str(image_path) if image_path else None,
                        "prompt": vlm_cfg["prompt"],
                        "response": content,
                        "elapsed": round(elapsed, 2),
                        "usage": usage,
                    }
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                except Exception as e:
                    # 任何错误都不终止监控，等待下一轮
                    print(f"❌ 分析失败: {type(e).__name__}: {e}")
                    print("   (继续下一轮采集)")
                    # 失败也记入日志，保证时间线连续可查
                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"timestamp": timestamp, "cycle": cycle,
                                            "status": None, "error": str(e)},
                                           ensure_ascii=False) + "\n")

            if args.once:
                break

            # 4. 定期清理旧图片(时间+数量双上限)
            interval_cycles = max(1, config["storage"]["cleanup_interval_cycles"])
            if cycle % interval_cycles == 0:
                try:
                    cleanup_images(output_dir, config["storage"])
                except OSError as e:
                    print(f"⚠️ 图片清理出错(不影响监控): {e}")

            # 5. 按固定间隔调度(扣除处理耗时)
            remaining = cap_cfg["interval"] - (time.time() - loop_start)
            if remaining > 0:
                print(f"😴 等待 {remaining:.1f}s 后进行下次采集...")
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n🛑 用户中断，退出")
    finally:
        cap.release()
        print_summary(status_counts, log_file)


if __name__ == "__main__":
    main()
