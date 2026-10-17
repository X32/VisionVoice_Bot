#!/usr/bin/env python3
"""
ChunkedPlayer 流式分块播放器（R2打断/R3流式共用底座）

为什么不用 sd.play/sd.wait：
  1) 无法立即停播（sd.stop与sd.wait在C层互锁，实测踩坑）
  2) 拿不到"正在播出哪一块"——AEC需要far-end参考与实际播出对齐
  3) R3流式TTS需要边生成边写

用法:
    player = ChunkedPlayer()
    complete = player.play(samples_16k, on_chunk=aec.feed_reference)
    if not complete: ...  # 被stop()打断
    player.stop()         # 任意时刻可调（其他线程），≤300ms内静音

    # 播放非16k音频（如TTS 22050输出）先重采样:
    from player import resample_to_16k
    player.play(resample_to_16k(samples, 22050))
"""
import threading
import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
BLOCK = 512          # 与 mic_loop 的 VAD_BLOCK 一致：far-end参考按块对齐


def resample_to_16k(audio: np.ndarray, from_rate: int) -> np.ndarray:
    """线性插值重采样到16k（round-trip实验已证不损检测精度）"""
    if from_rate == SAMPLE_RATE:
        return audio.astype(np.float32)
    n_out = int(len(audio) * SAMPLE_RATE / from_rate)
    return np.interp(np.linspace(0, len(audio) - 1, n_out),
                     np.arange(len(audio)), audio).astype(np.float32)


def trim_silence(audio: np.ndarray, sr: int, head_db: float = -40,
                 tail_db: float = -40, keep: float = 0.05) -> np.ndarray:
    """按能量修剪首尾静音（matcha实测头0.07~0.15s/尾0.10~0.18s，量不大；
    保留为安全网，防个别合成结果带长静音拖慢unmute恢复收听）

    Args:
        head_db/tail_db: 静音判定门限（相对峰值）
        keep: 修剪后首尾各保留的静音秒数（防爆音，听感自然）
    """
    peak = np.abs(audio).max()
    if peak < 1e-4:
        return audio
    w = max(1, int(sr * 0.01))
    env = np.convolve(np.abs(audio), np.ones(w) / w, 'same')
    th = peak * 10 ** (head_db / 20)
    idx = np.where(env > th)[0]
    if len(idx) == 0:
        return audio
    s = max(0, int(idx[0] - keep * sr))
    e = min(len(audio), int(idx[-1] + keep * sr))
    return audio[s:e]


def with_primer(audio: np.ndarray, sr: int, primer: float = 0.5) -> np.ndarray:
    """前置牺牲品静音：输出流/DAC冷启动会吃掉播放开头约0.5s（实测
    首write后音频即刻到达但时长缩水），让静音被吃而非语音本体。
    副产品：AEC收敛期（豁免窗0.6s）正好落在这段静音上，语音开始时
    打断检测已满状态。
    """
    if primer <= 0:
        return audio
    return np.concatenate([np.zeros(int(sr * primer), dtype=np.float32),
                           audio])


class ChunkedPlayer:
    """分块流式播放器：可打断、可吐far-end参考

    线程模型：play()在worker线程阻塞执行；stop()可从任意线程调用。
    停播路径用 OutputStream.abort()（丢弃缓冲立即静音），不经过
    sd.stop/sd.wait 全局互斥，规避C层死锁。
    """

    def __init__(self, samplerate: int = SAMPLE_RATE, block: int = BLOCK):
        self.samplerate = samplerate
        self.block = block
        self._stream = None
        self._stop_evt = threading.Event()
        self._lock = threading.Lock()   # 串行化play（单消费者，防御性）

    def play(self, samples: np.ndarray, on_chunk=None) -> bool:
        """播放一段音频，返回是否完整播完（False=被stop打断/写停滞）

        ⚠️ 返回有界保证（真机三次冻结的教训）：实际写循环在writer线程，
        本方法只等三态之一——写完/stop/超时(音频时长+2s)。即使writer被
        ALSA卡死在C调用的write()里，本方法也会超时返回，调用方（worker）
        得以执行unmute恢复收音，系统不会假死。被卡死的writer与流被
        daemon化放弃，下次play开新流。

        Args:
            samples: float32单声道，采样率须为self.samplerate
            on_chunk: 每块写入输出流后回调（writer线程调用）——AEC的
                far-end参考。回调时该块刚提交给输出缓冲，实际出声还有
                设备延迟，AEC3内部延迟估计器负责吸收这段偏差。
        """
        samples = np.asarray(samples, dtype=np.float32)
        with self._lock:
            self._stop_evt.clear()
            stream = sd.OutputStream(
                samplerate=self.samplerate,
                channels=1,
                dtype='float32',
                blocksize=self.block,
            )
            self._stream = stream
            stream.start()
            done_evt = threading.Event()
            state = {'complete': True}

            def _writer():
                try:
                    for i in range(0, len(samples), self.block):
                        if self._stop_evt.is_set():
                            state['complete'] = False
                            return
                        chunk = samples[i:i + self.block]
                        if len(chunk) < self.block:   # 末块补零到整块
                            chunk = np.concatenate([
                                chunk, np.zeros(self.block - len(chunk),
                                                dtype=np.float32)])
                        try:
                            stream.write(chunk.reshape(-1, 1))
                        except sd.PortAudioError:
                            # stop()的abort()会打断阻塞中的write并抛错——
                            # 这正是立即停播的机制，按被打断处理
                            state['complete'] = False
                            return
                        if on_chunk:
                            on_chunk(chunk)
                    state['complete'] = state['complete'] \
                        and not self._stop_evt.is_set()
                finally:
                    self._close_stream_async()
                    done_evt.set()

            threading.Thread(target=_writer, daemon=True).start()

            deadline = time.time() + len(samples) / self.samplerate + 2.0
            while not done_evt.wait(timeout=0.05):
                if self._stop_evt.is_set():
                    done_evt.wait(timeout=0.15)   # 给writer最后一次机会
                    return False
                if time.time() > deadline:
                    print("⚠️  播放写线程停滞（ALSA），放弃等待恢复收音")
                    return False
            return state['complete']

    def stop(self):
        """立即停止当前播放（线程安全；无播放时为no-op）

        ⚠️ 只置事件+异步abort，绝不在此线程同步调abort()——实测abort()
        在双工ALSA下会间歇性永久阻塞（同Pa_ReadStream停滞一族），
        而本方法的调用方是mic_loop线程（经on_barge_in），同步abort会
        把整个收音循环卡死（真机踩坑：stop_word打断后系统冻结）。
        异步代价：静音时刻取决于abort生效速度，极端情况播放完缓冲内
        剩余（数百ms）后自然停。
        """
        self._stop_evt.set()
        stream = self._stream
        if stream is not None:
            threading.Thread(target=self._safe_abort,
                             args=(stream,), daemon=True).start()

    @staticmethod
    def _safe_abort(stream):
        try:
            stream.abort()
        except Exception:
            pass   # 流已死/未启动：event已置位，play循环自然退出

    def _close_stream_async(self):
        """关闭输出流。fire-and-forget（ALSA的close也可能卡死，绝不join）"""
        stream = self._stream
        self._stream = None
        if stream is None:
            return

        def _close():
            try:
                stream.close()
            except Exception:
                pass
        threading.Thread(target=_close, daemon=True).start()

    def close(self):
        """实例生命周期结束（进程退出前可不调：daemon流随进程回收）"""
        self.stop()
        with self._lock:
            self._close_stream_async()


def main():
    """自测：播放正弦扫频 + 中途stop验证立即静音"""
    import time
    t = np.linspace(0, 3, SAMPLE_RATE * 3, endpoint=False)
    sweep = (0.3 * np.sin(2 * np.pi * (200 + 800 * t / 3) * t)).astype(np.float32)

    player = ChunkedPlayer()

    print("▶️ 播放3s扫频（完整）...")
    ok = player.play(sweep, on_chunk=lambda c: None)
    print(f"完整播放: {ok}")

    print("▶️ 播放3s扫频，1s后stop...")
    threading.Timer(1.0, player.stop).start()
    t0 = time.time()
    ok = player.play(sweep, on_chunk=lambda c: None)
    print(f"被打断: {not ok}，play()在stop后 {time.time()-t0-1.0:.3f}s 返回")

    print("▶️ 停止状态下stop()（应无异常no-op）...")
    player.stop()
    print("✅ 自测完成")


if __name__ == '__main__':
    main()
