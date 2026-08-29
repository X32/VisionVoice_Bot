#!/usr/bin/env python3
"""
USB摄像头测试程序
用法:
    python camera_test.py              # 实时预览
    python camera_test.py --capture    # 捕获单张图片
    python camera_test.py --device 1   # 使用 /dev/video1
"""
import argparse
import subprocess
import sys
import time

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️  OpenCV 未安装，将使用系统命令进行基本检测")
    print("   安装 OpenCV 以获得完整功能: pip install opencv-python")
    print("   或: sudo apt install python3-opencv")


def list_cameras():
    """列出可用的摄像头设备"""
    print("🔍 扫描摄像头设备...")
    available = []

    for i in range(4):  # 检查 /dev/video0 到 /dev/video3
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # 获取摄像头信息
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            backend = cap.getBackendName()

            print(f"   ✅ /dev/video{i}: {width}x{height} @ {fps}fps ({backend})")
            available.append(i)
            cap.release()
        else:
            print(f"   ❌ /dev/video{i}: 不可用")

    return available


def test_camera(device_id=0, resolution=(640, 480), fps=30):
    """测试摄像头实时预览"""
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"❌ 无法打开 /dev/video{device_id}")
        return False

    # 设置分辨率和帧率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"\n📹 摄像头实时预览 (/dev/video{device_id})")
    print(f"   分辨率: {actual_width}x{actual_height}")
    print(f"   帧率: {actual_fps}fps")
    print(f"   按 'q' 退出, 's' 保存截图")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法读取帧")
                break

            frame_count += 1
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0

            # 显示信息
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Res: {actual_width}x{actual_height}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Camera Test', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"camera_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"   📸 截图已保存: {filename}")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 统计: 共捕获 {frame_count} 帧, 平均 {current_fps:.1f} FPS")

    return True


def capture_single(device_id=0, resolution=(1920, 1080)):
    """捕获单张图片"""
    cap = cv2.VideoCapture(device_id)

    if not cap.isOpened():
        print(f"❌ 无法打开 /dev/video{device_id}")
        return False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    ret, frame = cap.read()
    cap.release()

    if ret:
        filename = f"camera_capture_{int(time.time())}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✅ 图片已保存: {filename} ({frame.shape[1]}x{frame.shape[0]})")
        return True
    else:
        print("❌ 捕获失败")
        return False


def test_camera_basic(device_id=0):
    """使用系统命令进行基本摄像头测试"""
    print(f"📹 基本摄像头测试 (/dev/video{device_id})")

    # 使用v4l2-ctl测试
    print("   测试摄像头访问...")
    result, code = subprocess.run(
        f"v4l2-ctl --device=/dev/video{device_id} --all",
        shell=True, capture_output=True, text=True
    )

    if code == 0:
        print("   ✅ 摄像头可访问")
        # 显示基本信息
        for line in result.stdout.split('\n'):
            if any(keyword in line for keyword in ['Width', 'Height', 'Frame Rate', 'Format']):
                print(f"   {line.strip()}")
    else:
        print(f"   ❌ 无法访问 /dev/video{device_id}")

    # 尝试捕获一帧
    print("   测试图像捕获...")
    capture_result, capture_code = subprocess.run(
        f"timeout 3 ffmpeg -f v4l2 -input_format mjpeg -i /dev/video{device_id} "
        f"-frames:v 1 /tmp/test_camera_{device_id}.jpg -y -v quiet 2>/dev/null",
        shell=True, capture_output=True, text=True
    )

    if capture_code == 0:
        print(f"   ✅ 成功捕获图像: /tmp/test_camera_{device_id}.jpg")
    else:
        print("   ❌ 图像捕获失败 (可能需要安装 ffmpeg: sudo apt install ffmpeg)")


def main():
    parser = argparse.ArgumentParser(description="USB摄像头测试程序")
    parser.add_argument("--device", "-d", type=int, default=0,
                       help="摄像头设备号 (默认: 0，即 /dev/video0)")
    parser.add_argument("--resolution", "-r", default="640x480",
                       help="分辨率 (默认: 640x480，支持: 1920x1080, 1280x720)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                       help="帧率 (默认: 30)")
    parser.add_argument("--list", "-l", action="store_true",
                       help="列出所有可用摄像头")
    parser.add_argument("--capture", "-c", action="store_true",
                       help="捕获单张图片后退出")

    args = parser.parse_args()

    print("=" * 60)
    print("📹 USB摄像头测试程序")
    print("=" * 60)

    if args.list:
        if CV2_AVAILABLE:
            list_cameras()
        else:
            print("   使用系统工具检测摄像头...")
            result, _ = subprocess.run("v4l2-ctl --list-devices", shell=True, capture_output=True, text=True)
            print(result.stdout if result.stdout else "   未检测到摄像头设备")
        return

    if not CV2_AVAILABLE:
        # 基本测试模式
        test_camera_basic(args.device)
        return

    # 解析分辨率
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"❌ 无效的分辨率格式: {args.resolution}")
        print("   格式应为: WIDTHxHEIGHT，例如: 1920x1080")
        sys.exit(1)

    if args.capture:
        capture_single(args.device, resolution)
    else:
        test_camera(args.device, resolution, args.fps)


if __name__ == "__main__":
    main()