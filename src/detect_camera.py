#!/usr/bin/env python3
"""
USB摄像头检测脚本
检测系统中的USB摄像头设备并保存详细信息到本地文件
"""
import subprocess
import json
import re
from datetime import datetime
from pathlib import Path


def run_command(cmd):
    """执行shell命令并返回输出"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip(), result.returncode
    except Exception as e:
        return str(e), -1


def detect_usb_cameras():
    """检测USB摄像头设备"""
    cameras = []

    # 1. 使用lsusb检测USB设备
    print("🔍 扫描USB摄像头设备...")
    usb_output, _ = run_command("lsusb")

    for line in usb_output.split('\n'):
        if any(keyword in line.lower() for keyword in ['camera', 'webcam', 'video']):
            # 解析lsusb输出，例如: Bus 001 Device 002: ID 0bda:5875 Realtek Semiconductor Corp. USB Camera
            match = re.search(r'Bus (\d+) Device (\d+): ID ([\da-f]{4}):([\da-f]{4}) (.+)', line)
            if match:
                cameras.append({
                    'bus': match.group(1),
                    'device': match.group(2),
                    'vendor_id': match.group(3),
                    'product_id': match.group(4),
                    'name': match.group(5).strip(),
                    'source': 'lsusb'
                })

    # 2. 检查video设备文件
    print("🔍 扫描video设备文件...")
    video_devices = []
    video_output, _ = run_command("ls -la /dev/video* 2>/dev/null")

    if video_output:
        for line in video_output.split('\n'):
            if 'video' in line:
                # 解析设备文件信息，处理可能不同的ls输出格式
                parts = line.split()
                if len(parts) >= 5:  # 至少需要基本字段
                    device_file = parts[-1]  # 最后一列是设备文件名
                    if device_file.startswith('/dev/video'):
                        device_num = device_file.replace('/dev/video', '')
                        video_devices.append({
                            'device_file': device_file,
                            'device_number': device_num,
                            'permissions': parts[0] if len(parts) > 0 else 'N/A',
                            'owner': parts[2] if len(parts) > 2 else 'N/A',
                            'group': parts[3] if len(parts) > 3 else 'N/A'
                        })

    # 3. 使用v4l2-ctl获取详细信息
    print("🔍 获取v4l2设备信息...")
    v4l2_info = {}
    v4l2_output, code = run_command("v4l2-ctl --list-devices 2>/dev/null")

    if v4l2_output and code == 0:
        current_camera = None
        for line in v4l2_output.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('/dev/video'):
                if current_camera:
                    current_camera['devices'].append(line)
            else:
                if current_camera:
                    v4l2_info[current_camera['name']] = current_camera
                current_camera = {'name': line, 'devices': [line]}

        if current_camera:
            v4l2_info[current_camera['name']] = current_camera

    return cameras, video_devices, v4l2_info


def get_camera_capabilities(device_num):
    """获取特定摄像头的能力信息"""
    capabilities = {}

    try:
        # 获取支持的格式
        cmd = f"v4l2-ctl --device=/dev/video{device_num} --list-formats-ext 2>/dev/null"
        output, code = run_command(cmd)

        if code == 0 and output:
            capabilities['formats'] = parse_formats(output)
            capabilities['device'] = f'/dev/video{device_num}'
    except Exception as e:
        capabilities['error'] = str(e)

    return capabilities


def parse_formats(v4l2_output):
    """解析v4l2格式输出"""
    formats = []
    current_format = None
    lines = v4l2_output.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # 检测格式，如: [0]: 'MJPG' (Motion-JPEG, compressed)
        format_match = re.match(r'\[(\d+)\]: \'([A-Z0-9]{4})\' (.+)', line)
        if format_match:
            if current_format:
                formats.append(current_format)

            current_format = {
                'index': format_match.group(1),
                'fourcc': format_match.group(2),
                'description': format_match.group(3).strip('()'),
                'resolutions': []
            }

        # 检测分辨率，如: Size: Discrete 1920x1080
        resolution_match = re.match(r'Size: Discrete (\d+)x(\d+)', line)
        if resolution_match and current_format:
            width, height = resolution_match.groups()

            # 检测帧率，在接下来的几行中查找
            fps = None
            j = i + 1
            while j < len(lines) and j < i + 3:  # 只检查接下来的2-3行
                fps_match = re.search(r'\(([\d.]+) fps\)', lines[j])
                if fps_match:
                    fps = float(fps_match.group(1))
                    break
                # 如果遇到新的分辨率或格式，停止查找fps
                if 'Size: Discrete' in lines[j] or re.match(r'\[\d+\]:', lines[j]):
                    break
                j += 1

            current_format['resolutions'].append({
                'width': int(width),
                'height': int(height),
                'fps': fps
            })

        i += 1

    if current_format:
        formats.append(current_format)

    return formats


def save_results(cameras, video_devices, v4l2_info, capabilities):
    """保存检测结果到文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path('./camera_detection_results')
    output_dir.mkdir(exist_ok=True)

    # 保存JSON格式
    json_file = output_dir / f'camera_info_{timestamp}.json'
    data = {
        'timestamp': timestamp,
        'usb_cameras': cameras,
        'video_devices': video_devices,
        'v4l2_info': v4l2_info,
        'capabilities': capabilities
    }

    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # 保存文本格式
    txt_file = output_dir / f'camera_info_{timestamp}.txt'
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write(f"USB摄像头检测报告\n")
        f.write(f"{'=' * 60}\n")
        f.write(f"检测时间: {timestamp}\n\n")

        f.write(f"USB摄像头设备 (lsusb):\n")
        f.write(f"{'-' * 40}\n")
        if cameras:
            for cam in cameras:
                f.write(f"  设备: {cam['name']}\n")
                f.write(f"  ID: {cam['vendor_id']}:{cam['product_id']}\n")
                f.write(f"  位置: Bus {cam['bus']}, Device {cam['device']}\n\n")
        else:
            f.write("  未检测到USB摄像头\n\n")

        f.write(f"视频设备文件:\n")
        f.write(f"{'-' * 40}\n")
        if video_devices:
            for dev in video_devices:
                f.write(f"  {dev['device_file']}\n")
                f.write(f"  权限: {dev['permissions']}\n")
                f.write(f"  所有者: {dev['owner']}:{dev['group']}\n\n")
        else:
            f.write("  未找到video设备文件\n\n")

        f.write(f"摄像头能力:\n")
        f.write(f"{'-' * 40}\n")
        for device_num, caps in capabilities.items():
            f.write(f"  设备: {device_num}\n")
            if 'formats' in caps:
                for fmt in caps['formats']:
                    f.write(f"    格式: {fmt['fourcc']} ({fmt['description']})\n")
                    for res in fmt['resolutions']:
                        fps_str = f"@ {res['fps']}fps" if res['fps'] else ""
                        f.write(f"      {res['width']}x{res['height']} {fps_str}\n")
            f.write("\n")

    return json_file, txt_file


def main():
    print("=" * 60)
    print("🔍 USB摄像头检测工具")
    print("=" * 60)
    print()

    # 检测摄像头
    cameras, video_devices, v4l2_info = detect_usb_cameras()

    # 获取每个摄像头的能力
    capabilities = {}
    for dev in video_devices:
        device_num = dev['device_number']
        print(f"📹 获取 /dev/video{device_num} 能力信息...")
        caps = get_camera_capabilities(device_num)
        if caps:
            capabilities[f'/dev/video{device_num}'] = caps

    # 保存结果
    print("💾 保存检测结果...")
    json_file, txt_file = save_results(cameras, video_devices, v4l2_info, capabilities)

    # 显示摘要
    print()
    print("✅ 检测完成!")
    print(f"📊 摘要:")
    print(f"  - 检测到 {len(cameras)} 个USB摄像头")
    print(f"  - 找到 {len(video_devices)} 个video设备")
    print(f"  - 获取了 {len(capabilities)} 个设备的能力信息")
    print()
    print(f"📁 结果已保存到:")
    print(f"  JSON: {json_file}")
    print(f"  TXT:  {txt_file}")

    # 推荐最佳配置
    if capabilities:
        print()
        print("💡 推荐配置:")
        for device, caps in capabilities.items():
            if 'formats' in caps:
                mjpeg_formats = [f for f in caps['formats'] if f['fourcc'] == 'MJPG']
                if mjpeg_formats:
                    best_resolution = max(mjpeg_formats[0]['resolutions'],
                                         key=lambda r: (r['width'] * r['height'], r['fps'] or 0))
                    print(f"  {device}: {best_resolution['width']}x{best_resolution['height']} @ {best_resolution['fps']}fps (MJPEG)")


if __name__ == "__main__":
    main()