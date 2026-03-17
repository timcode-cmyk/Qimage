"""
视频处理模块：抠像 + 深度检测 + 绿幕合成
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from depth_dpt import DepthEstimator


# 绿幕颜色 (BGR)
GREEN_BG = (0, 255, 0)


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "isnet-general-use",
    enable_depth: bool = False,
    depth_output_dir: str | Path | None = None,
) -> None:
    """
    视频抠像 + 绿幕合成，可选输出深度图。

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径（绿幕合成）
        model_name: rembg 模型名称
        enable_depth: 是否启用深度估计
        depth_output_dir: 深度图保存目录，None 则不保存
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    if depth_output_dir is not None:
        depth_output_dir = Path(depth_output_dir)
        depth_output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    session = new_session(model_name)
    depth_estimator = DepthEstimator() if (enable_depth and depth_output_dir) else None

    frame_idx = 0
    print(f"正在处理视频: {input_path.name}")
    print(f"帧率: {fps:.1f}, 分辨率: {width}x{height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if total_frames > 0 and frame_idx % 30 == 0:
                print(f"  进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

            # BGR -> RGB, 转 PIL
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)

            # rembg 去背
            result = remove(pil_img, session=session)
            result_np = np.array(result)

            # 提取 alpha 通道
            alpha = result_np[:, :, 3:4]
            rgb_foreground = result_np[:, :, :3]

            # 创建绿幕背景
            green_bg = np.full_like(rgb_foreground, GREEN_BG, dtype=np.uint8)
            green_bg = cv2.cvtColor(green_bg, cv2.COLOR_BGR2RGB)

            # 合成：前景 * alpha + 绿幕 * (1 - alpha)
            alpha_f = alpha.astype(np.float32) / 255.0
            composited = (rgb_foreground * alpha_f + green_bg * (1 - alpha_f)).astype(np.uint8)

            # 转回 BGR 并写入
            composited_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            out.write(composited_bgr)

            # 可选：保存深度图
            if enable_depth and depth_estimator is not None and depth_output_dir is not None:
                depth_map = depth_estimator.infer(frame)
                depth_path = depth_output_dir / f"depth_{frame_idx:06d}.png"
                cv2.imwrite(str(depth_path), depth_map)

    finally:
        cap.release()
        out.release()

    print(f"视频已保存: {output_path}")


def extract_video_frames_to_images(
    input_path: str | Path,
    output_dir: str | Path,
    matting_session,
    depth_estimator: DepthEstimator | None,
    save_depth: bool,
) -> None:
    """
    将视频帧提取为绿幕合成图，可选保存深度图。
    用于需要逐帧输出的场景。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        result = remove(pil_img, session=matting_session)
        result_np = np.array(result)
        alpha = result_np[:, :, 3:4]
        rgb_foreground = result_np[:, :, :3]
        green_bg = np.full((*rgb_foreground.shape[:2], 3), (0, 255, 0), dtype=np.uint8)
        alpha_f = alpha.astype(np.float32) / 255.0
        composited = (rgb_foreground * alpha_f + green_bg * (1 - alpha_f)).astype(np.uint8)
        cv2.imwrite(str(output_dir / f"frame_{frame_idx:06d}.png"), cv2.cvtColor(composited, cv2.COLOR_RGB2BGR))

        if save_depth and depth_estimator:
            depth_map = depth_estimator.infer(frame)
            cv2.imwrite(str(output_dir / f"depth_{frame_idx:06d}.png"), depth_map)

    cap.release()
