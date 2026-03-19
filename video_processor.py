"""
视频处理模块：抠像绿幕 / 深度视频
"""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from rembg import remove, new_session

from depth_dpt import DepthEstimator

# 绿幕颜色 (BGR)
GREEN_BG = (0, 255, 0)


def _temporal_smooth_depth(depth: np.ndarray, prev: np.ndarray | None, alpha: float = 0.7) -> np.ndarray:
    """时序平滑，减轻深度图闪烁。"""
    if prev is None:
        return depth.astype(np.float32)
    return alpha * prev + (1 - alpha) * depth.astype(np.float32)


def process_video_depth(
    input_path: str | Path,
    output_path: str | Path,
    smooth_alpha: float = 0.7,
) -> None:
    """
    生成深度视频：每帧为灰度深度图，可选时序平滑。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    estimator = DepthEstimator()
    prev_depth = None
    frame_idx = 0

    print(f"正在生成深度视频: {input_path.name}")
    print(f"帧率: {fps:.1f}, 分辨率: {width}x{height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if total_frames > 0 and frame_idx % 30 == 0:
                print(f"  进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

            depth = estimator.infer(frame)
            if smooth_alpha > 0:
                prev_depth = _temporal_smooth_depth(depth, prev_depth, smooth_alpha)
                depth = np.clip(prev_depth, 0, 255).astype(np.uint8)
            else:
                prev_depth = depth.astype(np.float32)

            # 灰度图写为 3 通道以便兼容部分编码器
            depth_bgr = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            out.write(depth_bgr)
    finally:
        cap.release()
        out.release()

    print(f"深度视频已保存: {output_path}")


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "isnet-general-use",
) -> None:
    """
    视频抠像 + 绿幕合成。
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

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
    frame_idx = 0

    print(f"正在处理视频(绿幕): {input_path.name}")
    print(f"帧率: {fps:.1f}, 分辨率: {width}x{height}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if total_frames > 0 and frame_idx % 30 == 0:
                print(f"  进度: {frame_idx}/{total_frames} ({100*frame_idx/total_frames:.1f}%)")

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            result = remove(pil_img, session=session)
            result_np = np.array(result)
            alpha = result_np[:, :, 3:4]
            rgb_foreground = result_np[:, :, :3]
            green_bg = np.full_like(rgb_foreground, GREEN_BG, dtype=np.uint8)
            green_bg = cv2.cvtColor(green_bg, cv2.COLOR_BGR2RGB)
            alpha_f = alpha.astype(np.float32) / 255.0
            composited = (rgb_foreground * alpha_f + green_bg * (1 - alpha_f)).astype(np.uint8)
            composited_bgr = cv2.cvtColor(composited, cv2.COLOR_RGB2BGR)
            out.write(composited_bgr)
    finally:
        cap.release()
        out.release()

    print(f"绿幕视频已保存: {output_path}")


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
